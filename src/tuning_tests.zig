//! Contract tests for the tuning table. The cached `get()` view and the
//! process environment are shared state, so the env-arm tests read through
//! the uncached `load()` around a save/set/restore window (libc targets
//! only), and the pin tests assert against the latched base instead of a
//! fixed default.

const std = @import("std");
const builtin = @import("builtin");
const tuning = @import("tuning.zig");

// std.c carries no setenv/unsetenv in Zig 0.16; the env-arm tests declare
// them directly and run on libc targets only.
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "env-name derivation reproduces the published spellings" {
    const cases = [_]struct { path: []const u8, name: []const u8 }{
        .{ .path = "winograd", .name = "FUCINA_WINOGRAD" },
        .{ .path = "winograd_f4", .name = "FUCINA_WINOGRAD_F4" },
        .{ .path = "norm_quant_fused", .name = "FUCINA_NORM_QUANT_FUSED" },
        .{ .path = "decode_compact", .name = "FUCINA_DECODE_COMPACT" },
        .{ .path = "conv_bwd_gemm", .name = "FUCINA_CONV_BWD_GEMM" },
        .{ .path = "attn_bwd_blas", .name = "FUCINA_ATTN_BWD_BLAS" },
        .{ .path = "attn_bwd_stats", .name = "FUCINA_ATTN_BWD_STATS" },
        .{ .path = "fused_distill", .name = "FUCINA_FUSED_DISTILL" },
        .{ .path = "cpu_f32_shadow", .name = "FUCINA_CPU_F32_SHADOW" },
        .{ .path = "ptqtp_fold", .name = "FUCINA_PTQTP_FOLD" },
        .{ .path = "moe_lru", .name = "FUCINA_MOE_LRU" },
        .{ .path = "moe_l2_cached", .name = "FUCINA_MOE_L2_CACHED" },
        .{ .path = "cpu_f32_shadow_min_m", .name = "FUCINA_CPU_F32_SHADOW_MIN_M" },
        .{ .path = "winograd_f4_min", .name = "FUCINA_WINOGRAD_F4_MIN" },
        .{ .path = "winograd_f4_maxcin", .name = "FUCINA_WINOGRAD_F4_MAXCIN" },
        .{ .path = "spin_budget", .name = "FUCINA_SPIN_BUDGET" },
        .{ .path = "pool_profile", .name = "FUCINA_POOL_PROFILE" },
        .{ .path = "gpu.enabled", .name = "FUCINA_GPU" },
        .{ .path = "gpu.decode", .name = "FUCINA_GPU_DECODE" },
        .{ .path = "gpu.tf32", .name = "FUCINA_GPU_TF32" },
        .{ .path = "gpu.trace", .name = "FUCINA_GPU_TRACE" },
        .{ .path = "gpu.quant_mma", .name = "FUCINA_GPU_QUANT_MMA" },
        .{ .path = "gpu.quant_split_k", .name = "FUCINA_GPU_QUANT_SPLIT_K" },
        .{ .path = "gpu.vram_budget", .name = "FUCINA_GPU_VRAM_BUDGET" },
        .{ .path = "gpu.qmoe_min_fill", .name = "FUCINA_GPU_QMOE_MIN_FILL" },
        .{ .path = "gpu.transient_min_m", .name = "FUCINA_GPU_TRANSIENT_MIN_M" },
        .{ .path = "gpu.min_work.base", .name = "FUCINA_GPU_MIN_WORK" },
        .{ .path = "gpu.min_work.resident", .name = "FUCINA_GPU_MIN_WORK_RESIDENT" },
        .{ .path = "gpu.min_work.f16", .name = "FUCINA_GPU_MIN_WORK_F16" },
        .{ .path = "gpu.min_work.f16_resident", .name = "FUCINA_GPU_MIN_WORK_F16_RESIDENT" },
        .{ .path = "gpu.min_work.16bit_resident", .name = "FUCINA_GPU_MIN_WORK_16BIT_RESIDENT" },
        .{ .path = "gpu.min_work.gemv", .name = "FUCINA_GPU_MIN_WORK_GEMV" },
        .{ .path = "gpu.min_work.attn", .name = "FUCINA_GPU_MIN_WORK_ATTN" },
        .{ .path = "gpu.min_work.qmoe", .name = "FUCINA_GPU_MIN_WORK_QMOE" },
        .{ .path = "gpu.min_work.transient", .name = "FUCINA_GPU_MIN_WORK_TRANSIENT" },
        .{ .path = "gpu.min_work.decode_q5", .name = "FUCINA_GPU_MIN_WORK_DECODE_Q5" },
        .{ .path = "gpu.min_work.dense_q4", .name = "FUCINA_GPU_MIN_WORK_DENSE_Q4" },
        .{ .path = "gpu.min_work.dense_q5", .name = "FUCINA_GPU_MIN_WORK_DENSE_Q5" },
        .{ .path = "gpu.min_work.dense_q6", .name = "FUCINA_GPU_MIN_WORK_DENSE_Q6" },
        .{ .path = "gpu.min_work.dense_q6_packed", .name = "FUCINA_GPU_MIN_WORK_DENSE_Q6_PACKED" },
        .{ .path = "gpu.min_work.dense_q8", .name = "FUCINA_GPU_MIN_WORK_DENSE_Q8" },
        .{ .path = "gpu.min_work.dense_tq2", .name = "FUCINA_GPU_MIN_WORK_DENSE_TQ2" },
    };
    inline for (cases) |case| {
        try std.testing.expectEqualStrings(case.name, tuning.envNameOf(case.path));
    }
    // A new leaf cannot dodge the golden list above.
    try std.testing.expectEqual(cases.len, comptime countLeaves(tuning.Table));
}

fn countLeaves(comptime T: type) usize {
    var n: usize = 0;
    for (@typeInfo(T).@"struct".fields) |f| {
        switch (@typeInfo(f.type)) {
            .@"struct" => n += countLeaves(f.type),
            else => n += 1,
        }
    }
    return n;
}

/// Save/set/restore for one env variable, so the env-arm tests leave the
/// process environment exactly as they found it (the outer run may pin
/// gates, e.g. the FUCINA_WINOGRAD=0 parity leg).
const EnvGuard = struct {
    name: [:0]const u8,
    saved: ?[256]u8 = null,
    saved_len: usize = 0,

    fn begin(name: [:0]const u8) EnvGuard {
        var self: EnvGuard = .{ .name = name };
        if (std.c.getenv(name.ptr)) |value| {
            const v = std.mem.sliceTo(value, 0);
            if (v.len < 256) {
                var buf: [256]u8 = undefined;
                @memcpy(buf[0..v.len], v);
                buf[v.len] = 0;
                self.saved = buf;
                self.saved_len = v.len;
            }
        }
        return self;
    }

    fn put(self: *const EnvGuard, value: [:0]const u8) void {
        _ = setenv(self.name.ptr, value.ptr, 1);
    }

    fn clear(self: *const EnvGuard) void {
        _ = unsetenv(self.name.ptr);
    }

    fn restore(self: *const EnvGuard) void {
        if (self.saved) |buf| {
            _ = setenv(self.name.ptr, buf[0..self.saved_len :0].ptr, 1);
        } else {
            _ = unsetenv(self.name.ptr);
        }
    }
};

test "boolean leaf: env on, env off, unset default" {
    if (comptime !builtin.link_libc) return error.SkipZigTest;
    const guard = EnvGuard.begin("FUCINA_CONV_BWD_GEMM");
    defer guard.restore();

    guard.put("1");
    try std.testing.expect(tuning.load().conv_bwd_gemm);
    guard.put("0");
    try std.testing.expect(!tuning.load().conv_bwd_gemm);
    guard.clear();
    try std.testing.expect(tuning.load().conv_bwd_gemm); // measured default
}

test "threshold leaf: parse, zero-is-unset, garbage-is-unset" {
    if (comptime !builtin.link_libc) return error.SkipZigTest;
    const guard = EnvGuard.begin("FUCINA_CPU_F32_SHADOW_MIN_M");
    defer guard.restore();

    guard.put("48");
    try std.testing.expectEqual(@as(u64, 48), tuning.load().cpu_f32_shadow_min_m);
    guard.put("0"); // positive-parse leaf: zero keeps the default
    try std.testing.expectEqual(@as(u64, 32), tuning.load().cpu_f32_shadow_min_m);
    guard.put("abc");
    try std.testing.expectEqual(@as(u64, 32), tuning.load().cpu_f32_shadow_min_m);
    guard.clear();
    try std.testing.expectEqual(@as(u64, 32), tuning.load().cpu_f32_shadow_min_m);
}

test "gpu leaf: zero is a meaningful value" {
    if (comptime !builtin.link_libc) return error.SkipZigTest;
    const guard = EnvGuard.begin("FUCINA_GPU_QMOE_MIN_FILL");
    defer guard.restore();

    guard.put("0"); // non-negative-parse leaf: occupancy-blind
    try std.testing.expectEqual(@as(u64, 0), tuning.load().gpu.qmoe_min_fill);
    guard.put("75");
    try std.testing.expectEqual(@as(u64, 75), tuning.load().gpu.qmoe_min_fill);
    guard.clear();
    try std.testing.expectEqual(@as(u64, 50), tuning.load().gpu.qmoe_min_fill);
}

test "setField pins over the latched base and null re-arms" {
    const before = tuning.get().decode_compact;
    tuning.setField("decode_compact", !before);
    try std.testing.expect(tuning.get().decode_compact == !before);
    tuning.setField("decode_compact", null);
    try std.testing.expect(tuning.get().decode_compact == before);

    const min_m = tuning.get().cpu_f32_shadow_min_m;
    tuning.setField("cpu_f32_shadow_min_m", min_m + 8);
    try std.testing.expectEqual(min_m + 8, tuning.get().cpu_f32_shadow_min_m);
    tuning.setField("cpu_f32_shadow_min_m", null);
    try std.testing.expectEqual(min_m, tuning.get().cpu_f32_shadow_min_m);
}

test "set replaces the whole pin set" {
    const shadow_before = tuning.get().cpu_f32_shadow;
    const base_before = tuning.get().gpu.min_work.base;
    tuning.set(.{ .cpu_f32_shadow = !shadow_before, .gpu = .{ .min_work = .{ .base = base_before + 1 } } });
    defer tuning.set(.{});
    try std.testing.expect(tuning.get().cpu_f32_shadow == !shadow_before);
    try std.testing.expectEqual(base_before + 1, tuning.get().gpu.min_work.base);
    tuning.set(.{});
    try std.testing.expect(tuning.get().cpu_f32_shadow == shadow_before);
    try std.testing.expectEqual(base_before, tuning.get().gpu.min_work.base);
}

test "resolve: per-context override wins, null follows the process value" {
    var overrides: tuning.Overrides = .{};
    try std.testing.expectEqual(
        tuning.get().cpu_f32_shadow_min_m,
        tuning.resolve(&overrides, "cpu_f32_shadow_min_m"),
    );
    overrides.cpu_f32_shadow = true;
    overrides.cpu_f32_shadow_min_m = 7;
    overrides.gpu.min_work.base = 123;
    try std.testing.expect(tuning.resolve(&overrides, "cpu_f32_shadow"));
    try std.testing.expectEqual(@as(u64, 7), tuning.resolve(&overrides, "cpu_f32_shadow_min_m"));
    try std.testing.expectEqual(@as(u64, 123), tuning.resolve(&overrides, "gpu.min_work.base"));
}
