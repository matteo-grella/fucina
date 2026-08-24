//! Provider-independent GPU parity tests: the behaviour every `-Dgpu`
//! provider must exhibit identically, written ONCE against the selected
//! provider (`gpu.zig`'s `impl`) instead of copied per provider. Metal runs
//! them on a `-Dgpu=metal` build, CUDA on a `-Dgpu=cuda` build (and compiles
//! them under `zig build cuda-check`); a new provider inherits the suite by
//! conforming to `gpu_provider.zig`.
//!
//! Provider-SPECIFIC behaviour stays in the provider file: format arms only
//! one side implements, residency/VRAM policy, decode gates, and each
//! provider's own async handoff semantics (Metal reads shared memory, CUDA
//! passes a device pointer). This file holds only what must agree.
//!
//! Every test skips when the provider is not selected or no device is
//! present, so the suite is inert on CPU-only builds and machines.

const std = @import("std");
const gpu = @import("gpu.zig");
const tensor = @import("../tensor.zig");
const test_util = @import("gpu_test_util.zig");

const impl = gpu.impl;
const Orient = gpu.provider.Orient;
const QMMTile = gpu.provider.QMMTile;
const Tensor = tensor.Tensor;
const TensorF16 = tensor.TensorOf(.f16);

const cpuReference = test_util.cpuReference;
const buildQuantWeights = test_util.buildQuantWeights;
const expectQuantGemmRows = test_util.expectQuantGemmRows;

/// Shared skip gate: provider not compiled in, or compiled in with no device.
fn requireDevice() !void {
    if (comptime !impl.enabled) return error.SkipZigTest;
    if (!impl.deviceAvailableForTest()) return error.SkipZigTest;
}

test "gpu conformance: gemm f32 parity vs reference (all orientations, edge tiles)" {
    try requireDevice();
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(3);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    const cases = [_]Case{
        .{ .m = 64, .n = 64, .k = 64 }, // fully aligned
        .{ .m = 33, .n = 47, .k = 17 }, // every edge path
        .{ .m = 128, .n = 96, .k = 33 }, // K tail only
        .{ .m = 65, .n = 64, .k = 48 }, // M edge only
    };
    for (cases) |case| {
        const m = case.m;
        const n = case.n;
        const k = case.k;
        const a = try allocator.alloc(f32, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, k * n);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, m * n);
        defer allocator.free(c);
        const expected = try allocator.alloc(f32, m * n);
        defer allocator.free(expected);

        for ([_]Orient{ .plain, .trans_a, .trans_b }) |orient| {
            for (a) |*x| x.* = random.floatNorm(f32);
            for (b) |*x| x.* = random.floatNorm(f32);
            @memset(c, std.math.nan(f32));
            try std.testing.expect(impl.gemmF32(orient, a, b, c, m, n, k));
            cpuReference(orient, a, b, expected, m, n, k);
            // Strict FP32 on both providers (cuBLAS default math mode, MLX
            // steel f32): one tolerance covers them.
            for (c, expected) |got, want| {
                const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "gpu conformance: gemm f32 batched matches per-matrix reference" {
    try requireDevice();
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(5);
    const random = prng.random();
    const m = 48;
    const n = 40;
    const k = 32;
    const batch = 3;

    const a = try allocator.alloc(f32, batch * m * k);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, batch * k * n);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, batch * m * n);
    defer allocator.free(c);
    const expected = try allocator.alloc(f32, batch * m * n);
    defer allocator.free(expected);
    for (a) |*x| x.* = random.floatNorm(f32);
    for (b) |*x| x.* = random.floatNorm(f32);
    @memset(c, 0);

    try std.testing.expect(impl.gemmBatchedF32(.plain, a, b, c, m, n, k, batch, m * k, k * n, m * n));
    for (0..batch) |bi| {
        cpuReference(.plain, a[bi * m * k ..][0 .. m * k], b[bi * k * n ..][0 .. k * n], expected[bi * m * n ..][0 .. m * n], m, n, k);
    }
    for (c, expected) |got, want| {
        const tol = @max(2e-5 * @max(@abs(want), @abs(got)), 2e-5);
        try std.testing.expect(@abs(got - want) <= tol);
    }
}

test "gpu conformance: gemm f16 NT parity vs f64 reference (f16-rounded output)" {
    try requireDevice();
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(9);
    const random = prng.random();

    const Case = struct { m: usize, n: usize, k: usize };
    const cases = [_]Case{
        .{ .m = 64, .n = 64, .k = 64 },
        .{ .m = 33, .n = 47, .k = 17 },
        .{ .m = 65, .n = 96, .k = 33 },
    };
    for (cases) |case| {
        const m = case.m;
        const n = case.n;
        const k = case.k;
        const a = try allocator.alloc(f16, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f16, n * k);
        defer allocator.free(b);
        for (a) |*x| x.* = @floatCast(random.floatNorm(f32));
        for (b) |*x| x.* = @floatCast(random.floatNorm(f32));

        impl.f16_lock.lock();
        defer impl.f16_lock.unlock();
        // transient test buffer: must not enter the wrap cache
        const staging = impl.gemmF16Nt(a, b, m, n, k, false) orelse return error.TestUnexpectedResult;
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0;
                for (0..k) |p| acc += @as(f64, a[i * k + p]) * @as(f64, b[j * k + p]);
                const got: f32 = staging[i * n + j];
                const want: f32 = @floatCast(acc);
                // f32 accumulate, f16-rounded store: tolerance = one f16 ulp
                // of the result magnitude plus accumulation slack.
                const tol = @max(2e-3 * @max(@abs(want), @abs(got)), 2e-3);
                try std.testing.expect(@abs(got - want) <= tol);
            }
        }
    }
}

test "gpu conformance: quant gemm grouped expert tiles parity" {
    try requireDevice();
    const allocator = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(23);
    const random = prng.random();

    // Every quantized format THIS provider declares, so a newly added format
    // is covered without editing the suite. The fused PTQTP pair layout has
    // no standalone dtype and keeps its provider-local parity test.
    inline for (comptime std.meta.fields(impl.KernelFormatTag)) |field| {
        if (comptime !std.mem.eql(u8, field.name, "tq2_0_folded")) {
            const fmt = @field(impl.KernelFormatTag, field.name);
            const Block = test_util.BlockFor(fmt);
            const k = 2 * comptime fmt.kMultiple();
            const n = 64;
            const bpr = k / comptime fmt.kMultiple();
            const n_expert = 4;
            // tile edges: 1 row, sub-tile, exactly one tile + 1, multi-tile
            const ms = [n_expert]usize{ 1, 7, 33, 40 };

            const blocks = try allocator.alloc(Block, n_expert * n * bpr);
            defer allocator.free(blocks);
            const wref = try buildQuantWeights(fmt, allocator, random, blocks, n_expert * n, k);
            defer allocator.free(wref);

            var total_rows: usize = 0;
            var tiles_buf: [16]QMMTile = undefined;
            var n_tiles: usize = 0;
            var bases: [n_expert]usize = undefined;
            for (ms, 0..) |m_e, e| {
                bases[e] = total_rows;
                var t: usize = 0;
                while (t * 32 < m_e) : (t += 1) {
                    tiles_buf[n_tiles] = .{
                        .expert = @intCast(e),
                        .base_row = @intCast(total_rows),
                        .m = @intCast(m_e),
                        .tile_m = @intCast(t),
                    };
                    n_tiles += 1;
                }
                total_rows += m_e;
            }

            const a = try allocator.alloc(f32, total_rows * k);
            defer allocator.free(a);
            for (a) |*x| x.* = random.floatNorm(f32);

            impl.qmoe_lock.lock();
            defer impl.qmoe_lock.unlock();
            const stage = impl.qmoeStage(total_rows * k * @sizeOf(f32), total_rows * n * @sizeOf(f32)) orelse
                return error.TestUnexpectedResult;
            @memcpy(stage.in[0 .. total_rows * k], a);
            try std.testing.expect(impl.gemmQGroupedNt(
                fmt,
                std.mem.sliceAsBytes(blocks),
                false, // transient test buffer: must not enter the wrap cache
                bpr * @sizeOf(Block),
                n * bpr * @sizeOf(Block),
                n,
                k,
                tiles_buf[0..n_tiles],
            ));
            for (ms, 0..) |m_e, e| {
                try expectQuantGemmRows(
                    a[bases[e] * k ..][0 .. m_e * k],
                    wref[e * n * k ..][0 .. n * k],
                    stage.out[bases[e] * n ..][0 .. m_e * n],
                    m_e,
                    n,
                    k,
                );
            }
        }
    }
}

test "gpu conformance: qmoe fill gate arithmetic" {
    if (comptime !impl.enabled) return error.SkipZigTest;
    const saved = impl.qmoeMinFillForTest();
    defer impl.setQmoeMinFillForTest(saved);

    impl.setQmoeMinFillForTest(50);
    // 2048 rows over 128 tiles = exactly 50% of the 32-row slots
    try std.testing.expect(impl.qmoeFillAcceptable(2048, 128));
    try std.testing.expect(!impl.qmoeFillAcceptable(2047, 128));
    // full tiles always pass
    try std.testing.expect(impl.qmoeFillAcceptable(4096, 128));
    // empty tile table never dispatches
    try std.testing.expect(!impl.qmoeFillAcceptable(0, 0));

    impl.setQmoeMinFillForTest(0); // occupancy-blind escape hatch
    try std.testing.expect(impl.qmoeFillAcceptable(1, 128));

    impl.setQmoeMinFillForTest(101); // >100 = grouped GPU path never engages
    try std.testing.expect(!impl.qmoeFillAcceptable(4096, 128));
}

test "gpu conformance: eager async f32 input mutation waits for the device reader" {
    try requireDevice();
    const allocator = std.testing.allocator;
    const m = 33;
    const n = 35;
    const k = 31;
    const av = try allocator.alloc(f32, m * k);
    defer allocator.free(av);
    const bv = try allocator.alloc(f32, k * n);
    defer allocator.free(bv);
    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);
    for (av, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) * 0.03125 - 0.125;
    for (bv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.015625 - 0.0625;
    cpuReference(.plain, av, bv, expected, m, n, k);

    var a = try Tensor.fromSlice(allocator, &.{ m, k }, av);
    defer a.deinit();
    var b = try Tensor.fromSlice(allocator, &.{ k, n }, bv);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();
    try std.testing.expect(impl.gemmF32Async(.plain, &a, &b, &out, m, n, k));
    try std.testing.expect(a.buffer.pending_use.load(.acquire) != null);
    a.data()[0] += 100; // mutable host boundary must wait for the old value's reader
    try std.testing.expect(a.buffer.pending_use.load(.acquire) == null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 2e-4);
}

test "gpu conformance: eager async f16 NT writes f32 directly and fences input mutation" {
    try requireDevice();
    const allocator = std.testing.allocator;
    const m = 33;
    const n = 47;
    const k = 17;
    const av = try allocator.alloc(f16, m * k);
    defer allocator.free(av);
    const bv = try allocator.alloc(f16, n * k);
    defer allocator.free(bv);
    const expected = try allocator.alloc(f32, m * n);
    defer allocator.free(expected);
    for (av, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i % 17)) * 0.03125 - 0.25);
    for (bv, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i % 13)) * 0.015625 - 0.125);
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |p| sum += @as(f32, av[row * k + p]) * @as(f32, bv[col * k + p]);
            expected[row * n + col] = sum;
        }
    }

    var a = try TensorF16.fromSlice(allocator, &.{ m, k }, av);
    defer a.deinit();
    var b = try TensorF16.fromSlice(allocator, &.{ n, k }, bv);
    defer b.deinit();
    var out = try Tensor.zeros(allocator, &.{ m, n });
    defer out.deinit();
    try std.testing.expect(impl.gemmF16NtAsync(&a, &b, &out, m, n, k));
    try std.testing.expect(out.buffer.pending() != null);
    try std.testing.expect(a.buffer.pending_use.load(.acquire) != null);
    a.data()[0] += 10;
    try std.testing.expect(a.buffer.pending_use.load(.acquire) == null);
    for (out.dataConst(), expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 4e-4);
    try std.testing.expect(out.buffer.pending() == null);
}
