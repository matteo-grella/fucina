const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const Allocator = std.mem.Allocator;

// The native backend is the production Zig backend. It uses portable Zig
// vector kernels for non-GEMM work and optional platform BLAS for GEMM.
const Impl = struct {
    const cpu_impl = @import("cpu.zig");
    const native_impl = @import("native.zig");
    const cpu = cpu_impl.kernels;
    const native = native_impl.kernels;
    const ParallelConfig = native_impl.ParallelConfig;
    const ops = @import("ops.zig");
    const pool = @import("vector/pool.zig");

    const elementwise_tolerance: f32 = 1e-6;
    const matmul_tolerance_scale: f32 = 1e-5;

    const elementwise_sizes = [_]usize{ 1, 3, 7, 8, 15, 16, 17, 31, 64, 128, 257, 1024 };
    const matmul_sizes = [_][3]usize{
        .{ 1, 1, 1 },
        .{ 2, 3, 5 },
        .{ 7, 11, 13 },
        .{ 16, 16, 16 },
        .{ 33, 17, 23 },
        .{ 64, 64, 64 },
    };

    fn fillRandom(rng: std.Random, slice: []f32) void {
        for (slice) |*v| v.* = rng.float(f32) * 2 - 1;
    }

    fn expectClose(
        expected: []const f32,
        actual: []const f32,
        tolerance: f32,
    ) !void {
        try std.testing.expectEqual(expected.len, actual.len);
        for (expected, actual, 0..) |e, a, i| {
            std.testing.expectApproxEqAbs(e, a, tolerance) catch |err| {
                std.debug.print(
                    "parity mismatch at index {}: cpu={d} native={d} (tol={d})\n",
                    .{ i, e, a, tolerance },
                );
                return err;
            };
        }
    }

    const BinaryFn = fn (out: *Tensor, a: *const Tensor, b: *const Tensor) anyerror!void;

    fn checkBinary(
        allocator: Allocator,
        rng: std.Random,
        cpu_fn: BinaryFn,
        native_fn: BinaryFn,
    ) !void {
        for (elementwise_sizes) |n| {
            const shape = [_]usize{n};

            const a_data = try allocator.alloc(f32, n);
            defer allocator.free(a_data);
            const b_data = try allocator.alloc(f32, n);
            defer allocator.free(b_data);
            fillRandom(rng, a_data);
            fillRandom(rng, b_data);

            var a = try Tensor.fromSlice(allocator, &shape, a_data);
            defer a.deinit();
            var b = try Tensor.fromSlice(allocator, &shape, b_data);
            defer b.deinit();

            var cpu_out = try Tensor.zeros(allocator, &shape);
            defer cpu_out.deinit();
            try cpu_fn(&cpu_out, &a, &b);

            var native_out = try Tensor.zeros(allocator, &shape);
            defer native_out.deinit();
            try native_fn(&native_out, &a, &b);

            try expectClose(cpu_out.dataConst(), native_out.dataConst(), elementwise_tolerance);
        }
    }

    const ReduceFn = fn (pc: ParallelConfig, out: *Tensor, a: *const Tensor) anyerror!void;

    fn checkReduce(
        allocator: Allocator,
        rng: std.Random,
        cpu_fn: ReduceFn,
        native_fn: ReduceFn,
        tolerance: f32,
    ) !void {
        for (elementwise_sizes) |n| {
            const shape = [_]usize{n};
            const a_data = try allocator.alloc(f32, n);
            defer allocator.free(a_data);
            fillRandom(rng, a_data);

            var a = try Tensor.fromSlice(allocator, &shape, a_data);
            defer a.deinit();

            var cpu_out = try Tensor.zeros(allocator, &.{1});
            defer cpu_out.deinit();
            try cpu_fn(.{}, &cpu_out, &a);

            var native_out = try Tensor.zeros(allocator, &.{1});
            defer native_out.deinit();
            try native_fn(.{}, &native_out, &a);

            // Scaled tolerance because both backends accumulate n values; the
            // SIMD pairwise/parallel summation diverges from a serial loop.
            const tol = tolerance * @as(f32, @floatFromInt(n));
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
        }
    }

    fn checkMatMul(
        allocator: Allocator,
        rng: std.Random,
        comptime variant: ops.MatmulKind,
    ) !void {
        for (matmul_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];

            const a_shape: [2]usize = switch (variant) {
                .plain, .trans_b => .{ m, k },
                .trans_a => .{ k, m },
            };
            const b_shape: [2]usize = switch (variant) {
                .plain, .trans_a => .{ k, n },
                .trans_b => .{ n, k },
            };

            const a_data = try allocator.alloc(f32, a_shape[0] * a_shape[1]);
            defer allocator.free(a_data);
            const b_data = try allocator.alloc(f32, b_shape[0] * b_shape[1]);
            defer allocator.free(b_data);
            fillRandom(rng, a_data);
            fillRandom(rng, b_data);

            var a = try Tensor.fromSlice(allocator, &a_shape, a_data);
            defer a.deinit();
            var b = try Tensor.fromSlice(allocator, &b_shape, b_data);
            defer b.deinit();

            var cpu_out = try Tensor.zeros(allocator, &.{ m, n });
            defer cpu_out.deinit();
            cpu.gemm(.{}, .{ .kind = variant }, &cpu_out, &a, &b, m, n, k);

            var native_out = try Tensor.zeros(allocator, &.{ m, n });
            defer native_out.deinit();
            native.gemm(.{}, .{ .kind = variant }, &native_out, &a, &b, m, n, k);

            // Each output element accumulates k products; the SIMD GEMM uses a
            // different reduction tree, so tolerance scales with k.
            const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
        }
    }

    fn scaleCpu(out: *Tensor, a: *const Tensor, _: *const Tensor) anyerror!void {
        try cpu.scaleInto(.{}, out, a, 2.5);
    }

    fn scaleNative(out: *Tensor, a: *const Tensor, _: *const Tensor) anyerror!void {
        try native.scaleInto(.{}, out, a, 2.5);
    }

    test "parity: addInto" {
        var prng = std.Random.DefaultPrng.init(0xa11ce);
        try checkBinary(std.testing.allocator, prng.random(), cpu.addInto, native.addInto);
    }

    test "parity: subInto" {
        var prng = std.Random.DefaultPrng.init(0xb0b);
        try checkBinary(std.testing.allocator, prng.random(), cpu.subInto, native.subInto);
    }

    test "parity: mulInto" {
        var prng = std.Random.DefaultPrng.init(0xcafe);
        try checkBinary(std.testing.allocator, prng.random(), cpu.mulInto, native.mulInto);
    }

    test "parity: scaleInto" {
        // scaleInto's signature is (out, a, scalar); wrap to reuse checkBinary,
        // passing the second tensor as an unused placeholder.
        var prng = std.Random.DefaultPrng.init(0xfeed);
        try checkBinary(std.testing.allocator, prng.random(), scaleCpu, scaleNative);
    }

    test "parity: sumInto" {
        var prng = std.Random.DefaultPrng.init(0xbeef);
        try checkReduce(std.testing.allocator, prng.random(), cpu.sumInto, native.sumInto, 1e-6);
    }

    test "parity: dot" {
        var prng = std.Random.DefaultPrng.init(0xdab);
        for (elementwise_sizes) |n| {
            const shape = [_]usize{n};
            const a_data = try std.testing.allocator.alloc(f32, n);
            defer std.testing.allocator.free(a_data);
            const b_data = try std.testing.allocator.alloc(f32, n);
            defer std.testing.allocator.free(b_data);
            fillRandom(prng.random(), a_data);
            fillRandom(prng.random(), b_data);

            var a = try Tensor.fromSlice(std.testing.allocator, &shape, a_data);
            defer a.deinit();
            var b = try Tensor.fromSlice(std.testing.allocator, &shape, b_data);
            defer b.deinit();

            var cpu_out = try Tensor.zeros(std.testing.allocator, &.{1});
            defer cpu_out.deinit();
            try cpu.dot(.{}, .f32, &cpu_out, &a, &b);

            var native_out = try Tensor.zeros(std.testing.allocator, &.{1});
            defer native_out.deinit();
            try native.dot(.{}, .f32, &native_out, &a, &b);

            const tol = 1e-6 * @as(f32, @floatFromInt(n));
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
        }
    }

    test "parity: large native elementwise and reductions" {
        const n: usize = 300_000;
        const shape = [_]usize{n};
        var prng = std.Random.DefaultPrng.init(0x1a2b3c);

        const a_data = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(a_data);
        const b_data = try std.testing.allocator.alloc(f32, n);
        defer std.testing.allocator.free(b_data);
        fillRandom(prng.random(), a_data);
        fillRandom(prng.random(), b_data);

        var a = try Tensor.fromSlice(std.testing.allocator, &shape, a_data);
        defer a.deinit();
        var b = try Tensor.fromSlice(std.testing.allocator, &shape, b_data);
        defer b.deinit();

        var cpu_vec = try Tensor.zeros(std.testing.allocator, &shape);
        defer cpu_vec.deinit();
        var native_vec = try Tensor.zeros(std.testing.allocator, &shape);
        defer native_vec.deinit();

        try cpu.addInto(&cpu_vec, &a, &b);
        try native.addInto(&native_vec, &a, &b);
        try expectClose(cpu_vec.dataConst(), native_vec.dataConst(), elementwise_tolerance);

        try cpu.mulInto(&cpu_vec, &a, &b);
        try native.mulInto(&native_vec, &a, &b);
        try expectClose(cpu_vec.dataConst(), native_vec.dataConst(), elementwise_tolerance);

        try cpu.scaleInto(.{}, &cpu_vec, &a, -0.75);
        try native.scaleInto(.{}, &native_vec, &a, -0.75);
        try expectClose(cpu_vec.dataConst(), native_vec.dataConst(), elementwise_tolerance);

        var cpu_scalar = try Tensor.zeros(std.testing.allocator, &.{1});
        defer cpu_scalar.deinit();
        var native_scalar = try Tensor.zeros(std.testing.allocator, &.{1});
        defer native_scalar.deinit();

        try cpu.sumInto(.{}, &cpu_scalar, &a);
        try native.sumInto(.{}, &native_scalar, &a);
        try expectClose(cpu_scalar.dataConst(), native_scalar.dataConst(), 1e-6 * @as(f32, @floatFromInt(n)));

        try cpu.dot(.{}, .f32, &cpu_scalar, &a, &b);
        try native.dot(.{}, .f32, &native_scalar, &a, &b);
        try expectClose(cpu_scalar.dataConst(), native_scalar.dataConst(), 1e-6 * @as(f32, @floatFromInt(n)));
    }

    test "parity: gemm plain" {
        var prng = std.Random.DefaultPrng.init(0xdeed);
        try checkMatMul(std.testing.allocator, prng.random(), .plain);
    }

    test "parity: gemm trans_a" {
        var prng = std.Random.DefaultPrng.init(0xfade);
        try checkMatMul(std.testing.allocator, prng.random(), .trans_a);
    }

    test "parity: gemm trans_b" {
        var prng = std.Random.DefaultPrng.init(0xface);
        try checkMatMul(std.testing.allocator, prng.random(), .trans_b);
    }

    test "parity: large native matmul variants" {
        var prng = std.Random.DefaultPrng.init(0x514e2d);
        const dims = .{ 48, 192, 128 };
        try checkOneMatMul(std.testing.allocator, prng.random(), dims, .plain);
        try checkOneMatMul(std.testing.allocator, prng.random(), dims, .trans_a);
        try checkOneMatMul(std.testing.allocator, prng.random(), dims, .trans_b);
    }

    fn checkBatched(
        allocator: Allocator,
        rng: std.Random,
        comptime variant: ops.MatmulKind,
    ) !void {
        const batch_counts = [_]usize{ 1, 2, 5, 8 };
        for (matmul_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];

            for (batch_counts) |batch| {
                // Test both fully-batched and broadcast-RHS (stride_b=0).
                const stride_b_options = [_]usize{ switch (variant) {
                    .plain, .trans_a => k * n,
                    .trans_b => n * k,
                }, 0 };
                for (stride_b_options) |stride_b| {
                    const a_per_batch: usize = switch (variant) {
                        .plain, .trans_b => m * k,
                        .trans_a => k * m,
                    };
                    const b_buf_len = if (stride_b == 0)
                        switch (variant) {
                            .plain, .trans_a => k * n,
                            .trans_b => n * k,
                        }
                    else
                        stride_b * batch;
                    const a_buf_len = a_per_batch * batch;
                    const out_buf_len = m * n * batch;

                    const a_data = try allocator.alloc(f32, a_buf_len);
                    defer allocator.free(a_data);
                    const b_data = try allocator.alloc(f32, b_buf_len);
                    defer allocator.free(b_data);
                    fillRandom(rng, a_data);
                    fillRandom(rng, b_data);

                    const a_shape: [3]usize = switch (variant) {
                        .plain, .trans_b => .{ batch, m, k },
                        .trans_a => .{ batch, k, m },
                    };
                    const b_shape_full: [3]usize = switch (variant) {
                        .plain, .trans_a => .{ batch, k, n },
                        .trans_b => .{ batch, n, k },
                    };
                    const b_shape_shared: [2]usize = switch (variant) {
                        .plain, .trans_a => .{ k, n },
                        .trans_b => .{ n, k },
                    };

                    var a = try Tensor.fromSlice(allocator, &a_shape, a_data);
                    defer a.deinit();
                    var b = if (stride_b == 0)
                        try Tensor.fromSlice(allocator, &b_shape_shared, b_data)
                    else
                        try Tensor.fromSlice(allocator, &b_shape_full, b_data);
                    defer b.deinit();

                    var cpu_out = try Tensor.zeros(allocator, &.{ batch, m, n });
                    defer cpu_out.deinit();
                    cpu.gemmBatched(.{}, variant, &cpu_out, &a, &b, m, n, k, batch, a_per_batch, stride_b, m * n);

                    var native_out = try Tensor.zeros(allocator, &.{ batch, m, n });
                    defer native_out.deinit();
                    native.gemmBatched(.{}, variant, &native_out, &a, &b, m, n, k, batch, a_per_batch, stride_b, m * n);

                    const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
                    try std.testing.expectEqual(cpu_out.dataConst().len, out_buf_len);
                    try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
                }
            }
        }
    }

    fn checkOneMatMul(
        allocator: Allocator,
        rng: std.Random,
        dims: [3]usize,
        comptime variant: ops.MatmulKind,
    ) !void {
        const m = dims[0];
        const k = dims[1];
        const n = dims[2];

        const a_shape: [2]usize = switch (variant) {
            .plain, .trans_b => .{ m, k },
            .trans_a => .{ k, m },
        };
        const b_shape: [2]usize = switch (variant) {
            .plain, .trans_a => .{ k, n },
            .trans_b => .{ n, k },
        };

        const a_data = try allocator.alloc(f32, a_shape[0] * a_shape[1]);
        defer allocator.free(a_data);
        const b_data = try allocator.alloc(f32, b_shape[0] * b_shape[1]);
        defer allocator.free(b_data);
        fillRandom(rng, a_data);
        fillRandom(rng, b_data);

        var a = try Tensor.fromSlice(allocator, &a_shape, a_data);
        defer a.deinit();
        var b = try Tensor.fromSlice(allocator, &b_shape, b_data);
        defer b.deinit();

        var cpu_out = try Tensor.zeros(allocator, &.{ m, n });
        defer cpu_out.deinit();
        cpu.gemm(.{}, .{ .kind = variant }, &cpu_out, &a, &b, m, n, k);

        var native_out = try Tensor.zeros(allocator, &.{ m, n });
        defer native_out.deinit();
        native.gemm(.{}, .{ .kind = variant }, &native_out, &a, &b, m, n, k);

        const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
        try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
    }

    fn checkBatchedSharedA(
        allocator: Allocator,
        rng: std.Random,
        comptime variant: ops.MatmulKind,
    ) !void {
        const batch: usize = 5;
        const m: usize = 5;
        const k: usize = 6;
        const n: usize = 7;

        const a_shape: [2]usize = switch (variant) {
            .plain, .trans_b => .{ m, k },
            .trans_a => .{ k, m },
        };
        const b_shape: [3]usize = switch (variant) {
            .plain, .trans_a => .{ batch, k, n },
            .trans_b => .{ batch, n, k },
        };
        const a_len = a_shape[0] * a_shape[1];
        const b_stride = b_shape[1] * b_shape[2];

        const a_data = try allocator.alloc(f32, a_len);
        defer allocator.free(a_data);
        const b_data = try allocator.alloc(f32, batch * b_stride);
        defer allocator.free(b_data);
        fillRandom(rng, a_data);
        fillRandom(rng, b_data);

        var a = try Tensor.fromSlice(allocator, &a_shape, a_data);
        defer a.deinit();
        var b = try Tensor.fromSlice(allocator, &b_shape, b_data);
        defer b.deinit();

        var cpu_out = try Tensor.zeros(allocator, &.{ batch, m, n });
        defer cpu_out.deinit();
        cpu.gemmBatched(.{}, variant, &cpu_out, &a, &b, m, n, k, batch, 0, b_stride, m * n);

        var native_out = try Tensor.zeros(allocator, &.{ batch, m, n });
        defer native_out.deinit();
        native.gemmBatched(.{}, variant, &native_out, &a, &b, m, n, k, batch, 0, b_stride, m * n);

        const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
        try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
    }

    test "parity: gemmBatched plain" {
        var prng = std.Random.DefaultPrng.init(0xb47ce0);
        try checkBatched(std.testing.allocator, prng.random(), .plain);
    }

    test "parity: gemmBatched trans_a" {
        var prng = std.Random.DefaultPrng.init(0xb47c71);
        try checkBatched(std.testing.allocator, prng.random(), .trans_a);
    }

    test "parity: gemmBatched trans_b" {
        var prng = std.Random.DefaultPrng.init(0xb47c72);
        try checkBatched(std.testing.allocator, prng.random(), .trans_b);
    }

    test "parity: native batched matmul accepts shared lhs stride" {
        var prng = std.Random.DefaultPrng.init(0xa571de);
        try checkBatchedSharedA(std.testing.allocator, prng.random(), .plain);
        try checkBatchedSharedA(std.testing.allocator, prng.random(), .trans_a);
        try checkBatchedSharedA(std.testing.allocator, prng.random(), .trans_b);
    }

    fn checkPool2dParity(comptime kind: pool.PoolKind, allocator: Allocator, rng: std.Random) !void {
        // h, w, c, k, s, p — odd channel counts exercise the SIMD remainder loop.
        const geoms = [_][6]usize{
            .{ 6, 6, 3, 2, 2, 0 },
            .{ 7, 5, 5, 3, 2, 1 },
            .{ 8, 8, 17, 3, 1, 1 },
            .{ 5, 9, 8, 2, 1, 0 },
        };
        for (geoms) |g| {
            const h = g[0];
            const w = g[1];
            const c = g[2];
            const k = g[3];
            const s = g[4];
            const p = g[5];
            const oh = (h + 2 * p - k) / s + 1;
            const ow = (w + 2 * p - k) / s + 1;

            const in_data = try allocator.alloc(f32, h * w * c);
            defer allocator.free(in_data);
            fillRandom(rng, in_data);
            var input = try Tensor.fromSlice(allocator, &[_]usize{ h, w, c }, in_data);
            defer input.deinit();

            const d: pool.Pool2dDims = .{ .h = h, .w = w, .c = c, .oh = oh, .ow = ow, .kh = k, .kw = k, .stride_h = s, .stride_w = s, .pad_h = p, .pad_w = p };
            var cpu_out = try Tensor.zeros(allocator, &[_]usize{ oh, ow, c });
            defer cpu_out.deinit();
            cpu.pool2dInto(.{}, kind, &cpu_out, &input, d);
            var native_out = try Tensor.zeros(allocator, &[_]usize{ oh, ow, c });
            defer native_out.deinit();
            native.pool2dInto(.{}, kind, &native_out, &input, d);
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), elementwise_tolerance);
        }
    }

    test "parity: pool2d avg/max, upsample2x, prelu, channelAffine" {
        var prng = std.Random.DefaultPrng.init(0x9e3779b9);
        const rng = prng.random();
        const allocator = std.testing.allocator;

        try checkPool2dParity(.max, allocator, rng);
        try checkPool2dParity(.avg, allocator, rng);
        try checkPool2dParity(.sum, allocator, rng);

        // 2× nearest upsample.
        {
            const h = 5;
            const w = 7;
            const c = 17;
            const in_data = try allocator.alloc(f32, h * w * c);
            defer allocator.free(in_data);
            fillRandom(rng, in_data);
            var input = try Tensor.fromSlice(allocator, &[_]usize{ h, w, c }, in_data);
            defer input.deinit();
            var cpu_out = try Tensor.zeros(allocator, &[_]usize{ 2 * h, 2 * w, c });
            defer cpu_out.deinit();
            cpu.upsample2xNearestInto(.{}, &cpu_out, &input, h, w, c);
            var native_out = try Tensor.zeros(allocator, &[_]usize{ 2 * h, 2 * w, c });
            defer native_out.deinit();
            native.upsample2xNearestInto(.{}, &native_out, &input, h, w, c);
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), elementwise_tolerance);
        }

        // prelu + channelAffine row kernels (with and without shift).
        for ([_]usize{ 1, 5, 17 }) |c| {
            const rows = 13;
            const x = try allocator.alloc(f32, rows * c);
            defer allocator.free(x);
            const alpha = try allocator.alloc(f32, c);
            defer allocator.free(alpha);
            const shift = try allocator.alloc(f32, c);
            defer allocator.free(shift);
            fillRandom(rng, x);
            fillRandom(rng, alpha);
            fillRandom(rng, shift);
            const zc = try allocator.alloc(f32, rows * c);
            defer allocator.free(zc);
            const zn = try allocator.alloc(f32, rows * c);
            defer allocator.free(zn);

            cpu.preluChannelsInto(.{}, zc, x, alpha, rows, c);
            native.preluChannelsInto(.{}, zn, x, alpha, rows, c);
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.channelAffineInto(.{}, zc, x, alpha, shift, rows, c);
            native.channelAffineInto(.{}, zn, x, alpha, shift, rows, c);
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.channelAffineInto(.{}, zc, x, alpha, null, rows, c);
            native.channelAffineInto(.{}, zn, x, alpha, null, rows, c);
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.preluChannelsBackwardInputInto(.{}, zc, x, x, alpha, rows, c);
            native.preluChannelsBackwardInputInto(.{}, zn, x, x, alpha, rows, c);
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.preluChannelsBackwardAlphaInto(.{}, zc[0..c], x, x, rows, c);
            native.preluChannelsBackwardAlphaInto(.{}, zn[0..c], x, x, rows, c);
            try expectClose(zc[0..c], zn[0..c], elementwise_tolerance);
        }
    }

    // ------------------------------------------------------------------
    // Quantized GEMM family: encode a random RHS, run both providers'
    // entries (the plain containers and the packed x4/x8 arms), and check
    // each against a dequantized f32 reference accumulated in f64. The
    // kernels' integer dots are exact, so the only divergence is the f32
    // epilogue's association order — tolerance scales with k like the dense
    // matmul cases above. On `-Dbackend=scalar` builds the format kernels
    // resolve to their scalar accumulators (`isa.tier == .scalar`), so this
    // suite is what actually holds the scalar reference and the SIMD tiles
    // to the same answer.

    const qm = @import("quant.zig");
    const dtype_mod = @import("../dtype.zig");
    const DType = dtype_mod.DType;
    const qk_k = qm.types.qk_k_block_size;

    // m hits the single-row, generic, and x4-LHS arms (native's
    // q4_k_x4_min_rows = 4; m = 16 also drives the Q8_0x4 packed-LHS path);
    // k covers one and two K-quant super-blocks.
    const quant_sizes = [_][3]usize{ .{ 1, 256, 8 }, .{ 3, 512, 16 }, .{ 16, 512, 16 } };

    fn quantRef(allocator: Allocator, lhs_deq: []const f32, rhs_deq: []const f32, m: usize, n: usize, k: usize) ![]f32 {
        const ref = try allocator.alloc(f32, m * n);
        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f64 = 0;
                for (0..k) |t| acc += @as(f64, lhs_deq[i * k + t]) * @as(f64, rhs_deq[j * k + t]);
                ref[i * n + j] = @floatCast(acc);
            }
        }
        return ref;
    }

    fn expectQuantClose(ref: []const f32, actual: []const f32, k: usize) !void {
        const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
        for (ref, actual, 0..) |r, x, i| {
            std.testing.expectApproxEqAbs(r, x, tol) catch |err| {
                std.debug.print("quant parity mismatch at index {}: ref={d} got={d} (tol={d})\n", .{ i, r, x, tol });
                return err;
            };
        }
    }

    /// Dequantize the [m, k] LHS exactly as the Q8_K-activation kernels see it.
    fn lhsDeqQ8_K(allocator: Allocator, a: *const Tensor, m: usize, k: usize) ![]f32 {
        const bpr = k / qk_k;
        const blocks = try qm.q8k.quantizeRowsQ8_K(allocator, a);
        defer allocator.free(blocks);
        const deq = try allocator.alloc(f32, m * k);
        errdefer allocator.free(deq);
        var buf: [qk_k]f32 = undefined;
        for (0..m) |i| {
            for (0..bpr) |bi| {
                qm.q8k.dequantizeBlockQ8_KInto(&buf, &blocks[i * bpr + bi]);
                @memcpy(deq[i * k + bi * qk_k ..][0..qk_k], &buf);
            }
        }
        return deq;
    }

    /// Dequantize the [m, k] LHS exactly as the Q8_0-activation kernels see it.
    fn lhsDeqQ8_0(allocator: Allocator, a: *const Tensor, m: usize, k: usize) ![]f32 {
        var rows = try qm.q8k.quantizeRowsQ8_0(allocator, a);
        defer rows.deinit();
        const deq = try allocator.alloc(f32, m * k);
        errdefer allocator.free(deq);
        for (0..m) |i| try qm.q8k.dequantizeRowQ8_0Into(deq[i * k ..][0..k], rows.rowBlocks(i));
        return deq;
    }

    fn runBothQuant(allocator: Allocator, a: *const Tensor, rhs: qm.AnyQuantizedMatmulRhs, ref: []const f32, m: usize, n: usize, k: usize) !void {
        var cpu_out = try Tensor.zeros(allocator, &.{ m, n });
        defer cpu_out.deinit();
        try cpu.matmul2DQuantizedRhs(.{}, allocator, &cpu_out, a, rhs, m, n, k);
        try expectQuantClose(ref, cpu_out.dataConst(), k);
        var native_out = try Tensor.zeros(allocator, &.{ m, n });
        defer native_out.deinit();
        try native.matmul2DQuantizedRhs(.{}, allocator, &native_out, a, rhs, m, n, k);
        try expectQuantClose(ref, native_out.dataConst(), k);
    }

    fn runBothPacked(allocator: Allocator, a: *const Tensor, pack: anytype, ref: []const f32, m: usize, n: usize, k: usize) !void {
        var cpu_out = try Tensor.zeros(allocator, &.{ m, n });
        defer cpu_out.deinit();
        try cpu.matmulPacked(.{}, allocator, &cpu_out, a, pack, m, n, k);
        try expectQuantClose(ref, cpu_out.dataConst(), k);
        var native_out = try Tensor.zeros(allocator, &.{ m, n });
        defer native_out.deinit();
        try native.matmulPacked(.{}, allocator, &native_out, a, pack, m, n, k);
        try expectQuantClose(ref, native_out.dataConst(), k);
    }

    fn checkQ8_0Parity(allocator: Allocator, rng: std.Random) !void {
        for (quant_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];

            const a_data = try allocator.alloc(f32, m * k);
            defer allocator.free(a_data);
            fillRandom(rng, a_data);
            var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_data);
            defer a.deinit();

            const b_data = try allocator.alloc(f32, k * n);
            defer allocator.free(b_data);
            fillRandom(rng, b_data);
            var b = try Tensor.fromSlice(allocator, &.{ k, n }, b_data);
            defer b.deinit();
            var rhs = try qm.quantizeMatmulRhsQ8_0(allocator, &b);
            defer rhs.deinit();

            const rhs_deq = try allocator.alloc(f32, n * k);
            defer allocator.free(rhs_deq);
            for (0..n) |j| try qm.q8k.dequantizeRowQ8_0Into(rhs_deq[j * k ..][0..k], rhs.rows.rowBlocks(j));
            const lhs_deq = try lhsDeqQ8_0(allocator, &a, m, k);
            defer allocator.free(lhs_deq);
            const ref = try quantRef(allocator, lhs_deq, rhs_deq, m, n, k);
            defer allocator.free(ref);

            try runBothQuant(allocator, &a, .{ .q8_0 = &rhs }, ref, m, n, k);

            var x4 = try qm.packRhsAs(qm.QuantizedMatmulRhsQ8_0x4, allocator, rhs.rows.blocks, n, k, rhs.rows.blocks_per_row);
            defer x4.deinit();
            try runBothPacked(allocator, &a, &x4, ref, m, n, k);
        }
    }

    fn checkKQuantParity(comptime dt: DType, allocator: Allocator, rng: std.Random) !void {
        for (quant_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];
            const bpc = k / qk_k;

            const a_data = try allocator.alloc(f32, m * k);
            defer allocator.free(a_data);
            fillRandom(rng, a_data);
            var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_data);
            defer a.deinit();

            // Encode the RHS column-wise from a random [k, n] matrix.
            const b_data = try allocator.alloc(f32, k * n);
            defer allocator.free(b_data);
            fillRandom(rng, b_data);
            const blocks = try allocator.alloc(dtype_mod.Storage(dt), n * bpc);
            defer allocator.free(blocks);
            const col = try allocator.alloc(f32, k);
            defer allocator.free(col);
            for (0..n) |j| {
                for (0..k) |t| col[t] = b_data[t * n + j];
                switch (dt) {
                    .q4_k => try qm.q4_k.quantizeRowQ4_KInto(blocks[j * bpc ..][0..bpc], col),
                    .q5_k => try qm.q5_k.quantizeRowQ5_KInto(blocks[j * bpc ..][0..bpc], col),
                    .q6_k => try qm.q6_k.quantizeRowQ6_KInto(blocks[j * bpc ..][0..bpc], col),
                    else => comptime unreachable,
                }
            }

            const rhs_deq = try allocator.alloc(f32, n * k);
            defer allocator.free(rhs_deq);
            var buf: [qk_k]f32 = undefined;
            for (0..n) |j| {
                for (0..bpc) |bi| {
                    switch (dt) {
                        .q4_k => qm.q4_k.dequantizeBlockQ4_KInto(&buf, &blocks[j * bpc + bi]),
                        .q5_k => qm.q5_k.dequantizeBlockQ5_KInto(&buf, &blocks[j * bpc + bi]),
                        .q6_k => qm.q6_k.dequantizeBlockQ6_KInto(&buf, &blocks[j * bpc + bi]),
                        else => comptime unreachable,
                    }
                    @memcpy(rhs_deq[j * k + bi * qk_k ..][0..qk_k], &buf);
                }
            }
            const lhs_deq = try lhsDeqQ8_K(allocator, &a, m, k);
            defer allocator.free(lhs_deq);
            const ref = try quantRef(allocator, lhs_deq, rhs_deq, m, n, k);
            defer allocator.free(ref);

            // Plain per-column container.
            var rhs = switch (dt) {
                .q4_k => try qm.q8k.quantizedMatmulRhsQ4_KFromBlocks(allocator, k, n, blocks),
                .q5_k => try qm.q8k.quantizedMatmulRhsQ5_KFromBlocks(allocator, k, n, blocks),
                .q6_k => try qm.q8k.quantizedMatmulRhsQ6_KFromBlocks(allocator, k, n, blocks),
                else => comptime unreachable,
            };
            defer rhs.deinit();
            const any: qm.AnyQuantizedMatmulRhs = switch (dt) {
                .q4_k => .{ .q4_k = &rhs },
                .q5_k => .{ .q5_k = &rhs },
                .q6_k => .{ .q6_k = &rhs },
                else => comptime unreachable,
            };
            try runBothQuant(allocator, &a, any, ref, m, n, k);

            // Packed lane arms.
            switch (dt) {
                .q4_k => {
                    var x4 = try qm.packRhsAs(qm.QuantizedMatmulRhsQ4_Kx4, allocator, blocks, n, k, bpc);
                    defer x4.deinit();
                    try runBothPacked(allocator, &a, &x4, ref, m, n, k);
                    var x8 = try qm.packRhsAs(qm.QuantizedMatmulRhsQ4_Kx8, allocator, blocks, n, k, bpc);
                    defer x8.deinit();
                    try runBothPacked(allocator, &a, &x8, ref, m, n, k);
                },
                .q5_k => {
                    var x8 = try qm.packRhsAs(qm.QuantizedMatmulRhsQ5_Kx8, allocator, blocks, n, k, bpc);
                    defer x8.deinit();
                    try runBothPacked(allocator, &a, &x8, ref, m, n, k);
                },
                .q6_k => {
                    var x4 = try qm.packRhsAs(qm.QuantizedMatmulRhsQ6_Kx4, allocator, blocks, n, k, bpc);
                    defer x4.deinit();
                    try runBothPacked(allocator, &a, &x4, ref, m, n, k);
                },
                else => comptime unreachable,
            }
        }
    }

    fn checkTQ2_0Parity(allocator: Allocator, rng: std.Random) !void {
        for (quant_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];

            const a_data = try allocator.alloc(f32, m * k);
            defer allocator.free(a_data);
            fillRandom(rng, a_data);
            var a = try Tensor.fromSlice(allocator, &.{ m, k }, a_data);
            defer a.deinit();

            // TQ2_0 encodes row-major [n][k] (RHS rows are output columns).
            const wt = try allocator.alloc(f32, n * k);
            defer allocator.free(wt);
            fillRandom(rng, wt);
            var rhs = try qm.ternary.quantizedMatmulRhsTQ2_0FromF32(allocator, k, n, wt);
            defer rhs.deinit();

            const rhs_deq = try allocator.alloc(f32, n * k);
            defer allocator.free(rhs_deq);
            for (0..n) |j| try qm.cold.dequantizeRowTQ2_0Into(rhs_deq[j * k ..][0..k], rhs.rows.rowBlocks(j));
            const lhs_deq = try lhsDeqQ8_K(allocator, &a, m, k);
            defer allocator.free(lhs_deq);
            const ref = try quantRef(allocator, lhs_deq, rhs_deq, m, n, k);
            defer allocator.free(ref);

            try runBothQuant(allocator, &a, .{ .tq2_0 = &rhs }, ref, m, n, k);
        }
    }

    test "parity: quantized GEMM q8_0 plain + x4 pack vs dequantized reference" {
        var prng = std.Random.DefaultPrng.init(0x9800);
        try checkQ8_0Parity(std.testing.allocator, prng.random());
    }

    test "parity: quantized GEMM q4_k plain + x4/x8 packs vs dequantized reference" {
        var prng = std.Random.DefaultPrng.init(0x94a0);
        try checkKQuantParity(.q4_k, std.testing.allocator, prng.random());
    }

    test "parity: quantized GEMM q5_k plain + x8 pack vs dequantized reference" {
        var prng = std.Random.DefaultPrng.init(0x95a0);
        try checkKQuantParity(.q5_k, std.testing.allocator, prng.random());
    }

    test "parity: quantized GEMM q6_k plain + x4 pack vs dequantized reference" {
        var prng = std.Random.DefaultPrng.init(0x96a0);
        try checkKQuantParity(.q6_k, std.testing.allocator, prng.random());
    }

    test "parity: quantized GEMM tq2_0 vs dequantized reference" {
        var prng = std.Random.DefaultPrng.init(0x9720);
        try checkTQ2_0Parity(std.testing.allocator, prng.random());
    }
};

test {
    _ = Impl;
}
