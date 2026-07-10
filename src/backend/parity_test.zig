const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const Allocator = std.mem.Allocator;

// The native backend is the production Zig backend. It uses portable Zig
// vector kernels for non-GEMM work and optional platform BLAS for GEMM.
const Impl = struct {
    const cpu = @import("cpu.zig");
    const native = @import("native.zig");

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

    const ReduceFn = fn (out: *Tensor, a: *const Tensor) anyerror!void;

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
            try cpu_fn(&cpu_out, &a);

            var native_out = try Tensor.zeros(allocator, &.{1});
            defer native_out.deinit();
            try native_fn(&native_out, &a);

            // Scaled tolerance because both backends accumulate n values; the
            // SIMD pairwise/parallel summation diverges from a serial loop.
            const tol = tolerance * @as(f32, @floatFromInt(n));
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
        }
    }

    const MatMulFn = fn (out: *Tensor, a: *const Tensor, b: *const Tensor) anyerror!void;

    fn checkMatMul(
        allocator: Allocator,
        rng: std.Random,
        cpu_fn: MatMulFn,
        native_fn: MatMulFn,
        comptime variant: enum { nn, tn, nt },
    ) !void {
        for (matmul_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];

            const a_shape: [2]usize = switch (variant) {
                .nn, .nt => .{ m, k },
                .tn => .{ k, m },
            };
            const b_shape: [2]usize = switch (variant) {
                .nn, .tn => .{ k, n },
                .nt => .{ n, k },
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
            try cpu_fn(&cpu_out, &a, &b);

            var native_out = try Tensor.zeros(allocator, &.{ m, n });
            defer native_out.deinit();
            try native_fn(&native_out, &a, &b);

            // Each output element accumulates k products; the SIMD GEMM uses a
            // different reduction tree, so tolerance scales with k.
            const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
            try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
        }
    }

    fn scaleCpu(out: *Tensor, a: *const Tensor, _: *const Tensor) anyerror!void {
        try cpu.scaleInto(out, a, 2.5);
    }

    fn scaleNative(out: *Tensor, a: *const Tensor, _: *const Tensor) anyerror!void {
        try native.scaleInto(out, a, 2.5);
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

    test "parity: dotInto" {
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
            try cpu.dotInto(&cpu_out, &a, &b);

            var native_out = try Tensor.zeros(std.testing.allocator, &.{1});
            defer native_out.deinit();
            try native.dotInto(&native_out, &a, &b);

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

        try cpu.scaleInto(&cpu_vec, &a, -0.75);
        try native.scaleInto(&native_vec, &a, -0.75);
        try expectClose(cpu_vec.dataConst(), native_vec.dataConst(), elementwise_tolerance);

        var cpu_scalar = try Tensor.zeros(std.testing.allocator, &.{1});
        defer cpu_scalar.deinit();
        var native_scalar = try Tensor.zeros(std.testing.allocator, &.{1});
        defer native_scalar.deinit();

        try cpu.sumInto(&cpu_scalar, &a);
        try native.sumInto(&native_scalar, &a);
        try expectClose(cpu_scalar.dataConst(), native_scalar.dataConst(), 1e-6 * @as(f32, @floatFromInt(n)));

        try cpu.dotInto(&cpu_scalar, &a, &b);
        try native.dotInto(&native_scalar, &a, &b);
        try expectClose(cpu_scalar.dataConst(), native_scalar.dataConst(), 1e-6 * @as(f32, @floatFromInt(n)));
    }

    test "parity: matmulInto" {
        var prng = std.Random.DefaultPrng.init(0xdeed);
        try checkMatMul(std.testing.allocator, prng.random(), cpu.matmulInto, native.matmulInto, .nn);
    }

    test "parity: matmulTransAInto" {
        var prng = std.Random.DefaultPrng.init(0xfade);
        try checkMatMul(std.testing.allocator, prng.random(), cpu.matmulTransAInto, native.matmulTransAInto, .tn);
    }

    test "parity: matmulTransBInto" {
        var prng = std.Random.DefaultPrng.init(0xface);
        try checkMatMul(std.testing.allocator, prng.random(), cpu.matmulTransBInto, native.matmulTransBInto, .nt);
    }

    test "parity: large native matmul variants" {
        var prng = std.Random.DefaultPrng.init(0x514e2d);
        const dims = .{ 48, 192, 128 };
        try checkOneMatMul(std.testing.allocator, prng.random(), dims, cpu.matmulInto, native.matmulInto, .nn);
        try checkOneMatMul(std.testing.allocator, prng.random(), dims, cpu.matmulTransAInto, native.matmulTransAInto, .tn);
        try checkOneMatMul(std.testing.allocator, prng.random(), dims, cpu.matmulTransBInto, native.matmulTransBInto, .nt);
    }

    const BatchedFn = fn (
        out: *Tensor,
        a: *const Tensor,
        b: *const Tensor,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
    ) void;

    fn checkBatched(
        allocator: Allocator,
        rng: std.Random,
        cpu_fn: BatchedFn,
        native_fn: BatchedFn,
        comptime variant: enum { nn, tn, nt },
    ) !void {
        const batch_counts = [_]usize{ 1, 2, 5, 8 };
        for (matmul_sizes) |dims| {
            const m = dims[0];
            const k = dims[1];
            const n = dims[2];

            for (batch_counts) |batch| {
                // Test both fully-batched and broadcast-RHS (stride_b=0).
                const stride_b_options = [_]usize{ switch (variant) {
                    .nn, .tn => k * n,
                    .nt => n * k,
                }, 0 };
                for (stride_b_options) |stride_b| {
                    const a_per_batch: usize = switch (variant) {
                        .nn, .nt => m * k,
                        .tn => k * m,
                    };
                    const b_buf_len = if (stride_b == 0)
                        switch (variant) {
                            .nn, .tn => k * n,
                            .nt => n * k,
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
                        .nn, .nt => .{ batch, m, k },
                        .tn => .{ batch, k, m },
                    };
                    const b_shape_full: [3]usize = switch (variant) {
                        .nn, .tn => .{ batch, k, n },
                        .nt => .{ batch, n, k },
                    };
                    const b_shape_shared: [2]usize = switch (variant) {
                        .nn, .tn => .{ k, n },
                        .nt => .{ n, k },
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
                    cpu_fn(&cpu_out, &a, &b, m, n, k, batch, a_per_batch, stride_b, m * n);

                    var native_out = try Tensor.zeros(allocator, &.{ batch, m, n });
                    defer native_out.deinit();
                    native_fn(&native_out, &a, &b, m, n, k, batch, a_per_batch, stride_b, m * n);

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
        cpu_fn: MatMulFn,
        native_fn: MatMulFn,
        comptime variant: enum { nn, tn, nt },
    ) !void {
        const m = dims[0];
        const k = dims[1];
        const n = dims[2];

        const a_shape: [2]usize = switch (variant) {
            .nn, .nt => .{ m, k },
            .tn => .{ k, m },
        };
        const b_shape: [2]usize = switch (variant) {
            .nn, .tn => .{ k, n },
            .nt => .{ n, k },
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
        try cpu_fn(&cpu_out, &a, &b);

        var native_out = try Tensor.zeros(allocator, &.{ m, n });
        defer native_out.deinit();
        try native_fn(&native_out, &a, &b);

        const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
        try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
    }

    fn checkBatchedSharedA(
        allocator: Allocator,
        rng: std.Random,
        cpu_fn: BatchedFn,
        native_fn: BatchedFn,
        comptime variant: enum { nn, tn, nt },
    ) !void {
        const batch: usize = 5;
        const m: usize = 5;
        const k: usize = 6;
        const n: usize = 7;

        const a_shape: [2]usize = switch (variant) {
            .nn, .nt => .{ m, k },
            .tn => .{ k, m },
        };
        const b_shape: [3]usize = switch (variant) {
            .nn, .tn => .{ batch, k, n },
            .nt => .{ batch, n, k },
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
        cpu_fn(&cpu_out, &a, &b, m, n, k, batch, 0, b_stride, m * n);

        var native_out = try Tensor.zeros(allocator, &.{ batch, m, n });
        defer native_out.deinit();
        native_fn(&native_out, &a, &b, m, n, k, batch, 0, b_stride, m * n);

        const tol = matmul_tolerance_scale * @as(f32, @floatFromInt(k));
        try expectClose(cpu_out.dataConst(), native_out.dataConst(), tol);
    }

    test "parity: matmulBatched2DIntoUnchecked" {
        var prng = std.Random.DefaultPrng.init(0xb47ce0);
        try checkBatched(std.testing.allocator, prng.random(), cpu.matmulBatched2DIntoUnchecked, native.matmulBatched2DIntoUnchecked, .nn);
    }

    test "parity: matmulBatchedTransA2DIntoUnchecked" {
        var prng = std.Random.DefaultPrng.init(0xb47c71);
        try checkBatched(std.testing.allocator, prng.random(), cpu.matmulBatchedTransA2DIntoUnchecked, native.matmulBatchedTransA2DIntoUnchecked, .tn);
    }

    test "parity: matmulBatchedTransB2DIntoUnchecked" {
        var prng = std.Random.DefaultPrng.init(0xb47c72);
        try checkBatched(std.testing.allocator, prng.random(), cpu.matmulBatchedTransB2DIntoUnchecked, native.matmulBatchedTransB2DIntoUnchecked, .nt);
    }

    test "parity: native batched matmul accepts shared lhs stride" {
        var prng = std.Random.DefaultPrng.init(0xa571de);
        try checkBatchedSharedA(std.testing.allocator, prng.random(), cpu.matmulBatched2DIntoUnchecked, native.matmulBatched2DIntoUnchecked, .nn);
        try checkBatchedSharedA(std.testing.allocator, prng.random(), cpu.matmulBatchedTransA2DIntoUnchecked, native.matmulBatchedTransA2DIntoUnchecked, .tn);
        try checkBatchedSharedA(std.testing.allocator, prng.random(), cpu.matmulBatchedTransB2DIntoUnchecked, native.matmulBatchedTransB2DIntoUnchecked, .nt);
    }

    fn checkPool2dParity(comptime kind: cpu.PoolKind, allocator: Allocator, rng: std.Random) !void {
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

            const d: cpu.Pool2dDims = .{ .h = h, .w = w, .c = c, .oh = oh, .ow = ow, .kh = k, .kw = k, .stride_h = s, .stride_w = s, .pad_h = p, .pad_w = p };
            var cpu_out = try Tensor.zeros(allocator, &[_]usize{ oh, ow, c });
            defer cpu_out.deinit();
            cpu.pool2dIntoWithConfig(kind, &cpu_out, &input, d, .{});
            var native_out = try Tensor.zeros(allocator, &[_]usize{ oh, ow, c });
            defer native_out.deinit();
            native.pool2dIntoWithConfig(kind, &native_out, &input, d, .{});
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
            cpu.upsample2xNearestIntoWithConfig(&cpu_out, &input, h, w, c, .{});
            var native_out = try Tensor.zeros(allocator, &[_]usize{ 2 * h, 2 * w, c });
            defer native_out.deinit();
            native.upsample2xNearestIntoWithConfig(&native_out, &input, h, w, c, .{});
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

            cpu.preluChannelsIntoWithConfig(zc, x, alpha, rows, c, .{});
            native.preluChannelsIntoWithConfig(zn, x, alpha, rows, c, .{});
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.channelAffineIntoWithConfig(zc, x, alpha, shift, rows, c, .{});
            native.channelAffineIntoWithConfig(zn, x, alpha, shift, rows, c, .{});
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.channelAffineIntoWithConfig(zc, x, alpha, null, rows, c, .{});
            native.channelAffineIntoWithConfig(zn, x, alpha, null, rows, c, .{});
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.preluChannelsBackwardInputIntoWithConfig(zc, x, x, alpha, rows, c, .{});
            native.preluChannelsBackwardInputIntoWithConfig(zn, x, x, alpha, rows, c, .{});
            try expectClose(zc, zn, elementwise_tolerance);

            cpu.preluChannelsBackwardAlphaIntoWithConfig(zc[0..c], x, x, rows, c, .{});
            native.preluChannelsBackwardAlphaIntoWithConfig(zn[0..c], x, x, rows, c, .{});
            try expectClose(zc[0..c], zn[0..c], elementwise_tolerance);
        }
    }
};

test {
    _ = Impl;
}
