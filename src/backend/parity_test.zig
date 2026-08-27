const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const Allocator = std.mem.Allocator;

// The parity gate of the one CPU provider: every case runs a kernel entry
// (`native.kernels`, the tier the build compiled) against its scalar
// reference twin (the `scalar` namespaces inside `vector/`) on the same
// data. On a native build that holds the SIMD/BLAS arms to the serial
// reference; on the `-Dbackend=scalar` build both sides resolve to the
// same scalar arms and the quantized cases below carry the load (they pin
// every entry against a dequantized f32/f64 reference instead).
const Impl = struct {
    const native_impl = @import("native.zig");
    const native = native_impl.kernels;
    const ParallelConfig = native_impl.ParallelConfig;
    const ops = @import("ops.zig");
    const vector_common = @import("vector/common.zig");
    const elementwise = @import("vector/elementwise.zig");
    const vector_gemm = @import("vector/gemm.zig");
    const vector_batched = @import("vector/batched.zig");
    const vector_mq = @import("vector/matmul_quant.zig");
    const pool = @import("vector/pool.zig");

    // The scalar twin set, spelled with the kernel-entry signatures the
    // cases below share with `native.kernels` (the twins themselves take no
    // `pc`; these shims adapt).
    const cpu = struct {
        fn addInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
            try tensor_mod.requireSameShape(a, b);
            try tensor_mod.requireSameShape(out, a);
            elementwise.scalar.addContiguousIntoUnchecked(out, a, b, a.len());
        }
        fn subInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
            try tensor_mod.requireSameShape(a, b);
            try tensor_mod.requireSameShape(out, a);
            elementwise.scalar.subContiguousIntoUnchecked(out, a, b, a.len());
        }
        fn mulInto(out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
            try tensor_mod.requireSameShape(a, b);
            try tensor_mod.requireSameShape(out, a);
            elementwise.scalar.mulContiguousIntoUnchecked(out, a, b, a.len());
        }
        fn scaleInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor, v: f32) !void {
            _ = pc;
            return elementwise.scalar.scaleInto(out, a, v);
        }
        fn sumInto(pc: ParallelConfig, out: *Tensor, a: *const Tensor) !void {
            _ = pc;
            return elementwise.scalar.sumInto(out, a);
        }
        fn dot(pc: ParallelConfig, comptime dtype: DType, out: *Tensor, a: *const Tensor, b: *const Tensor) !void {
            _ = pc;
            return elementwise.scalar.dot(dtype, out, a, b);
        }
        fn gemm(pc: ParallelConfig, comptime g: ops.Gemm, out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize) void {
            _ = pc;
            vector_gemm.scalar.gemm(
                g,
                vector_common.contiguousData(out, m * n),
                vector_common.contiguousDataConst(a, m * k),
                vector_common.contiguousDataConst(b, k * n),
                m,
                n,
                k,
            );
        }
        fn gemmBatched(pc: ParallelConfig, comptime kind: ops.MatmulKind, out: *Tensor, a: *const Tensor, b: *const Tensor, m: usize, n: usize, k: usize, batch_count: usize, stride_a: usize, stride_b: usize, stride_c: usize) void {
            _ = pc;
            vector_batched.scalar.gemmBatched(
                kind,
                vector_common.contiguousData(out, out.buffer.data.len - out.offset),
                vector_common.contiguousDataConst(a, a.buffer.data.len - a.offset),
                vector_common.contiguousDataConst(b, b.buffer.data.len - b.offset),
                m,
                n,
                k,
                batch_count,
                stride_a,
                stride_b,
                stride_c,
            );
        }
        fn pool2dInto(pc: ParallelConfig, comptime kind: pool.PoolKind, out: *Tensor, input: *const Tensor, d: pool.Pool2dDims) void {
            _ = pc;
            pool.scalar.pool2dInto(kind, out, input, d);
        }
        fn upsample2xNearestInto(pc: ParallelConfig, out: *Tensor, input: *const Tensor, h: usize, w: usize, c: usize) void {
            _ = pc;
            pool.scalar.upsample2xNearestInto(out, input, h, w, c);
        }
        fn preluChannelsInto(pc: ParallelConfig, z: []f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
            _ = pc;
            elementwise.scalar.preluChannelsInto(z, x, alpha, rows, cols);
        }
        fn channelAffineInto(pc: ParallelConfig, z: []f32, x: []const f32, scale: []const f32, shift: ?[]const f32, rows: usize, cols: usize) void {
            _ = pc;
            elementwise.scalar.channelAffineInto(z, x, scale, shift, rows, cols);
        }
        fn preluChannelsBackwardInputInto(pc: ParallelConfig, gx: []f32, gy: []const f32, x: []const f32, alpha: []const f32, rows: usize, cols: usize) void {
            _ = pc;
            elementwise.scalar.preluChannelsBackwardInputInto(gx, gy, x, alpha, rows, cols);
        }
        fn preluChannelsBackwardAlphaInto(pc: ParallelConfig, galpha: []f32, gy: []const f32, x: []const f32, rows: usize, cols: usize) void {
            _ = pc;
            elementwise.scalar.preluChannelsBackwardAlphaInto(galpha, gy, x, rows, cols);
        }
        fn matmul2DQuantizedRhs(pc: ParallelConfig, allocator: Allocator, out: *Tensor, a: *const Tensor, rhs: qm.AnyQuantizedMatmulRhs, m: usize, n: usize, k: usize) !void {
            _ = pc;
            return vector_mq.scalar.matmul2DQuantizedRhs(allocator, out, a, rhs, m, n, k);
        }
        fn matmulPacked(pc: ParallelConfig, allocator: Allocator, out: anytype, a: anytype, rhs: anytype, m: usize, n: usize, k: usize) !void {
            _ = pc;
            return vector_mq.scalar.matmulPacked(allocator, out, a, rhs, m, n, k);
        }
    };

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

    // ---- Row-kernel parity: the vector/rows.zig entries against their
    // scalar twins on the same task payloads. The rows kernels differ from
    // the twins in reduction order only (per-element transcendentals are
    // shared), so tolerances scale with the reduced length; the inner-lane
    // kernels keep per-output accumulation order, so those comparisons are
    // bitwise.
    const rows_impl = @import("vector/rows.zig");

    fn expectBitwise(expected: []const f32, actual: []const f32) !void {
        try std.testing.expectEqualSlices(u32, @ptrCast(expected), @ptrCast(actual));
    }

    test "parity: softmax row kernels vs scalar twins" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x50f7a);
        const rng = prng.random();
        const n_rows: usize = 5;
        for ([_]usize{ 7, 64, 257 }) |cols| {
            const len = n_rows * cols;
            const input = try allocator.alloc(f32, len);
            defer allocator.free(input);
            fillRandom(rng, input);
            const out_a = try allocator.alloc(f32, len);
            defer allocator.free(out_a);
            const out_b = try allocator.alloc(f32, len);
            defer allocator.free(out_b);
            const tol = 1e-6 * @as(f32, @floatFromInt(cols));

            rows_impl.softmaxRows(.{ .input = input, .output = out_a, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.softmaxRows(.{ .input = input, .output = out_b, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            try expectClose(out_b, out_a, tol);

            rows_impl.logSoftmaxRows(.{ .input = input, .output = out_a, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.logSoftmaxRows(.{ .input = input, .output = out_b, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            try expectClose(out_b, out_a, tol);

            rows_impl.logsumexpRows(.{ .input = input, .output = out_a, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.logsumexpRows(.{ .input = input, .output = out_b, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            try expectClose(out_b[0..n_rows], out_a[0..n_rows], tol);

            // Backward over a softmax output.
            const gy = try allocator.alloc(f32, len);
            defer allocator.free(gy);
            fillRandom(rng, gy);
            rows_impl.softmaxRows(.{ .input = input, .output = out_a, .axis_dim = cols, .row_start = 0, .row_end = n_rows });
            const y = out_a;
            const gx_a = try allocator.alloc(f32, len);
            defer allocator.free(gx_a);
            const gx_b = try allocator.alloc(f32, len);
            defer allocator.free(gx_b);
            rows_impl.softmaxBackwardRows(.{ .y = y, .gy = gy, .output = gx_a, .axis_dim = cols, .scale = 0.5, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.softmaxBackwardRows(.{ .y = y, .gy = gy, .output = gx_b, .axis_dim = cols, .scale = 0.5, .row_start = 0, .row_end = n_rows });
            try expectClose(gx_b, gx_a, tol);

            // Distill statistics + backward share the CE stats layout.
            const stats_a = try allocator.alloc(f32, 2 * n_rows);
            defer allocator.free(stats_a);
            const stats_b = try allocator.alloc(f32, 2 * n_rows);
            defer allocator.free(stats_b);
            rows_impl.distillStatsRows(.{ .input = input, .row_stats = stats_a, .class_count = cols, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.distillStatsRows(.{ .input = input, .row_stats = stats_b, .class_count = cols, .row_start = 0, .row_end = n_rows });
            try expectClose(stats_b, stats_a, tol);
            const mass = try allocator.alloc(f32, n_rows);
            defer allocator.free(mass);
            fillRandom(rng, mass);
            rows_impl.distillBackwardRows(.{ .input = input, .output = gx_a, .row_stats = stats_a, .row_mass = mass, .class_count = cols, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.distillBackwardRows(.{ .input = input, .output = gx_b, .row_stats = stats_a, .row_mass = mass, .class_count = cols, .row_start = 0, .row_end = n_rows });
            try expectClose(gx_b, gx_a, tol);
        }
    }

    test "parity: cross-entropy row kernels vs scalar twins" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0xce10);
        const rng = prng.random();
        const n_rows: usize = 6;
        const cols: usize = 129;
        const len = n_rows * cols;
        const input = try allocator.alloc(f32, len);
        defer allocator.free(input);
        fillRandom(rng, input);
        var labels: [n_rows]usize = .{ 3, 100, 7, 2, 128, 0 };
        labels[3] = 42; // one row hits the ignore_index below
        const tol = 1e-6 * @as(f32, @floatFromInt(cols));

        for ([_]f32{ 0, 0.1 }) |smoothing| {
            const losses_a = try allocator.alloc(f32, n_rows);
            defer allocator.free(losses_a);
            const losses_b = try allocator.alloc(f32, n_rows);
            defer allocator.free(losses_b);
            const stats_a = try allocator.alloc(f32, 2 * n_rows);
            defer allocator.free(stats_a);
            const stats_b = try allocator.alloc(f32, 2 * n_rows);
            defer allocator.free(stats_b);
            rows_impl.crossEntropyLossRows(.{ .input = input, .labels = &labels, .row_losses = losses_a, .row_stats = stats_a, .class_count = cols, .ignore_index = 42, .label_smoothing = smoothing, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.crossEntropyLossRows(.{ .input = input, .labels = &labels, .row_losses = losses_b, .row_stats = stats_b, .class_count = cols, .ignore_index = 42, .label_smoothing = smoothing, .row_start = 0, .row_end = n_rows });
            try expectClose(losses_b, losses_a, tol);
            try expectClose(stats_b, stats_a, tol);

            const gx_a = try allocator.alloc(f32, len);
            defer allocator.free(gx_a);
            const gx_b = try allocator.alloc(f32, len);
            defer allocator.free(gx_b);
            // The forward-saved-stats one-pass arm (same stats for both sides).
            rows_impl.crossEntropyBackwardRows(.{ .input = input, .labels = &labels, .output = gx_a, .per_row_scale = null, .row_stats = stats_a, .class_count = cols, .ignore_index = 42, .label_smoothing = smoothing, .grad_common = 0.25, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.crossEntropyBackwardRows(.{ .input = input, .labels = &labels, .output = gx_b, .per_row_scale = null, .row_stats = stats_a, .class_count = cols, .ignore_index = 42, .label_smoothing = smoothing, .grad_common = 0.25, .row_start = 0, .row_end = n_rows });
            try expectClose(gx_b, gx_a, tol);
            // The recompute arm.
            rows_impl.crossEntropyBackwardRows(.{ .input = input, .labels = &labels, .output = gx_a, .per_row_scale = null, .row_stats = null, .class_count = cols, .ignore_index = 42, .label_smoothing = smoothing, .grad_common = 0.25, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.crossEntropyBackwardRows(.{ .input = input, .labels = &labels, .output = gx_b, .per_row_scale = null, .row_stats = null, .class_count = cols, .ignore_index = 42, .label_smoothing = smoothing, .grad_common = 0.25, .row_start = 0, .row_end = n_rows });
            try expectClose(gx_b, gx_a, tol);
        }
    }

    test "parity: gated-activation row kernels vs scalar twins" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x619d);
        const rng = prng.random();
        const outer: usize = 4;
        const half: usize = 37;
        const len = outer * 2 * half;
        const input = try allocator.alloc(f32, len);
        defer allocator.free(input);
        fillRandom(rng, input);
        const grad = try allocator.alloc(f32, outer * half);
        defer allocator.free(grad);
        fillRandom(rng, grad);
        const out_a = try allocator.alloc(f32, len);
        defer allocator.free(out_a);
        const out_b = try allocator.alloc(f32, len);
        defer allocator.free(out_b);

        rows_impl.splitSwiGluRows(.{ .input = input, .output = out_a, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        rows_impl.scalar.splitSwiGluRows(.{ .input = input, .output = out_b, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        try expectClose(out_b[0 .. outer * half], out_a[0 .. outer * half], 1e-5);

        rows_impl.splitGluRows(.{ .input = input, .output = out_a, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        rows_impl.scalar.splitGluRows(.{ .input = input, .output = out_b, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        try expectClose(out_b[0 .. outer * half], out_a[0 .. outer * half], 1e-5);

        rows_impl.splitSwiGluBackwardRows(.{ .input = input, .grad = grad, .output = out_a, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        rows_impl.scalar.splitSwiGluBackwardRows(.{ .input = input, .grad = grad, .output = out_b, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        try expectClose(out_b, out_a, 1e-5);

        rows_impl.splitGluBackwardRows(.{ .input = input, .grad = grad, .output = out_a, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        rows_impl.scalar.splitGluBackwardRows(.{ .input = input, .grad = grad, .output = out_b, .axis_dim = 2 * half, .half = half, .outer_start = 0, .outer_end = outer });
        try expectClose(out_b, out_a, 1e-5);
    }

    test "parity: rms-norm row kernels vs scalar twins" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x4a45);
        const rng = prng.random();
        const n_rows: usize = 5;
        const cols: usize = 173;
        const len = n_rows * cols;
        const eps: f32 = 1e-5;
        const inv_cols = 1 / @as(f32, @floatFromInt(cols));
        const input = try allocator.alloc(f32, len);
        defer allocator.free(input);
        fillRandom(rng, input);
        const weights = try allocator.alloc(f32, cols);
        defer allocator.free(weights);
        fillRandom(rng, weights);
        const grad = try allocator.alloc(f32, len);
        defer allocator.free(grad);
        fillRandom(rng, grad);
        const residual = try allocator.alloc(f32, len);
        defer allocator.free(residual);
        fillRandom(rng, residual);
        const out_a = try allocator.alloc(f32, len);
        defer allocator.free(out_a);
        const out_b = try allocator.alloc(f32, len);
        defer allocator.free(out_b);
        const tol = 1e-6 * @as(f32, @floatFromInt(cols));

        rows_impl.rmsNormMulRows(.{ .input = input, .weights = weights, .output = out_a, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        rows_impl.scalar.rmsNormMulRows(.{ .input = input, .weights = weights, .output = out_b, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        try expectClose(out_b, out_a, tol);

        rows_impl.rmsNormMulAddRows(.{ .input = input, .weights = weights, .residual = residual, .output = out_a, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        rows_impl.scalar.rmsNormMulAddRows(.{ .input = input, .weights = weights, .residual = residual, .output = out_b, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        try expectClose(out_b, out_a, tol);

        rows_impl.rmsNormMulBackwardInputRows(.{ .input = input, .weights = weights, .grad = grad, .output = out_a, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        rows_impl.scalar.rmsNormMulBackwardInputRows(.{ .input = input, .weights = weights, .grad = grad, .output = out_b, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        try expectClose(out_b, out_a, tol);

        @memset(out_a[0..cols], 0);
        @memset(out_b[0..cols], 0);
        rows_impl.rmsNormMulBackwardWeightRows(.{ .input = input, .grad = grad, .output = out_a[0..cols], .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        rows_impl.scalar.rmsNormMulBackwardWeightRows(.{ .input = input, .grad = grad, .output = out_b[0..cols], .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        try expectClose(out_b[0..cols], out_a[0..cols], tol * @as(f32, @floatFromInt(n_rows)));

        // The block-partial reduce: same partials, vector vs serial column walk.
        const block_count: usize = 3;
        const partials = try allocator.alloc(f32, block_count * cols);
        defer allocator.free(partials);
        fillRandom(rng, partials);
        @memset(out_a[0..cols], 0);
        @memset(out_b[0..cols], 0);
        rows_impl.rmsNormWeightGradReduce(.{ .partials = partials, .output = out_a[0..cols], .block_count = block_count, .axis_dim = cols, .col_start = 0, .col_end = cols });
        rows_impl.scalar.rmsNormWeightGradReduce(.{ .partials = partials, .output = out_b[0..cols], .block_count = block_count, .axis_dim = cols, .col_start = 0, .col_end = cols });
        try expectBitwise(out_b[0..cols], out_a[0..cols]);
    }

    test "parity: layer-norm row kernels vs scalar twins" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x1a9e);
        const rng = prng.random();
        const n_rows: usize = 5;
        const cols: usize = 173;
        const len = n_rows * cols;
        const eps: f32 = 1e-5;
        const inv_cols = 1 / @as(f32, @floatFromInt(cols));
        const input = try allocator.alloc(f32, len);
        defer allocator.free(input);
        fillRandom(rng, input);
        const weights = try allocator.alloc(f32, cols);
        defer allocator.free(weights);
        fillRandom(rng, weights);
        const biases = try allocator.alloc(f32, cols);
        defer allocator.free(biases);
        fillRandom(rng, biases);
        const grad = try allocator.alloc(f32, len);
        defer allocator.free(grad);
        fillRandom(rng, grad);
        const out_a = try allocator.alloc(f32, len);
        defer allocator.free(out_a);
        const out_b = try allocator.alloc(f32, len);
        defer allocator.free(out_b);
        const tol = 1e-6 * @as(f32, @floatFromInt(cols));
        const grad_tol = tol * @as(f32, @floatFromInt(n_rows));

        for ([_]bool{ false, true }) |affine| {
            const w: ?[]const f32 = if (affine) weights else null;
            const b: ?[]const f32 = if (affine) biases else null;
            rows_impl.layerNormRows(.{ .input = input, .weights = w, .biases = b, .output = out_a, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.layerNormRows(.{ .input = input, .weights = w, .biases = b, .output = out_b, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
            try expectClose(out_b, out_a, tol);

            rows_impl.layerNormBackwardInputRows(.{ .input = input, .weights = w, .grad = grad, .output = out_a, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
            rows_impl.scalar.layerNormBackwardInputRows(.{ .input = input, .weights = w, .grad = grad, .output = out_b, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
            try expectClose(out_b, out_a, tol);
        }

        // Serial dweight/dbias fallback.
        const dw_a = try allocator.alloc(f32, cols);
        defer allocator.free(dw_a);
        const dw_b = try allocator.alloc(f32, cols);
        defer allocator.free(dw_b);
        const db_a = try allocator.alloc(f32, cols);
        defer allocator.free(db_a);
        const db_b = try allocator.alloc(f32, cols);
        defer allocator.free(db_b);
        @memset(dw_a, 0);
        @memset(dw_b, 0);
        @memset(db_a, 0);
        @memset(db_b, 0);
        rows_impl.layerNormAffineParamGradRows(input, grad, dw_a, db_a, n_rows, cols, inv_cols, eps);
        rows_impl.scalar.layerNormAffineParamGradRows(input, grad, dw_b, db_b, n_rows, cols, inv_cols, eps);
        try expectClose(dw_b, dw_a, grad_tol);
        try expectBitwise(db_b, db_a);

        // Stats pass + the column-partitioned accumulation.
        const stats_a = try allocator.alloc(f32, 2 * n_rows);
        defer allocator.free(stats_a);
        const stats_b = try allocator.alloc(f32, 2 * n_rows);
        defer allocator.free(stats_b);
        rows_impl.layerNormRowStats(.{ .input = input, .stats = stats_a, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        rows_impl.scalar.layerNormRowStats(.{ .input = input, .stats = stats_b, .axis_dim = cols, .inv_axis_dim = inv_cols, .eps = eps, .row_start = 0, .row_end = n_rows });
        try expectClose(stats_b, stats_a, tol);
        @memset(dw_a, 0);
        @memset(dw_b, 0);
        @memset(db_a, 0);
        @memset(db_b, 0);
        rows_impl.layerNormParamGradColumns(.{ .input = input, .grad = grad, .stats = stats_a, .dweight = dw_a, .dbias = db_a, .rows = n_rows, .axis_dim = cols, .col_start = 0, .col_end = cols });
        rows_impl.scalar.layerNormParamGradColumns(.{ .input = input, .grad = grad, .stats = stats_a, .dweight = dw_b, .dbias = db_b, .rows = n_rows, .axis_dim = cols, .col_start = 0, .col_end = cols });
        // Same per-column row-order accumulation on both sides: bitwise.
        try expectBitwise(dw_b, dw_a);
        try expectBitwise(db_b, db_a);
    }

    test "parity: scatter-add rows twin is the entry's bytes" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x5ca7);
        const rng = prng.random();
        const src_rows: usize = 9;
        const dst_rows: usize = 4;
        const row_len: usize = 21;
        const grad = try allocator.alloc(f32, src_rows * row_len);
        defer allocator.free(grad);
        fillRandom(rng, grad);
        const indices = [_]usize{ 0, 3, 1, 3, 0, 2, 1, 0, 3 };
        const out_a = try allocator.alloc(f32, dst_rows * row_len);
        defer allocator.free(out_a);
        const out_b = try allocator.alloc(f32, dst_rows * row_len);
        defer allocator.free(out_b);
        rows_impl.scatterAddRows(.{ .grad = grad, .output = out_a, .indices = &indices, .row_len = row_len, .row_start = 0, .row_end = dst_rows });
        rows_impl.scalar.scatterAddRows(.{ .grad = grad, .output = out_b, .indices = &indices, .row_len = row_len, .row_start = 0, .row_end = dst_rows });
        try expectBitwise(out_b, out_a);
    }

    test "parity: inner-lane row kernels are the scalar twins' bytes" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x1a2e5);
        const rng = prng.random();
        const outer: usize = 3;
        const axis_dim: usize = 19;
        const inner: usize = 43; // odd: exercises the vector tails
        const len = outer * axis_dim * inner;
        const eps: f32 = 1e-5;
        const inv_axis = 1 / @as(f32, @floatFromInt(axis_dim));
        const input = try allocator.alloc(f32, len);
        defer allocator.free(input);
        fillRandom(rng, input);
        const grad = try allocator.alloc(f32, len);
        defer allocator.free(grad);
        fillRandom(rng, grad);
        const weights = try allocator.alloc(f32, axis_dim);
        defer allocator.free(weights);
        fillRandom(rng, weights);
        const out_a = try allocator.alloc(f32, len);
        defer allocator.free(out_a);
        const out_b = try allocator.alloc(f32, len);
        defer allocator.free(out_b);
        const scratch = try allocator.alloc(f32, 5 * inner);
        defer allocator.free(scratch);

        rows_impl.softmaxInner(.{ .input = input, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.softmaxInner(.{ .input = input, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b, out_a);

        rows_impl.logsumexpInner(.{ .input = input, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.logsumexpInner(.{ .input = input, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b[0 .. outer * inner], out_a[0 .. outer * inner]);

        rows_impl.logSoftmaxInner(.{ .input = input, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.logSoftmaxInner(.{ .input = input, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b, out_a);

        rows_impl.softmaxBackwardInner(.{ .y = input, .gy = grad, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0..inner], .scale = 0.5, .outer = outer, .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.softmaxBackwardInner(.{ .y = input, .gy = grad, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0..inner], .scale = 0.5, .outer = outer, .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b, out_a);

        rows_impl.varianceInner(.{ .input = input, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inv_axis_dim = inv_axis, .inv_denom = 1 / @as(f32, @floatFromInt(axis_dim - 1)), .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.varianceInner(.{ .input = input, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inv_axis_dim = inv_axis, .inv_denom = 1 / @as(f32, @floatFromInt(axis_dim - 1)), .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b[0 .. outer * inner], out_a[0 .. outer * inner]);

        inline for ([_]type{ f32, f64 }) |Acc| {
            const acc_scratch = try allocator.alloc(Acc, 5 * inner);
            defer allocator.free(acc_scratch);
            rows_impl.standardizeInner(Acc, .{ .input = input, .output = out_a, .axis_dim = axis_dim, .inner = inner, .valid_count = axis_dim - 2, .ddof_count = 1, .eps = 1e-5, .eps_inside_sqrt = true, .scratch = acc_scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
            rows_impl.scalar.standardizeInner(Acc, .{ .input = input, .output = out_b, .axis_dim = axis_dim, .inner = inner, .valid_count = axis_dim - 2, .ddof_count = 1, .eps = 1e-5, .eps_inside_sqrt = true, .scratch = acc_scratch[0 .. 2 * inner], .outer = outer, .inner_start = 0, .inner_end = inner });
            try expectBitwise(out_b, out_a);

            @memset(out_a, 0);
            @memset(out_b, 0);
            rows_impl.standardizeBackwardInner(Acc, .{ .input = input, .grad = grad, .output = out_a, .axis_dim = axis_dim, .inner = inner, .valid_count = axis_dim - 2, .ddof_count = 1, .eps = 1e-5, .eps_inside_sqrt = false, .scratch = acc_scratch, .outer = outer, .inner_start = 0, .inner_end = inner });
            rows_impl.scalar.standardizeBackwardInner(Acc, .{ .input = input, .grad = grad, .output = out_b, .axis_dim = axis_dim, .inner = inner, .valid_count = axis_dim - 2, .ddof_count = 1, .eps = 1e-5, .eps_inside_sqrt = false, .scratch = acc_scratch, .outer = outer, .inner_start = 0, .inner_end = inner });
            try expectBitwise(out_b, out_a);
        }

        rows_impl.rmsNormInner(.{ .input = input, .weights = weights, .residual = grad, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0..inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps, .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.rmsNormInner(.{ .input = input, .weights = weights, .residual = grad, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0..inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps, .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b, out_a);

        for ([_]bool{ false, true }) |weighted| {
            const w: ?[]const f32 = if (weighted) weights else null;
            rows_impl.rmsNormBackwardInputInner(.{ .input = input, .weights = w, .grad = grad, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps, .inner_start = 0, .inner_end = inner });
            rows_impl.scalar.rmsNormBackwardInputInner(.{ .input = input, .weights = w, .grad = grad, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps, .inner_start = 0, .inner_end = inner });
            try expectBitwise(out_b, out_a);
        }

        @memset(out_a[0..axis_dim], 0);
        @memset(out_b[0..axis_dim], 0);
        rows_impl.rmsNormBackwardWeightInner(.{ .input = input, .grad = grad, .output = out_a[0..axis_dim], .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0..inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps });
        rows_impl.scalar.rmsNormBackwardWeightInner(.{ .input = input, .grad = grad, .output = out_b[0..axis_dim], .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0..inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps });
        try expectBitwise(out_b[0..axis_dim], out_a[0..axis_dim]);

        rows_impl.layerNormInner(.{ .input = input, .weights = weights, .biases = weights, .output = out_a, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps, .inner_start = 0, .inner_end = inner });
        rows_impl.scalar.layerNormInner(.{ .input = input, .weights = weights, .biases = weights, .output = out_b, .axis_dim = axis_dim, .inner = inner, .scratch = scratch[0 .. 2 * inner], .outer = outer, .inv_axis_dim = inv_axis, .eps = eps, .inner_start = 0, .inner_end = inner });
        try expectBitwise(out_b, out_a);

        // layerNormBackwardInner: dx keeps per-element order (bitwise); the
        // per-axis dweight/dbias reduce is the one lane-vec reduction in the
        // family, so those compare with tolerance.
        const dw_a = try allocator.alloc(f32, axis_dim);
        defer allocator.free(dw_a);
        const dw_b = try allocator.alloc(f32, axis_dim);
        defer allocator.free(dw_b);
        const db_a = try allocator.alloc(f32, axis_dim);
        defer allocator.free(db_a);
        const db_b = try allocator.alloc(f32, axis_dim);
        defer allocator.free(db_b);
        @memset(dw_a, 0);
        @memset(dw_b, 0);
        @memset(db_a, 0);
        @memset(db_b, 0);
        const ln_scratch = try allocator.alloc(f32, 4 * inner);
        defer allocator.free(ln_scratch);
        rows_impl.layerNormBackwardInner(.{ .input = input, .grad = grad, .weights = weights, .dx = out_a, .dweight = dw_a, .dbias = db_a, .axis_dim = axis_dim, .inner = inner, .scratch = ln_scratch, .inv_axis_dim = inv_axis, .eps = eps, .outer = outer });
        rows_impl.scalar.layerNormBackwardInner(.{ .input = input, .grad = grad, .weights = weights, .dx = out_b, .dweight = dw_b, .dbias = db_b, .axis_dim = axis_dim, .inner = inner, .scratch = ln_scratch, .inv_axis_dim = inv_axis, .eps = eps, .outer = outer });
        try expectBitwise(out_b, out_a);
        const inner_tol = 1e-6 * @as(f32, @floatFromInt(outer * inner));
        try expectClose(dw_b, dw_a, inner_tol);
        try expectClose(db_b, db_a, inner_tol);
    }

    // ---- Attention-kernel parity: the vector/attention.zig entries against
    // the serial three-pass twins. The per-query entries differ from the
    // twins in reduction order only; the query-tiled entry additionally
    // carries its documented online-softmax summation-order class.
    const attn_impl = @import("vector/attention.zig");

    fn attnParityCase(comptime KvElem: type, causal: bool, window: usize) !void {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0xa77e57);
        const rng = prng.random();
        const q_seq: usize = 5;
        const kv_seq: usize = 9;
        const heads: usize = 4;
        const kv_heads: usize = 2;
        const d: usize = 20;
        const scale: f32 = 0.25;
        const map = [_]usize{ 0, 0, 1, 1 };

        const q = try allocator.alloc(f32, q_seq * heads * d);
        defer allocator.free(q);
        fillRandom(rng, q);
        const kf = try allocator.alloc(f32, kv_seq * kv_heads * d);
        defer allocator.free(kf);
        fillRandom(rng, kf);
        const vf = try allocator.alloc(f32, kv_seq * kv_heads * d);
        defer allocator.free(vf);
        fillRandom(rng, vf);

        const k = try allocator.alloc(KvElem, kv_seq * kv_heads * d);
        defer allocator.free(k);
        const v = try allocator.alloc(KvElem, kv_seq * kv_heads * d);
        defer allocator.free(v);
        for (k, kf) |*dst, x| dst.* = if (KvElem == f32) x else @floatCast(x);
        for (v, vf) |*dst, x| dst.* = if (KvElem == f32) x else @floatCast(x);

        const out_a = try allocator.alloc(f32, q_seq * heads * d);
        defer allocator.free(out_a);
        const out_b = try allocator.alloc(f32, q_seq * heads * d);
        defer allocator.free(out_b);
        const scores = try allocator.alloc(f32, kv_seq * 2);
        defer allocator.free(scores);
        const stats_a = try allocator.alloc(f32, heads * q_seq * 2);
        defer allocator.free(stats_a);
        const stats_b = try allocator.alloc(f32, heads * q_seq * 2);
        defer allocator.free(stats_b);
        const tol: f32 = 1e-5;

        const base = attn_impl.GroupedCausalAttentionTask(KvElem){
            .q_data = q,
            .k_data = k,
            .v_data = v,
            .out_data = out_a,
            .kv_head_for_head = &map,
            .q_seq = q_seq,
            .kv_seq = kv_seq,
            .source_offset = kv_seq - q_seq,
            .heads = heads,
            .d = d,
            .kv_heads = kv_heads,
            .scale_value = scale,
            .window = window,
            .causal = causal,
            .head_start = 0,
            .head_end = heads,
            .scores = scores,
            .stats = stats_a,
        };
        attn_impl.groupedCausalAttentionHeads(KvElem, base);
        var twin = base;
        twin.out_data = out_b;
        twin.stats = stats_b;
        attn_twins.attnHeadsTwin(KvElem, twin);
        try expectClose(out_b, out_a, tol);
        try expectClose(stats_b, stats_a, tol * @as(f32, @floatFromInt(kv_seq)));

        // Adjacent-pair form on the same data.
        const pair = attn_impl.GroupedCausalAttentionPairTask(KvElem){
            .q_data = q,
            .k_data = k,
            .v_data = v,
            .out_data = out_a,
            .q_seq = q_seq,
            .kv_seq = kv_seq,
            .source_offset = kv_seq - q_seq,
            .heads = heads,
            .d = d,
            .kv_heads = kv_heads,
            .scale_value = scale,
            .window = window,
            .causal = causal,
            .kv_head_start = 0,
            .kv_head_end = kv_heads,
            .scores = scores,
        };
        attn_impl.groupedCausalAttentionHeadPairs(KvElem, pair);
        var pair_twin = pair;
        pair_twin.out_data = out_b;
        attn_twins.attnPairsTwin(KvElem, pair_twin);
        try expectClose(out_b, out_a, tol);

        // Query-tiled entry (head_group 1) over the full flattened work.
        const q_tile = attn_impl.attention_tile_rows;
        const n_tiles = (q_seq + q_tile - 1) / q_tile;
        const tiled = attn_impl.GroupedCausalAttentionTiledTask(KvElem){
            .q_data = q,
            .k_data = k,
            .v_data = v,
            .out_data = out_a,
            .kv_head_for_head = &map,
            .q_seq = q_seq,
            .kv_seq = kv_seq,
            .source_offset = kv_seq - q_seq,
            .heads = heads,
            .d = d,
            .kv_heads = kv_heads,
            .scale_value = scale,
            .window = window,
            .causal = causal,
            .n_tiles = n_tiles,
            .work_start = 0,
            .work_end = heads * n_tiles,
        };
        attn_impl.groupedCausalAttentionQueryTiles(KvElem, 1, tiled);
        var tiled_twin = tiled;
        tiled_twin.out_data = out_b;
        attn_twins.attnTilesTwin(KvElem, tiled_twin);
        try expectClose(out_b, out_a, tol * 10);
    }

    // Thin comptime shims so the case body above stays readable.
    const attn_twins = struct {
        fn attnHeadsTwin(comptime KvElem: type, task: attn_impl.GroupedCausalAttentionTask(KvElem)) void {
            attn_impl.scalar.groupedCausalAttentionHeads(KvElem, task);
        }
        fn attnPairsTwin(comptime KvElem: type, task: attn_impl.GroupedCausalAttentionPairTask(KvElem)) void {
            attn_impl.scalar.groupedCausalAttentionHeadPairs(KvElem, task);
        }
        fn attnTilesTwin(comptime KvElem: type, task: attn_impl.GroupedCausalAttentionTiledTask(KvElem)) void {
            attn_impl.scalar.groupedCausalAttentionQueryTiles(KvElem, 1, task);
        }
    };

    test "parity: attention forward kernels vs scalar twins" {
        try attnParityCase(f32, true, 0);
        try attnParityCase(f32, true, 4);
        try attnParityCase(f32, false, 0);
        try attnParityCase(f16, true, 0);
        try attnParityCase(f16, false, 0);
    }

    test "parity: attention backward kernels vs scalar twins" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0xba77e5);
        const rng = prng.random();
        const q_seq: usize = 6;
        const kv_seq: usize = 10;
        const heads: usize = 4;
        const kv_heads: usize = 2;
        const d: usize = 20;
        const scale: f32 = 0.25;
        const map = [_]usize{ 0, 0, 1, 1 };
        const len_q = q_seq * heads * d;
        const len_kv = kv_seq * kv_heads * d;

        const q = try allocator.alloc(f32, len_q);
        defer allocator.free(q);
        fillRandom(rng, q);
        const k = try allocator.alloc(f32, len_kv);
        defer allocator.free(k);
        fillRandom(rng, k);
        const v = try allocator.alloc(f32, len_kv);
        defer allocator.free(v);
        fillRandom(rng, v);
        const gy = try allocator.alloc(f32, len_q);
        defer allocator.free(gy);
        fillRandom(rng, gy);
        const tol: f32 = 1e-5 * @as(f32, @floatFromInt(kv_seq));

        // Per-query backward kernel vs twin.
        {
            const scores = try allocator.alloc(f32, kv_seq);
            defer allocator.free(scores);
            const dprob = try allocator.alloc(f32, kv_seq);
            defer allocator.free(dprob);
            const dq_a = try allocator.alloc(f32, len_q);
            defer allocator.free(dq_a);
            const dq_b = try allocator.alloc(f32, len_q);
            defer allocator.free(dq_b);
            const dk_a = try allocator.alloc(f32, len_kv);
            defer allocator.free(dk_a);
            const dk_b = try allocator.alloc(f32, len_kv);
            defer allocator.free(dk_b);
            const dv_a = try allocator.alloc(f32, len_kv);
            defer allocator.free(dv_a);
            const dv_b = try allocator.alloc(f32, len_kv);
            defer allocator.free(dv_b);
            @memset(dq_a, 0);
            @memset(dq_b, 0);
            @memset(dk_a, 0);
            @memset(dk_b, 0);
            @memset(dv_a, 0);
            @memset(dv_b, 0);
            const base = attn_impl.GroupedCausalAttentionBackwardTask{
                .q_data = q,
                .k_data = k,
                .v_data = v,
                .gy_data = gy,
                .q_grad = dq_a,
                .k_grad = dk_a,
                .v_grad = dv_a,
                .kv_head_for_head = &map,
                .q_seq = q_seq,
                .kv_seq = kv_seq,
                .source_offset = kv_seq - q_seq,
                .heads = heads,
                .d = d,
                .kv_heads = kv_heads,
                .scale_value = scale,
                .window = 0,
                .causal = true,
                .kv_head_start = 0,
                .kv_head_end = kv_heads,
                .scores = scores,
                .dprob = dprob,
            };
            attn_impl.groupedCausalAttentionBackwardKvHeads(base);
            var twin = base;
            twin.q_grad = dq_b;
            twin.k_grad = dk_b;
            twin.v_grad = dv_b;
            attn_impl.scalar.groupedCausalAttentionBackwardKvHeads(twin);
            try expectClose(dq_b, dq_a, tol);
            try expectClose(dk_b, dk_a, tol);
            try expectClose(dv_b, dv_a, tol);
        }

        // Tiled backward (direct mode over an identity head map) vs twin.
        {
            const id_map = [_]usize{ 0, 1 };
            const d_heads: usize = 2;
            const dlen_q = q_seq * d_heads * d;
            const qd = q[0..dlen_q];
            const gyd = gy[0..dlen_q];
            const scratch = try allocator.alloc(f32, 2 * attn_impl.attention_bwd_tile_rows * kv_seq);
            defer allocator.free(scratch);
            const dq_a = try allocator.alloc(f32, dlen_q);
            defer allocator.free(dq_a);
            const dq_b = try allocator.alloc(f32, dlen_q);
            defer allocator.free(dq_b);
            const dk_a = try allocator.alloc(f32, len_kv);
            defer allocator.free(dk_a);
            const dk_b = try allocator.alloc(f32, len_kv);
            defer allocator.free(dk_b);
            const dv_a = try allocator.alloc(f32, len_kv);
            defer allocator.free(dv_a);
            const dv_b = try allocator.alloc(f32, len_kv);
            defer allocator.free(dv_b);
            @memset(dq_a, 0);
            @memset(dq_b, 0);
            @memset(dk_a, 0);
            @memset(dk_b, 0);
            @memset(dv_a, 0);
            @memset(dv_b, 0);
            const base = attn_impl.GroupedCausalAttentionBackwardTiledTask{
                .q_data = qd,
                .k_data = k,
                .v_data = v,
                .gy_data = gyd,
                .stats = null,
                .q_grad = dq_a,
                .dk_target = dk_a,
                .dv_target = dv_a,
                .partial_mode = false,
                .kv_head_for_head = &id_map,
                .q_seq = q_seq,
                .kv_seq = kv_seq,
                .source_offset = kv_seq - q_seq,
                .heads = d_heads,
                .d = d,
                .kv_heads = kv_heads,
                .scale_value = scale,
                .window = 0,
                .causal = true,
                .scratch = scratch,
                .head_start = 0,
                .head_end = d_heads,
            };
            attn_impl.groupedCausalAttentionBackwardTiles(base);
            var twin = base;
            twin.q_grad = dq_b;
            twin.dk_target = dk_b;
            twin.dv_target = dv_b;
            attn_impl.scalar.groupedCausalAttentionBackwardTiles(twin);
            try expectClose(dq_b, dq_a, tol);
            try expectClose(dk_b, dk_a, tol);
            try expectClose(dv_b, dv_a, tol);
        }
    }

    test "parity: dtype cast rows vs scalar twins are bitwise" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0xca57);
        const rng = prng.random();
        for ([_]usize{ 1, 7, 64, 257 }) |n| {
            const src_f32 = try allocator.alloc(f32, n);
            defer allocator.free(src_f32);
            fillRandom(rng, src_f32);
            const f16_a = try allocator.alloc(f16, n);
            defer allocator.free(f16_a);
            const f16_b = try allocator.alloc(f16, n);
            defer allocator.free(f16_b);
            native.castF32ToF16(f16_a, src_f32);
            elementwise.scalar.castF32ToF16(f16_b, src_f32);
            try std.testing.expectEqualSlices(u16, @ptrCast(f16_b), @ptrCast(f16_a));

            const back_a = try allocator.alloc(f32, n);
            defer allocator.free(back_a);
            const back_b = try allocator.alloc(f32, n);
            defer allocator.free(back_b);
            native.castF16ToF32(back_a, f16_a);
            elementwise.scalar.castF16ToF32(back_b, f16_a);
            try expectBitwise(back_b, back_a);

            const bf16_a = try allocator.alloc(u16, n);
            defer allocator.free(bf16_a);
            const bf16_b = try allocator.alloc(u16, n);
            defer allocator.free(bf16_b);
            native.castF32ToBf16(bf16_a, src_f32);
            elementwise.scalar.castF32ToBf16(bf16_b, src_f32);
            try std.testing.expectEqualSlices(u16, bf16_b, bf16_a);

            native.castBf16ToF32(back_a, bf16_a);
            elementwise.scalar.castBf16ToF32(back_b, bf16_a);
            try expectBitwise(back_b, back_a);
        }
    }

    test "parity: straggler row kernels vs serial references" {
        const allocator = std.testing.allocator;
        var prng = std.Random.DefaultPrng.init(0x57a6);
        const rng = prng.random();
        const rows_n: usize = 4;
        const cols: usize = 133;
        const len = rows_n * cols;
        const input = try allocator.alloc(f32, len);
        defer allocator.free(input);
        fillRandom(rng, input);

        // Row extremum: @max/@min are exact selections, so the lane reduce
        // agrees with the serial walk bitwise.
        for (0..rows_n) |r| {
            const row = input[r * cols ..][0..cols];
            var smax = -std.math.inf(f32);
            var smin = std.math.inf(f32);
            for (row) |value| {
                smax = @max(smax, value);
                smin = @min(smin, value);
            }
            try std.testing.expectEqual(smax, native.extremumRowValue(true, row));
            try std.testing.expectEqual(smin, native.extremumRowValue(false, row));
        }

        // Variance rows: lane-reduced sums vs the serial twin (tolerance).
        {
            const out_a = try allocator.alloc(f32, rows_n);
            defer allocator.free(out_a);
            const out_b = try allocator.alloc(f32, rows_n);
            defer allocator.free(out_b);
            const inv_cols = 1 / @as(f32, @floatFromInt(cols));
            const inv_denom = 1 / @as(f32, @floatFromInt(cols - 1));
            native.varianceRowsInto(out_a, input, rows_n, cols, inv_cols, inv_denom);
            rows_impl.scalar.varianceRowsInto(out_b, input, rows_n, cols, inv_cols, inv_denom);
            try expectClose(out_b, out_a, 1e-6 * @as(f32, @floatFromInt(cols)));
        }

        // Fused-rope pair strip: purely elementwise, bitwise.
        {
            const pairs: usize = 37;
            const angles = try allocator.alloc(f32, 2 * pairs);
            defer allocator.free(angles);
            fillRandom(rng, angles);
            const weights = try allocator.alloc(f32, 2 * pairs);
            defer allocator.free(weights);
            fillRandom(rng, weights);
            const out_a = try allocator.alloc(f32, 2 * pairs);
            defer allocator.free(out_a);
            const out_b = try allocator.alloc(f32, 2 * pairs);
            defer allocator.free(out_b);
            const rms_scale: f32 = 0.75;
            native.ropeHalfPairsInto(out_a[0..pairs], out_a[pairs..], input[0..pairs], input[pairs..][0..pairs], weights[0..pairs], weights[pairs..], angles[0..pairs], angles[pairs..], rms_scale);
            for (0..pairs) |i| {
                const first = input[i] * rms_scale * weights[i];
                const second = input[pairs + i] * rms_scale * weights[pairs + i];
                out_b[i] = first * angles[pairs + i] - second * angles[i];
                out_b[pairs + i] = first * angles[i] + second * angles[pairs + i];
            }
            try expectBitwise(out_b, out_a);
        }

        // Scans: the strided-columns kernel is bitwise the serial twin on
        // every build; the last-axis kernel is bitwise on default builds and
        // the Hillis–Steele rounding class under -Dvector-scan.
        {
            const out_a = try allocator.alloc(f32, len);
            defer allocator.free(out_a);
            const out_b = try allocator.alloc(f32, len);
            defer allocator.free(out_b);
            inline for ([_]ops.ScanOp{ .sum, .prod }) |op| {
                inline for ([_]bool{ false, true }) |reverse| {
                    native.scanRows(op, reverse, input, out_a, rows_n, cols);
                    rows_impl.scalar.scanRows(op, reverse, input, out_b, rows_n, cols);
                    try expectClose(out_b, out_a, 1e-5 * @as(f32, @floatFromInt(cols)));
                    native.scanColumns(op, reverse, input, out_a, rows_n, cols);
                    rows_impl.scalar.scanColumns(op, reverse, input, out_b, rows_n, cols);
                    try expectBitwise(out_b, out_a);
                }
            }
        }

        // Masked select row: purely elementwise, bitwise for bool and f32
        // masks (the vectorized arms) on any build.
        {
            const flags = try allocator.alloc(bool, cols);
            defer allocator.free(flags);
            for (flags, 0..) |*flag, i| flag.* = (i % 3) != 0;
            const dst_a = try allocator.alloc(f32, cols);
            defer allocator.free(dst_a);
            const dst_b = try allocator.alloc(f32, cols);
            defer allocator.free(dst_b);
            native.selectRow(.bool, input[0..cols], flags, dst_a);
            for (dst_b, input[0..cols], flags) |*dst, value, keep| dst.* = if (keep) value else 0;
            try expectBitwise(dst_b, dst_a);
        }
    }
};

test {
    _ = Impl;
}
