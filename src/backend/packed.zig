const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

pub const PackedMatmulFormat = enum {
    f16_rhs_f32,
    bf16_rhs_f32,
};

pub fn preferredRhsFormat(comptime dtype: DType) PackedMatmulFormat {
    return switch (dtype) {
        .f16 => .f16_rhs_f32,
        .bf16 => .bf16_rhs_f32,
        else => @compileError("packed matmul RHS is not implemented for this dtype"),
    };
}

pub fn PackedMatmulRhsFor(comptime dtype: DType) type {
    return PackedMatmulRhs(preferredRhsFormat(dtype));
}

pub fn PackedMatmulRhs(comptime format_value: PackedMatmulFormat) type {
    return switch (format_value) {
        .f16_rhs_f32 => struct {
            rhs: Tensor,
            k: usize,
            n: usize,

            const Self = @This();
            pub const format = format_value;
            pub const source_dtype = DType.f16;
            pub const packed_dtype = DType.f32;

            pub fn deinit(self: *Self) void {
                self.rhs.deinit();
                self.* = undefined;
            }
        },
        .bf16_rhs_f32 => struct {
            rhs: Tensor,
            k: usize,
            n: usize,

            const Self = @This();
            pub const format = format_value;
            pub const source_dtype = DType.bf16;
            pub const packed_dtype = DType.f32;

            pub fn deinit(self: *Self) void {
                self.rhs.deinit();
                self.* = undefined;
            }
        },
    };
}

pub fn packRhs(
    allocator: Allocator,
    comptime dtype: DType,
    rhs: *const tensor.TensorOf(dtype),
) !PackedMatmulRhsFor(dtype) {
    return switch (comptime preferredRhsFormat(dtype)) {
        .f16_rhs_f32 => packF16RhsAsF32(allocator, rhs),
        .bf16_rhs_f32 => packBf16RhsAsF32(allocator, rhs),
    };
}

pub fn matmul2DIntoUncheckedPackedRhsTypedWithConfig(
    allocator: Allocator,
    comptime dtype: DType,
    out: *tensor.TensorOf(dtype_mod.outputDType(.matmul, dtype)),
    a: *const tensor.TensorOf(dtype),
    rhs: *const PackedMatmulRhsFor(dtype),
    m: usize,
    n: usize,
    k: usize,
    config: anytype,
    comptime matmul_f32: anytype,
) !void {
    if (rhs.k != k or rhs.n != n) return tensor.TensorError.ShapeMismatch;

    switch (comptime preferredRhsFormat(dtype)) {
        .f16_rhs_f32 => {
            const a16 = try a.dataConstChecked();
            const c16 = try out.dataChecked();
            if (comptime configHasPool(@TypeOf(config))) {
                if (m == 1) {
                    if (maybeParallelF16PackedGemv(config, c16[0..n], a16[0..k], rhs.rhs.dataConst(), n, k)) return;
                    matmulF16PackedGemvRange(c16[0..n], a16[0..k], rhs.rhs.dataConst(), n, k, 0, n);
                    return;
                }
            }

            var a32 = try Tensor.zeros(allocator, &.{ m, k });
            defer a32.deinit();
            var c32 = try Tensor.zeros(allocator, &.{ m, n });
            defer c32.deinit();

            widenF16ToF32(a32.data(), a16[0 .. m * k]);
            matmul_f32(&c32, &a32, &rhs.rhs, m, n, k, config);
            narrowF32ToF16(c16[0 .. m * n], c32.dataConst());
        },
        .bf16_rhs_f32 => {
            const a_bits = try a.dataConstChecked();
            const c_bits = try out.dataChecked();
            if (comptime configHasPool(@TypeOf(config))) {
                if (m == 1) {
                    if (maybeParallelBf16PackedGemv(config, c_bits[0..n], a_bits[0..k], rhs.rhs.dataConst(), n, k)) return;
                    matmulBf16PackedGemvRange(c_bits[0..n], a_bits[0..k], rhs.rhs.dataConst(), n, k, 0, n);
                    return;
                }
            }

            var a32 = try Tensor.zeros(allocator, &.{ m, k });
            defer a32.deinit();
            var c32 = try Tensor.zeros(allocator, &.{ m, n });
            defer c32.deinit();

            widenBf16ToF32(a32.data(), a_bits[0 .. m * k]);
            matmul_f32(&c32, &a32, &rhs.rhs, m, n, k, config);
            narrowF32ToBf16(c_bits[0 .. m * n], c32.dataConst());
        },
    }
}

fn packF16RhsAsF32(
    allocator: Allocator,
    rhs: *const tensor.TensorOf(.f16),
) !PackedMatmulRhs(.f16_rhs_f32) {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);

    const src = try rhs.dataConstChecked();
    var packed_rhs = try Tensor.zeros(allocator, &.{ k, n });
    errdefer packed_rhs.deinit();
    widenF16ToF32(packed_rhs.data(), src);
    return .{ .rhs = packed_rhs, .k = k, .n = n };
}

fn packBf16RhsAsF32(
    allocator: Allocator,
    rhs: *const tensor.TensorOf(.bf16),
) !PackedMatmulRhs(.bf16_rhs_f32) {
    const view = try rhs.rankView(2);
    const k = view.dim(0);
    const n = view.dim(1);

    const src = try rhs.dataConstChecked();
    var packed_rhs = try Tensor.zeros(allocator, &.{ k, n });
    errdefer packed_rhs.deinit();
    widenBf16ToF32(packed_rhs.data(), src);
    return .{ .rhs = packed_rhs, .k = k, .n = n };
}

const f16_bridge_vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
const Vf16Bridge = @Vector(f16_bridge_vector_len, f16);
const Vf32Bridge = @Vector(f16_bridge_vector_len, f32);
const Vu16Bridge = @Vector(f16_bridge_vector_len, u16);
const Vu32Bridge = @Vector(f16_bridge_vector_len, u32);

const F16PackedGemvTask = struct {
    out: []f16,
    lhs: []const f16,
    rhs: []const f32,
    n: usize,
    k: usize,
    c0: usize,
    c1: usize,
};

const Bf16PackedGemvTask = struct {
    out: []u16,
    lhs: []const u16,
    rhs: []const f32,
    n: usize,
    k: usize,
    c0: usize,
    c1: usize,
};

fn maybeParallelF16PackedGemv(config: anytype, out: []f16, lhs: []const f16, rhs: []const f32, n: usize, k: usize) bool {
    const pool = configPool(config) orelse return false;
    const thread_count = packedGemvThreadCount(n, k);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]F16PackedGemvTask = undefined;
    var wait_group: thread.WaitGroup = .{};
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .out = out,
            .lhs = lhs,
            .rhs = rhs,
            .n = n,
            .k = k,
            .c0 = ti * n / thread_count,
            .c1 = (ti + 1) * n / thread_count,
        };
    }
    for (tasks[1..thread_count]) |*task| _ = pool.spawnWg(&wait_group, runF16PackedGemvTask, .{task});
    runF16PackedGemvTask(&tasks[0]);
    pool.waitAndWork(&wait_group);
    return true;
}

fn maybeParallelBf16PackedGemv(config: anytype, out: []u16, lhs: []const u16, rhs: []const f32, n: usize, k: usize) bool {
    const pool = configPool(config) orelse return false;
    const thread_count = packedGemvThreadCount(n, k);
    if (thread_count == 1) return false;

    var tasks: [parallel.vector_max_threads]Bf16PackedGemvTask = undefined;
    var wait_group: thread.WaitGroup = .{};
    for (0..thread_count) |ti| {
        tasks[ti] = .{
            .out = out,
            .lhs = lhs,
            .rhs = rhs,
            .n = n,
            .k = k,
            .c0 = ti * n / thread_count,
            .c1 = (ti + 1) * n / thread_count,
        };
    }
    for (tasks[1..thread_count]) |*task| _ = pool.spawnWg(&wait_group, runBf16PackedGemvTask, .{task});
    runBf16PackedGemvTask(&tasks[0]);
    pool.waitAndWork(&wait_group);
    return true;
}

fn configPool(config: anytype) ?*thread.Pool {
    const Config = @TypeOf(config);
    return if (configHasPool(Config)) config.pool else null;
}

fn configHasPool(comptime Config: type) bool {
    return @typeInfo(Config) == .@"struct" and @hasField(Config, "pool");
}

fn packedGemvThreadCount(n: usize, k: usize) usize {
    if (n < parallel.vector_column_min_n or k == 0) return 1;
    const work = parallel.saturatedMul3(1, n, k);
    if (work < parallel.vector_matmul_work_threshold) return 1;
    const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
    return @max(@as(usize, 1), @min(@min(cpu_count, n / parallel.vector_column_chunk), n));
}

fn runF16PackedGemvTask(task: *const F16PackedGemvTask) void {
    matmulF16PackedGemvRange(task.out, task.lhs, task.rhs, task.n, task.k, task.c0, task.c1);
}

fn runBf16PackedGemvTask(task: *const Bf16PackedGemvTask) void {
    matmulBf16PackedGemvRange(task.out, task.lhs, task.rhs, task.n, task.k, task.c0, task.c1);
}

fn matmulF16PackedGemvRange(out: []f16, lhs: []const f16, rhs: []const f32, n: usize, k: usize, c0: usize, c1: usize) void {
    var j = c0;
    while (j + f16_bridge_vector_len <= c1) : (j += f16_bridge_vector_len) {
        var acc: Vf32Bridge = @splat(0);
        for (0..k) |p| {
            const a32: Vf32Bridge = @splat(@as(f32, @floatCast(lhs[p])));
            const b: Vf32Bridge = rhs[p * n + j ..][0..f16_bridge_vector_len].*;
            acc += a32 * b;
        }
        out[j..][0..f16_bridge_vector_len].* = @as(Vf16Bridge, @floatCast(acc));
    }
    while (j < c1) : (j += 1) {
        var acc: f32 = 0;
        for (0..k) |p| acc += @as(f32, @floatCast(lhs[p])) * rhs[p * n + j];
        out[j] = @floatCast(acc);
    }
}

fn matmulBf16PackedGemvRange(out: []u16, lhs: []const u16, rhs: []const f32, n: usize, k: usize, c0: usize, c1: usize) void {
    var j = c0;
    while (j + f16_bridge_vector_len <= c1) : (j += f16_bridge_vector_len) {
        var acc: Vf32Bridge = @splat(0);
        for (0..k) |p| {
            const a32: Vf32Bridge = @splat(dtype_mod.bf16ToF32(lhs[p]));
            const b: Vf32Bridge = rhs[p * n + j ..][0..f16_bridge_vector_len].*;
            acc += a32 * b;
        }
        out[j..][0..f16_bridge_vector_len].* = f32VecToBf16(acc);
    }
    while (j < c1) : (j += 1) {
        var acc: f32 = 0;
        for (0..k) |p| acc += dtype_mod.bf16ToF32(lhs[p]) * rhs[p * n + j];
        out[j] = dtype_mod.f32ToBf16(acc);
    }
}

inline fn f32VecToBf16(values: Vf32Bridge) Vu16Bridge {
    const bits: Vu32Bridge = @bitCast(values);
    const lsb = (bits >> @as(Vu32Bridge, @splat(16))) & @as(Vu32Bridge, @splat(1));
    const rounded = bits + @as(Vu32Bridge, @splat(0x7fff)) + lsb;
    return @truncate(rounded >> @as(Vu32Bridge, @splat(16)));
}

pub fn widenF16ToF32(dst: []f32, src: []const f16) void {
    var i: usize = 0;
    while (i + 4 * f16_bridge_vector_len <= src.len) : (i += 4 * f16_bridge_vector_len) {
        dst[i..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i..][0..f16_bridge_vector_len].*)));
        dst[i + f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i + f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
        dst[i + 2 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i + 2 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
        dst[i + 3 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i + 3 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
    }
    while (i + f16_bridge_vector_len <= src.len) : (i += f16_bridge_vector_len) {
        dst[i..][0..f16_bridge_vector_len].* = @as(Vf32Bridge, @floatCast(@as(Vf16Bridge, src[i..][0..f16_bridge_vector_len].*)));
    }
    while (i < src.len) : (i += 1) {
        dst[i] = @floatCast(src[i]);
    }
}

pub fn narrowF32ToF16(dst: []f16, src: []const f32) void {
    var i: usize = 0;
    while (i + 4 * f16_bridge_vector_len <= src.len) : (i += 4 * f16_bridge_vector_len) {
        dst[i..][0..f16_bridge_vector_len].* = @as(Vf16Bridge, @floatCast(@as(Vf32Bridge, src[i..][0..f16_bridge_vector_len].*)));
        dst[i + f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf16Bridge, @floatCast(@as(Vf32Bridge, src[i + f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
        dst[i + 2 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf16Bridge, @floatCast(@as(Vf32Bridge, src[i + 2 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
        dst[i + 3 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].* = @as(Vf16Bridge, @floatCast(@as(Vf32Bridge, src[i + 3 * f16_bridge_vector_len ..][0..f16_bridge_vector_len].*)));
    }
    while (i + f16_bridge_vector_len <= src.len) : (i += f16_bridge_vector_len) {
        dst[i..][0..f16_bridge_vector_len].* = @as(Vf16Bridge, @floatCast(@as(Vf32Bridge, src[i..][0..f16_bridge_vector_len].*)));
    }
    while (i < src.len) : (i += 1) {
        dst[i] = @floatCast(src[i]);
    }
}

// bf16 is stored as raw u16 bits, so widening/narrowing is bit manipulation
// rather than a float cast. Reuse the canonical scalar converters so rounding
// matches the rest of the runtime; the simple loops auto-vectorize in release.
pub fn widenBf16ToF32(dst: []f32, src: []const u16) void {
    for (dst, src) |*d, s| d.* = dtype_mod.bf16ToF32(s);
}

pub fn narrowF32ToBf16(dst: []u16, src: []const f32) void {
    for (dst, src) |*d, s| d.* = dtype_mod.f32ToBf16(s);
}

test {
    _ = @import("packed_tests.zig");
}
