//! Reference oracles shared by the GPU provider test suites: the f64 GEMM
//! reference, quantized weight construction, and the quantized-GEMM row
//! comparison. Provider-agnostic on purpose — `backend/gpu_conformance.zig`
//! and each provider's own tests (`metal.zig`, `cuda.zig`) all call these
//! instead of carrying a copy.
//!
//! `KernelFormatTag` is per-provider (see `gpu_provider.zig`), so the format-keyed
//! helpers take the tag generically and resolve it against `dtype`'s block
//! registry rather than restating a switch per provider.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const gpu_provider = @import("gpu_provider.zig");
const quant = @import("quant.zig");

const DType = dtype_mod.DType;
const Orient = gpu_provider.Orient;

/// The `DType` a provider `KernelFormatTag` tag names. Provider tags are spelled
/// exactly like their dtypes, so this is a lookup, not a mapping table that
/// can drift.
pub fn dtypeForFormat(comptime fmt: anytype) DType {
    const name = @tagName(fmt);
    if (comptime std.mem.eql(u8, name, "tq2_0_folded")) @compileError(
        "tq2_0_folded is a fused PTQTP plane-pair layout with no DType of its own; " ++
            "it is covered by the provider's dedicated parity test",
    );
    if (!@hasField(DType, name)) @compileError("no DType named `" ++ name ++ "` for KernelFormatTag tag");
    return @field(DType, name);
}

/// The block struct backing a provider `KernelFormatTag` tag (`dtype.block_formats`).
pub fn BlockFor(comptime fmt: anytype) type {
    return dtype_mod.Storage(dtypeForFormat(fmt));
}

/// f64-accumulated dense GEMM in every operand orientation: the oracle both
/// providers' f32 parity tests compare against.
pub fn cpuReference(orient: Orient, a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f64 = 0;
            for (0..k) |p| {
                const av: f64 = switch (orient) {
                    .nn, .nt => a[i * k + p],
                    .tn => a[p * m + i],
                };
                const bv: f64 = switch (orient) {
                    .nn, .tn => b[p * n + j],
                    .nt => b[j * k + p],
                };
                acc += av * bv;
            }
            c[i * n + j] = @floatCast(acc);
        }
    }
}

/// Fill `blocks` with `n` quantized rows of `k` random values and return the
/// dequantized reference the GPU result is compared against (caller owns it).
pub fn buildQuantWeights(
    comptime fmt: anytype,
    allocator: std.mem.Allocator,
    random: std.Random,
    blocks: anytype,
    n: usize,
    k: usize,
) ![]f32 {
    const dt = comptime dtypeForFormat(fmt);
    const bpr = blocks.len / n;
    const wref = try allocator.alloc(f32, n * k);
    errdefer allocator.free(wref);
    const row_src = try allocator.alloc(f32, k);
    defer allocator.free(row_src);
    for (0..n) |r| {
        for (row_src) |*x| x.* = random.floatNorm(f32);
        try quant.quantizeRowForDType(dt, blocks[r * bpr ..][0..bpr], row_src);
        try quant.dequantizeRowForDType(dt, wref[r * k ..][0..k], blocks[r * bpr ..][0..bpr]);
    }
    return wref;
}

/// Compare a quantized GEMM result against an f16-operand reference: the GPU
/// stores both the dequantized weights and the f32 activations as half in
/// threadgroup/shared memory and accumulates in f32, so the oracle rounds its
/// operands the same way.
pub fn expectQuantGemmRows(
    a: []const f32,
    wref: []const f32,
    c: []const f32,
    m: usize,
    n: usize,
    k: usize,
) !void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f64 = 0;
            for (0..k) |p| {
                const av: f16 = @floatCast(a[i * k + p]);
                const wv: f16 = @floatCast(wref[j * k + p]);
                acc += @as(f64, av) * @as(f64, wv);
            }
            const got: f32 = c[i * n + j];
            const want: f32 = @floatCast(acc);
            const tol = @max(5e-3 * @max(@abs(want), @abs(got)), 5e-3);
            if (@abs(got - want) > tol) {
                std.debug.print(
                    "quant gemm mismatch m={d} n={d} k={d} at ({d},{d}): got={e} want={e}\n",
                    .{ m, n, k, i, j, got, want },
                );
                return error.TestUnexpectedResult;
            }
        }
    }
}
