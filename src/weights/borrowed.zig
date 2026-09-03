//! Zero-copy linears over caller-owned immutable bytes: the borrowed f16
//! and block-quantized RHS forwards (`linearSeqBorrowedF16`,
//! `linearSeqBorrowedQuantized`) and the fused grouped q8_0 low-rank GEMV
//! (`packGroupedQ8_0Rhs`, `groupedQ8_0GemvFusedInto`).

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const ag_mod = @import("../ag.zig");

const common = @import("common.zig");
const backend_kernels = @import("../backend.zig").kernels;

const Tensor = ag_mod.Tensor;
const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const RhsLifetime = exec_mod.RhsLifetime;
const Error = common.Error;
const backend_quant = common.backend_quant;
const blockSlice = common.blockSlice;
const BlockStorage = common.BlockStorage;
const Tag = @TypeOf(.tag);
const Allocator = std.mem.Allocator;

pub const BorrowedQuantLinearOptions = struct {
    allow_gpu: bool = true,
    rhs_lifetime: RhsLifetime = .transient,
};

/// Zero-copy f16 RHS linear over caller-owned immutable bytes. This stays in
/// the Tensor world: the bytes become a borrowed typed Tensor and the public
/// `dot` facade chooses the f16 matmul implementation.
pub fn linearSeqBorrowedF16(
    ctx: *ExecContext,
    input: anytype,
    bytes: []const u8,
    shape: [2]usize,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !Tensor(.{ .seq, out_tag }) {
    const values = try f16Slice(bytes, try std.math.mul(usize, shape[0], shape[1]));
    var rhs = try Tensor(.{ .dtype = .f16, .tags = .{ out_tag, in_tag } }).fromBorrowedConstSlice(ctx, shape, values);
    defer rhs.deinit();
    return input.dot(ctx, &rhs, in_tag);
}

/// Fused grouped low-rank GEMV (the deepseek4 attention output's stage-A
/// stack): `n_groups` independent q8_0 linears of `rank x group_dim`,
/// group g reading activation slice g, results written in group order —
/// exactly the concat a per-group facade path would produce. One LHS
/// quantization per group and ONE worker-team dispatch replace n_groups
/// facade linears + concat, bit-identically: the same q8_0 tile kernel
/// computes the same integer dots, and i32 accumulation is order-free.
/// Load-time x4 packing for `groupedQ8_0GemvFusedInto`: one pack per
/// group — same bytes column-interleaved, the difference between the
/// generic q8_0 row kernel and the hot sdot tile path (the bench-ternary
/// "x4 pack (load-time, same bytes)" pattern). Caller owns the slice;
/// free each pack's deinit then the slice.
pub fn packGroupedQ8_0Rhs(
    allocator: Allocator,
    weight_bytes: []const u8,
    n_groups: usize,
    rank: usize,
    group_dim: usize,
) ![]backend_quant.QuantizedMatmulRhsQ8_0x4 {
    if (group_dim % 32 != 0 or rank % 4 != 0 or n_groups == 0) return Error.InvalidWeightShape;
    const bpr = group_dim / 32;
    const row_bytes = bpr * @sizeOf(dtype_mod.BlockQ8_0);
    if (weight_bytes.len != n_groups * rank * row_bytes) return Error.InvalidWeightShape;
    const packs = try allocator.alloc(backend_quant.QuantizedMatmulRhsQ8_0x4, n_groups);
    var built: usize = 0;
    errdefer {
        for (packs[0..built]) |*p| p.deinit();
        allocator.free(packs);
    }
    const all = std.mem.bytesAsSlice(dtype_mod.BlockQ8_0, weight_bytes);
    for (0..n_groups) |g| {
        packs[g] = try backend_kernels.packMatmulRhsQ8_0x4(allocator, @alignCast(all[g * rank * bpr ..][0 .. rank * bpr]), rank, group_dim, bpr);
        built += 1;
    }
    return packs;
}

pub fn groupedQ8_0GemvFusedInto(
    ctx: *ExecContext,
    x: []const f32,
    rhs_packs: []const backend_quant.QuantizedMatmulRhsQ8_0x4,
    rank: usize,
    group_dim: usize,
    out: []f32,
) !void {
    const n_groups = rhs_packs.len;
    if (group_dim % 32 != 0 or n_groups == 0 or n_groups > 8) return Error.InvalidWeightShape;
    const bpr = group_dim / 32;
    if (x.len != n_groups * group_dim or out.len != n_groups * rank) return Error.InvalidWeightShape;

    const allocator = ctx.allocator();
    const lhs = try allocator.alloc(dtype_mod.BlockQ8_0, n_groups * bpr);
    defer allocator.free(lhs);
    for (0..n_groups) |g| {
        try backend_kernels.quantizeRowQ8_0Into(lhs[g * bpr ..][0..bpr], x[g * group_dim ..][0..group_dim]);
    }

    const Task = struct {
        out: []f32,
        lhs: []const dtype_mod.BlockQ8_0,
        rhs: *const backend_quant.QuantizedMatmulRhsQ8_0x4,
        n: usize,

        fn run(task: *const @This()) void {
            backend_kernels.matmulQ8_0x4RhsTile(task.out, task.lhs, task.rhs, task.n, 0, 1, 0, task.n);
        }
    };
    var tasks: [8]Task = undefined;
    for (0..n_groups) |g| {
        tasks[g] = .{
            .out = out[g * rank ..][0..rank],
            .lhs = lhs[g * bpr ..][0..bpr],
            .rhs = &rhs_packs[g],
            .n = rank,
        };
    }
    if (ctx.workPool()) |pool| {
        pool.parallelChunks(Task, tasks[0..n_groups], Task.run);
    } else {
        for (tasks[0..n_groups]) |*t| Task.run(t);
    }
}

/// Zero-copy block-quantized RHS linear over caller-owned immutable bytes. This
/// remains an LLM/runtime helper because the call carries raw GGUF block bytes
/// plus a backend RHS lifetime policy; callers still pass and receive ordinary
/// Tensor values.
pub fn linearSeqBorrowedQuantized(
    comptime dtype: DType,
    ctx: *ExecContext,
    input: anytype,
    bytes: []const u8,
    shape: [2]usize,
    options: BorrowedQuantLinearOptions,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !Tensor(.{ .seq, out_tag }) {
    comptime switch (dtype) {
        .q8_0, .q4_k, .q5_k, .q6_k => {},
        else => @compileError("borrowed quantized linear supports q8_0/q4_k/q5_k/q6_k"),
    };
    if (input.requiresGrad()) return Error.UnsupportedGradient;
    if (input.dim(in_tag) != shape[1]) return Error.InvalidWeightShape;

    const blocks = try blockSlice(BlockStorage(dtype), bytes);
    const qrhs = try ctx.compactMatmulRhsFromBlocks(dtype, blocks, shape[0], shape[1]);
    const lhs: exec_mod.QuantMatmulLhs = .{ .plain = input.asRawTensor() };
    var value = if (!options.allow_gpu)
        try ctx.matmulQuant(lhs, &qrhs, .{ .placement = .cpu })
    else switch (options.rhs_lifetime) {
        .transient => try ctx.matmulQuant(lhs, &qrhs, .{}),
        .stable_process => try ctx.matmulQuant(lhs, &qrhs, .{ .rhs_lifetime = .stable_process }),
    };
    errdefer value.deinit();
    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, value);
}

fn f16Slice(bytes: []const u8, expected_len: usize) ![]const f16 {
    if (bytes.len != expected_len * @sizeOf(u16)) return Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(f16) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(f16)) const u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(f16, aligned);
}
