//! PTQTP trit-plane weight containers (docs/PTQTP.md): the plane-pair
//! `WeightPtqtp`, the native-folded `WeightPtqtpFx4`, and their fused
//! forward paths (`linearSeqPtqtpFused`, `linearSeqFx4`). `LinearWeight`'s
//! `.ptqtp` / `.tq2_0_fx4` arms dispatch here.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const ag_mod = @import("../ag.zig");
const tensor_mod = @import("../tensor.zig");
const parallel = @import("../parallel.zig");

const common = @import("common.zig");

const offload = backend_mod.offload;
const Tensor = ag_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const Error = common.Error;
const QuantWeight = common.QuantWeight;
const backend_quant = common.backend_quant;
const Tag = @TypeOf(.tag);
const Allocator = std.mem.Allocator;

/// Fused multi-plane ternary linear for PTQTP-decorated weights: quantize
/// the activation rows to Q8_K ONCE, then run every plane inside a SINGLE
/// worker-team dispatch — column-partitioned tasks each compute all K
/// planes for their column range and sum in the fixed plane order. The
/// per-element arithmetic and the plane-add order match the facade
/// per-plane dot chain exactly, so results are bitwise equal to the
/// fallback; what changes is the dispatch count — one fork-join per
/// linear instead of one per plane. At decode-sized GEMVs the pool's
/// per-dispatch barrier cost rivals the ternary kernel itself, so
/// per-plane dispatches would serialize the team; sharing one dispatch
/// (and one LHS quantization) across planes keeps the multi-plane cost
/// near K kernel passes. Returns null when the fast path does not apply
/// (gradient-tracking or non-contiguous input) — the caller falls back
/// to facade dots.
pub fn linearSeqPtqtpFused(
    weight: *const WeightPtqtp,
    ctx: *ExecContext,
    input: anytype,
    comptime out_tag: Tag,
) !?Tensor(.{ .seq, out_tag }) {
    if (input.requiresGrad()) return null;
    const x = input.asRawTensor().dataConstChecked() catch return null;

    const n = weight.p1.dim(.out);
    const k = weight.p1.dim(.in);
    const m = input.dim(.seq);
    if (n == 0 or k == 0 or k % 256 != 0 or m == 0 or m * k != x.len) return null;
    const blocks_per_row = k / 256;

    // GPU prefill arm: with resident plane bytes and prefill-sized m, each
    // plane runs as one Metal ternary dequant-in-kernel dispatch and the K
    // plane outputs sum on the CPU (K=1 returns the async tensor directly).
    // NOT bitwise vs the CPU chain (half dequant, simdgroup f32 accumulate)
    // — the same accepted numerics stance as the q4_k/q6_k/q8_0 dense
    // offload. The seam's gates decide; any refusal falls through to the
    // CPU path wholesale.
    if (comptime offload.enabled and offload.supportsQuant(.tq2_0)) {
        // Folded resident form: ONE dispatch, async return, no plane sum.
        if (comptime offload.supportsQuant(.tq2_0_folded)) {
            if (m >= 32 and weight.gpu_fold != null) {
                const nb01 = blocks_per_row * @sizeOf(backend_quant.types.BlockTQ2_0Folded);
                if (try ctx.tryMatmulTernaryFolded(weight.gpu_fold.?, .stable_process, nb01, input.asRawTensor(), m, n, k)) |out_raw| {
                    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, out_raw);
                }
            }
        }
        if (m >= 32 and weight.gpu_planes[0] != null) gpu_blk: {
            const nb01 = blocks_per_row * @sizeOf(dtype_mod.BlockTQ2_0);
            const raw_input = input.asRawTensor();
            const dev_planes = [3]?[]const u8{ weight.gpu_planes[0], weight.gpu_planes[1], weight.gpu_planes[2] };
            var first: ?tensor_mod.Tensor = null;
            errdefer if (first) |*t| t.deinit();
            for (dev_planes) |maybe_dev| {
                const dev = maybe_dev orelse continue;
                var plane_out = (try ctx.tryMatmulQuantRhs(.tq2_0, dev, .stable_process, nb01, raw_input, m, n, k)) orelse {
                    if (first) |*t| t.deinit();
                    first = null;
                    break :gpu_blk; // gates refused: CPU path, all planes
                };
                if (first == null) {
                    first = plane_out;
                } else {
                    const dst = first.?.data();
                    for (dst, plane_out.dataConst()) |*d, s| d.* += s;
                    plane_out.deinit();
                }
            }
            if (first) |t| {
                first = null;
                return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, t);
            }
        }
    }

    // With packs built (WeightPtqtp.init, all-or-nothing) the planes run on
    // the x4 column-interleaved kernels — same bits, no per-block reduces,
    // and the accumulating twin folds extra planes straight into `out` with
    // no scratch pass. Without packs, the row kernels + scratch add.
    const px4_ready = weight.px4_allocator != null;
    var rhs: [3]backend_quant.types.QuantizedMatmulRhsTQ2_0 = undefined;
    var px4s: [3][]const backend_quant.types.BlockTQ2_0x4 = undefined;
    var plane_count: usize = 0;
    inline for ([_][]const u8{ "p1", "p2", "p3" }, 0..) |plane_field, slot| {
        const plane: ?*const QuantWeight(.tq2_0) = if (comptime std.mem.eql(u8, plane_field, "p1"))
            &weight.p1
        else if (@field(weight, plane_field)) |*p| p else null;
        if (plane) |p| {
            const blocks = p.asRawTensor().dataConstChecked() catch return null;
            // Borrow is sound: the matmul path never mutates RHS blocks
            // (same stance as the exec-tier tensor-RHS wrapper).
            rhs[plane_count] = backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(blocks)) catch return null;
            if (px4_ready) px4s[plane_count] = weight.px4[slot].?;
            plane_count += 1;
        }
    }

    const allocator = ctx.allocator();
    const lhs = try allocator.alloc(dtype_mod.BlockQ8_K, m * blocks_per_row);
    defer allocator.free(lhs);
    for (0..m) |r| {
        try backend_mod.kernels.quantizeRowQ8_KInto(lhs[r * blocks_per_row ..][0..blocks_per_row], x[r * k ..][0..k]);
    }
    // The kernels tile straight into the result tensor; only the multi-plane
    // accumulate keeps a scratch.
    var out_t = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ m, n });
    errdefer out_t.deinit();
    const out = try out_t.data();
    const tmp = try allocator.alloc(f32, if (plane_count > 1 and !px4_ready) m * n else 0);
    defer allocator.free(tmp);

    const Task = struct {
        out: []f32,
        tmp: []f32,
        lhs: []const dtype_mod.BlockQ8_K,
        rhs: []const backend_quant.types.QuantizedMatmulRhsTQ2_0,
        px4: []const []const backend_quant.types.BlockTQ2_0x4, // empty = row-kernel path
        pfold: []const backend_quant.types.BlockTQ2_0Foldedx4, // nonempty = one-pass fold
        bpr: usize,
        m: usize,
        n: usize,
        c0: usize,
        c1: usize,

        fn run(task: *const @This()) void {
            if (task.pfold.len != 0) {
                backend_mod.kernels.matmulTQ2_0FoldedX4RhsTile(task.out, task.lhs, task.pfold, task.bpr, task.n, 0, task.m, task.c0, task.c1);
                return;
            }
            if (task.px4.len != 0) {
                backend_mod.kernels.matmulTQ2_0X4RhsTile(task.out, task.lhs, task.px4[0], task.bpr, task.n, 0, task.m, task.c0, task.c1);
                for (task.px4[1..]) |pack| {
                    backend_mod.kernels.matmulTQ2_0X4RhsTileAcc(task.out, task.lhs, pack, task.bpr, task.n, 0, task.m, task.c0, task.c1);
                }
                return;
            }
            backend_mod.kernels.matmulTQ2_0RhsTile(task.out, task.lhs, &task.rhs[0], task.n, 0, task.m, task.c0, task.c1);
            for (task.rhs[1..]) |*plane_rhs| {
                backend_mod.kernels.matmulTQ2_0RhsTile(task.tmp, task.lhs, plane_rhs, task.n, 0, task.m, task.c0, task.c1);
                for (0..task.m) |r| {
                    const orow = task.out[r * task.n ..][0..task.n];
                    const srow = task.tmp[r * task.n ..][0..task.n];
                    for (task.c0..task.c1) |c| orow[c] += srow[c];
                }
            }
        }
    };
    const base = Task{
        .out = out,
        .tmp = tmp,
        .lhs = lhs,
        .rhs = rhs[0..plane_count],
        .px4 = if (px4_ready) px4s[0..plane_count] else &.{},
        .pfold = if (px4_ready and weight.pfold != null) weight.pfold.? else &.{},
        .bpr = blocks_per_row,
        .m = m,
        .n = n,
        .c0 = 0,
        .c1 = n,
    };
    // Column split over the worker team; per-column results are independent
    // and each column is computed by exactly one task, so any partition is
    // bitwise identical to the serial run.
    const work = m * n * k * plane_count;
    var tasks_run = false;
    // The dense threshold stands for ternary too: swept 1M/8M/32M/96M on an
    // M1 Max (2026-07) — pooled beats serial even at seq=1 talker shapes
    // (36.4 vs 14.4 fps; 1.7B chat 43 vs 30 tok/s). A profile's "main
    // thread waits in the barrier" is not evidence of waste: the team
    // finishes the columns faster than one core runs them.
    if (work >= parallel.vector_matmul_work_threshold) {
        if (ctx.workPool()) |pool| {
            const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
            const task_count = @max(@as(usize, 1), @min(cpu_count, n / parallel.vector_column_chunk));
            if (task_count > 1) {
                var tasks: [parallel.vector_max_threads]Task = undefined;
                for (0..task_count) |ti| {
                    tasks[ti] = base;
                    if (px4_ready) {
                        // Partition in 4-column group units: the pack has no
                        // finer addressing. Exact cover since n % 4 == 0.
                        const groups = n / 4;
                        tasks[ti].c0 = (ti * groups / task_count) * 4;
                        tasks[ti].c1 = ((ti + 1) * groups / task_count) * 4;
                    } else {
                        tasks[ti].c0 = ti * n / task_count;
                        tasks[ti].c1 = (ti + 1) * n / task_count;
                    }
                }
                pool.parallelChunks(Task, tasks[0..task_count], Task.run);
                tasks_run = true;
            }
        }
    }
    if (!tasks_run) Task.run(&base);

    return out_t;
}

/// Native-folded (`tq2_0_fx4`) dense serve: the WeightPtqtp pfold path with
/// the pack unconditionally present — same guards, one Q8_K quantize of the
/// m rows, one fork-join column-split dispatch of the one-pass folded
/// kernel, partitioned in 4-column-group units. Bitwise identical to the
/// tied `.ptqtp` pfold serve by construction (same kernel, same pack
/// algebra). The GPU arm is one async folded dispatch at prefill m, same
/// accepted non-bitwise stance as ptqtp's. No plane facade exists to fall
/// back to: gradient-tracking inputs error (an fx4 file is a serving
/// artifact, not a training checkpoint), non-contiguous views surface the
/// view error truthfully, and m == 0 returns the empty result like every
/// other arm.
pub fn linearSeqFx4(
    weight: *const WeightPtqtpFx4,
    ctx: *ExecContext,
    input: anytype,
    comptime out_tag: Tag,
) !Tensor(.{ .seq, out_tag }) {
    if (input.requiresGrad()) return Error.UnsupportedWeightType;
    if (input.dim(.seq) == 0) return try Tensor(.{ .seq, out_tag }).empty(ctx, .{ 0, weight.n });
    const x = try input.asRawTensor().dataConstChecked();

    const n = weight.n;
    const k = weight.k;
    const m = input.dim(.seq);
    if (n == 0 or k == 0 or m * k != x.len) return Error.InvalidWeightShape;
    const blocks_per_row = k / 256;

    if (comptime offload.enabled and offload.supportsQuant(.tq2_0)) {
        if (comptime offload.supportsQuant(.tq2_0_folded)) {
            if (m >= 32 and weight.gpu_fold != null) {
                const nb01 = blocks_per_row * @sizeOf(backend_quant.types.BlockTQ2_0Folded);
                if (try ctx.tryMatmulTernaryFolded(weight.gpu_fold.?, .stable_process, nb01, input.asRawTensor(), m, n, k)) |out_raw| {
                    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, out_raw);
                }
            }
        }
    }

    const allocator = ctx.allocator();

    // Prefill arm: dequantized weight panels through BLAS/AMX (the
    // backend's gate decides; decode and short bursts stay on the
    // mul-free ternary tile below).
    if (comptime backend_mod.blas.available) {
        var out_blas = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ m, n });
        errdefer out_blas.deinit();
        if (try backend_mod.blas.matmulFoldedx4(
            .{ .pool = ctx.workPool() },
            allocator,
            try out_blas.data(),
            x,
            weight.pack,
            blocks_per_row,
            m,
            n,
            k,
        )) return out_blas;
        out_blas.deinit();
    }

    const lhs = try allocator.alloc(dtype_mod.BlockQ8_K, m * blocks_per_row);
    defer allocator.free(lhs);
    for (0..m) |r| {
        try backend_mod.kernels.quantizeRowQ8_KInto(lhs[r * blocks_per_row ..][0..blocks_per_row], x[r * k ..][0..k]);
    }
    var out_t = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ m, n });
    errdefer out_t.deinit();
    const out = try out_t.data();

    const Task = struct {
        out: []f32,
        lhs: []const dtype_mod.BlockQ8_K,
        pack: []const backend_quant.types.BlockTQ2_0Foldedx4,
        bpr: usize,
        m: usize,
        n: usize,
        c0: usize,
        c1: usize,

        fn run(task: *const @This()) void {
            backend_mod.kernels.matmulTQ2_0FoldedX4RhsTile(task.out, task.lhs, task.pack, task.bpr, task.n, 0, task.m, task.c0, task.c1);
        }
    };
    const base = Task{ .out = out, .lhs = lhs, .pack = weight.pack, .bpr = blocks_per_row, .m = m, .n = n, .c0 = 0, .c1 = n };
    // plane_count 2 in the work formula for pooling-decision parity with
    // the tied .ptqtp pfold path (results are bitwise for any partition).
    const work = m * n * k * 2;
    var tasks_run = false;
    if (work >= parallel.vector_matmul_work_threshold) {
        if (ctx.workPool()) |pool| {
            const cpu_count = parallel.cpuThreadCount(parallel.vector_max_threads);
            const task_count = @max(@as(usize, 1), @min(cpu_count, n / parallel.vector_column_chunk));
            if (task_count > 1) {
                var tasks: [parallel.vector_max_threads]Task = undefined;
                for (0..task_count) |ti| {
                    tasks[ti] = base;
                    // 4-column group units: the pack has no finer addressing.
                    const groups = n / 4;
                    tasks[ti].c0 = (ti * groups / task_count) * 4;
                    tasks[ti].c1 = ((ti + 1) * groups / task_count) * 4;
                }
                pool.parallelChunks(Task, tasks[0..task_count], Task.run);
                tasks_run = true;
            }
        }
    }
    if (!tasks_run) Task.run(&base);

    return out_t;
}

/// PTQTP-decorated linear (arXiv:2509.16989; docs/PTQTP.md): the weight is
/// two packed TQ2_0 trit-planes with per-block group scales — each plane a
/// standalone valid TQ2_0 tensor — and the product is p1·x + p2·x through
/// the stock ternary RHS dot. `p2` is null for a single-plane decoration
/// (`ptqtp.Options.planes = 1`). Built by `LinearWeight.toPtqtp`, or loaded
/// from persisted `<name>.ptqtpK` plane tensors (ptqtp_gguf.zig).
pub const WeightPtqtp = struct {
    p1: QuantWeight(.tq2_0),
    p2: ?QuantWeight(.tq2_0),
    p3: ?QuantWeight(.tq2_0) = null,
    /// Column-interleaved x4 packs of the planes (same bytes rearranged —
    /// docs/TERNARY.md), the fused linear's fast operands: zero per-block
    /// reduces, bitwise identical to the row kernel. All-or-nothing: either
    /// every present plane has its pack (slot i mirrors pN) or all slots are
    /// null and the fused path falls back to the row kernels (odd n,
    /// unreadable plane storage, or allocation failure at build time).
    px4: [3]?[]backend_quant.types.BlockTQ2_0x4 = .{ null, null, null },
    px4_allocator: ?Allocator = null,
    /// GPU-resident copies of the plane blocks (`backend.offload`
    /// residency): stable device-shared bytes the Metal ternary
    /// dequant-in-kernel prefill dispatches against with zero per-call wrap
    /// cost. All-or-nothing like `px4`; null slots = CPU-only.
    gpu_planes: [3]?[]u8 = .{ null, null, null },
    /// Tied K=2: ONE resident buffer of row-major folded blocks
    /// (BlockTQ2_0Folded) — the GPU serves the linear as a single folded
    /// dispatch instead of one per plane, and its output returns async with
    /// no CPU plane-sum sync. Half the resident bytes of two plane copies.
    gpu_fold: ?[]u8 = null,
    /// True when the planes were fit with ptqtp.Options.tie_scales (scales
    /// locked to exact ratio 3). At K=2 the fused linear then serves through
    /// `pfold` — the 4-bit pack folding both planes into one 9-level code,
    /// ONE dot pass (matmulTQ2_0FoldedX4RhsTile). K=3's 27 levels exceed a
    /// nibble, so tied K=3 serves through the multi-pass x4 path. Persisted
    /// by the GGUF sidecars (`ptqtp.tie` — ptqtp_gguf.zig), so loaded
    /// tied K=2 decorations rebuild the fold and serve one-pass.
    tied: bool = false,
    pfold: ?[]backend_quant.types.BlockTQ2_0Foldedx4 = null,

    /// Construct with eager x4 pack building (and, on ternary-capable GPU
    /// builds, resident plane copies). Failure of either is silent — the
    /// weight works identically without them, just slower.
    pub fn init(allocator: Allocator, p1: QuantWeight(.tq2_0), p2: ?QuantWeight(.tq2_0), p3: ?QuantWeight(.tq2_0), tied: bool) WeightPtqtp {
        var self = WeightPtqtp{ .p1 = p1, .p2 = p2, .p3 = p3, .tied = tied and p2 != null };
        self.buildX4Packs(allocator);
        self.buildGpuResidency();
        return self;
    }

    pub fn outDim(self: *const WeightPtqtp) usize {
        return self.p1.dim(.out);
    }

    pub fn inDim(self: *const WeightPtqtp) usize {
        return self.p1.dim(.in);
    }

    /// Storage-sharing clone: plane views are retagged and the packs,
    /// fold, and GPU residency rebuild from the views. The clone drops the
    /// tie flag (and with it the fold) -- rebuilding is free, unlike the
    /// fx4 pack copy where the fold IS the weight.
    pub fn cloneView(self: *const WeightPtqtp, ctx: *ExecContext) !WeightPtqtp {
        var p1 = try self.p1.withTags(ctx, .{ .out, .in });
        errdefer p1.deinit();
        var p2: ?QuantWeight(.tq2_0) = if (self.p2) |*plane|
            try plane.withTags(ctx, .{ .out, .in })
        else
            null;
        errdefer if (p2) |*plane| plane.deinit();
        const p3: ?QuantWeight(.tq2_0) = if (self.p3) |*plane|
            try plane.withTags(ctx, .{ .out, .in })
        else
            null;
        return WeightPtqtp.init(ctx.allocator(), p1, p2, p3, false);
    }

    /// Multi-plane linear: the fused single-dispatch path when it applies
    /// (`linearSeqPtqtpFused`), else the per-plane facade dot chain --
    /// bitwise-equal results either way.
    pub fn linearSeq(self: *const WeightPtqtp, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        if (try linearSeqPtqtpFused(self, ctx, input, out_tag)) |fused| return fused;
        var p1 = try self.p1.withTags(ctx, .{ out_tag, in_tag });
        defer p1.deinit();
        var acc = try input.dot(ctx, &p1, in_tag);
        inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
            if (@field(self, plane_field)) |*plane| {
                errdefer acc.deinit();
                var tagged = try plane.withTags(ctx, .{ out_tag, in_tag });
                defer tagged.deinit();
                var y = try input.dot(ctx, &tagged, in_tag);
                defer y.deinit();
                const sum = try acc.add(ctx, &y);
                acc.deinit();
                acc = sum;
            }
        }
        return acc;
    }

    /// Row gather: per-plane f32 gathers summed in plane order.
    pub fn getRowsAs(self: *const WeightPtqtp, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        var acc = try self.p1.getRows(ctx, .out, token_ids, .seq);
        defer acc.deinit();
        inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
            if (@field(self, plane_field)) |*plane| {
                var rows2 = try plane.getRows(ctx, .out, token_ids, .seq);
                defer rows2.deinit();
                const sum = try acc.add(ctx, &rows2);
                acc.deinit();
                acc = sum;
            }
        }
        return acc.withTags(ctx, .{ .seq, out_tag });
    }

    fn buildGpuResidency(self: *WeightPtqtp) void {
        const gpu = offload;
        if (comptime !(gpu.enabled and gpu.has_quant_gemm and offload.supportsQuant(.tq2_0))) return;
        // Tied K=2 prefers the single folded resident buffer; falls through
        // to per-plane residency on any failure.
        if (comptime offload.supportsQuant(.tq2_0_folded)) {
            if (self.tied and self.p2 != null and self.p3 == null) fold: {
                const n = self.p1.dim(.out);
                const k = self.p1.dim(.in);
                const b1 = self.p1.asRawTensor().dataConstChecked() catch break :fold;
                const b2 = self.p2.?.asRawTensor().dataConstChecked() catch break :fold;
                const r1 = backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b1)) catch break :fold;
                const r2 = backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b2)) catch break :fold;
                const px4_alloc = self.px4_allocator orelse break :fold;
                const rows = backend_mod.kernels.packMatmulRhsTQ2_0FoldedRows(px4_alloc, &r1, &r2) catch break :fold;
                defer px4_alloc.free(rows);
                const bytes = std.mem.sliceAsBytes(rows);
                const dev = gpu.allocResidentBytes(bytes.len) orelse break :fold;
                @memcpy(dev, bytes);
                self.gpu_fold = dev;
                return; // folded residency replaces the per-plane copies
            }
        }
        const planes = [3]?*const QuantWeight(.tq2_0){
            &self.p1,
            if (self.p2) |*p| p else null,
            if (self.p3) |*p| p else null,
        };
        for (planes, 0..) |maybe_plane, i| {
            const plane = maybe_plane orelse continue;
            const ok = blk: {
                const blocks = plane.asRawTensor().dataConstChecked() catch break :blk false;
                const bytes = std.mem.sliceAsBytes(blocks);
                const dev = gpu.allocResidentBytes(bytes.len) orelse break :blk false;
                @memcpy(dev, bytes);
                self.gpu_planes[i] = dev;
                break :blk true;
            };
            if (!ok) {
                self.freeGpuResidency();
                return;
            }
        }
    }

    fn freeGpuResidency(self: *WeightPtqtp) void {
        const gpu = offload;
        if (comptime !gpu.enabled) return;
        if (self.gpu_fold) |dev| gpu.freeResidentBytes(dev);
        self.gpu_fold = null;
        for (&self.gpu_planes) |*slot| {
            if (slot.*) |dev| gpu.freeResidentBytes(dev);
            slot.* = null;
        }
    }

    fn buildX4Packs(self: *WeightPtqtp, allocator: Allocator) void {
        const n = self.p1.dim(.out);
        const k = self.p1.dim(.in);
        if (n == 0 or n % 4 != 0 or k == 0 or k % 256 != 0) return;
        const planes = [3]?*const QuantWeight(.tq2_0){
            &self.p1,
            if (self.p2) |*p| p else null,
            if (self.p3) |*p| p else null,
        };
        for (planes, 0..) |maybe_plane, i| {
            const plane = maybe_plane orelse continue;
            const ok = blk: {
                const blocks = plane.asRawTensor().dataConstChecked() catch break :blk false;
                const rhs = backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(blocks)) catch break :blk false;
                self.px4[i] = backend_mod.kernels.packMatmulRhsTQ2_0x4(allocator, &rhs) catch break :blk false;
                break :blk true;
            };
            if (!ok) {
                self.freeX4Packs(allocator);
                return;
            }
        }
        self.px4_allocator = allocator;
        // K=2 tie-fitted planes additionally fold into the 4-bit pack —
        // the fused linear's single-pass operand. Failure just leaves the
        // 2-pass x4 path (correct either way).
        if (self.tied and self.p2 != null and self.p3 == null) fold: {
            const b1 = self.p1.asRawTensor().dataConstChecked() catch break :fold;
            const b2 = self.p2.?.asRawTensor().dataConstChecked() catch break :fold;
            const r1 = backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b1)) catch break :fold;
            const r2 = backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b2)) catch break :fold;
            self.pfold = backend_mod.kernels.packMatmulRhsTQ2_0Foldedx4(allocator, &r1, &r2) catch null;
        }
    }

    fn freeX4Packs(self: *WeightPtqtp, allocator: Allocator) void {
        for (&self.px4) |*slot| {
            if (slot.*) |pack| allocator.free(pack);
            slot.* = null;
        }
        if (self.pfold) |pack| allocator.free(pack);
        self.pfold = null;
        self.px4_allocator = null;
    }

    pub fn planeCount(self: *const WeightPtqtp) usize {
        var count: usize = 1;
        if (self.p2 != null) count += 1;
        if (self.p3 != null) count += 1;
        return count;
    }

    pub fn deinit(self: *WeightPtqtp) void {
        self.freeGpuResidency();
        if (self.px4_allocator) |allocator| self.freeX4Packs(allocator);
        if (self.p3) |*plane| plane.deinit();
        if (self.p2) |*plane| plane.deinit();
        self.p1.deinit();
        self.* = undefined;
    }
};

/// Native-folded tied-K=2 PTQTP dense weight (`gguf.GgmlType.tq2_0_fx4`,
/// docs/PTQTP.md): the weight IS the one-pass 4-bit pack — no planes, no
/// x4 repacks, no load-time fold. 4.0625 bpw resident vs the sibling-plane
/// `WeightPtqtp`'s planes + x4 packs + pfold (~3x the bytes for the same
/// tied serving numbers). Serving is bitwise the `WeightPtqtp.pfold` CPU
/// path for every m (same kernel, same pack algebra); on Metal builds the
/// pack relayouts into the row-major folded device form for the single
/// async prefill dispatch (same accepted non-bitwise GPU stance as ptqtp).
/// Autograd walks and frozen-training dots reject the arm (no plane facade
/// to fall back to) — an fx4 file is a serving artifact, not a training
/// checkpoint.
pub const WeightPtqtpFx4 = struct {
    /// Column-group-major pack: `pack[g * blocks_per_row + bi]`, group `g`
    /// = output columns `4g..4g+3`.
    pack: []const backend_quant.types.BlockTQ2_0Foldedx4,
    /// Frees `pack`; null = borrowed storage kept alive by the owner.
    allocator: ?Allocator,
    /// Out / in dims — the pack slice carries no shape.
    n: usize,
    k: usize,
    /// Metal-resident row-major folded bytes (`BlockTQ2_0Folded`), built
    /// best-effort at init on capable builds.
    gpu_fold: ?[]u8 = null,

    pub fn init(allocator: Allocator, pack: []const backend_quant.types.BlockTQ2_0Foldedx4, pack_allocator: ?Allocator, n: usize, k: usize, gpu_resident: bool) WeightPtqtpFx4 {
        var self = WeightPtqtpFx4{ .pack = pack, .allocator = pack_allocator, .n = n, .k = k };
        if (gpu_resident) self.buildGpuResidency(allocator);
        return self;
    }

    pub fn buildGpuResidency(self: *WeightPtqtpFx4, allocator: Allocator) void {
        const gpu = offload;
        if (comptime !(gpu.enabled and gpu.has_quant_gemm and offload.supportsQuant(.tq2_0))) return;
        if (comptime !offload.supportsQuant(.tq2_0_folded)) return;
        if (self.gpu_fold != null) return;
        const rows = backend_mod.kernels.packMatmulRhsTQ2_0FoldedRowsFromX4(allocator, self.pack, self.n, self.k / 256) catch return;
        defer allocator.free(rows);
        const bytes = std.mem.sliceAsBytes(rows);
        const dev = gpu.allocResidentBytes(bytes.len) orelse return;
        @memcpy(dev, bytes);
        self.gpu_fold = dev;
    }

    pub fn outDim(self: *const WeightPtqtpFx4) usize {
        return self.n;
    }

    pub fn inDim(self: *const WeightPtqtpFx4) usize {
        return self.k;
    }

    /// The fold IS the weight: clone copies the pack (unlike the ptqtp
    /// clone, which drops fold/tie and rebuilds free).
    pub fn cloneView(self: *const WeightPtqtpFx4, ctx: *ExecContext) !WeightPtqtpFx4 {
        const owned = try ctx.allocator().alloc(backend_quant.types.BlockTQ2_0Foldedx4, self.pack.len);
        @memcpy(owned, self.pack);
        return WeightPtqtpFx4.init(ctx.allocator(), owned, ctx.allocator(), self.n, self.k, true);
    }

    /// One-pass folded linear (`linearSeqFx4`); `in_tag` is fixed by the
    /// stored pack geometry.
    pub fn linearSeq(self: *const WeightPtqtpFx4, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        _ = in_tag;
        return linearSeqFx4(self, ctx, input, out_tag);
    }

    /// Row gather through `decodeRow` -- decoded straight into the result.
    pub fn getRowsAs(self: *const WeightPtqtpFx4, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        var out_t = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ token_ids.len, self.k });
        errdefer out_t.deinit();
        const dst = try out_t.data();
        for (token_ids, 0..) |row, i| self.decodeRow(row, dst[i * self.k ..][0..self.k]);
        return out_t;
    }

    /// Decode output row `r` to f32 — the embedding/util path (`getRowsAs`,
    /// `toResidentF16`). value = fine_scale * (cu - 4), cu the stored
    /// 9-level code; exact-equal to the tied plane sum by the tie identity.
    pub fn decodeRow(self: *const WeightPtqtpFx4, r: usize, dst: []f32) void {
        const bpr = self.k / 256;
        const g = r / 4;
        const ci = r % 4;
        for (0..bpr) |bi| {
            const block = &self.pack[g * bpr + bi];
            const d: f32 = @floatCast(@as(f16, @bitCast(block.d[ci])));
            const out = dst[bi * 256 ..][0..256];
            for (0..256) |e| {
                const sub = e / 32;
                const rem = e % 32;
                const idx = rem % 16;
                const byte = block.qs[sub * 64 + (idx / 4) * 16 + ci * 4 + (idx % 4)];
                const cu: i32 = if (rem >= 16) (byte >> 4) else (byte & 0xF);
                out[e] = d * @as(f32, @floatFromInt(cu - 4));
            }
        }
    }

    pub fn deinit(self: *WeightPtqtpFx4) void {
        const gpu = offload;
        if (comptime gpu.enabled) {
            if (self.gpu_fold) |dev| gpu.freeResidentBytes(dev);
            self.gpu_fold = null;
        }
        if (self.allocator) |allocator| allocator.free(self.pack);
        self.* = undefined;
    }
};
