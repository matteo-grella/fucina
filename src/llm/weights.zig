const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");

const DType = fucina.DType;
const ExecContext = fucina.ExecContext;
const Tag = @TypeOf(.tag);
const gguf = fucina.gguf;
const Allocator = std.mem.Allocator;
const RhsLifetime = fucina.RhsLifetime;

pub const Error = error{
    InvalidWeightShape,
    UnsupportedWeightType,
    GradUnsupported,
};

pub const WeightF32 = fucina.Tensor(.{ .out, .in });
pub const WeightF16 = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } });
pub const WeightBf16 = fucina.Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
const RawWeightQ4_K = QuantWeight(.q4_k);
const RawWeightQ5_K = QuantWeight(.q5_k);
const RawWeightQ6_K = QuantWeight(.q6_k);
const RawWeightQ8_0 = QuantWeight(.q8_0);

pub const WeightQ4_K = struct {
    value: RawWeightQ4_K,
    packed_rhs: fucina.PackedRhs(.q4_k),
    rhs_lifetime: RhsLifetime = .transient,

    pub fn init(ctx: *ExecContext, value: RawWeightQ4_K) !WeightQ4_K {
        return initWithRhsLifetime(ctx, value, .transient);
    }

    pub fn initWithRhsLifetime(ctx: *ExecContext, value: RawWeightQ4_K, rhs_lifetime: RhsLifetime) !WeightQ4_K {
        var owned = value;
        errdefer owned.deinit();
        var packed_rhs = try owned.packRhs(ctx);
        errdefer packed_rhs.deinit();
        return .{ .value = owned, .packed_rhs = packed_rhs, .rhs_lifetime = rhs_lifetime };
    }

    pub fn deinit(self: *WeightQ4_K) void {
        self.packed_rhs.deinit();
        self.value.deinit();
        self.* = undefined;
    }

    pub fn cloneView(self: *const WeightQ4_K, ctx: *ExecContext) !WeightQ4_K {
        const value = try self.value.withTags(ctx, .{ .out, .in });
        return initWithRhsLifetime(ctx, value, self.rhs_lifetime);
    }

    pub fn concat(self: *const WeightQ4_K, ctx: *ExecContext, comptime tag: Tag, others: []const *const WeightQ4_K) !WeightQ4_K {
        var raw_others = try ctx.allocator.alloc(*const RawWeightQ4_K, others.len);
        defer ctx.allocator.free(raw_others);
        for (others, 0..) |other, i| raw_others[i] = &other.value;

        var value = try self.value.concat(ctx, tag, raw_others);
        var owns_value = true;
        errdefer if (owns_value) value.deinit();
        const rhs_lifetime: RhsLifetime = if (try makeGpuResidentQuantWeight(.q4_k, ctx, &value)) .stable_process else .transient;
        return initWithRhsLifetime(ctx, value, rhs_lifetime) catch |err| {
            owns_value = false;
            return err;
        };
    }
};

/// Session/model-owned registry for immutable byte payloads that should be
/// copied once into device-owned storage when a capable GPU provider is built.
/// The returned bytes are still CPU-readable and remain ordinary RHS storage;
/// this only changes the backing allocation used by GPU matmul accelerators.
pub const ResidentByteRegistry = struct {
    allocator: Allocator,
    map: std.AutoHashMapUnmanaged(usize, []const u8) = .empty,

    pub fn init(allocator: Allocator) ResidentByteRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ResidentByteRegistry) void {
        if (comptime fucina.internal.gpu.enabled) {
            var it = self.map.iterator();
            while (it.next()) |e| {
                fucina.internal.gpu.freeResidentBytes(e.value_ptr.*);
            }
        }
        self.map.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn bytes(self: *ResidentByteRegistry, src: []const u8) []const u8 {
        if (comptime !fucina.internal.gpu.enabled) return src;
        const key = @intFromPtr(src.ptr);
        if (self.map.get(key)) |dev| return dev;
        self.map.ensureUnusedCapacity(self.allocator, 1) catch return src;
        const dev = fucina.internal.gpu.allocResidentBytes(src.len) orelse return src;
        @memcpy(dev, src);
        self.map.putAssumeCapacityNoClobber(key, dev);
        return dev;
    }
};

pub const WeightQ5_K = struct {
    value: RawWeightQ5_K,
    packed_rhs: fucina.PackedRhs(.q5_k),
    rhs_lifetime: RhsLifetime = .transient,

    pub fn init(ctx: *ExecContext, value: RawWeightQ5_K) !WeightQ5_K {
        return initWithRhsLifetime(ctx, value, .transient);
    }

    pub fn initWithRhsLifetime(ctx: *ExecContext, value: RawWeightQ5_K, rhs_lifetime: RhsLifetime) !WeightQ5_K {
        var owned = value;
        errdefer owned.deinit();
        var packed_rhs = try owned.packRhs(ctx);
        errdefer packed_rhs.deinit();
        return .{ .value = owned, .packed_rhs = packed_rhs, .rhs_lifetime = rhs_lifetime };
    }

    pub fn deinit(self: *WeightQ5_K) void {
        self.packed_rhs.deinit();
        self.value.deinit();
        self.* = undefined;
    }

    pub fn cloneView(self: *const WeightQ5_K, ctx: *ExecContext) !WeightQ5_K {
        const value = try self.value.withTags(ctx, .{ .out, .in });
        return initWithRhsLifetime(ctx, value, self.rhs_lifetime);
    }

    pub fn concat(self: *const WeightQ5_K, ctx: *ExecContext, comptime tag: Tag, others: []const *const WeightQ5_K) !WeightQ5_K {
        var raw_others = try ctx.allocator.alloc(*const RawWeightQ5_K, others.len);
        defer ctx.allocator.free(raw_others);
        for (others, 0..) |other, i| raw_others[i] = &other.value;

        var value = try self.value.concat(ctx, tag, raw_others);
        var owns_value = true;
        errdefer if (owns_value) value.deinit();
        const rhs_lifetime: RhsLifetime = if (try makeGpuResidentQuantWeight(.q5_k, ctx, &value)) .stable_process else .transient;
        return initWithRhsLifetime(ctx, value, rhs_lifetime) catch |err| {
            owns_value = false;
            return err;
        };
    }
};
pub const WeightQ6_K = struct {
    value: RawWeightQ6_K,
    packed_rhs: fucina.PackedRhs(.q6_k),
    rhs_lifetime: RhsLifetime = .transient,

    pub fn init(ctx: *ExecContext, value: RawWeightQ6_K) !WeightQ6_K {
        return initWithRhsLifetime(ctx, value, .transient);
    }

    pub fn initWithRhsLifetime(ctx: *ExecContext, value: RawWeightQ6_K, rhs_lifetime: RhsLifetime) !WeightQ6_K {
        var owned = value;
        errdefer owned.deinit();
        var packed_rhs = try owned.packRhs(ctx);
        errdefer packed_rhs.deinit();
        return .{ .value = owned, .packed_rhs = packed_rhs, .rhs_lifetime = rhs_lifetime };
    }

    pub fn deinit(self: *WeightQ6_K) void {
        self.packed_rhs.deinit();
        self.value.deinit();
        self.* = undefined;
    }

    pub fn cloneView(self: *const WeightQ6_K, ctx: *ExecContext) !WeightQ6_K {
        const value = try self.value.withTags(ctx, .{ .out, .in });
        return initWithRhsLifetime(ctx, value, self.rhs_lifetime);
    }

    pub fn concat(self: *const WeightQ6_K, ctx: *ExecContext, comptime tag: Tag, others: []const *const WeightQ6_K) !WeightQ6_K {
        var raw_others = try ctx.allocator.alloc(*const RawWeightQ6_K, others.len);
        defer ctx.allocator.free(raw_others);
        for (others, 0..) |other, i| raw_others[i] = &other.value;

        var value = try self.value.concat(ctx, tag, raw_others);
        var owns_value = true;
        errdefer if (owns_value) value.deinit();
        const rhs_lifetime: RhsLifetime = if (try makeGpuResidentQuantWeight(.q6_k, ctx, &value)) .stable_process else .transient;
        return initWithRhsLifetime(ctx, value, rhs_lifetime) catch |err| {
            owns_value = false;
            return err;
        };
    }
};
pub const WeightQ8_0 = struct {
    value: RawWeightQ8_0,
    packed_rhs: fucina.PackedRhs(.q8_0),
    rhs_lifetime: RhsLifetime = .transient,

    pub fn init(ctx: *ExecContext, value: RawWeightQ8_0) !WeightQ8_0 {
        return initWithRhsLifetime(ctx, value, .transient);
    }

    pub fn initWithRhsLifetime(ctx: *ExecContext, value: RawWeightQ8_0, rhs_lifetime: RhsLifetime) !WeightQ8_0 {
        var owned = value;
        errdefer owned.deinit();
        var packed_rhs = try owned.packRhs(ctx);
        errdefer packed_rhs.deinit();
        return .{ .value = owned, .packed_rhs = packed_rhs, .rhs_lifetime = rhs_lifetime };
    }

    pub fn deinit(self: *WeightQ8_0) void {
        self.packed_rhs.deinit();
        self.value.deinit();
        self.* = undefined;
    }

    pub fn cloneView(self: *const WeightQ8_0, ctx: *ExecContext) !WeightQ8_0 {
        const value = try self.value.withTags(ctx, .{ .out, .in });
        return initWithRhsLifetime(ctx, value, self.rhs_lifetime);
    }

    pub fn concat(self: *const WeightQ8_0, ctx: *ExecContext, comptime tag: Tag, others: []const *const WeightQ8_0) !WeightQ8_0 {
        var raw_others = try ctx.allocator.alloc(*const RawWeightQ8_0, others.len);
        defer ctx.allocator.free(raw_others);
        for (others, 0..) |other, i| raw_others[i] = &other.value;

        var value = try self.value.concat(ctx, tag, raw_others);
        var owns_value = true;
        errdefer if (owns_value) value.deinit();
        const rhs_lifetime: RhsLifetime = if (try makeGpuResidentQuantWeight(.q8_0, ctx, &value)) .stable_process else .transient;
        return initWithRhsLifetime(ctx, value, rhs_lifetime) catch |err| {
            owns_value = false;
            return err;
        };
    }
};

pub fn QuantWeight(comptime dtype: DType) type {
    return fucina.Tensor(.{ .dtype = dtype, .tags = .{ .out, .in } });
}

const backend_quant = fucina.internal.backend_mod.quantized_matmul;

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
fn linearSeqPtqtpFused(
    weight: *const WeightPtqtp,
    ctx: *ExecContext,
    input: anytype,
    comptime out_tag: Tag,
) !?fucina.Tensor(.{ .seq, out_tag }) {
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
    if (comptime fucina.internal.gpu.enabled and fucina.internal.gpu.has_tq2_0_quant) {
        // Folded resident form: ONE dispatch, async return, no plane sum.
        if (comptime fucina.internal.backend_mod.gpu_impl.has_tq2_0_folded_quant) {
            if (m >= 32 and weight.gpu_fold != null) {
                const nb01 = blocks_per_row * @sizeOf(backend_quant.BlockTQ2_0Folded);
                if (try ctx.foldedTernaryMatmulGpu(weight.gpu_fold.?, .stable_process, nb01, input.asRawTensor(), m, n, k)) |out_raw| {
                    return try fucina.Tensor(.{ .seq, out_tag }).fromTensor(ctx, out_raw);
                }
            }
        }
        if (m >= 32 and weight.gpu_planes[0] != null) gpu_blk: {
            const nb01 = blocks_per_row * @sizeOf(backend_quant.BlockTQ2_0);
            const raw_input = input.asRawTensor();
            const dev_planes = [3]?[]const u8{ weight.gpu_planes[0], weight.gpu_planes[1], weight.gpu_planes[2] };
            var first: ?fucina.internal.RawTensor = null;
            errdefer if (first) |*t| t.deinit();
            for (dev_planes) |maybe_dev| {
                const dev = maybe_dev orelse continue;
                var plane_out = (try ctx.denseQuantMatmulGpu(.tq2_0, dev, .stable_process, nb01, raw_input, m, n, k)) orelse {
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
                return try fucina.Tensor(.{ .seq, out_tag }).fromTensor(ctx, t);
            }
        }
    }

    // With packs built (WeightPtqtp.init, all-or-nothing) the planes run on
    // the x4 column-interleaved kernels — same bits, no per-block reduces,
    // and the accumulating twin folds extra planes straight into `out` with
    // no scratch pass. Without packs, the row kernels + scratch add.
    const px4_ready = weight.px4_allocator != null;
    var rhs: [3]backend_quant.QuantizedMatmulRhsTQ2_0 = undefined;
    var px4s: [3][]const backend_quant.BlockTQ2_0x4 = undefined;
    var plane_count: usize = 0;
    inline for ([_][]const u8{ "p1", "p2", "p3" }, 0..) |plane_field, slot| {
        const plane: ?*const QuantWeight(.tq2_0) = if (comptime std.mem.eql(u8, plane_field, "p1"))
            &weight.p1
        else if (@field(weight, plane_field)) |*p| p else null;
        if (plane) |p| {
            const blocks = p.asRawTensor().dataConstChecked() catch return null;
            // Borrow is sound: the matmul path never mutates RHS blocks
            // (same stance as the exec-tier tensor-RHS wrapper).
            rhs[plane_count] = backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(blocks)) catch return null;
            if (px4_ready) px4s[plane_count] = weight.px4[slot].?;
            plane_count += 1;
        }
    }

    const allocator = ctx.allocator;
    const lhs = try allocator.alloc(backend_quant.BlockQ8_K, m * blocks_per_row);
    defer allocator.free(lhs);
    for (0..m) |r| {
        try backend_quant.quantizeRowQ8_KInto(lhs[r * blocks_per_row ..][0..blocks_per_row], x[r * k ..][0..k]);
    }
    const out = try allocator.alloc(f32, m * n);
    defer allocator.free(out);
    const tmp = try allocator.alloc(f32, if (plane_count > 1 and !px4_ready) m * n else 0);
    defer allocator.free(tmp);

    const Task = struct {
        out: []f32,
        tmp: []f32,
        lhs: []const backend_quant.BlockQ8_K,
        rhs: []const backend_quant.QuantizedMatmulRhsTQ2_0,
        px4: []const []const backend_quant.BlockTQ2_0x4, // empty = row-kernel path
        pfold: []const backend_quant.BlockTQ2_0Foldedx4, // nonempty = one-pass fold
        bpr: usize,
        m: usize,
        n: usize,
        c0: usize,
        c1: usize,

        fn run(task: *const @This()) void {
            if (task.pfold.len != 0) {
                backend_quant.matmulTQ2_0FoldedX4RhsTile(task.out, task.lhs, task.pfold, task.bpr, task.n, 0, task.m, task.c0, task.c1);
                return;
            }
            if (task.px4.len != 0) {
                backend_quant.matmulTQ2_0X4RhsTile(task.out, task.lhs, task.px4[0], task.bpr, task.n, 0, task.m, task.c0, task.c1);
                for (task.px4[1..]) |pack| {
                    backend_quant.matmulTQ2_0X4RhsTileAcc(task.out, task.lhs, pack, task.bpr, task.n, 0, task.m, task.c0, task.c1);
                }
                return;
            }
            backend_quant.matmulTQ2_0RhsTile(task.out, task.lhs, &task.rhs[0], task.n, 0, task.m, task.c0, task.c1);
            for (task.rhs[1..]) |*plane_rhs| {
                backend_quant.matmulTQ2_0RhsTile(task.tmp, task.lhs, plane_rhs, task.n, 0, task.m, task.c0, task.c1);
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
    if (work >= fucina.parallel.vector_matmul_work_threshold) {
        if (ctx.workPool()) |pool| {
            const cpu_count = fucina.parallel.cpuThreadCount(fucina.parallel.vector_max_threads);
            const task_count = @max(@as(usize, 1), @min(cpu_count, n / fucina.parallel.vector_column_chunk));
            if (task_count > 1) {
                var tasks: [fucina.parallel.vector_max_threads]Task = undefined;
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

    return try fucina.Tensor(.{ .seq, out_tag }).fromSlice(ctx, .{ m, n }, out);
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
    px4: [3]?[]backend_quant.BlockTQ2_0x4 = .{ null, null, null },
    px4_allocator: ?Allocator = null,
    /// GPU-resident copies of the plane blocks (`fucina.internal.gpu`
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
    /// nibble, so tied K=3 serves through the 2-pass x4 path. Not persisted
    /// by the GGUF sidecars yet, so loaded decorations run unfolded
    /// (correct, just K passes).
    tied: bool = false,
    pfold: ?[]backend_quant.BlockTQ2_0Foldedx4 = null,

    /// Construct with eager x4 pack building (and, on ternary-capable GPU
    /// builds, resident plane copies). Failure of either is silent — the
    /// weight works identically without them, just slower.
    pub fn init(allocator: Allocator, p1: QuantWeight(.tq2_0), p2: ?QuantWeight(.tq2_0), p3: ?QuantWeight(.tq2_0), tied: bool) WeightPtqtp {
        var self = WeightPtqtp{ .p1 = p1, .p2 = p2, .p3 = p3, .tied = tied and p2 != null };
        self.buildX4Packs(allocator);
        self.buildGpuResidency();
        return self;
    }

    fn buildGpuResidency(self: *WeightPtqtp) void {
        const gpu = fucina.internal.gpu;
        if (comptime !(gpu.enabled and gpu.has_quant_gemm and gpu.has_tq2_0_quant)) return;
        // Tied K=2 prefers the single folded resident buffer; falls through
        // to per-plane residency on any failure.
        if (comptime fucina.internal.backend_mod.gpu_impl.has_tq2_0_folded_quant) {
            if (self.tied and self.p2 != null and self.p3 == null) fold: {
                const n = self.p1.dim(.out);
                const k = self.p1.dim(.in);
                const b1 = self.p1.asRawTensor().dataConstChecked() catch break :fold;
                const b2 = self.p2.?.asRawTensor().dataConstChecked() catch break :fold;
                const r1 = backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b1)) catch break :fold;
                const r2 = backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b2)) catch break :fold;
                const px4_alloc = self.px4_allocator orelse break :fold;
                const rows = backend_quant.packMatmulRhsTQ2_0FoldedRows(px4_alloc, &r1, &r2) catch break :fold;
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
        const gpu = fucina.internal.gpu;
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
                const rhs = backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(blocks)) catch break :blk false;
                self.px4[i] = backend_quant.packMatmulRhsTQ2_0x4(allocator, &rhs) catch break :blk false;
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
            const r1 = backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b1)) catch break :fold;
            const r2 = backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b2)) catch break :fold;
            self.pfold = backend_quant.packMatmulRhsTQ2_0Foldedx4(allocator, &r1, &r2) catch null;
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

/// One raw quantized linear weight to be copied into a contiguous stack.
/// `data` is the GGUF block payload for a `[out, in]` matrix.
pub const QuantByteStackPart = struct {
    data: []const u8,
    in: usize,
    out: usize,
};

pub const QuantByteStackOptions = struct {
    /// Prefer device-owned storage when the active provider implements the
    /// dtype's quantized kernel.
    prefer_device: bool = true,
    /// Return null instead of heap-allocating when device-owned storage is not
    /// available. Use this for GPU-only command-batched paths.
    require_device: bool = false,
};

/// Internal byte stack for same-shaped quantized linear weights. The stack is
/// CPU-readable either way; `device_owned=true` additionally means the bytes
/// live in provider-owned storage and are safe for cached wraps until deinit.
pub const QuantByteStack = struct {
    dtype: DType,
    count: usize,
    in: usize,
    out: usize,
    bytes_per_weight: usize,
    data: []u8,
    device_owned: bool,

    pub fn deinit(self: *QuantByteStack, allocator: Allocator) void {
        if (self.device_owned) {
            if (comptime fucina.internal.gpu.enabled) fucina.internal.gpu.freeResidentBytes(self.data);
        } else {
            allocator.free(self.data);
        }
        self.* = undefined;
    }

    pub fn bytesPerRow(self: *const QuantByteStack) usize {
        return self.bytes_per_weight / self.out;
    }

    pub fn totalOutRows(self: *const QuantByteStack) usize {
        return self.count * self.out;
    }
};

fn dtypeHasDenseQuantGpuKernel(comptime dtype: DType) bool {
    return switch (comptime dtype) {
        .q4_k, .q6_k, .q8_0 => true,
        .q5_k => fucina.internal.gpu.has_q5_k_quant,
        else => false,
    };
}

/// Build a contiguous stack of same-shaped quantized linear weights with the
/// same residency policy used by the generic loaders. This is a private
/// building block for eager dispatch batching: callers still return ordinary
/// tensors and still fall back when the backend declines the GPU path.
pub fn makeQuantByteStack(
    comptime dtype: DType,
    allocator: Allocator,
    parts: []const QuantByteStackPart,
    options: QuantByteStackOptions,
) !?QuantByteStack {
    if (parts.len == 0) return null;
    const first = parts[0];
    if (first.in == 0 or first.out == 0 or first.data.len == 0) return Error.InvalidWeightShape;
    if (first.data.len % first.out != 0) return Error.InvalidWeightShape;
    const bytes_per_weight = first.data.len;
    for (parts[1..]) |part| {
        if (part.in != first.in or part.out != first.out or part.data.len != bytes_per_weight) {
            return Error.InvalidWeightShape;
        }
    }

    const total_len = try std.math.mul(usize, bytes_per_weight, parts.len);
    _ = try std.math.mul(usize, first.out, parts.len);
    var device_owned = false;
    const data: []u8 = blk: {
        if (options.prefer_device and comptime fucina.internal.gpu.enabled and dtypeHasDenseQuantGpuKernel(dtype)) {
            if (fucina.internal.gpu.allocResidentBytes(total_len)) |dev| {
                device_owned = true;
                break :blk dev;
            }
        }
        if (options.require_device) return null;
        break :blk try allocator.alloc(u8, total_len);
    };
    errdefer if (!device_owned) allocator.free(data);

    for (parts, 0..) |part, i| {
        @memcpy(data[i * bytes_per_weight ..][0..bytes_per_weight], part.data);
    }

    return .{
        .dtype = dtype,
        .count = parts.len,
        .in = first.in,
        .out = first.out,
        .bytes_per_weight = bytes_per_weight,
        .data = data,
        .device_owned = device_owned,
    };
}

pub const LinearWeight = union(enum) {
    f32: WeightF32,
    f16: WeightF16,
    bf16: WeightBf16,
    q1_0: QuantWeight(.q1_0),
    q2_0: QuantWeight(.q2_0),
    q4_0: QuantWeight(.q4_0),
    q4_1: QuantWeight(.q4_1),
    q5_0: QuantWeight(.q5_0),
    q5_1: QuantWeight(.q5_1),
    q8_0: WeightQ8_0,
    q2_k: QuantWeight(.q2_k),
    q3_k: QuantWeight(.q3_k),
    q4_k: WeightQ4_K,
    q5_k: WeightQ5_K,
    q6_k: WeightQ6_K,
    iq1_s: QuantWeight(.iq1_s),
    iq1_m: QuantWeight(.iq1_m),
    iq2_xxs: QuantWeight(.iq2_xxs),
    iq2_xs: QuantWeight(.iq2_xs),
    iq2_s: QuantWeight(.iq2_s),
    iq3_xxs: QuantWeight(.iq3_xxs),
    iq3_s: QuantWeight(.iq3_s),
    iq4_nl: QuantWeight(.iq4_nl),
    iq4_xs: QuantWeight(.iq4_xs),
    tq1_0: QuantWeight(.tq1_0),
    tq2_0: QuantWeight(.tq2_0),
    mxfp4: QuantWeight(.mxfp4),
    nvfp4: QuantWeight(.nvfp4),
    ptqtp: WeightPtqtp,

    pub const LoadOptions = struct {
        /// Copy provider-supported quant payloads into device-owned storage
        /// (q4_k/q6_k/q8_0 on Metal; those plus q5_k on CUDA) for stable RHS
        /// dequant-in-kernel GEMM. `loadForFusion` turns this off: parts are consumed
        /// by `fuseLinear`, whose concat re-acquires residency for the fused
        /// result, so per-part device copies would be alloc+memcpy+free waste.
        gpu_resident: bool = true,
    };

    pub fn load(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize) !LinearWeight {
        return loadWithOptions(ctx, info, expected_rows, expected_cols, .{});
    }

    /// Load a weight that exists only to be consumed by `fuseLinear`: skips
    /// the transient device-residency copy the fused result would immediately
    /// free. Should fusion decline (mixed part formats), the parts remain
    /// fully usable on the CPU packed path — they just stay CPU-resident.
    pub fn loadForFusion(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize) !LinearWeight {
        return loadWithOptions(ctx, info, expected_rows, expected_cols, .{ .gpu_resident = false });
    }

    pub fn loadWithOptions(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize, options: LoadOptions) !LinearWeight {
        const shape = try requireMatrixShape(info, expected_rows, expected_cols);
        // Every dense linear is read in full here (copy/widen/repack), so kick
        // off readahead for its (possibly cold-mapped) bytes before we touch
        // them. No-op once resident; borrowed MoE experts skip this path.
        gguf.prefetch(info.data);
        return switch (info.ggml_type) {
            .f32, .f64 => .{ .f32 = try loadDenseF32Weight(ctx, info, shape) },
            .f16 => .{ .f16 = try loadDenseF16Weight(ctx, info, shape, options) },
            .bf16 => .{ .bf16 = try loadDenseBf16Weight(ctx, info, shape) },
            .q1_0 => .{ .q1_0 = try loadQuantizedWeight(.q1_0, ctx, info, shape) },
            .q2_0 => .{ .q2_0 = try loadQuantizedWeight(.q2_0, ctx, info, shape) },
            .q4_0 => .{ .q4_0 = try loadQuantizedWeight(.q4_0, ctx, info, shape) },
            .q4_1 => .{ .q4_1 = try loadQuantizedWeight(.q4_1, ctx, info, shape) },
            .q5_0 => .{ .q5_0 = try loadQuantizedWeight(.q5_0, ctx, info, shape) },
            .q5_1 => .{ .q5_1 = try loadQuantizedWeight(.q5_1, ctx, info, shape) },
            .q8_0 => .{ .q8_0 = try loadQ8_0Weight(ctx, info, shape, options) },
            .q2_k => .{ .q2_k = try loadQuantizedWeight(.q2_k, ctx, info, shape) },
            .q3_k => .{ .q3_k = try loadQuantizedWeight(.q3_k, ctx, info, shape) },
            .q4_k => .{ .q4_k = try loadQ4_KWeight(ctx, info, shape, options) },
            .q5_k => .{ .q5_k = try loadQ5_KWeight(ctx, info, shape, options) },
            .q6_k => .{ .q6_k = try loadQ6_KWeight(ctx, info, shape, options) },
            .iq1_s => .{ .iq1_s = try loadQuantizedWeight(.iq1_s, ctx, info, shape) },
            .iq1_m => .{ .iq1_m = try loadQuantizedWeight(.iq1_m, ctx, info, shape) },
            .iq2_xxs => .{ .iq2_xxs = try loadQuantizedWeight(.iq2_xxs, ctx, info, shape) },
            .iq2_xs => .{ .iq2_xs = try loadQuantizedWeight(.iq2_xs, ctx, info, shape) },
            .iq2_s => .{ .iq2_s = try loadQuantizedWeight(.iq2_s, ctx, info, shape) },
            .iq3_xxs => .{ .iq3_xxs = try loadQuantizedWeight(.iq3_xxs, ctx, info, shape) },
            .iq3_s => .{ .iq3_s = try loadQuantizedWeight(.iq3_s, ctx, info, shape) },
            .iq4_nl => .{ .iq4_nl = try loadQuantizedWeight(.iq4_nl, ctx, info, shape) },
            .iq4_xs => .{ .iq4_xs = try loadQuantizedWeight(.iq4_xs, ctx, info, shape) },
            .tq1_0 => .{ .tq1_0 = try loadQuantizedWeight(.tq1_0, ctx, info, shape) },
            .tq2_0 => .{ .tq2_0 = try loadQuantizedWeight(.tq2_0, ctx, info, shape) },
            .mxfp4 => .{ .mxfp4 = try loadQuantizedWeight(.mxfp4, ctx, info, shape) },
            .nvfp4 => .{ .nvfp4 = try loadQuantizedWeight(.nvfp4, ctx, info, shape) },
            else => Error.UnsupportedWeightType,
        };
    }

    pub fn deinit(self: *LinearWeight) void {
        switch (self.*) {
            inline else => |*value| value.deinit(),
        }
        self.* = undefined;
    }

    pub fn cloneView(self: *const LinearWeight, ctx: *ExecContext) !LinearWeight {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            .q4_k => |*value| .{ .q4_k = try value.cloneView(ctx) },
            .q5_k => |*value| .{ .q5_k = try value.cloneView(ctx) },
            .q6_k => |*value| .{ .q6_k = try value.cloneView(ctx) },
            .q8_0 => |*value| .{ .q8_0 = try value.cloneView(ctx) },
            .ptqtp => |*value| blk: {
                var p1 = try value.p1.withTags(ctx, .{ .out, .in });
                errdefer p1.deinit();
                var p2: ?QuantWeight(.tq2_0) = if (value.p2) |*plane|
                    try plane.withTags(ctx, .{ .out, .in })
                else
                    null;
                errdefer if (p2) |*plane| plane.deinit();
                const p3: ?QuantWeight(.tq2_0) = if (value.p3) |*plane|
                    try plane.withTags(ctx, .{ .out, .in })
                else
                    null;
                break :blk .{ .ptqtp = WeightPtqtp.init(ctx.allocator, p1, p2, p3, false) };
            },
            inline else => |*value, tag| blk: {
                const view = try value.withTags(ctx, .{ .out, .in });
                break :blk @unionInit(LinearWeight, @tagName(tag), view);
            },
        };
    }

    pub fn outDim(self: *const LinearWeight) usize {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            .q4_k => |*w| w.value.dim(.out),
            .q5_k => |*w| w.value.dim(.out),
            .q6_k => |*w| w.value.dim(.out),
            .q8_0 => |*w| w.value.dim(.out),
            .ptqtp => |*w| w.p1.dim(.out),
            inline else => |*w| w.dim(.out),
        };
    }

    pub fn inDim(self: *const LinearWeight) usize {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            .q4_k => |*w| w.value.dim(.in),
            .q5_k => |*w| w.value.dim(.in),
            .q6_k => |*w| w.value.dim(.in),
            .q8_0 => |*w| w.value.dim(.in),
            .ptqtp => |*w| w.p1.dim(.in),
            inline else => |*w| w.dim(.in),
        };
    }

    /// Replace this weight with a RESIDENT dequantized f16 copy (2 B/weight):
    /// the f16-operands GEMM path's weight format — and the `-Dgpu=metal`
    /// f16 offload operand. Rows are dequantized in chunks through the same
    /// row-gather the embedding lookup uses, so the transient peak stays a
    /// few MB. No-op when the weight is already f16. Supported for the arms
    /// `getRowsAs` covers (f32/f16/bf16/q4_k/q5_k/q6_k/q8_0).
    pub fn toResidentF16(self: *LinearWeight, ctx: *ExecContext) !void {
        switch (self.*) {
            .f16 => return,
            else => {},
        }
        const rows = self.outDim();
        const cols = self.inDim();
        const allocator = ctx.allocator;

        const values = try allocator.alloc(f16, rows * cols);
        defer allocator.free(values);
        const chunk_max: usize = 4096;
        const ids = try allocator.alloc(usize, @min(chunk_max, rows));
        defer allocator.free(ids);
        var row0: usize = 0;
        while (row0 < rows) : (row0 += chunk_max) {
            const chunk = @min(chunk_max, rows - row0);
            for (ids[0..chunk], 0..) |*id, i| id.* = row0 + i;
            var rows_f32 = try self.getRowsAs(ctx, ids[0..chunk], .in);
            defer rows_f32.deinit();
            const src = try rows_f32.dataConst();
            for (values[row0 * cols ..][0 .. chunk * cols], src) |*dst, v| dst.* = @floatCast(v);
        }

        const fresh = try WeightF16.fromSlice(ctx, .{ rows, cols }, values);
        self.deinit();
        self.* = .{ .f16 = fresh };
    }

    /// Whether `toPtqtp` accepts this weight: any non-ptqtp arm whose
    /// contract dim satisfies the TQ2_0 256-element block granularity.
    pub fn ptqtpEligible(self: *const LinearWeight) bool {
        return switch (self.*) {
            .ptqtp => false,
            else => self.inDim() % fucina.ptqtp.block_len == 0,
        };
    }

    /// Replace this weight with its PTQTP trit-plane decoration
    /// (arXiv:2509.16989; docs/PTQTP.md): rows are dequantized in chunks
    /// through the same row-gather the embedding lookup uses — so ANY
    /// loadable source dtype (f32/f16/bf16/K-quants/legacy/cold formats)
    /// quantizes through one code path — then the solver packs two TQ2_0
    /// planes and the original storage is dropped (the "purge"): the weight
    /// becomes ~2 x 2.0625 bits. Returns the solver diagnostics. Requires
    /// `ptqtpEligible`; `Error.UnsupportedWeightType` otherwise.
    pub fn toPtqtp(self: *LinearWeight, ctx: *ExecContext, options: fucina.ptqtp.Options) !fucina.ptqtp.MatrixStats {
        if (!self.ptqtpEligible()) return Error.UnsupportedWeightType;
        const rows = self.outDim();
        const cols = self.inDim();
        const allocator = ctx.allocator;

        const values = try allocator.alloc(f32, rows * cols);
        defer allocator.free(values);
        const chunk_max: usize = 4096;
        const ids = try allocator.alloc(usize, @min(chunk_max, rows));
        defer allocator.free(ids);
        var row0: usize = 0;
        while (row0 < rows) : (row0 += chunk_max) {
            const chunk = @min(chunk_max, rows - row0);
            for (ids[0..chunk], 0..) |*id, i| id.* = row0 + i;
            var rows_f32 = try self.getRowsAs(ctx, ids[0..chunk], .in);
            defer rows_f32.deinit();
            const src = try rows_f32.dataConst();
            @memcpy(values[row0 * cols ..][0 .. chunk * cols], src);
        }

        var pair = try fucina.ptqtp.quantizeMatrix(ctx, values, rows, cols, options);
        defer pair.deinit(ctx.allocator);
        var p1 = try QuantWeight(.tq2_0).fromBlocks(ctx, .{ rows, cols }, pair.plane1);
        errdefer p1.deinit();
        var p2: ?QuantWeight(.tq2_0) = if (pair.plane2.len != 0)
            try QuantWeight(.tq2_0).fromBlocks(ctx, .{ rows, cols }, pair.plane2)
        else
            null;
        errdefer if (p2) |*plane| plane.deinit();
        const p3: ?QuantWeight(.tq2_0) = if (pair.plane3.len != 0)
            try QuantWeight(.tq2_0).fromBlocks(ctx, .{ rows, cols }, pair.plane3)
        else
            null;
        const stats = pair.stats;
        self.deinit();
        self.* = .{ .ptqtp = WeightPtqtp.init(ctx.allocator, p1, p2, p3, options.tie_scales) };
        return stats;
    }

    pub fn linearSeq(self: *const LinearWeight, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag }) {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            .q4_k => |*weight| try linearSeqQ4_K(weight, ctx, input, in_tag, out_tag),
            .q5_k => |*weight| try linearSeqQ5_K(weight, ctx, input, in_tag, out_tag),
            .q6_k => |*weight| try linearSeqQ6_K(weight, ctx, input, in_tag, out_tag),
            .q8_0 => |*weight| try linearSeqQ8_0(weight, ctx, input, in_tag, out_tag),
            .ptqtp => |*weight| blk: {
                if (try linearSeqPtqtpFused(weight, ctx, input, out_tag)) |fused| break :blk fused;
                var p1 = try weight.p1.withTags(ctx, .{ out_tag, in_tag });
                defer p1.deinit();
                var acc = try input.dot(ctx, &p1, in_tag);
                inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
                    if (@field(weight, plane_field)) |*plane| {
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
                break :blk acc;
            },
            inline else => |*weight| blk: {
                var tagged_weight = try weight.withTags(ctx, .{ out_tag, in_tag });
                defer tagged_weight.deinit();
                break :blk try input.dot(ctx, &tagged_weight, in_tag);
            },
        };
    }

    /// True when `linearSeqNormed` takes the fused normalize+quantize+packed
    /// GEMM route for an m-row input: the packed CPU arms only (GPU builds
    /// keep their offload policy, the x86 m<4 decode-compact routes keep
    /// their byte win, MMLA q4_k has no fused kernel, float/ptqtp/ternary
    /// arms have no LHS quantization to fuse into), gated by
    /// FUCINA_NO_NORM_QUANT_FUSED=1 (the A/B and emergency-revert switch —
    /// the fused route matches the unfused pair to f32 roundoff, not
    /// bitwise). Callers fanning ONE normalized input into several
    /// projections should require this for every projection before
    /// switching to the normed calls — the fallback re-normalizes per call.
    /// Norm-into-quantize fusion gate: FUCINA_NO_NORM_QUANT_FUSED=1 forces
    /// the unfused rmsNormMul + linearSeq pair, FUCINA_NORM_QUANT_FUSED=1
    /// forces the fused route. Read once, cached (winograd-style).
    pub fn setNormQuantFused(on: ?bool) void {
        const s: u8 = if (on) |o| (if (o) 1 else 2) else 0;
        norm_quant_fused_state.store(s, .release);
    }

    pub fn supportsNormedFusion(self: *const LinearWeight, m: usize) bool {
        if (comptime fucina.internal.gpu.enabled) return false;
        if (!normQuantFusedEnabled()) return false;
        // Prefill-only: at decode shapes (m < 4) the fused route pays pooled
        // scratch acquisitions plus a padded 4-row-group quantize for one
        // real row, where the unfused internal quantizer has an m=1 stack
        // fast path — measured 2-3% decode LOSS on M1 Q4_K_M/Q8_0, against
        // a +11-23% pp32 win. (The x86 m<4 decode-compact routes bypass the
        // packed path anyway.)
        if (m < 4) return false;
        return switch (self.*) {
            .q4_k => comptime !fucina.supports_q4_k_mmla,
            .q8_0 => true,
            .q5_k => true,
            .q6_k => true,
            else => false,
        };
    }

    /// `linearSeq` over `rmsNormMul(x, norm_weight, eps)` — on the packed CPU
    /// routes (see `supportsNormedFusion`) the normalized [m, k] tensor is
    /// never materialized: the fused kernel normalizes up to 4 rows into
    /// task-private scratch with the exact kernels the unfused dispatch uses
    /// and quantizes in place — results match the unfused pair to f32
    /// roundoff (<= 1 ulp observed). Every other arm (and
    /// FUCINA_NO_NORM_QUANT_FUSED=1) normalizes and delegates.
    pub fn linearSeqNormed(
        self: *const LinearWeight,
        ctx: *ExecContext,
        x: anytype,
        norm_weight: anytype,
        eps: f32,
        comptime in_tag: Tag,
        comptime out_tag: Tag,
    ) !fucina.Tensor(.{ .seq, out_tag }) {
        @setEvalBranchQuota(20_000);
        if (comptime !fucina.internal.gpu.enabled) {
            if (!x.requiresGrad() and x.dim(.seq) >= 4 and normQuantFusedEnabled()) switch (self.*) {
                .q4_k => |*weight| if (comptime !fucina.supports_q4_k_mmla) {
                    return x.rmsNormMulDotPacked(ctx, norm_weight, eps, &weight.packed_rhs, in_tag, out_tag);
                },
                .q8_0 => |*weight| return x.rmsNormMulDotPacked(ctx, norm_weight, eps, &weight.packed_rhs, in_tag, out_tag),
                .q5_k => |*weight| return x.rmsNormMulDotPacked(ctx, norm_weight, eps, &weight.packed_rhs, in_tag, out_tag),
                .q6_k => |*weight| return x.rmsNormMulDotPacked(ctx, norm_weight, eps, &weight.packed_rhs, in_tag, out_tag),
                else => {},
            };
        }
        var normed = try x.rmsNormMul(ctx, in_tag, norm_weight, eps);
        defer normed.deinit();
        return self.linearSeq(ctx, &normed, in_tag, out_tag);
    }

    pub fn getRowsAs(self: *const LinearWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !fucina.Tensor(.{ .seq, out_tag }) {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            .f32 => |*table| blk: {
                var rows = try table.gather(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .f16 => |*table| blk: {
                var rows_f16 = try table.gather(ctx, .out, token_ids, .seq);
                defer rows_f16.deinit();
                var rows = try rows_f16.to(ctx, .f32);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .bf16 => |*table| blk: {
                var rows_bf16 = try table.gather(ctx, .out, token_ids, .seq);
                defer rows_bf16.deinit();
                var rows = try rows_bf16.to(ctx, .f32);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .q6_k => |*table| blk: {
                var rows = try table.value.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .q8_0 => |*table| blk: {
                var rows = try table.value.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .q4_k => |*table| blk: {
                var rows = try table.value.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .q5_k => |*table| blk: {
                var rows = try table.value.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            .ptqtp => |*table| blk: {
                var acc = try table.p1.getRows(ctx, .out, token_ids, .seq);
                defer acc.deinit();
                inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
                    if (@field(table, plane_field)) |*plane| {
                        var rows2 = try plane.getRows(ctx, .out, token_ids, .seq);
                        defer rows2.deinit();
                        const sum = try acc.add(ctx, &rows2);
                        acc.deinit();
                        acc = sum;
                    }
                }
                break :blk try acc.withTags(ctx, .{ .seq, out_tag });
            },
            inline else => |*table| blk: {
                var rows = try table.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
        };
    }
};

/// Load all experts of one MoE projection from a 3D stacked tensor
/// (`blk.N.ffn_{gate,up,down}_exps.weight`, GGUF shape `[in, out, n_expert]`)
/// into a SINGLE packed matmul RHS. The 3D tensor is expert-major contiguous, so
/// it is logically a `(n_expert*out, in)` matrix: we pack it once (fast load, no
/// per-expert allocations) and the fused MoE kernel slices each expert as a
/// zero-copy row-block sub-view. Only K-quant experts (q4_k/q5_k/q6_k) are
/// supported — the formats every real MoE GGUF uses, all sharing the Q8_K LHS
/// hot path; the raw blocks are dropped after packing to avoid doubling memory.
pub fn loadMoeRhs(
    ctx: *ExecContext,
    info: *const gguf.TensorInfo,
    expected_in_dim: usize,
    expected_out_dim: usize,
    expected_n_expert: usize,
    borrow: bool,
) !fucina.MoeRhs {
    if (info.n_dims != 3) return Error.InvalidWeightShape;
    const in_dim = info.dims[0];
    const out_dim = info.dims[1];
    const n_expert = info.dims[2];
    if (in_dim != expected_in_dim or out_dim != expected_out_dim or n_expert != expected_n_expert) return Error.InvalidWeightShape;
    const rows = try std.math.mul(usize, n_expert, out_dim);

    return switch (info.ggml_type) {
        .q2_k => .{ .q2_k = try copyOrBorrowMoeRhs(fucina.QuantizedMatmulRhsQ2_K, fucina.BlockQ2_K, ctx, info, rows, in_dim, borrow) },
        .q3_k => .{ .q3_k = try copyOrBorrowMoeRhs(fucina.QuantizedMatmulRhsQ3_K, fucina.BlockQ3_K, ctx, info, rows, in_dim, borrow) },
        .q4_k => .{ .q4_k = try copyOrBorrowMoeRhs(fucina.QuantizedMatmulRhsQ4_K, fucina.BlockQ4_K, ctx, info, rows, in_dim, borrow) },
        .q5_k => .{ .q5_k = try copyOrBorrowMoeRhs(fucina.QuantizedMatmulRhsQ5_K, fucina.BlockQ5_K, ctx, info, rows, in_dim, borrow) },
        .q6_k => .{ .q6_k = try copyOrBorrowMoeRhs(fucina.QuantizedMatmulRhsQ6_K, fucina.BlockQ6_K, ctx, info, rows, in_dim, borrow) },
        // q8_0: what llama.cpp falls back to when an expert dim is not a
        // 256 multiple (deepseek2). Nested rows container, so it gets its
        // own copy-or-borrow.
        .q8_0 => blk: {
            const src = try blockSlice(fucina.BlockQ8_0, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (bpc * 32 != in_dim) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .q8_0 = .{ .rows = .{ .allocator = null, .blocks = src, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockQ8_0, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .q8_0 = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        // iq2_xxs experts: nested generic rows container with mutable
        // blocks, so the borrow arm needs the sound @constCast.
        .iq2_xxs => blk: {
            const src = try blockSlice(fucina.BlockIQ2_XXS, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (try fucina.internal.backend_mod.quantized_matmul.qkBlockCount(in_dim) != bpc) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .iq2_xxs = .{ .rows = .{ .allocator = null, .blocks = @constCast(src), .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockIQ2_XXS, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .iq2_xxs = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        .iq2_s => blk: {
            const src = try blockSlice(fucina.BlockIQ2_S, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (try fucina.internal.backend_mod.quantized_matmul.qkBlockCount(in_dim) != bpc) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .iq2_s = .{ .rows = .{ .allocator = null, .blocks = @constCast(src), .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockIQ2_S, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .iq2_s = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        .iq4_xs => blk: {
            const src = try blockSlice(fucina.BlockIQ4_XS, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (try fucina.internal.backend_mod.quantized_matmul.qkBlockCount(in_dim) != bpc) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .iq4_xs = .{ .rows = .{ .allocator = null, .blocks = @constCast(src), .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockIQ4_XS, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .iq4_xs = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        .iq3_xxs => blk: {
            const src = try blockSlice(fucina.BlockIQ3_XXS, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (try fucina.internal.backend_mod.quantized_matmul.qkBlockCount(in_dim) != bpc) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .iq3_xxs = .{ .rows = .{ .allocator = null, .blocks = @constCast(src), .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockIQ3_XXS, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .iq3_xxs = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        // Ternary experts (TQ2_0): nested generic rows container with
        // mutable blocks, so the borrow arm needs the sound @constCast.
        .tq2_0 => blk: {
            const src = try blockSlice(fucina.BlockTQ2_0, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (try fucina.internal.backend_mod.quantized_matmul.qkBlockCount(in_dim) != bpc) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .tq2_0 = .{ .rows = .{ .allocator = null, .blocks = @constCast(src), .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockTQ2_0, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .tq2_0 = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        else => Error.UnsupportedWeightType,
    };
}

/// PTQTP counterpart of `loadMoeRhs`: build the multi-plane `ptqtp` MoeRhs
/// arm from persisted `<name>.ptqtpK` sibling plane tensors (plane-major,
/// each with the base expert stack's `[in, out, n_expert]` shape and
/// standalone-valid TQ2_0 payload — `llm/ptqtp_gguf.zig` owns the
/// naming/version pair-detection and calls this). Planes are borrowed from
/// the mapping or copied, exactly as `loadMoeRhs` treats a tq2_0 stack.
pub fn loadMoeRhsPtqtp(
    ctx: *ExecContext,
    plane_infos: []const *const gguf.TensorInfo,
    expected_in_dim: usize,
    expected_out_dim: usize,
    expected_n_expert: usize,
    borrow: bool,
    tied: bool,
) !fucina.MoeRhs {
    if (plane_infos.len == 0 or plane_infos.len > 3) return Error.InvalidWeightShape;
    const rows = try std.math.mul(usize, expected_n_expert, expected_out_dim);
    const bpc = try fucina.internal.backend_mod.quantized_matmul.qkBlockCount(expected_in_dim);
    const blocks_per_plane = try std.math.mul(usize, rows, bpc);

    var planes: [3][]const fucina.BlockTQ2_0 = .{ &.{}, &.{}, &.{} };
    var owned_count: usize = 0;
    errdefer for (planes[0..owned_count]) |plane| ctx.allocator.free(@constCast(plane));
    for (plane_infos, 0..) |info, p| {
        if (info.ggml_type != .tq2_0) return Error.UnsupportedWeightType;
        if (info.n_dims != 3) return Error.InvalidWeightShape;
        if (info.dims[0] != expected_in_dim or info.dims[1] != expected_out_dim or info.dims[2] != expected_n_expert) return Error.InvalidWeightShape;
        const src = try blockSlice(fucina.BlockTQ2_0, info.data);
        if (src.len != blocks_per_plane) return Error.InvalidWeightShape;
        if (borrow) {
            planes[p] = src;
        } else {
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(fucina.BlockTQ2_0, src.len);
            @memcpy(owned, src);
            planes[p] = owned;
            owned_count += 1;
        }
    }
    // Tie-fitted K=2 stacks fold into the 4-bit one-pass pack, expert by
    // expert (docs/PTQTP.md). Errors propagate rather than degrade: the
    // streamed tier serves a tied K=2 file folded-or-error (ProjSpec.fold
    // has no fallback), so the resident tier folding under the same file
    // condition keeps the two tiers bitwise-identical on every file both
    // can load — a silent 2-pass fallback here would diverge from a
    // streamed run of the same file in final f32 ulps.
    var folded: []const backend_quant.BlockTQ2_0Foldedx4 = &.{};
    var folded_allocator: ?Allocator = null;
    if (tied and plane_infos.len == 2 and expected_out_dim % 4 == 0) {
        const fg = (expected_out_dim / 4) * bpc;
        const buf = try ctx.allocator.alloc(backend_quant.BlockTQ2_0Foldedx4, expected_n_expert * fg);
        errdefer ctx.allocator.free(buf);
        const expert_blocks = expected_out_dim * bpc;
        for (0..expected_n_expert) |e| {
            var views: [2]backend_quant.QuantizedMatmulRhsTQ2_0 = undefined;
            for (0..2) |p| {
                const blocks = planes[p][e * expert_blocks ..][0..expert_blocks];
                views[p] = try backend_quant.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(expected_in_dim, expected_out_dim, @constCast(blocks));
            }
            try backend_quant.packMatmulRhsTQ2_0Foldedx4Into(buf[e * fg ..][0..fg], &views[0], &views[1]);
        }
        folded = buf;
        folded_allocator = ctx.allocator;
    }
    return .{ .ptqtp = .{
        .allocator = if (borrow) null else ctx.allocator,
        .planes = planes,
        .plane_count = plane_infos.len,
        .k = expected_in_dim,
        .n = rows,
        .blocks_per_column = bpc,
        .folded = folded,
        .folded_allocator = folded_allocator,
    } };
}

/// Opt-in disk streaming for the MoE expert stacks, shared by every MoE
/// loader (`LoadOptions.moe_stream`): experts stay on disk and are `pread`
/// on demand through a tiered store (pinned + LRU + working set), so a
/// mixture model loads with only its dense weights resident. Decode then
/// pays disk reads for expert-cache misses — the explicit trade that lets a
/// bigger-than-RAM model run at all (docs: out-of-core MoE).
pub const MoeStreamOptions = struct {
    /// Path of the same GGUF being loaded; the store opens its own read fds
    /// (every part of a split GGUF), so the load-time mmap can be released
    /// after load — resident memory stays dense weights + expert cache.
    gguf_path: []const u8,
    /// Total RAM budget for the streamed tiers across all layers. Default:
    /// half of available memory at load time.
    cache_bytes: ?usize = null,
    /// Fixed LRU slots per layer; wins over `cache_bytes` when set.
    cache_slots_per_layer: ?usize = null,
    /// OS readahead hints for miss batches.
    readahead: bool = true,
    /// The learning cache: pin the hottest experts from the persisted usage
    /// sidecar (`<gguf>.experts`) at load; save updated counts with
    /// `ExpertStore.saveUsage` at generation/turn boundaries.
    auto_pin: bool = true,
    /// RAM for the pinned tier (default: half the budget when history
    /// qualifies).
    pin_bytes: ?usize = null,
    /// Router-lookahead prefetch: predict each next layer's experts from the
    /// current post-attention state and readahead them from a background I/O
    /// thread while the current layer computes. Honored by the models that
    /// implement the lookahead hook (qwen3, deepseek2); never changes
    /// output. Prediction recall is measured in `ExpertStore.Stats`.
    pilot: bool = false,
    /// Extra full copies of the model, one path per copy (part-1 path for
    /// split GGUFs; siblings resolve like `gguf_path`), typically each on
    /// its own drive. Expert reads split across every copy by a
    /// deterministic weighted hash, so aggregate streaming bandwidth
    /// scales with the drives holding one — the lever for disk-bound
    /// decode. Output is unchanged; a mirror read error falls back to the
    /// primary.
    mirror_paths: []const []const u8 = &.{},
    /// Read share per mirror relative to the primary's 1 (parallel to
    /// `mirror_paths`); null = 1 each, an even split across all copies.
    mirror_weights: ?[]const f32 = null,
    /// Demand-miss reads fan out across this many persistent I/O worker
    /// threads (the acquiring thread participates too); 0 = sequential.
    /// Parallelism is what lets disk queue depth — and mirror copies on
    /// separate drives — add bandwidth within one acquire.
    io_workers: usize = 8,
};

/// The runners' shared `--moe-mirror-weights=` comma list, parsed into
/// `buf` and validated against the number of `--moe-mirror` flags given.
/// A null argument returns null (even split).
pub fn parseMirrorWeights(arg: ?[]const u8, n_mirrors: usize, buf: []f32) !?[]const f32 {
    const list = arg orelse return null;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |tok| {
        if (n >= buf.len) return error.TooManyMirrors;
        buf[n] = try std.fmt.parseFloat(f32, tok);
        n += 1;
    }
    if (n != n_mirrors) return error.MirrorWeightsMismatch;
    return buf[0..n];
}

/// The store-create block shared by the MoE loaders: expand split-GGUF part
/// paths (single files pass through as one entry) and open the ExpertStore
/// over them. The caller registers layers (`loadMoeRhsStreamed`) and then
/// calls `ExpertStore.finalize`.
pub fn createExpertStore(allocator: Allocator, options: MoeStreamOptions, n_layers: usize) !*fucina.ExpertStore {
    const split_paths = try gguf.File.splitPartPaths(allocator, options.gguf_path);
    defer if (split_paths) |paths| {
        for (paths) |part| allocator.free(part);
        allocator.free(paths);
    };
    var one_path = [_][]const u8{options.gguf_path};
    const store_paths: []const []const u8 = if (split_paths) |paths| blk: {
        const view = try allocator.alloc([]const u8, paths.len);
        for (view, paths) |*d, src| d.* = src;
        break :blk view;
    } else &one_path;
    defer if (split_paths != null) allocator.free(store_paths);
    if (options.mirror_weights) |ws| {
        if (ws.len != options.mirror_paths.len) return error.MirrorWeightsMismatch;
    }
    const store = try fucina.ExpertStore.create(allocator, store_paths, n_layers, .{
        .cache_bytes = options.cache_bytes,
        .cache_slots_per_layer = options.cache_slots_per_layer,
        .readahead = options.readahead,
        .auto_pin = options.auto_pin,
        .pin_bytes = options.pin_bytes,
        .io_workers = options.io_workers,
    });
    errdefer store.destroy();
    for (options.mirror_paths, 0..) |mirror_path, m| {
        const weight = if (options.mirror_weights) |ws| ws[m] else 1.0;
        const mirror_split = try gguf.File.splitPartPaths(allocator, mirror_path);
        defer if (mirror_split) |paths| {
            for (paths) |part| allocator.free(part);
            allocator.free(paths);
        };
        var one_mirror = [_][]const u8{mirror_path};
        const mirror_parts: []const []const u8 = if (mirror_split) |paths| blk: {
            const view = try allocator.alloc([]const u8, paths.len);
            for (view, paths) |*d, src| d.* = src;
            break :blk view;
        } else &one_mirror;
        defer if (mirror_split != null) allocator.free(mirror_parts);
        try store.addMirror(mirror_parts, weight);
    }
    return store;
}

/// Exit-time streamed-tier report shared by the MoE runners: print the
/// stats line(s) and persist the usage histogram (the learning cache)
/// unless `learn` is false. Failures lose only the report/learning.
pub fn reportAndSaveMoeStream(store: *fucina.ExpertStore, learn: bool, writer: anytype) void {
    if (learn) store.saveUsage() catch {};
    const s = store.stats;
    writer.print(
        "moe stream: {d} acquires, hits {d} / misses {d} ({d:.1}% hit, {d} pin hits), {d:.2} GB read in {d:.2}s, cap {d} slots/layer, pinned {d} experts ({d:.2} GB)\n",
        .{ s.acquires, s.hits, s.misses, s.hitRate() * 100, s.pin_hits, @as(f64, @floatFromInt(s.bytes_read)) / 1e9, @as(f64, @floatFromInt(s.read_ns)) / 1e9, store.cap, store.pinned_experts, @as(f64, @floatFromInt(store.pinned_bytes)) / 1e9 },
    ) catch {};
    if (s.pilot_recall_total > 0) writer.print(
        "moe pilot: recall {d:.1}% ({d}/{d} routed experts predicted), {d} experts hinted\n",
        .{ s.pilotRecall() * 100, s.pilot_recall_hits, s.pilot_recall_total, s.pilot_ranges },
    ) catch {};
    if (s.staged_loads > 0) writer.print(
        "moe prefetch: staged {d} loads ({d:.2} GB), consumed {d}, wasted {d}\n",
        .{ s.staged_loads, @as(f64, @floatFromInt(s.staged_bytes)) / 1e9, s.staged_consumed, s.staged_wasted },
    ) catch {};
    if (store.mirrors.len > 0) {
        var total: u64 = 0;
        for (store.copy_bytes) |*b| total += b.load(.monotonic);
        writer.print("moe mirror: {d} copies, reads", .{store.mirrors.len + 1}) catch {};
        for (store.copy_bytes, 0..) |*b, i| {
            const bytes = b.load(.monotonic);
            const pct = if (total == 0) 0 else @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(total)) * 100;
            writer.print("{s}{d:.1}% ({d:.2} GB)", .{ if (i == 0) " " else " / ", pct, @as(f64, @floatFromInt(bytes)) / 1e9 }) catch {};
        }
        const fallbacks = store.mirror_fallbacks.load(.monotonic);
        if (fallbacks > 0) {
            writer.print(", {d} mirror reads fell back to the primary\n", .{fallbacks}) catch {};
        } else {
            writer.print("\n", .{}) catch {};
        }
    }
}

/// Streamed counterpart of three `loadMoeRhs` calls: registers one layer's
/// gate/up/down stacked expert tensors with the ExpertStore (which will
/// `pread` individual experts on demand) instead of materializing or
/// borrowing them. Nothing of the expert stacks is read here — only the
/// geometry is validated, exactly as `loadMoeRhs` would.
pub const StreamedMoeFfnRhs = struct {
    gate: fucina.MoeRhs,
    up: fucina.MoeRhs,
    down: fucina.MoeRhs,
};

pub fn loadMoeRhsStreamed(
    store: *fucina.ExpertStore,
    file: *const gguf.File,
    layer_i: usize,
    gate_info: *const gguf.TensorInfo,
    up_info: *const gguf.TensorInfo,
    down_info: *const gguf.TensorInfo,
    expected_in_dim: usize,
    expected_out_dim: usize,
    expected_n_expert: usize,
) !StreamedMoeFfnRhs {
    return registerStreamedMoeLayer(store, layer_i, .{
        try streamedProjSpec(file, gate_info, expected_in_dim, expected_out_dim, expected_n_expert),
        try streamedProjSpec(file, up_info, expected_in_dim, expected_out_dim, expected_n_expert),
        // down transposes the FFN: (out_pe -> hidden).
        try streamedProjSpec(file, down_info, expected_out_dim, expected_in_dim, expected_n_expert),
    }, expected_n_expert);
}

/// Register one layer's three ProjSpecs (`streamedProjSpec` /
/// `streamedProjSpecPtqtp` — the specs may mix plain and PTQTP
/// projections) and hand out the streamed arms.
pub fn registerStreamedMoeLayer(
    store: *fucina.ExpertStore,
    layer_i: usize,
    specs: [3]fucina.expert_store.ProjSpec,
    expected_n_expert: usize,
) !StreamedMoeFfnRhs {
    try store.addLayer(layer_i, specs, expected_n_expert);
    return .{
        .gate = .{ .streamed = store.streamedRhs(layer_i, .gate) },
        .up = .{ .streamed = store.streamedRhs(layer_i, .up) },
        .down = .{ .streamed = store.streamedRhs(layer_i, .down) },
    };
}

pub fn streamedProjSpec(
    file: *const gguf.File,
    info: *const gguf.TensorInfo,
    expected_in_dim: usize,
    expected_out_dim: usize,
    expected_n_expert: usize,
) !fucina.expert_store.ProjSpec {
    if (info.n_dims != 3) return Error.InvalidWeightShape;
    if (info.dims[0] != expected_in_dim or info.dims[1] != expected_out_dim or info.dims[2] != expected_n_expert) return Error.InvalidWeightShape;
    const quant: fucina.expert_store.StreamedQuant = switch (info.ggml_type) {
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .q8_0 => .q8_0,
        .tq2_0 => .tq2_0,
        .q2_k => .q2_k,
        .iq2_xxs => .iq2_xxs,
        .iq3_xxs => .iq3_xxs,
        .iq2_s => .iq2_s,
        .iq4_xs => .iq4_xs,
        .q3_k => .q3_k,
        else => return Error.UnsupportedWeightType,
    };
    return .{
        .quant = quant,
        .part = info.part,
        .file_offset = file.partDataOffset(info.part) + info.offset,
        .byte_len = info.data.len,
        .in_dim = expected_in_dim,
        .out_dim = expected_out_dim,
    };
}

/// PTQTP counterpart of `streamedProjSpec`: one ProjSpec whose
/// `plane_count`/`plane_offsets` point at the `<name>.ptqtpK` sibling
/// plane tensors. The planes stay plane-major on disk (the same
/// standalone-valid TQ2_0 tensors the dense decoration writes — no
/// expert-major interleave); the ExpertStore gathers one expert's K plane
/// row-blocks by offset into a contiguous slab section per acquire.
pub fn streamedProjSpecPtqtp(
    file: *const gguf.File,
    plane_infos: []const *const gguf.TensorInfo,
    expected_in_dim: usize,
    expected_out_dim: usize,
    expected_n_expert: usize,
    tied: bool,
) !fucina.expert_store.ProjSpec {
    if (plane_infos.len == 0 or plane_infos.len > 3) return Error.InvalidWeightShape;
    var offsets: [3]u64 = .{ 0, 0, 0 };
    for (plane_infos, 0..) |info, p| {
        if (info.ggml_type != .tq2_0) return Error.UnsupportedWeightType;
        if (info.n_dims != 3) return Error.InvalidWeightShape;
        if (info.dims[0] != expected_in_dim or info.dims[1] != expected_out_dim or info.dims[2] != expected_n_expert) return Error.InvalidWeightShape;
        if (info.part != plane_infos[0].part or info.data.len != plane_infos[0].data.len) return Error.InvalidWeightShape;
        offsets[p] = file.partDataOffset(info.part) + info.offset;
    }
    return .{
        .quant = .tq2_0,
        .part = plane_infos[0].part,
        .file_offset = offsets[0],
        .byte_len = plane_infos[0].data.len,
        .in_dim = expected_in_dim,
        .out_dim = expected_out_dim,
        .plane_count = @intCast(plane_infos.len),
        .plane_offsets = .{ offsets[1], offsets[2] },
        // Tie-fitted K=2 streams fold at fill into the one-pass 4-bit pack
        // (ExpertStore.readExpert); other shapes stream plane-per-plane.
        .fold = tied and plane_infos.len == 2 and expected_out_dim % 4 == 0,
    };
}

/// Tensor-valued wrapper for the generic Qwen-style SwiGLU MoE FFN. This keeps
/// model code in public Tensor values while preserving the exact eager raw
/// kernels and decode/prefill split underneath.
pub fn moeSwiGluFfnSeq(
    ctx: *ExecContext,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    gate: *const fucina.MoeRhs,
    up: *const fucina.MoeRhs,
    down: *const fucina.MoeRhs,
    selected: []const usize,
    routing_weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*fucina.MoeBatchProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    return moeGatedFfnSeq(ctx, input, gate, up, down, selected, routing_weights, top_k, out_pe, .swiglu, io, profile);
}

/// As `moeSwiGluFfnSeq`, with the gated activation chosen by the caller
/// (deepseek4 routes through the clamped SwiGLU).
pub fn moeGatedFfnSeq(
    ctx: *ExecContext,
    input: *const fucina.Tensor(.{ .seq, .embed }),
    gate: *const fucina.MoeRhs,
    up: *const fucina.MoeRhs,
    down: *const fucina.MoeRhs,
    selected: []const usize,
    routing_weights: []const f32,
    top_k: usize,
    out_pe: usize,
    act: fucina.GatedOp,
    io: ?std.Io,
    profile: ?*fucina.MoeBatchProfile,
) !fucina.Tensor(.{ .seq, .embed }) {
    if (input.requiresGrad()) return Error.GradUnsupported;
    const raw_input = input.asRawTensor();
    var raw = if (input.dim(.seq) == 1)
        try ctx.moeExpertFfn(
            raw_input,
            gate,
            up,
            down,
            selected,
            routing_weights,
            out_pe,
            act,
            io,
            profile,
        )
    else
        try ctx.moeExpertFfnBatch(
            raw_input,
            gate,
            up,
            down,
            selected,
            routing_weights,
            top_k,
            out_pe,
            act,
            io,
            profile,
        );
    errdefer raw.deinit();
    return fucina.Tensor(.{ .seq, .embed }).fromTensor(ctx, raw);
}

/// Copy a stacked-expert tensor's raw K-quant blocks into a compact matmul RHS.
/// The GGUF block layout is already `[output_row][in_block]` row-major, which is
/// exactly what the raw tile kernels index, so this is a plain memcpy — no
/// repack (fast load), and ~5.5 bits/weight stays resident (bandwidth-optimal
/// for the m=1 MoE GEMVs).
fn copyOrBorrowMoeRhs(comptime Rhs: type, comptime Block: type, ctx: *ExecContext, info: *const gguf.TensorInfo, rows: usize, in_dim: usize, borrow: bool) !Rhs {
    const src = try blockSlice(Block, info.data);
    if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
    const blocks_per_column = src.len / rows;
    if (borrow) {
        // Borrow the blocks straight from the (mmap'd) GGUF: the caller keeps
        // the mapping alive for the model's lifetime (gguf.File.takeMapping).
        // Skips the multi-GB expert copy and lets the OS reclaim clean pages.
        return .{ .allocator = null, .blocks = src, .k = in_dim, .n = rows, .blocks_per_column = blocks_per_column };
    }
    gguf.prefetch(info.data); // about to copy the whole stack — warm it first
    const owned = try ctx.allocator.alloc(Block, src.len);
    errdefer ctx.allocator.free(owned);
    @memcpy(owned, src);
    return .{ .allocator = ctx.allocator, .blocks = owned, .k = in_dim, .n = rows, .blocks_per_column = blocks_per_column };
}

pub fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_i, suffix });
}

pub fn loadVector(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_len: usize, comptime tag: Tag) !fucina.Tensor(.{tag}) {
    if (info.n_dims != 1 or info.dims[0] != expected_len) return Error.InvalidWeightShape;

    const values = try ctx.allocator.alloc(f32, expected_len);
    defer ctx.allocator.free(values);
    try fillF32(values, info);
    return fucina.Tensor(.{tag}).fromSlice(ctx, .{expected_len}, values);
}

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
) !fucina.Tensor(.{ .seq, out_tag }) {
    const values = try f16Slice(bytes, try std.math.mul(usize, shape[0], shape[1]));
    var rhs = try fucina.Tensor(.{ .dtype = .f16, .tags = .{ out_tag, in_tag } }).fromBorrowedConstSlice(ctx, shape, values);
    defer rhs.deinit();
    return input.dot(ctx, &rhs, in_tag);
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
) !fucina.Tensor(.{ .seq, out_tag }) {
    comptime switch (dtype) {
        .q8_0, .q4_k, .q5_k, .q6_k => {},
        else => @compileError("borrowed quantized linear supports q8_0/q4_k/q5_k/q6_k"),
    };
    if (input.requiresGrad()) return Error.GradUnsupported;
    if (input.dim(in_tag) != shape[1]) return Error.InvalidWeightShape;

    const blocks = try blockSlice(BlockStorage(dtype), bytes);
    var value = try ctx.matmul2DWithQuantizedBlocksRhsOptions(dtype, input.asRawTensor(), blocks, shape[0], shape[1], .{
        .allow_gpu = options.allow_gpu,
        .rhs_lifetime = options.rhs_lifetime,
    });
    errdefer value.deinit();
    return try fucina.Tensor(.{ .seq, out_tag }).fromTensor(ctx, value);
}

/// Try the dense quantized GPU matmul: `out = in · dequant(W)ᵀ` over the raw
/// GGUF blocks (`weight.value`), via the provider's dequant-in-kernel
/// GEMM. Returns null — caller falls back to the CPU packed path — when the GPU
/// is off, the input needs gradients (training), or the exec gate declines
/// (shape/work threshold). Comptime-elided on non-gpu builds.
fn denseQuantGpuTry(
    comptime dtype: DType,
    weight: anytype,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !?fucina.Tensor(.{ .seq, out_tag }) {
    if (comptime !fucina.internal.gpu.enabled) return null;
    if (comptime dtype == .q5_k and !fucina.internal.gpu.has_q5_k_quant) return null;
    if (input.requiresGrad()) return null;
    const m = input.dim(.seq);
    const k = input.dim(in_tag);
    const n = weight.value.dim(.out);
    if (weight.value.dim(.in) != k) return null;
    const wraw = weight.value.asRawTensor();
    if (!wraw.isContiguous()) return null;
    const wbytes = std.mem.sliceAsBytes(wraw.dataConst());
    const nb01 = std.math.divExact(usize, wbytes.len, n) catch return null;
    var out = (try ctx.denseQuantMatmulGpu(dtype, wbytes, weight.rhs_lifetime, nb01, input.asRawTensor(), m, n, k)) orelse return null;
    errdefer out.deinit();
    return try fucina.Tensor(.{ .seq, out_tag }).fromTensor(ctx, out);
}

pub fn linearSeqQ8_0(
    weight: *const WeightQ8_0,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    if (try denseQuantGpuTry(.q8_0, weight, ctx, input, in_tag, out_tag)) |r| return r;
    return input.dotPacked(ctx, &weight.packed_rhs, in_tag, out_tag);
}

pub fn linearSeqQ4_K(
    weight: *const WeightQ4_K,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    if (try denseQuantGpuTry(.q4_k, weight, ctx, input, in_tag, out_tag)) |r| return r;
    return input.dotPacked(ctx, &weight.packed_rhs, in_tag, out_tag);
}

/// Q5_K decode-route gate: at decode shapes (m < 4) the dense Q5_K matmul is a
/// DRAM-bound GEMV, and the byte-expanded packed layout (BlockQ5_Kx8, 276 B per
/// 256-weight block per column = 8.625 bpw) streams 1.57x the bytes of the
/// GGUF-native compact blocks already resident in `weight.value` (176 B =
/// 5.5 bpw). Routing decode through the compact tensor-RHS path is
/// bitwise-equal (same Q8_K LHS quantization, same order-independent i32
/// integer stage, same f32 epilogue association — proven by the cross-layout
/// test in q5_k_tests.zig) and wins where bandwidth is the limit: default ON
/// on x86_64 (the measured ~10% decode loss), OFF on aarch64 (pending
/// measurement). Runtime overrides: FUCINA_Q5K_DECODE_COMPACT=1 forces on,
/// FUCINA_NO_Q5K_DECODE_COMPACT=1 forces off (the A/B and emergency-revert
/// switches, winograd-style). Read once, cached.
var norm_quant_fused_state = std.atomic.Value(u8).init(0); // 0 = unread, 1 = enabled, 2 = disabled
fn normQuantFusedEnabled() bool {
    const s = norm_quant_fused_state.load(.acquire);
    if (s != 0) return s == 1;
    const on = if (fucina.parallel.envPositiveUsize("FUCINA_NO_NORM_QUANT_FUSED") != null)
        false
    else if (fucina.parallel.envPositiveUsize("FUCINA_NORM_QUANT_FUSED") != null)
        true
    else
        true;
    norm_quant_fused_state.store(if (on) 1 else 2, .release);
    return on;
}

const q5k_decode_compact_default_on = builtin.cpu.arch == .x86_64;
var q5k_decode_compact_state = std.atomic.Value(u8).init(0); // 0 = unread, 1 = enabled, 2 = disabled
fn q5kDecodeCompactEnabled() bool {
    const s = q5k_decode_compact_state.load(.acquire);
    if (s != 0) return s == 1;
    const on = if (fucina.parallel.envPositiveUsize("FUCINA_NO_Q5K_DECODE_COMPACT") != null)
        false
    else if (fucina.parallel.envPositiveUsize("FUCINA_Q5K_DECODE_COMPACT") != null)
        true
    else
        q5k_decode_compact_default_on;
    q5k_decode_compact_state.store(if (on) 1 else 2, .release);
    return on;
}

/// Programmatic override for the Q5_K decode-route gate (pre-seeds the
/// read-once cache, like `parallel.setMaxThreads`): `true`/`false` force the
/// compact/packed route, `null` resets to unread so the next query re-reads
/// the env/arch default. The tests' A/B hook; also usable from a CLI flag.
pub fn setQ5kDecodeCompact(on: ?bool) void {
    const s: u8 = if (on) |o| (if (o) 1 else 2) else 0;
    q5k_decode_compact_state.store(s, .release);
}

pub fn linearSeqQ5_K(
    weight: *const WeightQ5_K,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    if (try denseQuantGpuTry(.q5_k, weight, ctx, input, in_tag, out_tag)) |r| return r;
    // Decode shapes: contract against the resident GGUF-native compact blocks
    // (`weight.value`) through the public quantized-RHS `dot` (exec's
    // matmul2DWithQuantizedTensorRhsOptions -> matmulQ5_KRhsTile kernels with
    // column-split threading) instead of the byte-expanded packed layout —
    // bitwise-equal outputs, ~1.57x fewer weight bytes streamed (see the gate
    // comment above). Grad inputs keep the packed path's explicit
    // GradientQuantizedMatmulUnsupported error.
    if (input.dim(.seq) < 4 and !input.requiresGrad() and q5kDecodeCompactEnabled()) {
        var tagged = try weight.value.withTags(ctx, .{ out_tag, in_tag });
        defer tagged.deinit();
        return input.dot(ctx, &tagged, in_tag);
    }
    return input.dotPacked(ctx, &weight.packed_rhs, in_tag, out_tag);
}

/// Q6_K decode-route gate, the Q5_K gate's ride-along (same bytes-not-kernels
/// story, smaller expansion): the packed Q6_Kx4 layout is byte-expanded to
/// 274 B per 256-weight block per column vs the GGUF-native compact blocks'
/// 210 B (6.5625 bpw) — 1.30x the necessary traffic on the DRAM-bound m < 4
/// GEMV. Bitwise-equal outputs (same Q8_K LHS quantization, same exact i32
/// iacc = sum dot*scale — Q6_K has no separate mins path — and the identical
/// f32 epilogue association acc + float(iacc)*(f16(d_w)*a.d) in ascending
/// block order; proven by the cross-layout test in q6_k_tests.zig). Default
/// ON on x86_64, OFF on aarch64. Runtime overrides:
/// FUCINA_Q6K_DECODE_COMPACT=1 forces on, FUCINA_NO_Q6K_DECODE_COMPACT=1
/// forces off. Read once, cached.
const q6k_decode_compact_default_on = builtin.cpu.arch == .x86_64;
var q6k_decode_compact_state = std.atomic.Value(u8).init(0); // 0 = unread, 1 = enabled, 2 = disabled
fn q6kDecodeCompactEnabled() bool {
    const s = q6k_decode_compact_state.load(.acquire);
    if (s != 0) return s == 1;
    const on = if (fucina.parallel.envPositiveUsize("FUCINA_NO_Q6K_DECODE_COMPACT") != null)
        false
    else if (fucina.parallel.envPositiveUsize("FUCINA_Q6K_DECODE_COMPACT") != null)
        true
    else
        q6k_decode_compact_default_on;
    q6k_decode_compact_state.store(if (on) 1 else 2, .release);
    return on;
}

/// Programmatic override for the Q6_K decode-route gate; same contract as
/// `setQ5kDecodeCompact` (null resets the read-once cache to unread).
pub fn setQ6kDecodeCompact(on: ?bool) void {
    const s: u8 = if (on) |o| (if (o) 1 else 2) else 0;
    q6k_decode_compact_state.store(s, .release);
}

pub fn linearSeqQ6_K(
    weight: *const WeightQ6_K,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    if (try denseQuantGpuTry(.q6_k, weight, ctx, input, in_tag, out_tag)) |r| return r;
    // Decode shapes: compact GGUF-native blocks instead of the byte-expanded
    // packed layout — see the Q5_K gate above; identical structure here. The
    // GPU try stays first so `-Dgpu` builds keep their existing offload
    // policy (weight-lifetime-aware wraps) ahead of the CPU route choice.
    if (input.dim(.seq) < 4 and !input.requiresGrad() and q6kDecodeCompactEnabled()) {
        var tagged = try weight.value.withTags(ctx, .{ out_tag, in_tag });
        defer tagged.deinit();
        return input.dot(ctx, &tagged, in_tag);
    }
    return input.dotPacked(ctx, &weight.packed_rhs, in_tag, out_tag);
}

fn restoreGpuResidencyAfterDeclinedFusion(ctx: *ExecContext, parts: []const *LinearWeight) !void {
    if (comptime !fucina.internal.gpu.enabled) return;
    for (parts) |part| switch (part.*) {
        .f32 => |*value| _ = try makeGpuResidentDenseWeight(.f32, WeightF32, ctx, value),
        .f16 => |*value| _ = try makeGpuResidentDenseWeight(.f16, WeightF16, ctx, value),
        .q4_k => |*weight| if (!weight.rhs_lifetime.isCacheable() and try makeGpuResidentQuantWeight(.q4_k, ctx, &weight.value)) {
            weight.rhs_lifetime = .stable_process;
        },
        .q5_k => |*weight| if (!weight.rhs_lifetime.isCacheable() and try makeGpuResidentQuantWeight(.q5_k, ctx, &weight.value)) {
            weight.rhs_lifetime = .stable_process;
        },
        .q6_k => |*weight| if (!weight.rhs_lifetime.isCacheable() and try makeGpuResidentQuantWeight(.q6_k, ctx, &weight.value)) {
            weight.rhs_lifetime = .stable_process;
        },
        .q8_0 => |*weight| if (!weight.rhs_lifetime.isCacheable() and try makeGpuResidentQuantWeight(.q8_0, ctx, &weight.value)) {
            weight.rhs_lifetime = .stable_process;
        },
        else => {},
    };
}

fn declinedFusion(ctx: *ExecContext, parts: []const *LinearWeight) !?LinearWeight {
    // `loadForFusion` deliberately skips per-part device copies. If fusion is
    // impossible (mixed GGUF quant types are common in *_K_M files), restore
    // the residency policy those independent linears would have received from
    // ordinary `load`; otherwise every prefill streams the same weights.
    try restoreGpuResidencyAfterDeclinedFusion(ctx, parts);
    return null;
}

/// Fuse same-format weights into one output-stacked matrix (one GEMM instead
/// of N on the forward path), consuming the parts on success. Returns null
/// with every part still valid when the formats differ or the format has no
/// fused fast path; capable GPU builds restore the skipped per-part residency
/// before returning.
pub fn fuseLinear(ctx: *ExecContext, parts: []const *LinearWeight) !?LinearWeight {
    if (parts.len < 2 or parts.len > 4) return Error.InvalidWeightShape;
    const fusable = [_]std.meta.Tag(LinearWeight){ .f32, .f16, .bf16, .q4_k, .q5_k, .q6_k, .q8_0 };
    inline for (fusable) |tag| {
        if (std.meta.activeTag(parts[0].*) == tag) {
            for (parts[1..]) |part| {
                if (std.meta.activeTag(part.*) != tag) return declinedFusion(ctx, parts);
            }
            const name = @tagName(tag);
            var others: [3]*const @FieldType(LinearWeight, name) = undefined;
            for (parts[1..], 0..) |part, i| others[i] = &@field(part.*, name);
            var fused = try @field(parts[0].*, name).concat(ctx, .out, others[0 .. parts.len - 1]);
            for (parts) |part| part.deinit();
            // Dense fused results re-acquire GPU residency like the quant
            // arms (the parts were loaded fusion-only, skipping residency).
            if (comptime tag == .f16) {
                _ = try makeGpuResidentDenseWeight(.f16, WeightF16, ctx, &fused);
            } else if (comptime tag == .f32) {
                _ = try makeGpuResidentDenseWeight(.f32, WeightF32, ctx, &fused);
            }
            return @unionInit(LinearWeight, name, fused);
        }
    }
    // PTQTP arms fuse per plane: the solver treats every 256-column group
    // independently, so a plane-wise row concat is byte-identical to
    // decorating the fused matrix (ptqtp_gguf.zig persists per-part planes
    // on the strength of the same property). Requires a uniform plane
    // count; mixed counts stay separate like any mixed-format parts.
    if (std.meta.activeTag(parts[0].*) == .ptqtp) {
        for (parts[1..]) |part| {
            if (std.meta.activeTag(part.*) != .ptqtp) return declinedFusion(ctx, parts);
        }
        const plane_count = parts[0].ptqtp.planeCount();
        for (parts[1..]) |part| {
            if (part.ptqtp.planeCount() != plane_count) return declinedFusion(ctx, parts);
        }

        var others: [3]*const QuantWeight(.tq2_0) = undefined;
        for (parts[1..], 0..) |part, i| others[i] = &part.ptqtp.p1;
        var p1 = try parts[0].ptqtp.p1.concat(ctx, .out, others[0 .. parts.len - 1]);
        errdefer p1.deinit();
        var p2: ?QuantWeight(.tq2_0) = null;
        errdefer if (p2) |*plane| plane.deinit();
        if (plane_count >= 2) {
            for (parts[1..], 0..) |part, i| others[i] = &part.ptqtp.p2.?;
            p2 = try parts[0].ptqtp.p2.?.concat(ctx, .out, others[0 .. parts.len - 1]);
        }
        var p3: ?QuantWeight(.tq2_0) = null;
        if (plane_count >= 3) {
            for (parts[1..], 0..) |part, i| others[i] = &part.ptqtp.p3.?;
            p3 = try parts[0].ptqtp.p3.?.concat(ctx, .out, others[0 .. parts.len - 1]);
        }
        // Folding survives fusion only when every part was tie-fitted.
        var all_tied = true;
        for (parts) |part| all_tied = all_tied and part.ptqtp.tied;
        for (parts) |part| part.deinit();
        return .{ .ptqtp = WeightPtqtp.init(ctx.allocator, p1, p2, p3, all_tied) };
    }
    return declinedFusion(ctx, parts);
}

/// Aggregate PTQTP decoration diagnostics over a model walk.
pub const PtqtpReport = struct {
    decorated: usize = 0,
    skipped: usize = 0,
    /// Whole layers excluded by skip-first/skip-last decoration options.
    skipped_layers: usize = 0,
    elements: u64 = 0,
    /// Σ elements x planes over decorated weights — the packed-size basis
    /// when per-projection plane counts are mixed.
    plane_weights: u64 = 0,
    err2_weighted: f64 = 0,
    worst_rel_err: f64 = 0,
    unconverged_groups: usize = 0,
    group_count: usize = 0,

    /// Element-weighted RMS of the per-tensor relative Frobenius errors.
    pub fn rmsRelErr(self: *const PtqtpReport) f64 {
        if (self.elements == 0) return 0;
        return @sqrt(self.err2_weighted / @as(f64, @floatFromInt(self.elements)));
    }
};

/// Decorate one weight if eligible, else count it as skipped. Family model
/// walks (e.g. qwen3's decoratePtqtp) drive this per projection.
pub fn decoratePtqtpInto(
    weight: *LinearWeight,
    ctx: *ExecContext,
    options: fucina.ptqtp.Options,
    report: *PtqtpReport,
) !void {
    if (!weight.ptqtpEligible()) {
        report.skipped += 1;
        return;
    }
    const elems: u64 = @intCast(weight.outDim() * weight.inDim());
    const stats = try weight.toPtqtp(ctx, options);
    report.decorated += 1;
    report.elements += elems;
    report.plane_weights += elems * options.planes;
    report.err2_weighted += stats.rel_frob_err * stats.rel_frob_err * @as(f64, @floatFromInt(elems));
    report.worst_rel_err = @max(report.worst_rel_err, stats.rel_frob_err);
    report.unconverged_groups += stats.unconverged_groups;
    report.group_count += stats.group_count;
}

fn requireMatrixShape(info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize) ![2]usize {
    const shape = try info.logicalMatrixShape();
    if (shape[0] != expected_rows or shape[1] != expected_cols) return Error.InvalidWeightShape;
    return shape;
}

fn loadDenseF32Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !WeightF32 {
    const len = try std.math.mul(usize, shape[0], shape[1]);
    const values = try ctx.allocator.alloc(f32, len);
    defer ctx.allocator.free(values);
    try fillF32(values, info);
    return WeightF32.fromSlice(ctx, shape, values);
}

fn loadDenseF16Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !WeightF16 {
    if (info.ggml_type != .f16) return Error.UnsupportedWeightType;
    const len = try std.math.mul(usize, shape[0], shape[1]);
    if (info.data.len != len * @sizeOf(u16)) return Error.InvalidWeightShape;

    // GPU builds: the weight lives in managed/shared resident bytes, so the
    // f16 GEMM offload uses it with zero per-call transfer (registry hit;
    // this path never adopt-copies) while the bytes stay CPU-readable and
    // in-place-trainable. Fallback: plain heap storage.
    if (comptime fucina.internal.gpu.enabled) {
        if (options.gpu_resident) {
            if (fucina.internal.gpu.allocResidentBytes(info.data.len)) |dev| {
                @memcpy(dev, info.data);
                return gpuResidentDenseTensor(.f16, WeightF16, ctx, shape, dev);
            }
        }
    }

    const values = try ctx.allocator.alloc(f16, len);
    defer ctx.allocator.free(values);
    for (values, 0..) |*dst, i| {
        const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
        dst.* = @bitCast(bits);
    }
    return WeightF16.fromSlice(ctx, shape, values);
}

/// bf16 stays RESIDENT (2 B/weight, like llama.cpp): the linearSeq fast path
/// streams the raw bits through the mixed f32 x bf16 TransB kernel
/// (`matmulTransB2DWithBf16Rhs`), which widens in-register (u16 << 16, exact)
/// and accumulates in f32. `Scalar(.bf16)` is the raw `u16` bit pattern.
fn loadDenseBf16Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !WeightBf16 {
    if (info.ggml_type != .bf16) return Error.UnsupportedWeightType;
    const len = try std.math.mul(usize, shape[0], shape[1]);
    if (info.data.len != len * @sizeOf(u16)) return Error.InvalidWeightShape;

    const values = try ctx.allocator.alloc(u16, len);
    defer ctx.allocator.free(values);
    for (values, 0..) |*dst, i| {
        dst.* = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
    }
    return WeightBf16.fromSlice(ctx, shape, values);
}

pub fn fillF32(out: []f32, info: *const gguf.TensorInfo) !void {
    switch (info.ggml_type) {
        .f32 => {
            if (info.data.len != out.len * @sizeOf(f32)) return Error.InvalidWeightShape;
            @memcpy(std.mem.sliceAsBytes(out), info.data);
        },
        .f16 => {
            if (info.data.len != out.len * @sizeOf(u16)) return Error.InvalidWeightShape;
            for (out, 0..) |*dst, i| {
                const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
                const half: f16 = @bitCast(bits);
                dst.* = @floatCast(half);
            }
        },
        .bf16 => {
            if (info.data.len != out.len * @sizeOf(u16)) return Error.InvalidWeightShape;
            for (out, 0..) |*dst, i| {
                const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
                dst.* = bf16ToF32(bits);
            }
        },
        .f64 => {
            if (info.data.len != out.len * @sizeOf(f64)) return Error.InvalidWeightShape;
            for (out, 0..) |*dst, i| {
                const bits = std.mem.readInt(u64, info.data[i * 8 ..][0..8], .little);
                const value: f64 = @bitCast(bits);
                dst.* = @floatCast(value);
            }
        },
        else => return Error.UnsupportedWeightType,
    }
}

fn loadQuantizedWeight(comptime dtype: DType, ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !QuantWeight(dtype) {
    const Elem = BlockStorage(dtype);
    const blocks = try blockSlice(Elem, info.data);
    return QuantWeight(dtype).fromBlocks(ctx, shape, blocks);
}

fn LoadedQuantWeight(comptime dtype: DType) type {
    return struct {
        value: QuantWeight(dtype),
        rhs_lifetime: RhsLifetime,
    };
}

/// Wrap freshly copied device-resident blocks (`internal.gpu.allocResidentBytes`)
/// in a tensor whose storage OWNS the device bytes: the final buffer release
/// (refs==0, counting cloneView'd weights that share it) frees them via
/// `freeResidentBytes`, which also evicts the shim's cached wrap for that base
/// address. Takes ownership of `dev` — freed here on error.
fn gpuResidentQuantTensor(comptime dtype: DType, ctx: *ExecContext, shape: [2]usize, dev: []u8) !QuantWeight(dtype) {
    const Raw = fucina.internal.tensor_mod.TensorOf(dtype);
    const DevBuffer = std.meta.Child(@FieldType(Raw, "buffer"));
    const hook = struct {
        fn releaseDeviceBytes(_: *anyopaque, buffer: *DevBuffer) void {
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            fucina.internal.gpu.freeResidentBytes(bytes);
        }
    };
    var dev_owned: ?[]u8 = dev;
    errdefer if (dev_owned) |bytes| fucina.internal.gpu.freeResidentBytes(bytes);
    const dev_blocks = try blockSliceMut(BlockStorage(dtype), dev);
    const buffer = try DevBuffer.fromBorrowedSliceWithRelease(ctx.allocator, dev_blocks, hook.releaseDeviceBytes);
    dev_owned = null; // from here the buffer's release hook frees the device bytes
    var raw = Raw.fromOwnedBuffer(buffer, &shape) catch |err| {
        buffer.release();
        return err;
    };
    errdefer raw.deinit();
    return QuantWeight(dtype).fromTensor(ctx, raw);
}

/// Dense scalar analog of `gpuResidentQuantTensor`: wrap device-resident
/// bytes as a dense [out, in] weight tensor whose storage OWNS them (same
/// release-hook contract). Managed residency keeps the bytes CPU-readable
/// AND CPU-writable at the same pointer, so in-place trainers (fucina.es)
/// can mutate resident weights and GPU dispatches read the live values —
/// the dense f32/f16 GEMM paths never adopt-copy, so there is no stale
/// snapshot to fence.
fn gpuResidentDenseTensor(comptime dtype: DType, comptime Facade: type, ctx: *ExecContext, shape: [2]usize, dev: []u8) !Facade {
    const Raw = fucina.internal.tensor_mod.TensorOf(dtype);
    const DevBuffer = std.meta.Child(@FieldType(Raw, "buffer"));
    const hook = struct {
        fn releaseDeviceBytes(_: *anyopaque, buffer: *DevBuffer) void {
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            fucina.internal.gpu.freeResidentBytes(bytes);
        }
    };
    var dev_owned: ?[]u8 = dev;
    errdefer if (dev_owned) |bytes| fucina.internal.gpu.freeResidentBytes(bytes);
    const Elem = std.meta.Child(@FieldType(DevBuffer, "data"));
    if (dev.len % @sizeOf(Elem) != 0) return Error.InvalidWeightShape;
    const elems: []Elem = @alignCast(std.mem.bytesAsSlice(Elem, dev));
    const buffer = try DevBuffer.fromBorrowedSliceWithRelease(ctx.allocator, elems, hook.releaseDeviceBytes);
    dev_owned = null; // from here the buffer's release hook frees the device bytes
    var raw = Raw.fromOwnedBuffer(buffer, &shape) catch |err| {
        buffer.release();
        return err;
    };
    errdefer raw.deinit();
    return Facade.fromTensor(ctx, raw);
}

fn loadGpuResidentQuantizedWeight(comptime dtype: DType, ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !LoadedQuantWeight(dtype) {
    const Elem = BlockStorage(dtype);
    const blocks = try blockSlice(Elem, info.data);
    if (comptime fucina.internal.gpu.enabled) {
        if (options.gpu_resident) {
            switch (comptime dtype) {
                .q4_k, .q6_k, .q8_0 => {
                    if (fucina.internal.gpu.allocResidentBytes(info.data.len)) |dev| {
                        @memcpy(dev, info.data);
                        return .{ .value = try gpuResidentQuantTensor(dtype, ctx, shape, dev), .rhs_lifetime = .stable_process };
                    }
                },
                .q5_k => if (comptime fucina.internal.gpu.has_q5_k_quant) {
                    if (fucina.internal.gpu.allocResidentBytes(info.data.len)) |dev| {
                        @memcpy(dev, info.data);
                        return .{ .value = try gpuResidentQuantTensor(dtype, ctx, shape, dev), .rhs_lifetime = .stable_process };
                    }
                },
                else => {},
            }
        }
    }
    return .{ .value = try QuantWeight(dtype).fromBlocks(ctx, shape, blocks), .rhs_lifetime = .transient };
}

/// Dense analog of `makeGpuResidentQuantWeight`: move a dense weight's
/// storage into GPU-resident bytes (fused concat results are heap tensors —
/// their fusion parts were loaded with `loadForFusion`, skipping per-part
/// residency on purpose). No-op (false) when the GPU is off or the budget
/// is exhausted.
fn makeGpuResidentDenseWeight(comptime dtype: DType, comptime Facade: type, ctx: *ExecContext, value: *Facade) !bool {
    if (comptime !fucina.internal.gpu.enabled) return false;
    const elems = try value.dataConst();
    const bytes = std.mem.sliceAsBytes(elems);
    const dev = fucina.internal.gpu.allocResidentBytes(bytes.len) orelse return false;
    @memcpy(dev, bytes);
    const raw_shape = value.asRawTensor().shape.slice();
    const shape = [2]usize{ raw_shape[0], raw_shape[1] };
    var resident = gpuResidentDenseTensor(dtype, Facade, ctx, shape, dev) catch |err| {
        return err;
    };
    errdefer resident.deinit();
    value.deinit();
    value.* = resident;
    return true;
}

fn makeGpuResidentQuantWeight(comptime dtype: DType, ctx: *ExecContext, value: *QuantWeight(dtype)) !bool {
    if (comptime !fucina.internal.gpu.enabled) return false;
    switch (comptime dtype) {
        .q4_k, .q6_k, .q8_0 => {},
        .q5_k => if (!fucina.internal.gpu.has_q5_k_quant) return false,
        else => return false,
    }
    const blocks = try value.dataConst();
    const bytes = std.mem.sliceAsBytes(blocks);
    const dev = fucina.internal.gpu.allocResidentBytes(bytes.len) orelse return false;
    @memcpy(dev, bytes);
    var resident = try gpuResidentQuantTensor(dtype, ctx, value.shape(), dev);
    errdefer resident.deinit();
    value.deinit();
    value.* = resident;
    return true;
}

fn loadQ6_KWeight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !WeightQ6_K {
    const loaded = try loadGpuResidentQuantizedWeight(.q6_k, ctx, info, shape, options);
    return WeightQ6_K.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

fn loadQ4_KWeight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !WeightQ4_K {
    const loaded = try loadGpuResidentQuantizedWeight(.q4_k, ctx, info, shape, options);
    return WeightQ4_K.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

fn loadQ5_KWeight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !WeightQ5_K {
    const loaded = try loadGpuResidentQuantizedWeight(.q5_k, ctx, info, shape, options);
    return WeightQ5_K.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

fn loadQ8_0Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !WeightQ8_0 {
    const loaded = try loadGpuResidentQuantizedWeight(.q8_0, ctx, info, shape, options);
    return WeightQ8_0.initWithRhsLifetime(ctx, loaded.value, loaded.rhs_lifetime);
}

fn blockSlice(comptime Elem: type, bytes: []const u8) ![]const Elem {
    if (bytes.len % @sizeOf(Elem) != 0) return Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(Elem) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(Elem)) const u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(Elem, aligned);
}

fn blockSliceMut(comptime Elem: type, bytes: []u8) ![]Elem {
    if (bytes.len % @sizeOf(Elem) != 0) return Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(Elem) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(Elem)) u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(Elem, aligned);
}

fn f16Slice(bytes: []const u8, expected_len: usize) ![]const f16 {
    if (bytes.len != expected_len * @sizeOf(u16)) return Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(f16) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(f16)) const u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(f16, aligned);
}

fn BlockStorage(comptime dtype: DType) type {
    return switch (dtype) {
        .q1_0 => fucina.BlockQ1_0,
        .q2_0 => fucina.BlockQ2_0,
        .q4_0 => fucina.BlockQ4_0,
        .q4_1 => fucina.BlockQ4_1,
        .q5_0 => fucina.BlockQ5_0,
        .q5_1 => fucina.BlockQ5_1,
        .q8_0 => fucina.BlockQ8_0,
        .q2_k => fucina.BlockQ2_K,
        .q3_k => fucina.BlockQ3_K,
        .q4_k => fucina.BlockQ4_K,
        .q5_k => fucina.BlockQ5_K,
        .q6_k => fucina.BlockQ6_K,
        .iq1_s => fucina.BlockIQ1_S,
        .iq1_m => fucina.BlockIQ1_M,
        .iq2_xxs => fucina.BlockIQ2_XXS,
        .iq2_xs => fucina.BlockIQ2_XS,
        .iq2_s => fucina.BlockIQ2_S,
        .iq3_xxs => fucina.BlockIQ3_XXS,
        .iq3_s => fucina.BlockIQ3_S,
        .iq4_nl => fucina.BlockIQ4_NL,
        .iq4_xs => fucina.BlockIQ4_XS,
        .tq1_0 => fucina.BlockTQ1_0,
        .tq2_0 => fucina.BlockTQ2_0,
        .mxfp4 => fucina.BlockMXFP4,
        .nvfp4 => fucina.BlockNVFP4,
        else => @compileError("unsupported quantized weight dtype"),
    };
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

// ---------------------------------------------------------------------------
// Tests — resident bf16 weights.
//
// Exactness note: every test below uses small-integer values. Those are
// exactly representable in bf16 (8-bit mantissa), bf16 -> f32 widening is
// always exact (u16 << 16), and the per-element products / partial sums are
// small integers that f32 represents exactly under ANY accumulation order —
// so the bf16 path must match the f32 reference BITWISE on every backend
// (native/scalar, BLAS or not), no tolerance needed.
//
// Most named tests live in the sibling `weights_tests.zig`; the test below
// stays here because it references the non-`pub` `bf16ToF32` helper.
// ---------------------------------------------------------------------------

test {
    _ = @import("weights_tests.zig");
}

/// f32 -> bf16 bit truncation; exact (== round-to-nearest) for the
/// small-integer test values used here.
fn testBf16Bits(value: f32) u16 {
    return @truncate(@as(u32, @bitCast(value)) >> 16);
}

/// Build the same logical weight twice: the resident-bf16 arm and the
/// f32-widened reference arm (what `load` produced before bf16 residency).
fn testBf16AndF32Pair(ctx: *ExecContext, values: []const f32, out_dim: usize, in_dim: usize) ![2]LinearWeight {
    var w32 = try WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values);
    defer w32.deinit();
    var w_bf16 = try w32.to(ctx, .bf16);
    errdefer w_bf16.deinit();
    const w_ref = try WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values);
    return .{ .{ .bf16 = w_bf16 }, .{ .f32 = w_ref } };
}

test "getRowsAs: bf16 embedding rows match the f32-widened table bitwise" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const vocab = 8;
    const hidden = 16;
    var t_vals: [vocab * hidden]f32 = undefined;
    // Arbitrary bf16-exact values, not just integers: gather + widen is exact
    // for ANY bf16 bit pattern, so snap to bf16 first and widen the snapped
    // values for the reference.
    fucina.rng.uniformFill(0xB16, &t_vals, -2, 2);
    for (&t_vals) |*v| v.* = bf16ToF32(testBf16Bits(v.*));

    var pair = try testBf16AndF32Pair(&ctx, &t_vals, vocab, hidden);
    defer pair[0].deinit();
    defer pair[1].deinit();

    const ids = [_]usize{ 3, 0, 7, 3 };
    var rows_bf16 = try pair[0].getRowsAs(&ctx, &ids, .embed);
    defer rows_bf16.deinit();
    var rows_ref = try pair[1].getRowsAs(&ctx, &ids, .embed);
    defer rows_ref.deinit();
    try std.testing.expectEqualSlices(f32, try rows_ref.dataConst(), try rows_bf16.dataConst());
}
