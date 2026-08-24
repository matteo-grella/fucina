//! Model weight I/O and the fused linear/MoE weight containers: GGUF-backed
//! dense and quantized weights (packed-RHS `LinearWeight`, PTQTP plane
//! weights, `LookupWeight`), MoE expert-stack loading (resident, mmap-
//! borrowed, or disk-streamed through `ExpertStore`), the fused forward
//! helpers the LLM families share (`moeSwiGluFfnSeq`, `linearSeq*`), GGUF
//! host-side vector/matrix readers, and PTQTP decoration. Public as
//! `fucina.weights`.
//!
//! Home modules are imported directly: `fucina.zig` re-exports this file,
//! so the facade cannot be imported from here — the production import
//! graph is cycle-checked (`zig build arch-check`).
//!
//! WHAT BELONGS HERE, and what does not. Shared model code in this tree has
//! three homes, and the subject of the code decides which:
//!
//!   * here (`fucina.weights`, core band) — the subject is a WEIGHT
//!     CONTAINER: how to build one from GGUF bytes, and how to multiply by
//!     it. `linearSeq*` and `moe*FfnSeq` are forward compute but they are
//!     not stray: `LinearWeight.linearSeq` is a union dispatch INTO the
//!     per-format arms and the arms take the container types back, so the
//!     container and its multiply are one mutually-dependent unit.
//!   * `models/model_common.zig` (models band) — the subject is a GGUF FILE's
//!     layout: which tensor names a family's layer trio has, how an
//!     embed/head/norm set is read. Naming conventions, not numerics.
//!   * `models/host_ops.zig` (models band) — the subject is raw f32 HOST SLICES,
//!     for the host-reference ports that run below the Tensor facade.
//!
//! A helper that fits none of these wants a new home with a stated subject,
//! not a fourth un-ruled one.

const std = @import("std");
const builtin = @import("builtin");

const dtype_mod = @import("dtype.zig");
const backend_mod = @import("backend.zig");
const exec_mod = @import("exec.zig");
const ag_mod = @import("ag.zig");
const tensor_mod = @import("tensor.zig");
const gguf = @import("gguf.zig");
const ptqtp = @import("ptqtp.zig");
const parallel = @import("parallel.zig");
const tuning = @import("tuning.zig");

const weights_mod = @This();
const gpu_impl = backend_mod.gpu_impl;
const Tensor = ag_mod.Tensor;
const PackedRhs = ag_mod.PackedRhs;
const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const RhsLifetime = exec_mod.RhsLifetime;
const MoeRhs = exec_mod.ExecContext.MoeRhs;
const MoeBatchProfile = exec_mod.MoeBatchProfile;
const GatedOp = exec_mod.GatedOp;
const expert_store = @import("store/expert_store.zig");
const ExpertStore = expert_store.ExpertStore;
const Tag = @TypeOf(.tag);
const Allocator = std.mem.Allocator;

pub const Error = error{
    InvalidWeightShape,
    UnsupportedWeightType,
    GradUnsupported,
};

pub const WeightF32 = Tensor(.{ .out, .in });
pub const WeightF16 = Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } });
pub const WeightBf16 = Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
/// Packed quantized linear weight: the raw block tensor plus its
/// column-interleaved packed RHS, built once at init. One comptime body
/// serves every packed format — the four public names below are distinct
/// instantiations, so `LinearWeight`'s union arms stay nominal and every
/// change to the ownership/residency contract lands in all four at once.
fn PackedQuantWeight(comptime dt: DType) type {
    return struct {
        value: QuantWeight(dt),
        packed_rhs: PackedRhs(dt),
        rhs_lifetime: RhsLifetime = .transient,

        const Self = @This();

        /// The container's block format; `weights.linearSeq` infers its
        /// route from this.
        pub const dtype: DType = dt;

        pub fn init(ctx: *ExecContext, value: QuantWeight(dt)) !Self {
            return initWithRhsLifetime(ctx, value, .transient);
        }

        pub fn initWithRhsLifetime(ctx: *ExecContext, value: QuantWeight(dt), rhs_lifetime: RhsLifetime) !Self {
            var owned = value;
            errdefer owned.deinit();
            var packed_rhs = try owned.packRhs(ctx);
            errdefer packed_rhs.deinit();
            return .{ .value = owned, .packed_rhs = packed_rhs, .rhs_lifetime = rhs_lifetime };
        }

        pub fn deinit(self: *Self) void {
            self.packed_rhs.deinit();
            self.value.deinit();
            self.* = undefined;
        }

        pub fn cloneView(self: *const Self, ctx: *ExecContext) !Self {
            const value = try self.value.withTags(ctx, .{ .out, .in });
            return initWithRhsLifetime(ctx, value, self.rhs_lifetime);
        }

        pub fn concat(self: *const Self, ctx: *ExecContext, comptime tag: Tag, others: []const *const Self) !Self {
            var raw_others = try ctx.allocator.alloc(*const QuantWeight(dt), others.len);
            defer ctx.allocator.free(raw_others);
            for (others, 0..) |other, i| raw_others[i] = &other.value;

            var value = try self.value.concat(ctx, tag, raw_others);
            var owns_value = true;
            errdefer if (owns_value) value.deinit();
            const rhs_lifetime: RhsLifetime = if (try makeGpuResidentQuantWeight(dt, ctx, &value)) .stable_process else .transient;
            return initWithRhsLifetime(ctx, value, rhs_lifetime) catch |err| {
                owns_value = false;
                return err;
            };
        }
    };
}

pub const WeightQ4_K = PackedQuantWeight(.q4_k);
pub const WeightQ5_K = PackedQuantWeight(.q5_k);
pub const WeightQ6_K = PackedQuantWeight(.q6_k);
pub const WeightQ8_0 = PackedQuantWeight(.q8_0);

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
        if (comptime gpu_impl.enabled) {
            var it = self.map.iterator();
            while (it.next()) |e| {
                gpu_impl.freeResidentBytes(e.value_ptr.*);
            }
        }
        self.map.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn bytes(self: *ResidentByteRegistry, src: []const u8) []const u8 {
        if (comptime !gpu_impl.enabled) return src;
        const key = @intFromPtr(src.ptr);
        if (self.map.get(key)) |dev| return dev;
        self.map.ensureUnusedCapacity(self.allocator, 1) catch return src;
        const dev = gpu_impl.allocResidentBytes(src.len) orelse return src;
        @memcpy(dev, src);
        self.map.putAssumeCapacityNoClobber(key, dev);
        return dev;
    }
};

pub fn QuantWeight(comptime dtype: DType) type {
    return Tensor(.{ .dtype = dtype, .tags = .{ .out, .in } });
}

const backend_quant = backend_mod.quantized_matmul;

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
    if (comptime gpu_impl.enabled and gpu_impl.has_tq2_0_quant) {
        // Folded resident form: ONE dispatch, async return, no plane sum.
        if (comptime gpu_impl.has_tq2_0_folded_quant) {
            if (m >= 32 and weight.gpu_fold != null) {
                const nb01 = blocks_per_row * @sizeOf(backend_quant.BlockTQ2_0Folded);
                if (try ctx.foldedTernaryMatmulGpu(weight.gpu_fold.?, .stable_process, nb01, input.asRawTensor(), m, n, k)) |out_raw| {
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
                return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, t);
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
            rhs[plane_count] = backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(blocks)) catch return null;
            if (px4_ready) px4s[plane_count] = weight.px4[slot].?;
            plane_count += 1;
        }
    }

    const allocator = ctx.allocator;
    const lhs = try allocator.alloc(dtype_mod.BlockQ8_K, m * blocks_per_row);
    defer allocator.free(lhs);
    for (0..m) |r| {
        try backend_quant.q8k.quantizeRowQ8_KInto(lhs[r * blocks_per_row ..][0..blocks_per_row], x[r * k ..][0..k]);
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
                backend_quant.ternary.matmulTQ2_0FoldedX4RhsTile(task.out, task.lhs, task.pfold, task.bpr, task.n, 0, task.m, task.c0, task.c1);
                return;
            }
            if (task.px4.len != 0) {
                backend_quant.ternary.matmulTQ2_0X4RhsTile(task.out, task.lhs, task.px4[0], task.bpr, task.n, 0, task.m, task.c0, task.c1);
                for (task.px4[1..]) |pack| {
                    backend_quant.ternary.matmulTQ2_0X4RhsTileAcc(task.out, task.lhs, pack, task.bpr, task.n, 0, task.m, task.c0, task.c1);
                }
                return;
            }
            backend_quant.ternary.matmulTQ2_0RhsTile(task.out, task.lhs, &task.rhs[0], task.n, 0, task.m, task.c0, task.c1);
            for (task.rhs[1..]) |*plane_rhs| {
                backend_quant.ternary.matmulTQ2_0RhsTile(task.tmp, task.lhs, plane_rhs, task.n, 0, task.m, task.c0, task.c1);
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
fn linearSeqFx4(
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

    if (comptime gpu_impl.enabled and gpu_impl.has_tq2_0_quant) {
        if (comptime gpu_impl.has_tq2_0_folded_quant) {
            if (m >= 32 and weight.gpu_fold != null) {
                const nb01 = blocks_per_row * @sizeOf(backend_quant.BlockTQ2_0Folded);
                if (try ctx.foldedTernaryMatmulGpu(weight.gpu_fold.?, .stable_process, nb01, input.asRawTensor(), m, n, k)) |out_raw| {
                    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, out_raw);
                }
            }
        }
    }

    const allocator = ctx.allocator;

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
        try backend_quant.q8k.quantizeRowQ8_KInto(lhs[r * blocks_per_row ..][0..blocks_per_row], x[r * k ..][0..k]);
    }
    var out_t = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ m, n });
    errdefer out_t.deinit();
    const out = try out_t.data();

    const Task = struct {
        out: []f32,
        lhs: []const dtype_mod.BlockQ8_K,
        pack: []const backend_quant.BlockTQ2_0Foldedx4,
        bpr: usize,
        m: usize,
        n: usize,
        c0: usize,
        c1: usize,

        fn run(task: *const @This()) void {
            backend_quant.ternary.matmulTQ2_0FoldedX4RhsTile(task.out, task.lhs, task.pack, task.bpr, task.n, 0, task.m, task.c0, task.c1);
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
    px4: [3]?[]backend_quant.BlockTQ2_0x4 = .{ null, null, null },
    px4_allocator: ?Allocator = null,
    /// GPU-resident copies of the plane blocks (`gpu_impl`
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
        const gpu = gpu_impl;
        if (comptime !(gpu.enabled and gpu.has_quant_gemm and gpu.has_tq2_0_quant)) return;
        // Tied K=2 prefers the single folded resident buffer; falls through
        // to per-plane residency on any failure.
        if (comptime gpu_impl.has_tq2_0_folded_quant) {
            if (self.tied and self.p2 != null and self.p3 == null) fold: {
                const n = self.p1.dim(.out);
                const k = self.p1.dim(.in);
                const b1 = self.p1.asRawTensor().dataConstChecked() catch break :fold;
                const b2 = self.p2.?.asRawTensor().dataConstChecked() catch break :fold;
                const r1 = backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b1)) catch break :fold;
                const r2 = backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b2)) catch break :fold;
                const px4_alloc = self.px4_allocator orelse break :fold;
                const rows = backend_quant.ternary.packMatmulRhsTQ2_0FoldedRows(px4_alloc, &r1, &r2) catch break :fold;
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
        const gpu = gpu_impl;
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
                const rhs = backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(blocks)) catch break :blk false;
                self.px4[i] = backend_quant.ternary.packMatmulRhsTQ2_0x4(allocator, &rhs) catch break :blk false;
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
            const r1 = backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b1)) catch break :fold;
            const r2 = backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(k, n, @constCast(b2)) catch break :fold;
            self.pfold = backend_quant.ternary.packMatmulRhsTQ2_0Foldedx4(allocator, &r1, &r2) catch null;
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
    pack: []const backend_quant.BlockTQ2_0Foldedx4,
    /// Frees `pack`; null = borrowed storage kept alive by the owner.
    allocator: ?Allocator,
    /// Out / in dims — the pack slice carries no shape.
    n: usize,
    k: usize,
    /// Metal-resident row-major folded bytes (`BlockTQ2_0Folded`), built
    /// best-effort at init on capable builds.
    gpu_fold: ?[]u8 = null,

    pub fn init(allocator: Allocator, pack: []const backend_quant.BlockTQ2_0Foldedx4, pack_allocator: ?Allocator, n: usize, k: usize, gpu_resident: bool) WeightPtqtpFx4 {
        var self = WeightPtqtpFx4{ .pack = pack, .allocator = pack_allocator, .n = n, .k = k };
        if (gpu_resident) self.buildGpuResidency(allocator);
        return self;
    }

    pub fn buildGpuResidency(self: *WeightPtqtpFx4, allocator: Allocator) void {
        const gpu = gpu_impl;
        if (comptime !(gpu.enabled and gpu.has_quant_gemm and gpu.has_tq2_0_quant)) return;
        if (comptime !gpu_impl.has_tq2_0_folded_quant) return;
        if (self.gpu_fold != null) return;
        const rows = backend_quant.ternary.packMatmulRhsTQ2_0FoldedRowsFromX4(allocator, self.pack, self.n, self.k / 256) catch return;
        defer allocator.free(rows);
        const bytes = std.mem.sliceAsBytes(rows);
        const dev = gpu.allocResidentBytes(bytes.len) orelse return;
        @memcpy(dev, bytes);
        self.gpu_fold = dev;
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
        const gpu = gpu_impl;
        if (comptime gpu.enabled) {
            if (self.gpu_fold) |dev| gpu.freeResidentBytes(dev);
            self.gpu_fold = null;
        }
        if (self.allocator) |allocator| allocator.free(self.pack);
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
            if (comptime gpu_impl.enabled) gpu_impl.freeResidentBytes(self.data);
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
        .q5_k => gpu_impl.has_q5_k_quant,
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
        if (options.prefer_device and comptime gpu_impl.enabled and dtypeHasDenseQuantGpuKernel(dtype)) {
            if (gpu_impl.allocResidentBytes(total_len)) |dev| {
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
    tq2_0_fx4: WeightPtqtpFx4,

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
            .q8_0 => .{ .q8_0 = try loadQ8_0Weight(ctx, info, shape, options) },
            .q4_k => .{ .q4_k = try loadQ4_KWeight(ctx, info, shape, options) },
            .q5_k => .{ .q5_k = try loadQ5_KWeight(ctx, info, shape, options) },
            .q6_k => .{ .q6_k = try loadQ6_KWeight(ctx, info, shape, options) },
            // Every remaining block format loads as a plain `QuantWeight`
            // arm named exactly like its GGML type.
            inline .q1_0, .q2_0, .q4_0, .q4_1, .q5_0, .q5_1, .q2_k, .q3_k, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_nl, .iq4_xs, .tq1_0, .tq2_0, .mxfp4, .nvfp4 => |t| @unionInit(
                LinearWeight,
                @tagName(t),
                try loadQuantizedWeight(@field(DType, @tagName(t)), ctx, info, shape),
            ),
            .tq2_0_fx4 => blk: {
                const n = shape[0];
                const k = shape[1];
                if (n % 4 != 0 or k % 256 != 0) return Error.InvalidWeightShape;
                const src = try blockSlice(backend_quant.BlockTQ2_0Foldedx4, info.data);
                if (src.len != (n / 4) * (k / 256)) return Error.InvalidWeightShape;
                // Copy (like every dense quant arm): load() has no file
                // handle, so mmap lifetime cannot be promised here. Still
                // 4.0625 bpw once — no planes, no x4 packs, no fold pass.
                const owned = try ctx.allocator.alloc(backend_quant.BlockTQ2_0Foldedx4, src.len);
                errdefer ctx.allocator.free(owned);
                @memcpy(owned, src);
                break :blk .{ .tq2_0_fx4 = WeightPtqtpFx4.init(ctx.allocator, owned, ctx.allocator, n, k, options.gpu_resident) };
            },
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
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |*value, tag| @unionInit(LinearWeight, @tagName(tag), try value.cloneView(ctx)),
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
            // The fold IS the weight: clone copies the pack (unlike the
            // ptqtp clone above, which drops fold/tie and rebuilds free).
            .tq2_0_fx4 => |*value| blk: {
                const owned = try ctx.allocator.alloc(backend_quant.BlockTQ2_0Foldedx4, value.pack.len);
                @memcpy(owned, value.pack);
                break :blk .{ .tq2_0_fx4 = WeightPtqtpFx4.init(ctx.allocator, owned, ctx.allocator, value.n, value.k, true) };
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
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |*w| w.value.dim(.out),
            .ptqtp => |*w| w.p1.dim(.out),
            .tq2_0_fx4 => |*w| w.n,
            inline else => |*w| w.dim(.out),
        };
    }

    pub fn inDim(self: *const LinearWeight) usize {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |*w| w.value.dim(.in),
            .ptqtp => |*w| w.p1.dim(.in),
            .tq2_0_fx4 => |*w| w.k,
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
            // Both PTQTP forms are already the decoration — re-decorating
            // would double-quantize.
            .ptqtp, .tq2_0_fx4 => false,
            else => self.inDim() % ptqtp.block_len == 0,
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
    pub fn toPtqtp(self: *LinearWeight, ctx: *ExecContext, options: ptqtp.Options) !ptqtp.MatrixStats {
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

        var pair = try ptqtp.quantizeMatrix(ctx, values, rows, cols, options);
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

    pub fn linearSeq(self: *const LinearWeight, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |*weight| try weights_mod.linearSeq(weight, ctx, input, in_tag, out_tag),
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
            .tq2_0_fx4 => |*weight| try linearSeqFx4(weight, ctx, input, out_tag),
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
    /// FUCINA_NORM_QUANT_FUSED=0 (the A/B and emergency-revert switch —
    /// the fused route matches the unfused pair to f32 roundoff, not
    /// bitwise). Callers fanning ONE normalized input into several
    /// projections should require this for every projection before
    /// switching to the normed calls — the fallback re-normalizes per call.
    /// Norm-into-quantize fusion gate: FUCINA_NORM_QUANT_FUSED=0 forces
    /// the unfused rmsNormMul + linearSeq pair, =1 forces the fused route.
    /// Read once, cached (`tuning.Table.norm_quant_fused`).
    pub fn setNormQuantFused(on: ?bool) void {
        tuning.setField("norm_quant_fused", on);
    }

    pub fn supportsNormedFusion(self: *const LinearWeight, m: usize) bool {
        if (comptime gpu_impl.enabled) return false;
        if (!tuning.get().norm_quant_fused) return false;
        // Prefill-only: at decode shapes (m < 4) the fused route pays pooled
        // scratch acquisitions plus a padded 4-row-group quantize for one
        // real row, where the unfused internal quantizer has an m=1 stack
        // fast path — measured 2-3% decode LOSS on M1 Q4_K_M/Q8_0, against
        // a +11-23% pp32 win. (The x86 m<4 decode-compact routes bypass the
        // packed path anyway.)
        if (m < 4) return false;
        return switch (self.*) {
            .q4_k => comptime !backend_mod.supports_q4_k_mmla,
            .q8_0, .q5_k, .q6_k => true,
            else => false,
        };
    }

    /// `linearSeq` over `rmsNormMul(x, norm_weight, eps)` — on the packed CPU
    /// routes (see `supportsNormedFusion`) the normalized [m, k] tensor is
    /// never materialized: the fused kernel normalizes up to 4 rows into
    /// task-private scratch with the exact kernels the unfused dispatch uses
    /// and quantizes in place — results match the unfused pair to f32
    /// roundoff (<= 1 ulp observed). Every other arm (and
    /// FUCINA_NORM_QUANT_FUSED=0) normalizes and delegates.
    pub fn linearSeqNormed(
        self: *const LinearWeight,
        ctx: *ExecContext,
        x: anytype,
        norm_weight: anytype,
        eps: f32,
        comptime in_tag: Tag,
        comptime out_tag: Tag,
    ) !Tensor(.{ .seq, out_tag }) {
        @setEvalBranchQuota(20_000);
        if (comptime !gpu_impl.enabled) {
            if (!x.requiresGrad() and x.dim(.seq) >= 4 and tuning.get().norm_quant_fused) switch (self.*) {
                .q4_k => |*weight| if (comptime !backend_mod.supports_q4_k_mmla) {
                    return x.rmsNormMulDotPacked(ctx, norm_weight, eps, &weight.packed_rhs, in_tag, out_tag);
                },
                inline .q8_0, .q5_k, .q6_k => |*weight| return x.rmsNormMulDotPacked(ctx, norm_weight, eps, &weight.packed_rhs, in_tag, out_tag),
                else => {},
            };
        }
        var normed = try x.rmsNormMul(ctx, in_tag, norm_weight, eps);
        defer normed.deinit();
        return self.linearSeq(ctx, &normed, in_tag, out_tag);
    }

    pub fn getRowsAs(self: *const LinearWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        @setEvalBranchQuota(20_000);
        return switch (self.*) {
            .f32 => |*table| blk: {
                var rows = try table.gather(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            inline .f16, .bf16 => |*table| blk: {
                var rows_half = try table.gather(ctx, .out, token_ids, .seq);
                defer rows_half.deinit();
                var rows = try rows_half.to(ctx, .f32);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |*table| blk: {
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
            .tq2_0_fx4 => |*table| blk: {
                var out_t = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ token_ids.len, table.k });
                errdefer out_t.deinit();
                const dst = try out_t.data();
                for (token_ids, 0..) |row, i| table.decodeRow(row, dst[i * table.k ..][0..table.k]);
                break :blk out_t;
            },
            inline else => |*table| blk: {
                var rows = try table.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
        };
    }
};

/// Lookup-only table: a weight consumed exclusively through `getRowsAs`
/// (per-layer-embedding tables and the like — never a matmul operand).
/// The `mapped` arm serves rows straight out of the GGUF mapping: no heap
/// copy of the table, no matmul-RHS packing, rows decoded on demand with
/// the shared `gguf.decodeF32` kernels. The mapping must outlive the
/// weight — the owning model keeps it alive via `gguf.File.takeMapping`
/// whenever `borrowsMapping()` reports a borrow. `resident` is the copying
/// `LinearWeight` fallback for heap-read files, split mappings (their part
/// mappings die with the `File`), GPU builds, and dtypes without a row
/// decoder.
pub const LookupWeight = union(enum) {
    resident: LinearWeight,
    mapped: MappedTable,

    pub const MappedTable = struct {
        table: gguf.RowTable,
        rows: usize,
    };

    /// Borrow when the caller will be able to keep the mapping alive
    /// (single-file mmap — the same predicate as `gguf.File.takeMapping`)
    /// and the dtype decodes per row; copy otherwise. CPU builds only: GPU
    /// builds keep the resident load path and its device placement policy.
    pub fn load(ctx: *ExecContext, file: *const gguf.File, info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize) !LookupWeight {
        if (comptime !gpu_impl.enabled) {
            if (file.is_mmap and !file.isSplit()) {
                const shape = try requireMatrixShape(info, expected_rows, expected_cols);
                if (gguf.RowTable.init(ctx.allocator, info)) |table| {
                    return .{ .mapped = .{ .table = table, .rows = shape[0] } };
                } else |_| {} // no row decoder for this dtype: fall through to the copy
            }
        }
        return .{ .resident = try LinearWeight.load(ctx, info, expected_rows, expected_cols) };
    }

    /// True when this weight points into the GGUF mapping — the loader must
    /// then take ownership of the mapping for the model's lifetime.
    pub fn borrowsMapping(self: *const LookupWeight) bool {
        return self.* == .mapped;
    }

    pub fn deinit(self: *LookupWeight) void {
        switch (self.*) {
            .resident => |*w| w.deinit(),
            .mapped => {}, // borrows the mapping; nothing owned
        }
        self.* = undefined;
    }

    /// Same contract as `LinearWeight.getRowsAs` (ids must be < rows, as
    /// for the resident gather); the mapped arm decodes each requested row
    /// straight into the result tensor — the same shape the resident
    /// quantized gather has (`getRowsTensorInto` decodes into its output),
    /// so neither arm pays a scratch pass.
    pub fn getRowsAs(self: *const LookupWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        switch (self.*) {
            .resident => |*w| return w.getRowsAs(ctx, token_ids, out_tag),
            .mapped => |*m| {
                const width = m.table.width;
                var out = try Tensor(.{ .seq, out_tag }).empty(ctx, .{ token_ids.len, width });
                errdefer out.deinit();
                const out_data = try out.data();
                for (token_ids, 0..) |id, i| {
                    std.debug.assert(id < m.rows);
                    const dst = out_data[i * width ..][0..width];
                    const src = m.table.row(id, dst);
                    if (src.ptr != dst.ptr) @memcpy(dst, src);
                }
                return out;
            },
        }
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
) !MoeRhs {
    if (info.n_dims != 3) return Error.InvalidWeightShape;
    const in_dim = info.dims[0];
    const out_dim = info.dims[1];
    const n_expert = info.dims[2];
    if (in_dim != expected_in_dim or out_dim != expected_out_dim or n_expert != expected_n_expert) return Error.InvalidWeightShape;
    const rows = try std.math.mul(usize, n_expert, out_dim);

    return switch (info.ggml_type) {
        inline .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => |t| @unionInit(MoeRhs, @tagName(t), try copyOrBorrowMoeRhs(
            @FieldType(MoeRhs, @tagName(t)),
            dtype_mod.Storage(@field(DType, @tagName(t))),
            ctx,
            info,
            rows,
            in_dim,
            borrow,
        )),
        // q8_0: what llama.cpp falls back to when an expert dim is not a
        // 256 multiple (deepseek2). Nested rows container, so it gets its
        // own copy-or-borrow.
        .q8_0 => blk: {
            const src = try blockSlice(dtype_mod.BlockQ8_0, info.data);
            if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
            const bpc = src.len / rows;
            if (bpc * 32 != in_dim) return Error.InvalidWeightShape;
            if (borrow) {
                break :blk .{ .q8_0 = .{ .rows = .{ .allocator = null, .blocks = src, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(dtype_mod.BlockQ8_0, src.len);
            errdefer ctx.allocator.free(owned);
            @memcpy(owned, src);
            break :blk .{ .q8_0 = .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows } };
        },
        // IQ*/ternary experts: nested-rows containers, one shared
        // copy-or-borrow (copyOrBorrowMoeRhsRows).
        inline .iq2_xxs, .iq2_s, .iq4_xs, .iq3_xxs, .tq2_0 => |t| @unionInit(MoeRhs, @tagName(t), try copyOrBorrowMoeRhsRows(@field(DType, @tagName(t)), ctx, info, rows, in_dim, borrow)),
        // Native pre-folded tied-K=2 PTQTP (docs/PTQTP.md): the on-disk
        // bytes ARE the expert-major one-pass pack, so the resident arm is
        // the folded-only `ptqtp` form — no planes, no load-time fold, and
        // a borrow serves straight from the mapping's clean pages.
        .tq2_0_fx4 => blk: {
            if (out_dim % 4 != 0 or in_dim % 256 != 0) return Error.InvalidWeightShape;
            const src = try blockSlice(backend_quant.BlockTQ2_0Foldedx4, info.data);
            if (src.len != (rows / 4) * (in_dim / 256)) return Error.InvalidWeightShape;
            var folded: []const backend_quant.BlockTQ2_0Foldedx4 = src;
            var folded_allocator: ?Allocator = null;
            if (!borrow) {
                gguf.prefetch(info.data);
                const owned = try ctx.allocator.alloc(backend_quant.BlockTQ2_0Foldedx4, src.len);
                @memcpy(owned, src);
                folded = owned;
                folded_allocator = ctx.allocator;
            }
            break :blk .{ .ptqtp = .{
                .allocator = null,
                .planes = .{ &.{}, &.{}, &.{} },
                .plane_count = 0,
                .k = in_dim,
                .n = rows,
                .blocks_per_column = in_dim / 256,
                .folded = folded,
                .folded_allocator = folded_allocator,
            } };
        },
        else => Error.UnsupportedWeightType,
    };
}

/// PTQTP counterpart of `loadMoeRhs`: build the multi-plane `ptqtp` MoeRhs
/// arm from persisted `<name>.ptqtpK` sibling plane tensors (plane-major,
/// each with the base expert stack's `[in, out, n_expert]` shape and
/// standalone-valid TQ2_0 payload — `ptqtp_gguf.zig` owns the
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
) !MoeRhs {
    if (plane_infos.len == 0 or plane_infos.len > 3) return Error.InvalidWeightShape;
    const rows = try std.math.mul(usize, expected_n_expert, expected_out_dim);
    const bpc = try backend_mod.quantized_matmul.q8k.qkBlockCount(expected_in_dim);
    const blocks_per_plane = try std.math.mul(usize, rows, bpc);

    var planes: [3][]const dtype_mod.BlockTQ2_0 = .{ &.{}, &.{}, &.{} };
    var owned_count: usize = 0;
    errdefer for (planes[0..owned_count]) |plane| ctx.allocator.free(@constCast(plane));
    for (plane_infos, 0..) |info, p| {
        if (info.ggml_type != .tq2_0) return Error.UnsupportedWeightType;
        if (info.n_dims != 3) return Error.InvalidWeightShape;
        if (info.dims[0] != expected_in_dim or info.dims[1] != expected_out_dim or info.dims[2] != expected_n_expert) return Error.InvalidWeightShape;
        const src = try blockSlice(dtype_mod.BlockTQ2_0, info.data);
        if (src.len != blocks_per_plane) return Error.InvalidWeightShape;
        if (borrow) {
            planes[p] = src;
        } else {
            gguf.prefetch(info.data);
            const owned = try ctx.allocator.alloc(dtype_mod.BlockTQ2_0, src.len);
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
                views[p] = try backend_quant.ternary.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(expected_in_dim, expected_out_dim, @constCast(blocks));
            }
            try backend_quant.ternary.packMatmulRhsTQ2_0Foldedx4Into(buf[e * fg ..][0..fg], &views[0], &views[1]);
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

// The streamed-experts options struct lives in weights/moe_stream.zig; the
// shared `--moe-*` argv parser and the exit-time report belong to the
// runners (`models.moe_stream_cli`), the cache-aware routing policy to
// `models.moe_router`.
const moe_stream = @import("weights/moe_stream.zig");
pub const MoeStreamOptions = moe_stream.MoeStreamOptions;

/// The store-create block shared by the MoE loaders: expand split-GGUF part
/// paths (single files pass through as one entry) and open the ExpertStore
/// over them. The caller registers layers (`loadMoeRhsStreamed`) and then
/// calls `ExpertStore.finalize`.
pub fn createExpertStore(allocator: Allocator, options: MoeStreamOptions, n_layers: usize) !*ExpertStore {
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
    const store = try ExpertStore.create(allocator, store_paths, n_layers, .{
        .cache_bytes = options.cache_bytes,
        .cache_slots_per_layer = options.cache_slots_per_layer,
        .readahead = options.readahead,
        .auto_pin = options.auto_pin,
        .pin_bytes = options.pin_bytes,
        .io_workers = options.io_workers,
        .uncached = options.uncached,
        .trace_path = options.trace_path,
        .l2_path = options.l2_path,
        .l2_build_bytes = options.l2_build_bytes,
        .cache_route = if (options.cache_route) .{
            .sacred = options.route_sacred,
            .window = options.route_window,
        } else null,
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

/// Streamed counterpart of three `loadMoeRhs` calls: registers one layer's
/// gate/up/down stacked expert tensors with the ExpertStore (which will
/// `pread` individual experts on demand) instead of materializing or
/// borrowing them. Nothing of the expert stacks is read here — only the
/// geometry is validated, exactly as `loadMoeRhs` would.
pub const StreamedMoeFfnRhs = struct {
    gate: MoeRhs,
    up: MoeRhs,
    down: MoeRhs,
};

pub fn loadMoeRhsStreamed(
    store: *ExpertStore,
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
    store: *ExpertStore,
    layer_i: usize,
    specs: [3]expert_store.ProjSpec,
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
) !expert_store.ProjSpec {
    if (info.n_dims != 3) return Error.InvalidWeightShape;
    if (info.dims[0] != expected_in_dim or info.dims[1] != expected_out_dim or info.dims[2] != expected_n_expert) return Error.InvalidWeightShape;
    // The streamable subset is `StreamedQuant`'s member list: a block
    // dtype maps by name, and the folded pack is the one non-DType format.
    const quant: expert_store.StreamedQuant = if (info.ggml_type == .tq2_0_fx4)
        .tq2_0_fx4
    else
        expert_store.StreamedQuant.fromDType(gguf.dtypeForGgmlType(info.ggml_type) orelse return Error.UnsupportedWeightType) orelse return Error.UnsupportedWeightType;
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
) !expert_store.ProjSpec {
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
    input: *const Tensor(.{ .seq, .embed }),
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    selected: []const usize,
    routing_weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor(.{ .seq, .embed }) {
    return moeGatedFfnSeq(ctx, input, gate, up, down, selected, routing_weights, top_k, out_pe, .swiglu, io, profile);
}

/// As `moeSwiGluFfnSeq`, with the gated activation chosen by the caller
/// (deepseek4 routes through the clamped SwiGLU).
pub fn moeGatedFfnSeq(
    ctx: *ExecContext,
    input: *const Tensor(.{ .seq, .embed }),
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    selected: []const usize,
    routing_weights: []const f32,
    top_k: usize,
    out_pe: usize,
    act: GatedOp,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor(.{ .seq, .embed }) {
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
    return Tensor(.{ .seq, .embed }).fromTensor(ctx, raw);
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

/// `copyOrBorrowMoeRhs` for the nested-rows expert containers
/// (`QuantizedMatmulRhsRowsFor` — mutable blocks, so the borrow arm needs
/// the sound `@constCast`: every consumer only reads through the
/// container). The IQ*/TQ2_0 arms of `loadMoeRhs` are one-line users, so a
/// new streamed expert format cannot silently drop the shape validation or
/// the prefetch.
fn copyOrBorrowMoeRhsRows(
    comptime dtype: DType,
    ctx: *ExecContext,
    info: *const gguf.TensorInfo,
    rows: usize,
    in_dim: usize,
    borrow: bool,
) !backend_quant.QuantizedMatmulRhsRowsFor(dtype) {
    const Block = switch (dtype) {
        .iq2_xxs => dtype_mod.BlockIQ2_XXS,
        .iq2_s => dtype_mod.BlockIQ2_S,
        .iq4_xs => dtype_mod.BlockIQ4_XS,
        .iq3_xxs => dtype_mod.BlockIQ3_XXS,
        .tq2_0 => dtype_mod.BlockTQ2_0,
        else => @compileError("copyOrBorrowMoeRhsRows: no nested-rows expert container for this dtype"),
    };
    const src = try blockSlice(Block, info.data);
    if (rows == 0 or src.len % rows != 0) return Error.InvalidWeightShape;
    const bpc = src.len / rows;
    if (try backend_quant.q8k.qkBlockCount(in_dim) != bpc) return Error.InvalidWeightShape;
    if (borrow) {
        return .{ .rows = .{ .allocator = null, .blocks = @constCast(src), .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows };
    }
    gguf.prefetch(info.data);
    const owned = try ctx.allocator.alloc(Block, src.len);
    errdefer ctx.allocator.free(owned);
    @memcpy(owned, src);
    return .{ .rows = .{ .allocator = ctx.allocator, .blocks = owned, .rows = rows, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = rows };
}

pub fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_i, suffix });
}

/// 1-D tensor as an owned host f32 slice (any decodable source dtype).
pub fn hostVector(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, expected: usize) ![]f32 {
    return hostVectorInfo(allocator, try file.get(tensor_name), expected);
}

pub fn hostVectorInfo(allocator: Allocator, info: *const gguf.TensorInfo, expected: usize) ![]f32 {
    if (info.n_dims != 1 or info.dims[0] != expected) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, expected);
    errdefer allocator.free(out);
    try fillF32(out, info);
    return out;
}

/// 2-D tensor (GGUF dims `{cols, rows}`) as an owned host row-major
/// `[rows][cols]` f32 slice.
pub fn hostMatrix(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, cols: usize, rows: usize) ![]f32 {
    const info = try file.get(tensor_name);
    if (info.n_dims != 2 or info.dims[0] != cols or info.dims[1] != rows) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);
    try fillF32(out, info);
    return out;
}

pub fn loadVector(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_len: usize, comptime tag: Tag) !Tensor(.{tag}) {
    if (info.n_dims != 1 or info.dims[0] != expected_len) return Error.InvalidWeightShape;

    var v = try Tensor(.{tag}).empty(ctx, .{expected_len});
    errdefer v.deinit();
    try fillF32(try v.data(), info);
    return v;
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
    const qm = backend_quant;
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
        packs[g] = try qm.q8_0.packMatmulRhsQ8_0x4(allocator, @alignCast(all[g * rank * bpr ..][0 .. rank * bpr]), rank, group_dim, bpr);
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
    const qm = backend_quant;
    const n_groups = rhs_packs.len;
    if (group_dim % 32 != 0 or n_groups == 0 or n_groups > 8) return Error.InvalidWeightShape;
    const bpr = group_dim / 32;
    if (x.len != n_groups * group_dim or out.len != n_groups * rank) return Error.InvalidWeightShape;

    const allocator = ctx.allocator;
    const lhs = try allocator.alloc(dtype_mod.BlockQ8_0, n_groups * bpr);
    defer allocator.free(lhs);
    for (0..n_groups) |g| {
        try qm.q8k.quantizeRowQ8_0Into(lhs[g * bpr ..][0..bpr], x[g * group_dim ..][0..group_dim]);
    }

    const Task = struct {
        out: []f32,
        lhs: []const dtype_mod.BlockQ8_0,
        rhs: *const backend_quant.QuantizedMatmulRhsQ8_0x4,
        n: usize,

        fn run(task: *const @This()) void {
            qm.q8_0.matmulQ8_0x4RhsTile(task.out, task.lhs, task.rhs, task.n, 0, 1, 0, task.n);
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
    if (input.requiresGrad()) return Error.GradUnsupported;
    if (input.dim(in_tag) != shape[1]) return Error.InvalidWeightShape;

    const blocks = try blockSlice(BlockStorage(dtype), bytes);
    var value = try ctx.matmul2DWithQuantizedBlocksRhs(dtype, input.asRawTensor(), blocks, shape[0], shape[1], .{
        .allow_gpu = options.allow_gpu,
        .rhs_lifetime = options.rhs_lifetime,
    });
    errdefer value.deinit();
    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, value);
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
) !?Tensor(.{ .seq, out_tag }) {
    if (comptime !gpu_impl.enabled) return null;
    if (comptime dtype == .q5_k and !gpu_impl.has_q5_k_quant) return null;
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
    return try Tensor(.{ .seq, out_tag }).fromTensor(ctx, out);
}

/// Programmatic override for the decode-route gate
/// (`tuning.Table.decode_compact`, where the bandwidth argument lives):
/// `true`/`false` force the compact/packed route, `null` restores the
/// env/default value. The tests' A/B hook; also usable from a CLI flag.
pub fn setDecodeCompact(on: ?bool) void {
    tuning.setField("decode_compact", on);
}

/// Formats whose decode-shape GEMV wins by reading the compact GGUF-native
/// blocks instead of the byte-expanded packed layout (the
/// `tuning.Table.decode_compact` byte ratios). Q8_0's x4 pack carries the
/// same bytes as its compact blocks, so it goes straight to the packed dot.
fn hasCompactDecodeRoute(comptime dt: DType) bool {
    return switch (dt) {
        .q4_k, .q5_k, .q6_k => true,
        .q8_0 => false,
        else => false,
    };
}

/// Packed quantized linear forward, one body for every
/// `PackedQuantWeight` format: the GPU offload try first, then (where the
/// format has one) the decode-shape compact route, then the packed CPU
/// dot. Grad inputs keep the packed path's explicit
/// GradientQuantizedMatmulUnsupported error.
pub fn linearSeq(
    weight: anytype,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !Tensor(.{ .seq, out_tag }) {
    const dt = @TypeOf(weight.*).dtype;
    if (try denseQuantGpuTry(dt, weight, ctx, input, in_tag, out_tag)) |r| return r;
    // Decode shapes: contract against the resident GGUF-native compact
    // blocks (`weight.value`) through the public quantized-RHS `dot` —
    // bitwise-equal outputs, fewer weight bytes streamed (see the
    // `tuning.Table.decode_compact` doc). The GPU try stays first so `-Dgpu`
    // builds keep their offload policy ahead of the CPU route choice.
    if (comptime hasCompactDecodeRoute(dt)) {
        if (input.dim(.seq) < 4 and !input.requiresGrad() and tuning.get().decode_compact) {
            var tagged = try weight.value.withTags(ctx, .{ out_tag, in_tag });
            defer tagged.deinit();
            return input.dot(ctx, &tagged, in_tag);
        }
    }
    return input.dotPacked(ctx, &weight.packed_rhs, in_tag, out_tag);
}

fn restoreGpuResidencyAfterDeclinedFusion(ctx: *ExecContext, parts: []const *LinearWeight) !void {
    if (comptime !gpu_impl.enabled) return;
    for (parts) |part| switch (part.*) {
        .f32 => |*value| _ = try makeGpuResidentDenseWeight(.f32, WeightF32, ctx, value),
        .f16 => |*value| _ = try makeGpuResidentDenseWeight(.f16, WeightF16, ctx, value),
        inline .q4_k, .q5_k, .q6_k, .q8_0 => |*weight| if (!weight.rhs_lifetime.isCacheable() and try makeGpuResidentQuantWeight(@TypeOf(weight.*).dtype, ctx, &weight.value)) {
            weight.rhs_lifetime = .stable_process;
        },
        // loadForFusion skipped the fold residency; a declined fx4 part
        // serves standalone and wants it back.
        .tq2_0_fx4 => |*weight| weight.buildGpuResidency(ctx.allocator),
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
    // Native-folded parts concat by plain pack append: the layout is
    // column-group-major and every part's out dim is a multiple of 4 (the
    // format rejects anything else), so each part's groups land whole on
    // group boundaries — byte-identical to folding the fused matrix (fold
    // is per-column with no cross-column state).
    if (std.meta.activeTag(parts[0].*) == .tq2_0_fx4) {
        for (parts[1..]) |part| {
            if (std.meta.activeTag(part.*) != .tq2_0_fx4) return declinedFusion(ctx, parts);
        }
        const k = parts[0].tq2_0_fx4.k;
        var total_blocks: usize = 0;
        var total_n: usize = 0;
        for (parts) |part| {
            if (part.tq2_0_fx4.k != k) return declinedFusion(ctx, parts);
            total_blocks += part.tq2_0_fx4.pack.len;
            total_n += part.tq2_0_fx4.n;
        }
        const fused = try ctx.allocator.alloc(backend_quant.BlockTQ2_0Foldedx4, total_blocks);
        errdefer ctx.allocator.free(fused);
        var off: usize = 0;
        for (parts) |part| {
            @memcpy(fused[off..][0..part.tq2_0_fx4.pack.len], part.tq2_0_fx4.pack);
            off += part.tq2_0_fx4.pack.len;
        }
        for (parts) |part| part.deinit();
        return .{ .tq2_0_fx4 = WeightPtqtpFx4.init(ctx.allocator, fused, ctx.allocator, total_n, k, true) };
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
    options: ptqtp.Options,
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
    var w = try WeightF32.empty(ctx, shape);
    errdefer w.deinit();
    try fillF32(try w.data(), info); // validates info.data.len against the shape
    return w;
}

fn loadDenseF16Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize, options: LinearWeight.LoadOptions) !WeightF16 {
    if (info.ggml_type != .f16) return Error.UnsupportedWeightType;
    const len = try std.math.mul(usize, shape[0], shape[1]);
    if (info.data.len != len * @sizeOf(u16)) return Error.InvalidWeightShape;

    // GPU builds: the weight lives in managed/shared resident bytes, so the
    // f16 GEMM offload uses it with zero per-call transfer (registry hit;
    // this path never adopt-copies) while the bytes stay CPU-readable and
    // in-place-trainable. Fallback: plain heap storage.
    if (comptime gpu_impl.enabled) {
        if (options.gpu_resident) {
            if (gpu_impl.allocResidentBytes(info.data.len)) |dev| {
                @memcpy(dev, info.data);
                return gpuResidentDenseTensor(.f16, WeightF16, ctx, shape, dev);
            }
        }
    }

    var w = try WeightF16.empty(ctx, shape);
    errdefer w.deinit();
    for (try w.data(), 0..) |*dst, i| {
        const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
        dst.* = @bitCast(bits);
    }
    return w;
}

/// bf16 stays RESIDENT (2 B/weight, like llama.cpp): the linearSeq fast path
/// streams the raw bits through the mixed f32 x bf16 TransB kernel
/// (`matmulTransB2DWithBf16Rhs`), which widens in-register (u16 << 16, exact)
/// and accumulates in f32. `Scalar(.bf16)` is the raw `u16` bit pattern.
fn loadDenseBf16Weight(ctx: *ExecContext, info: *const gguf.TensorInfo, shape: [2]usize) !WeightBf16 {
    if (info.ggml_type != .bf16) return Error.UnsupportedWeightType;
    const len = try std.math.mul(usize, shape[0], shape[1]);
    if (info.data.len != len * @sizeOf(u16)) return Error.InvalidWeightShape;

    var w = try WeightBf16.empty(ctx, shape);
    errdefer w.deinit();
    for (try w.data(), 0..) |*dst, i| {
        dst.* = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
    }
    return w;
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
                dst.* = dtype_mod.bf16ToF32(bits);
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
    const Raw = tensor_mod.TensorOf(dtype);
    const DevBuffer = std.meta.Child(@FieldType(Raw, "buffer"));
    const hook = struct {
        fn releaseDeviceBytes(_: *anyopaque, buffer: *DevBuffer) void {
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            gpu_impl.freeResidentBytes(bytes);
        }
    };
    var dev_owned: ?[]u8 = dev;
    errdefer if (dev_owned) |bytes| gpu_impl.freeResidentBytes(bytes);
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
    const Raw = tensor_mod.TensorOf(dtype);
    const DevBuffer = std.meta.Child(@FieldType(Raw, "buffer"));
    const hook = struct {
        fn releaseDeviceBytes(_: *anyopaque, buffer: *DevBuffer) void {
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            gpu_impl.freeResidentBytes(bytes);
        }
    };
    var dev_owned: ?[]u8 = dev;
    errdefer if (dev_owned) |bytes| gpu_impl.freeResidentBytes(bytes);
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
    if (comptime gpu_impl.enabled) {
        if (options.gpu_resident) {
            switch (comptime dtype) {
                .q4_k, .q6_k, .q8_0 => {
                    if (gpu_impl.allocResidentBytes(info.data.len)) |dev| {
                        @memcpy(dev, info.data);
                        return .{ .value = try gpuResidentQuantTensor(dtype, ctx, shape, dev), .rhs_lifetime = .stable_process };
                    }
                },
                .q5_k => if (comptime gpu_impl.has_q5_k_quant) {
                    if (gpu_impl.allocResidentBytes(info.data.len)) |dev| {
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
    if (comptime !gpu_impl.enabled) return false;
    const elems = try value.dataConst();
    const bytes = std.mem.sliceAsBytes(elems);
    const dev = gpu_impl.allocResidentBytes(bytes.len) orelse return false;
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
    if (comptime !gpu_impl.enabled) return false;
    switch (comptime dtype) {
        .q4_k, .q6_k, .q8_0 => {},
        .q5_k => if (!gpu_impl.has_q5_k_quant) return false,
        else => return false,
    }
    const blocks = try value.dataConst();
    const bytes = std.mem.sliceAsBytes(blocks);
    const dev = gpu_impl.allocResidentBytes(bytes.len) orelse return false;
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
        .q1_0 => dtype_mod.BlockQ1_0,
        .q2_0 => dtype_mod.BlockQ2_0,
        .q4_0 => dtype_mod.BlockQ4_0,
        .q4_1 => dtype_mod.BlockQ4_1,
        .q5_0 => dtype_mod.BlockQ5_0,
        .q5_1 => dtype_mod.BlockQ5_1,
        .q8_0 => dtype_mod.BlockQ8_0,
        .q2_k => dtype_mod.BlockQ2_K,
        .q3_k => dtype_mod.BlockQ3_K,
        .q4_k => dtype_mod.BlockQ4_K,
        .q5_k => dtype_mod.BlockQ5_K,
        .q6_k => dtype_mod.BlockQ6_K,
        .iq1_s => dtype_mod.BlockIQ1_S,
        .iq1_m => dtype_mod.BlockIQ1_M,
        .iq2_xxs => dtype_mod.BlockIQ2_XXS,
        .iq2_xs => dtype_mod.BlockIQ2_XS,
        .iq2_s => dtype_mod.BlockIQ2_S,
        .iq3_xxs => dtype_mod.BlockIQ3_XXS,
        .iq3_s => dtype_mod.BlockIQ3_S,
        .iq4_nl => dtype_mod.BlockIQ4_NL,
        .iq4_xs => dtype_mod.BlockIQ4_XS,
        .tq1_0 => dtype_mod.BlockTQ1_0,
        .tq2_0 => dtype_mod.BlockTQ2_0,
        .mxfp4 => dtype_mod.BlockMXFP4,
        .nvfp4 => dtype_mod.BlockNVFP4,
        else => @compileError("unsupported quantized weight dtype"),
    };
}

test {
    _ = @import("weights_tests.zig");
}
