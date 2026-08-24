//! The `LinearWeight` union — one nominal arm per loadable GGUF weight
//! format — with its load / forward / decoration methods, and the
//! lookup-only `LookupWeight` table. The union dispatches into the
//! per-format bodies: packed dense arms in `dense.zig`, PTQTP arms in
//! `ptqtp.zig`.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const ag_mod = @import("../ag.zig");
const gguf = @import("../gguf.zig");
const ptqtp = @import("../ptqtp.zig");
const tuning = @import("../tuning.zig");

const common = @import("common.zig");
const dense = @import("dense.zig");
const ptqtp_w = @import("ptqtp.zig");

const gpu_impl = backend_mod.gpu_impl;
const Tensor = ag_mod.Tensor;
const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const Error = common.Error;
const QuantWeight = common.QuantWeight;
const backend_quant = common.backend_quant;
const blockSlice = common.blockSlice;
const WeightF32 = dense.WeightF32;
const WeightF16 = dense.WeightF16;
const WeightBf16 = dense.WeightBf16;
const WeightQ4_K = dense.WeightQ4_K;
const WeightQ5_K = dense.WeightQ5_K;
const WeightQ6_K = dense.WeightQ6_K;
const WeightQ8_0 = dense.WeightQ8_0;
const WeightPtqtp = ptqtp_w.WeightPtqtp;
const WeightPtqtpFx4 = ptqtp_w.WeightPtqtpFx4;
const Tag = @TypeOf(.tag);

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

    pub const LoadOptions = common.LoadOptions;

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
            .f32, .f64 => .{ .f32 = try dense.loadDenseF32Weight(ctx, info, shape) },
            .f16 => .{ .f16 = try dense.loadDenseF16Weight(ctx, info, shape, options) },
            .bf16 => .{ .bf16 = try dense.loadDenseBf16Weight(ctx, info, shape) },
            .q8_0 => .{ .q8_0 = try dense.loadQ8_0Weight(ctx, info, shape, options) },
            .q4_k => .{ .q4_k = try dense.loadQ4_KWeight(ctx, info, shape, options) },
            .q5_k => .{ .q5_k = try dense.loadQ5_KWeight(ctx, info, shape, options) },
            .q6_k => .{ .q6_k = try dense.loadQ6_KWeight(ctx, info, shape, options) },
            // Every remaining block format loads as a plain `QuantWeight`
            // arm named exactly like its GGML type.
            inline .q1_0, .q2_0, .q4_0, .q4_1, .q5_0, .q5_1, .q2_k, .q3_k, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_nl, .iq4_xs, .tq1_0, .tq2_0, .mxfp4, .nvfp4 => |t| @unionInit(
                LinearWeight,
                @tagName(t),
                try dense.loadQuantizedWeight(@field(DType, @tagName(t)), ctx, info, shape),
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
            inline .q4_k, .q5_k, .q6_k, .q8_0 => |*weight| try dense.linearSeq(weight, ctx, input, in_tag, out_tag),
            .ptqtp => |*weight| blk: {
                if (try ptqtp_w.linearSeqPtqtpFused(weight, ctx, input, out_tag)) |fused| break :blk fused;
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
            .tq2_0_fx4 => |*weight| try ptqtp_w.linearSeqFx4(weight, ctx, input, out_tag),
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

fn requireMatrixShape(info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize) ![2]usize {
    const shape = try info.logicalMatrixShape();
    if (shape[0] != expected_rows or shape[1] != expected_cols) return Error.InvalidWeightShape;
    return shape;
}
