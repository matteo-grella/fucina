//! The `LinearWeight` shape -- one container arm per BEHAVIOUR, not per
//! GGUF format: `dense` (f32/f16/bf16 facade dots), `quant` (every other
//! block format behind one dtype-erased compact container), `packed_quant`
//! (the four packed-RHS formats), `ptqtp` (trit planes), `tq2_0_fx4`
//! (native fold). The ~30 GGML formats exist as distinct comptime types
//! ONLY inside `loadWithOptions` -- the one runtime-dtype -> comptime
//! switch -- which builds the `quant` container's per-dtype vtable; every
//! behaviour method dispatches over the five containers once. Also home to
//! the lookup-only `LookupWeight` table.

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

const offload = backend_mod.offload;
const Tensor = ag_mod.Tensor;
const PackedRhs = ag_mod.PackedRhs;
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
const Allocator = std.mem.Allocator;

/// The dense float container: one nominal arm per storage width, all three
/// serving through the differentiable facade `dot` (f16/bf16 stream their
/// raw bits through the mixed-precision TransB kernels).
pub const DenseWeight = union(enum) {
    f32: WeightF32,
    f16: WeightF16,
    bf16: WeightBf16,

    pub fn dtype(self: *const DenseWeight) DType {
        return switch (self.*) {
            inline else => |*w| @TypeOf(w.*).dtype,
        };
    }

    pub fn deinit(self: *DenseWeight) void {
        switch (self.*) {
            inline else => |*w| w.deinit(),
        }
        self.* = undefined;
    }

    pub fn cloneView(self: *const DenseWeight, ctx: *ExecContext) !DenseWeight {
        return switch (self.*) {
            inline else => |*w, tag| @unionInit(DenseWeight, @tagName(tag), try w.withTags(ctx, .{ .out, .in })),
        };
    }

    pub fn outDim(self: *const DenseWeight) usize {
        return switch (self.*) {
            inline else => |*w| w.dim(.out),
        };
    }

    pub fn inDim(self: *const DenseWeight) usize {
        return switch (self.*) {
            inline else => |*w| w.dim(.in),
        };
    }

    pub fn linearSeq(self: *const DenseWeight, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        return switch (self.*) {
            inline else => |*w| blk: {
                var tagged = try w.withTags(ctx, .{ out_tag, in_tag });
                defer tagged.deinit();
                break :blk try input.dot(ctx, &tagged, in_tag);
            },
        };
    }

    pub fn getRowsAs(self: *const DenseWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
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
        };
    }

    /// Snapshot as f32 output-row panels for `dotPacked` (all three arms
    /// share the dense panel container -- f16/bf16 widen once here).
    pub fn packRhs(self: *const DenseWeight, ctx: *ExecContext) !PackedRhs(.f32) {
        return switch (self.*) {
            inline else => |*w| try w.packRhs(ctx),
        };
    }
};

/// The packed-RHS container: the four formats with column-interleaved
/// packed kernels (`dense.PackedQuantWeight` bodies), each arm carrying
/// the raw block tensor plus its packed RHS and lifetime.
pub const PackedWeight = union(enum) {
    q4_k: WeightQ4_K,
    q5_k: WeightQ5_K,
    q6_k: WeightQ6_K,
    q8_0: WeightQ8_0,

    pub fn dtype(self: *const PackedWeight) DType {
        return switch (self.*) {
            inline else => |*w| @TypeOf(w.*).dtype,
        };
    }

    pub fn deinit(self: *PackedWeight) void {
        switch (self.*) {
            inline else => |*w| w.deinit(),
        }
        self.* = undefined;
    }

    pub fn cloneView(self: *const PackedWeight, ctx: *ExecContext) !PackedWeight {
        return switch (self.*) {
            inline else => |*w, tag| @unionInit(PackedWeight, @tagName(tag), try w.cloneView(ctx)),
        };
    }

    pub fn outDim(self: *const PackedWeight) usize {
        return switch (self.*) {
            inline else => |*w| w.value.dim(.out),
        };
    }

    pub fn inDim(self: *const PackedWeight) usize {
        return switch (self.*) {
            inline else => |*w| w.value.dim(.in),
        };
    }

    pub fn linearSeq(self: *const PackedWeight, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        return switch (self.*) {
            inline else => |*w| try dense.linearSeq(w, ctx, input, in_tag, out_tag),
        };
    }

    pub fn getRowsAs(self: *const PackedWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        return switch (self.*) {
            inline else => |*w| blk: {
                var rows = try w.value.getRows(ctx, .out, token_ids, .seq);
                defer rows.deinit();
                break :blk try rows.withTags(ctx, .{ .seq, out_tag });
            },
        };
    }

    /// Whether this arm has a fused normalize+quantize+packed-GEMM kernel
    /// (MMLA q4_k has none). Shape/tuning gates live on the caller
    /// (`LinearWeight.supportsNormedFusion`).
    pub fn supportsNormedFusion(self: *const PackedWeight) bool {
        return switch (self.*) {
            .q4_k => comptime !backend_mod.quant.supports_q4_k_mmla,
            .q8_0, .q5_k, .q6_k => true,
        };
    }

    /// The fused normed route; null when this arm has no fused kernel
    /// (q4_k on MMLA targets -- the caller re-normalizes and delegates).
    pub fn tryLinearSeqNormed(
        self: *const PackedWeight,
        ctx: *ExecContext,
        x: anytype,
        norm_weight: anytype,
        eps: f32,
        comptime in_tag: Tag,
        comptime out_tag: Tag,
    ) !?Tensor(.{ .seq, out_tag }) {
        switch (self.*) {
            .q4_k => |*w| {
                if (comptime backend_mod.quant.supports_q4_k_mmla) return null;
                return try x.rmsNormMulDotPacked(ctx, norm_weight, eps, &w.packed_rhs, in_tag, out_tag);
            },
            inline .q8_0, .q5_k, .q6_k => |*w| return try x.rmsNormMulDotPacked(ctx, norm_weight, eps, &w.packed_rhs, in_tag, out_tag),
        }
    }
};

/// Dtype-erased compact container for every block format outside the
/// packed set (the cold formats: legacy q4_0/q5_1..., i-quants, tq1_0,
/// mxfp4, ...). The typed `QuantWeight(dt)` facade tensor lives in a heap
/// box behind a per-dtype vtable built at load -- the behaviour methods
/// are dtype-free, and results are the same facade calls the typed tensor
/// would make (canonical `[.seq, .in] x [.out, .in]` tags, retagged at the
/// boundary through grad-transparent views).
pub const ColdQuantWeight = struct {
    /// The loaded GGML block format (runtime).
    dtype: DType,
    out_dim: usize,
    in_dim: usize,
    /// Heap box holding the typed `QuantWeight(dtype)`.
    box: *anyopaque,
    allocator: Allocator,
    vt: *const VTable,

    pub const VTable = struct {
        deinit: *const fn (*anyopaque, Allocator) void,
        clone_view: *const fn (*anyopaque, *ExecContext) anyerror!*anyopaque,
        linear: *const fn (*anyopaque, *ExecContext, *const Tensor(.{ .seq, .in })) anyerror!Tensor(.{ .seq, .out }),
        get_rows: *const fn (*anyopaque, *ExecContext, []const usize) anyerror!Tensor(.{ .seq, .in }),
    };

    fn Impl(comptime dt: DType) type {
        return struct {
            const W = QuantWeight(dt);

            fn cast(box: *anyopaque) *W {
                return @ptrCast(@alignCast(box));
            }

            fn deinitFn(box: *anyopaque, allocator: Allocator) void {
                const w = cast(box);
                w.deinit();
                allocator.destroy(w);
            }

            fn cloneViewFn(box: *anyopaque, ctx: *ExecContext) anyerror!*anyopaque {
                var view = try cast(box).withTags(ctx, .{ .out, .in });
                errdefer view.deinit();
                const fresh = try ctx.allocator().create(W);
                fresh.* = view;
                return fresh;
            }

            fn linearFn(box: *anyopaque, ctx: *ExecContext, x: *const Tensor(.{ .seq, .in })) anyerror!Tensor(.{ .seq, .out }) {
                return x.dot(ctx, cast(box), .in);
            }

            fn getRowsFn(box: *anyopaque, ctx: *ExecContext, ids: []const usize) anyerror!Tensor(.{ .seq, .in }) {
                return cast(box).getRows(ctx, .out, ids, .seq);
            }

            const vtable = VTable{
                .deinit = &deinitFn,
                .clone_view = &cloneViewFn,
                .linear = &linearFn,
                .get_rows = &getRowsFn,
            };
        };
    }

    /// Box a typed compact tensor. The ONLY caller is `loadWithOptions`'s
    /// format switch -- each `dt` instantiation exists exactly there.
    pub fn of(comptime dt: DType, ctx: *ExecContext, value: QuantWeight(dt)) !ColdQuantWeight {
        var owned = value;
        errdefer owned.deinit();
        const out_dim = owned.dim(.out);
        const in_dim = owned.dim(.in);
        const box = try ctx.allocator().create(QuantWeight(dt));
        box.* = owned;
        return .{
            .dtype = dt,
            .out_dim = out_dim,
            .in_dim = in_dim,
            .box = box,
            .allocator = ctx.allocator(),
            .vt = &Impl(dt).vtable,
        };
    }

    /// Unbox the typed tensor when the runtime dtype matches, consuming
    /// the container (the ptqtp_gguf plane reader's exit).
    pub fn take(self: *ColdQuantWeight, comptime dt: DType) ?QuantWeight(dt) {
        if (self.dtype != dt) return null;
        const w: *QuantWeight(dt) = @ptrCast(@alignCast(self.box));
        const out = w.*;
        self.allocator.destroy(w);
        self.* = undefined;
        return out;
    }

    pub fn deinit(self: *ColdQuantWeight) void {
        self.vt.deinit(self.box, self.allocator);
        self.* = undefined;
    }

    pub fn cloneView(self: *const ColdQuantWeight, ctx: *ExecContext) !ColdQuantWeight {
        const box = try self.vt.clone_view(self.box, ctx);
        return .{
            .dtype = self.dtype,
            .out_dim = self.out_dim,
            .in_dim = self.in_dim,
            .box = box,
            .allocator = ctx.allocator(),
            .vt = self.vt,
        };
    }

    pub fn outDim(self: *const ColdQuantWeight) usize {
        return self.out_dim;
    }

    pub fn inDim(self: *const ColdQuantWeight) usize {
        return self.in_dim;
    }

    /// The generic compact dot (differentiable in the input, exactly the
    /// typed facade chain): retag to the canonical pair, contract, retag
    /// back -- the interposed views are grad-transparent and value-exact.
    pub fn linearSeq(self: *const ColdQuantWeight, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        _ = in_tag;
        var x = try input.withTags(ctx, .{ .seq, .in });
        defer x.deinit();
        var y = try self.vt.linear(self.box, ctx, &x);
        defer y.deinit();
        return y.withTags(ctx, .{ .seq, out_tag });
    }

    pub fn getRowsAs(self: *const ColdQuantWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        var rows = try self.vt.get_rows(self.box, ctx, token_ids);
        defer rows.deinit();
        return rows.withTags(ctx, .{ .seq, out_tag });
    }
};

pub const LinearWeight = union(enum) {
    dense: DenseWeight,
    quant: ColdQuantWeight,
    packed_quant: PackedWeight,
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
            .f32, .f64 => .{ .dense = .{ .f32 = try dense.loadDenseF32Weight(ctx, info, shape) } },
            .f16 => .{ .dense = .{ .f16 = try dense.loadDenseF16Weight(ctx, info, shape, options) } },
            .bf16 => .{ .dense = .{ .bf16 = try dense.loadDenseBf16Weight(ctx, info, shape) } },
            .q8_0 => .{ .packed_quant = .{ .q8_0 = try dense.loadQ8_0Weight(ctx, info, shape, options) } },
            .q4_k => .{ .packed_quant = .{ .q4_k = try dense.loadQ4_KWeight(ctx, info, shape, options) } },
            .q5_k => .{ .packed_quant = .{ .q5_k = try dense.loadQ5_KWeight(ctx, info, shape, options) } },
            .q6_k => .{ .packed_quant = .{ .q6_k = try dense.loadQ6_KWeight(ctx, info, shape, options) } },
            // Every remaining block format loads into the dtype-erased
            // compact container: the ONE runtime-dtype -> comptime switch,
            // and the only site instantiating the per-format vtables.
            inline .q1_0, .q2_0, .q4_0, .q4_1, .q5_0, .q5_1, .q2_k, .q3_k, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_nl, .iq4_xs, .tq1_0, .tq2_0, .mxfp4, .nvfp4 => |t| .{
                .quant = try ColdQuantWeight.of(
                    @field(DType, @tagName(t)),
                    ctx,
                    try dense.loadQuantizedWeight(@field(DType, @tagName(t)), ctx, info, shape),
                ),
            },
            .tq2_0_fx4 => blk: {
                const n = shape[0];
                const k = shape[1];
                if (n % 4 != 0 or k % 256 != 0) return Error.InvalidWeightShape;
                const src = try blockSlice(backend_quant.BlockTQ2_0Foldedx4, info.data);
                if (src.len != (n / 4) * (k / 256)) return Error.InvalidWeightShape;
                // Copy (like every dense quant arm): load() has no file
                // handle, so mmap lifetime cannot be promised here. Still
                // 4.0625 bpw once — no planes, no x4 packs, no fold pass.
                const owned = try ctx.allocator().alloc(backend_quant.BlockTQ2_0Foldedx4, src.len);
                errdefer ctx.allocator().free(owned);
                @memcpy(owned, src);
                break :blk .{ .tq2_0_fx4 = WeightPtqtpFx4.init(ctx.allocator(), owned, ctx.allocator(), n, k, options.gpu_resident) };
            },
            else => Error.UnsupportedWeightType,
        };
    }

    /// The storage/block format actually loaded (runtime). Both PTQTP
    /// containers report `.tq2_0` — the number format of their planes; the
    /// container arm carries the layout.
    pub fn dtype(self: *const LinearWeight) DType {
        return switch (self.*) {
            .dense => |*d| d.dtype(),
            .quant => |*c| c.dtype,
            .packed_quant => |*p| p.dtype(),
            .ptqtp, .tq2_0_fx4 => .tq2_0,
        };
    }

    pub fn deinit(self: *LinearWeight) void {
        switch (self.*) {
            inline else => |*container| container.deinit(),
        }
        self.* = undefined;
    }

    pub fn cloneView(self: *const LinearWeight, ctx: *ExecContext) !LinearWeight {
        return switch (self.*) {
            inline else => |*container, tag| @unionInit(LinearWeight, @tagName(tag), try container.cloneView(ctx)),
        };
    }

    pub fn outDim(self: *const LinearWeight) usize {
        return switch (self.*) {
            inline else => |*container| container.outDim(),
        };
    }

    pub fn inDim(self: *const LinearWeight) usize {
        return switch (self.*) {
            inline else => |*container| container.inDim(),
        };
    }

    /// Replace this weight with a RESIDENT dequantized f16 copy (2 B/weight):
    /// the f16-operands GEMM path's weight format — and the `-Dgpu=metal`
    /// f16 offload operand. Rows are dequantized in chunks through the same
    /// row-gather the embedding lookup uses (`getRowsAs`, every container),
    /// so the transient peak stays a few MB. No-op when the weight is
    /// already f16.
    pub fn toResidentF16(self: *LinearWeight, ctx: *ExecContext) !void {
        switch (self.*) {
            .dense => |*d| if (d.* == .f16) return,
            else => {},
        }
        const rows = self.outDim();
        const cols = self.inDim();
        const allocator = ctx.allocator();

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
        self.* = .{ .dense = .{ .f16 = fresh } };
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
        const allocator = ctx.allocator();

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
        defer pair.deinit(ctx.allocator());
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
        self.* = .{ .ptqtp = WeightPtqtp.init(ctx.allocator(), p1, p2, p3, options.tie_scales) };
        return stats;
    }

    pub fn linearSeq(self: *const LinearWeight, ctx: *ExecContext, input: anytype, comptime in_tag: Tag, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        return switch (self.*) {
            inline else => |*container| try container.linearSeq(ctx, input, in_tag, out_tag),
        };
    }

    /// Norm-into-quantize fusion gate: FUCINA_NORM_QUANT_FUSED=0 forces
    /// the unfused rmsNormMul + linearSeq pair, =1 forces the fused route.
    /// Read once, cached (`tuning.Table.norm_quant_fused`).
    pub fn setNormQuantFused(on: ?bool) void {
        tuning.setField("norm_quant_fused", on);
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
    pub fn supportsNormedFusion(self: *const LinearWeight, m: usize) bool {
        if (comptime offload.enabled) return false;
        if (!tuning.get().norm_quant_fused) return false;
        // Prefill-only: at decode shapes (m < 4) the fused route pays pooled
        // scratch acquisitions plus a padded 4-row-group quantize for one
        // real row, where the unfused internal quantizer has an m=1 stack
        // fast path — measured 2-3% decode LOSS on M1 Q4_K_M/Q8_0, against
        // a +11-23% pp32 win. (The x86 m<4 decode-compact routes bypass the
        // packed path anyway.)
        if (m < 4) return false;
        return switch (self.*) {
            .packed_quant => |*p| p.supportsNormedFusion(),
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
        if (comptime !offload.enabled) {
            if (self.* == .packed_quant and !x.requiresGrad() and x.dim(.seq) >= 4 and tuning.get().norm_quant_fused) {
                if (try self.packed_quant.tryLinearSeqNormed(ctx, x, norm_weight, eps, in_tag, out_tag)) |out| return out;
            }
        }
        var normed = try x.rmsNormMul(ctx, in_tag, norm_weight, eps);
        defer normed.deinit();
        return self.linearSeq(ctx, &normed, in_tag, out_tag);
    }

    pub fn getRowsAs(self: *const LinearWeight, ctx: *ExecContext, token_ids: []const usize, comptime out_tag: Tag) !Tensor(.{ .seq, out_tag }) {
        return switch (self.*) {
            inline else => |*container| try container.getRowsAs(ctx, token_ids, out_tag),
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
        if (comptime !offload.enabled) {
            if (file.is_mmap and !file.isSplit()) {
                const shape = try requireMatrixShape(info, expected_rows, expected_cols);
                if (gguf.RowTable.init(ctx.allocator(), info)) |table| {
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
