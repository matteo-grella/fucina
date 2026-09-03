//! MoE expert-stack loading and forward: the resident/borrowed stacked
//! RHS loaders (`loadMoeRhs`, `loadMoeRhsPtqtp`), the disk-streamed tier
//! over `ExpertStore` (`createExpertStore`, `loadMoeRhsStreamed`, the
//! ProjSpec builders), and the fused gated FFN forwards
//! (`moeSwiGluFfnSeq`, `moeGatedFfnSeq`).

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const ag_mod = @import("../ag.zig");
const gguf = @import("../gguf.zig");
const expert_store = @import("../store/expert_store.zig");

const common = @import("common.zig");

const Tensor = ag_mod.Tensor;
const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const moe_mod = @import("../moe.zig");
const MoeRhs = moe_mod.MoeRhs;
const MoeBatchProfile = moe_mod.MoeBatchProfile;
const Gated = exec_mod.Gated;
const ExpertStore = expert_store.ExpertStore;
const Error = common.Error;
const backend_quant = common.backend_quant;
const blockSlice = common.blockSlice;
const Allocator = std.mem.Allocator;

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
                break :blk .{ .q8_0 = .{ .allocator = null, .blocks = src, .blocks_per_column = bpc, .k = in_dim, .n = rows } };
            }
            gguf.prefetch(info.data);
            const owned = try ctx.allocator().alloc(dtype_mod.BlockQ8_0, src.len);
            errdefer ctx.allocator().free(owned);
            @memcpy(owned, src);
            break :blk .{ .q8_0 = .{ .allocator = ctx.allocator(), .blocks = owned, .blocks_per_column = bpc, .k = in_dim, .n = rows } };
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
            const src = try blockSlice(backend_quant.types.BlockTQ2_0Foldedx4, info.data);
            if (src.len != (rows / 4) * (in_dim / 256)) return Error.InvalidWeightShape;
            var folded: []const backend_quant.types.BlockTQ2_0Foldedx4 = src;
            var folded_allocator: ?Allocator = null;
            if (!borrow) {
                gguf.prefetch(info.data);
                const owned = try ctx.allocator().alloc(backend_quant.types.BlockTQ2_0Foldedx4, src.len);
                @memcpy(owned, src);
                folded = owned;
                folded_allocator = ctx.allocator();
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
    const bpc = try backend_mod.quant.types.blockCountForDType(.q8_k, expected_in_dim);
    const blocks_per_plane = try std.math.mul(usize, rows, bpc);

    var planes: [3][]const dtype_mod.BlockTQ2_0 = .{ &.{}, &.{}, &.{} };
    var owned_count: usize = 0;
    errdefer for (planes[0..owned_count]) |plane| ctx.allocator().free(@constCast(plane));
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
            const owned = try ctx.allocator().alloc(dtype_mod.BlockTQ2_0, src.len);
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
    var folded: []const backend_quant.types.BlockTQ2_0Foldedx4 = &.{};
    var folded_allocator: ?Allocator = null;
    if (tied and plane_infos.len == 2 and expected_out_dim % 4 == 0) {
        const fg = (expected_out_dim / 4) * bpc;
        const buf = try ctx.allocator().alloc(backend_quant.types.BlockTQ2_0Foldedx4, expected_n_expert * fg);
        errdefer ctx.allocator().free(buf);
        const expert_blocks = expected_out_dim * bpc;
        for (0..expected_n_expert) |e| {
            var views: [2]backend_quant.types.QuantizedMatmulRhsTQ2_0 = undefined;
            for (0..2) |p| {
                const blocks = planes[p][e * expert_blocks ..][0..expert_blocks];
                views[p] = try backend_mod.kernels.quantizedMatmulRhsTQ2_0FromBorrowedBlocks(expected_in_dim, expected_out_dim, @constCast(blocks));
            }
            try backend_mod.kernels.packMatmulRhsTQ2_0Foldedx4Into(buf[e * fg ..][0..fg], &views[0], &views[1]);
        }
        folded = buf;
        folded_allocator = ctx.allocator();
    }
    return .{ .ptqtp = .{
        .allocator = if (borrow) null else ctx.allocator(),
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
const moe_stream = @import("moe_stream.zig");
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
    return moeGatedFfnSeq(ctx, input, gate, up, down, selected, routing_weights, top_k, out_pe, .{ .op = .swiglu }, io, profile);
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
    act: Gated,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor(.{ .seq, .embed }) {
    if (input.requiresGrad()) return Error.UnsupportedGradient;
    const raw_input = input.asRawTensor();
    var raw = if (input.dim(.seq) == 1)
        try moe_mod.expertFfn(
            ctx,
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
        try moe_mod.expertFfnBatch(
            ctx,
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
    const owned = try ctx.allocator().alloc(Block, src.len);
    errdefer ctx.allocator().free(owned);
    @memcpy(owned, src);
    return .{ .allocator = ctx.allocator(), .blocks = owned, .k = in_dim, .n = rows, .blocks_per_column = blocks_per_column };
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
) !backend_quant.types.QuantizedMatmulRhsRowsFor(dtype) {
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
    if (try backend_quant.types.blockCountForDType(.q8_k, in_dim) != bpc) return Error.InvalidWeightShape;
    if (borrow) {
        return .{ .allocator = null, .blocks = @constCast(src), .blocks_per_column = bpc, .k = in_dim, .n = rows };
    }
    gguf.prefetch(info.data);
    const owned = try ctx.allocator().alloc(Block, src.len);
    errdefer ctx.allocator().free(owned);
    @memcpy(owned, src);
    return .{ .allocator = ctx.allocator(), .blocks = owned, .blocks_per_column = bpc, .k = in_dim, .n = rows };
}
