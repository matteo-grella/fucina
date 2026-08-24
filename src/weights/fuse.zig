//! Weight fusion and PTQTP decoration over `LinearWeight`: `fuseLinear`
//! stacks same-format parts into one output-fused matrix (restoring the
//! GPU residency `loadForFusion` skipped when fusion declines), and
//! `decoratePtqtpInto` with `PtqtpReport` drives and aggregates PTQTP
//! decoration over a model walk.

const std = @import("std");

const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const ptqtp = @import("../ptqtp.zig");

const common = @import("common.zig");
const dense = @import("dense.zig");
const gpu = @import("gpu.zig");
const linear = @import("linear.zig");
const ptqtp_w = @import("ptqtp.zig");

const gpu_impl = backend_mod.gpu_impl;
const ExecContext = exec_mod.ExecContext;
const Error = common.Error;
const QuantWeight = common.QuantWeight;
const backend_quant = common.backend_quant;
const WeightF32 = dense.WeightF32;
const WeightF16 = dense.WeightF16;
const LinearWeight = linear.LinearWeight;
const WeightPtqtp = ptqtp_w.WeightPtqtp;
const WeightPtqtpFx4 = ptqtp_w.WeightPtqtpFx4;

fn restoreGpuResidencyAfterDeclinedFusion(ctx: *ExecContext, parts: []const *LinearWeight) !void {
    if (comptime !gpu_impl.enabled) return;
    for (parts) |part| switch (part.*) {
        .f32 => |*value| _ = try gpu.makeGpuResidentDenseWeight(.f32, WeightF32, ctx, value),
        .f16 => |*value| _ = try gpu.makeGpuResidentDenseWeight(.f16, WeightF16, ctx, value),
        inline .q4_k, .q5_k, .q6_k, .q8_0 => |*weight| if (!weight.rhs_lifetime.isCacheable() and try gpu.makeGpuResidentQuantWeight(@TypeOf(weight.*).dtype, ctx, &weight.value)) {
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
                _ = try gpu.makeGpuResidentDenseWeight(.f16, WeightF16, ctx, &fused);
            } else if (comptime tag == .f32) {
                _ = try gpu.makeGpuResidentDenseWeight(.f32, WeightF32, ctx, &fused);
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
