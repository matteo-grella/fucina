//! Gemma-family surface over the fused gate|up MoE kernels, which live
//! next door in `moe_gu.zig`: the raw entries re-exported under the
//! family's names, plus the tagged `Tensor(.{ .seq, .embed })` wrappers
//! the gemma forward calls.
const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;
const MoeBatchProfile = fucina.MoeBatchProfile;
const SeqEmbedTensor = fucina.Tensor(.{ .seq, .embed });
const RawTensor = fucina.internal.RawTensor;
const backend_mod = fucina.internal.backend_mod;

/// The fused gate|up kernel bodies (packed x4 arms, raw block arm, GPU
/// batch path).
pub const moe_gu = @import("moe_gu.zig");

pub const RawExpertWeights = moe_gu.RawExpertWeights;
pub const decodePacked = moe_gu.decodePacked;
pub const batchPacked = moe_gu.batchPacked;
pub const decodeRaw = moe_gu.decodeRaw;
pub const batchRaw = moe_gu.batchRaw;

fn wrapSeqEmbedTensor(ctx: *ExecContext, raw: RawTensor) !SeqEmbedTensor {
    var owned = raw;
    errdefer owned.deinit();
    return SeqEmbedTensor.fromTensor(ctx, owned);
}

pub fn decodePackedTensor(
    self: *ExecContext,
    x: *const SeqEmbedTensor,
    gate: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !SeqEmbedTensor {
    if (x.requiresGrad()) return error.UnsupportedGradient;
    const raw = try decodePacked(self, x.asRawTensor(), gate, up, down, selected, weights, out_pe, io, profile);
    return wrapSeqEmbedTensor(self, raw);
}

pub fn batchPackedTensor(
    self: *ExecContext,
    x: *const SeqEmbedTensor,
    gate: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4,
    selected: []const usize,
    weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !SeqEmbedTensor {
    if (x.requiresGrad()) return error.UnsupportedGradient;
    const raw = try batchPacked(self, x.asRawTensor(), gate, up, down, selected, weights, top_k, out_pe, io, profile);
    return wrapSeqEmbedTensor(self, raw);
}

pub fn decodeRawTensor(
    self: *ExecContext,
    x: *const SeqEmbedTensor,
    gw: RawExpertWeights,
    n_expert: usize,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !SeqEmbedTensor {
    if (x.requiresGrad()) return error.UnsupportedGradient;
    const raw = try decodeRaw(self, x.asRawTensor(), gw, n_expert, selected, weights, out_pe, io, profile);
    return wrapSeqEmbedTensor(self, raw);
}

pub fn batchRawTensor(
    self: *ExecContext,
    x: *const SeqEmbedTensor,
    gw: RawExpertWeights,
    n_expert: usize,
    selected: []const usize,
    weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !SeqEmbedTensor {
    if (x.requiresGrad()) return error.UnsupportedGradient;
    const raw = try batchRaw(self, x.asRawTensor(), gw, n_expert, selected, weights, top_k, out_pe, io, profile);
    return wrapSeqEmbedTensor(self, raw);
}

test {
    _ = @import("moe_tests.zig");
}
