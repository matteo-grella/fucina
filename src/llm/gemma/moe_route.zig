//! Gemma-side surface over the shared exec-layer MoE route plan
//! (`exec/moe_chain.zig`): the counting-sort plan itself is family-agnostic
//! and lives with the rest of the batched-MoE scheduling scaffolding; this
//! module keeps the gemma callers' names and holds the gemma-specific
//! expert-major scatter.
const std = @import("std");
const fucina = @import("fucina");

const moe_chain = fucina.ExecContext.moe_chain;

pub const Plan = moe_chain.MoeRoutePlan;
pub const BuildResult = moe_chain.MoeRouteBuildResult;
pub const build = moe_chain.buildMoeRoutePlan;

/// Expert-major weighted scatter of the grouped down rows back into token
/// order. Serial by design: a token-parallel split needs the plan's inverse
/// mapping (`inv`) and changes each token's floating-point summation order,
/// which requires a tolerance argument against the gemma parity oracles.
pub fn scatterInto(
    out: []f32,
    down_rows: []const f32,
    route: *const Plan,
    weights: []const f32,
    top_k: usize,
    hidden: usize,
) void {
    @memset(out, 0);
    for (0..route.expertCount()) |e| {
        const m = route.count[e];
        if (m == 0) continue;
        const base = route.offset[e];
        for (0..m) |i| {
            const pair = route.order[base + i];
            const token = pair / top_k;
            const w = weights[pair];
            const src = down_rows[(base + i) * hidden ..][0..hidden];
            for (out[token * hidden ..][0..hidden], src) |*dst, value| dst.* += w * value;
        }
    }
}
