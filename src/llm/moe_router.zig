//! Cache-aware expert routing policy shared by the streamed-MoE decoders:
//! the resident-preferring near-tie selection (`cacheRouteSel`) and the
//! router-lookahead prefetch tail (`pilotHintTopK`). Public as
//! `llm.moe_router`. The family forwards own the family-specific part
//! (finding the MoE arm, building the normed hidden rows).

const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;
const MoeRhs = fucina.MoeRhs;
const ExpertStore = fucina.ExpertStore;
const LinearWeight = fucina.weights.LinearWeight;

/// Cache-aware expert selection when `gate` streams from an ExpertStore
/// that opted into cache routing (`MoeStreamOptions.cache_route`); false =
/// the caller keeps its plain top-k.
pub fn cacheRouteSel(gate: *const MoeRhs, choice: []const f32, sel: []usize) bool {
    return switch (gate.*) {
        .streamed => |*sw| sw.store.cacheRouteTopK(sw.layer, choice, sel),
        else => false,
    };
}

/// Router-lookahead tail shared by streamed-MoE decoders: score the normed
/// hidden rows through `router`, take the un-normalized top-k per row, and
/// hand the selection to the store as a prefetch hint for `layer_i`.
/// Callers own the family-specific part (finding the next layer's MoE arm
/// and building `nrm` with that layer's FFN norm).
pub fn pilotHintTopK(ctx: *ExecContext, nrm: *const fucina.Tensor(.{ .seq, .embed }), router: *const LinearWeight, top_k: usize, store: *ExpertStore, layer_i: usize) !void {
    var logits = try router.linearSeq(ctx, nrm, .embed, .expert);
    defer logits.deinit();
    const allocator = ctx.allocator;
    const seq = nrm.dim(.seq);
    const sel = try allocator.alloc(usize, seq * top_k);
    defer allocator.free(sel);
    const wgt = try allocator.alloc(f32, seq * top_k);
    defer allocator.free(wgt);
    try logits.routerTopK(ctx, .expert, top_k, .{ .normalize_selected = false }, sel, wgt);
    store.pilotHint(layer_i, sel);
}
