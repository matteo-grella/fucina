const std = @import("std");
const fucina = @import("fucina");
const route_mod = @import("moe_route.zig");

const tensor = fucina.internal.tensor_mod;

const ExecContext = fucina.ExecContext;
const MoeBatchProfile = fucina.MoeBatchProfile;
const Tensor = tensor.Tensor;

const profileStart = ExecContext.moe_chain.moeBatchProfileStart;
const profileElapsed = ExecContext.moe_chain.moeBatchProfileElapsed;

pub fn scatterGrouped(
    self: *ExecContext,
    seq: usize,
    hidden: usize,
    top_k: usize,
    route: *const route_mod.Plan,
    weights: []const f32,
    down_rows: []const f32,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const profile_enabled = profile != null;
    const out_alloc_start = profileStart(profile_enabled, io);
    var out = try self.emptyRank(2, .{ seq, hidden });
    errdefer out.deinit();
    if (profile) |p| p.alloc_ns += profileElapsed(out_alloc_start, io);

    const scatter_start = profileStart(profile_enabled, io);
    route_mod.scatterInto(out.data(), down_rows, route, weights, top_k, hidden);
    if (profile) |p| p.scatter_ns += profileElapsed(scatter_start, io);
    return out;
}

pub fn recordBatch(profile: ?*MoeBatchProfile, total_start: i128, io: ?std.Io, route: *const route_mod.Plan) void {
    if (profile) |p| {
        p.total_ns += profileElapsed(total_start, io);
        p.batches += 1;
        p.pairs += route.pairCount();
        p.active_experts += route.active_experts;
        p.max_expert_m = @max(p.max_expert_m, route.max_expert_m);
    }
}
