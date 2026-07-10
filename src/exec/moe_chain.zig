//! Shared batched-MoE scheduling scaffolding: the expert-grouped route plan
//! (counting sort), the gather → gate/up → act → down phase-chain machinery,
//! the phase chunking constants/helpers, and the profile timer pair. Consumed
//! by the qwen-style MoE op (`exec/moe.zig`) and — through the
//! `ExecContext.moe_chain` re-export — by the gemma MoE engines at the llm
//! layer, so scheduler fixes and chunk retunes land once for every family.
const std = @import("std");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;

pub fn moeBatchProfileStart(enabled: bool, io: ?std.Io) i128 {
    return if (enabled) std.Io.Clock.awake.now(io.?).nanoseconds else 0;
}

// Returns i64 (not i128): elapsed deltas fit comfortably, accumulators
// coerce, and i64 keeps the carved MoE task structs at align <= 8 so they
// can live in the u64-backed decode scratch (`carveMoeDecodeScratch`).
pub fn moeBatchProfileElapsed(start: i128, io: ?std.Io) i64 {
    return @intCast(std.Io.Clock.awake.now(io.?).nanoseconds - start);
}

pub fn moeDecodeColumnSplit(dim: usize, alignment: usize) usize {
    if (dim <= 1) return dim;
    var split = dim / 2;
    if (alignment > 1) split = (split / alignment) * alignment;
    if (split == 0 or split >= dim) split = dim / 2;
    return split;
}

pub const moe_phase_split_m_threshold = 16;
pub const moe_phase_col_chunk = 256;
/// Column width for small-m (m < moe_phase_split_m_threshold) splitting.
/// Multiples of 256 keep every c0 superblock-aligned for the Q4_K/Q5_K/Q6_K
/// column kernels (decode's split only needs 32; 256 satisfies both). A
/// 256-column gate/up chunk at k=2048 streams ~360-430 KB of K-quant weights
/// per task, keeping per-task chain overhead (two CAS ops) under 1%.
pub const moe_phase_small_m_col_chunk = 256;
/// Task-budget multiplier: small-m phases split only while the layer has
/// fewer than workers * this many active experts (each contributing one
/// task per projection phase). At the Qwen3-30B 128-expert/top-8 shape with
/// an 8-way team this fires for the pp15-33 band (~77-111 active experts)
/// and stays off at pp128+ (~128 active), where full-width tasks already
/// saturate the team.
pub const moe_phase_small_m_task_budget_mul = 16;
/// Phased-chain gate for batched MoE prefill, in routed (token, expert)
/// pairs. 64 pairs (seq >= 8 at top-8) puts the mid-batch prefill band on
/// the gather -> gate/up -> act -> down chain where small-m column chunking
/// applies; below it the monolithic per-expert task avoids the chain-task
/// allocations.
pub const moe_batch_phase_min_pairs = 64;

/// Per-layer-call decision: column width for small-m phase splitting, or 0
/// to keep one full-width task per expert per projection.
pub fn moeSmallMColWidth(active_experts: usize, workers: usize) usize {
    if (active_experts == 0) return 0;
    if (active_experts >= workers * moe_phase_small_m_task_budget_mul) return 0;
    return moe_phase_small_m_col_chunk;
}

/// Effective column width for one expert's projection phase. Count and
/// bounds both derive from this width, so the task-counting and
/// task-construction loops cannot disagree.
pub fn moePhaseColWidth(m: usize, out_dim: usize, small_m_width: usize) usize {
    if (m >= moe_phase_split_m_threshold) return moe_phase_col_chunk;
    return if (small_m_width == 0) out_dim else small_m_width;
}

pub fn moePhaseChunkCount(width: usize, out_dim: usize) usize {
    if (width >= out_dim) return 1;
    return (out_dim + width - 1) / width;
}

pub fn moePhaseChunkBounds(chunk: usize, width: usize, out_dim: usize) struct { c0: usize, c1: usize } {
    const c0 = chunk * width;
    return .{ .c0 = c0, .c1 = @min(out_dim, c0 + width) };
}

pub const MoeBatchPhaseChainKind = enum { gather, gate_up, act, down };

pub const MoeBatchPhaseChainState = struct {
    gate_start: usize,
    gate_count: usize,
    act_index: usize,
    down_start: usize,
    down_count: usize,
    remaining_gate_up: std.atomic.Value(u32),
};

pub const MoeBatchPhaseChainTask = struct {
    state: *MoeBatchPhaseChainState,
    kind: MoeBatchPhaseChainKind,
    run_ctx: *anyopaque,
    run_fn: *const fn (*anyopaque) void,
};

fn enqueueMoeBatchPhaseRange(chain: *const thread.Chain, start: usize, count: usize) void {
    var i: usize = 0;
    while (i < count) : (i += 1) chain.enqueue(start + i);
}

pub fn runMoeBatchPhaseChainTask(task: *MoeBatchPhaseChainTask, chain: *const thread.Chain) void {
    task.run_fn(task.run_ctx);
    const state = task.state;
    switch (task.kind) {
        .gather => enqueueMoeBatchPhaseRange(chain, state.gate_start, state.gate_count),
        .gate_up => {
            if (state.remaining_gate_up.fetchSub(1, .acq_rel) == 1) {
                chain.enqueue(state.act_index);
            }
        },
        .act => enqueueMoeBatchPhaseRange(chain, state.down_start, state.down_count),
        .down => {},
    }
}

/// Wire the per-phase task arrays into one dependency-chained task list:
/// per active expert, gather → (all gate/up chunks) → act → (all down
/// chunks). `chain_states[e].gate_start/gate_count/down_start/down_count`
/// arrive holding indices into the per-phase arrays and leave holding
/// indices into `chain_tasks`; `remaining_gate_up` is armed here. Returns
/// the number of initially-runnable tasks (the gather prefix), i.e. the
/// `initial_count` for `Pool.parallelChained`.
pub fn wireMoeBatchPhaseChain(
    comptime GatherTask: type,
    comptime GateUpTask: type,
    comptime ActTask: type,
    comptime DownTask: type,
    chain_tasks: []MoeBatchPhaseChainTask,
    chain_states: []MoeBatchPhaseChainState,
    count: []const usize,
    gather_tasks: []GatherTask,
    gate_up_tasks: []GateUpTask,
    act_tasks: []ActTask,
    down_tasks: []DownTask,
    run_gather: *const fn (*anyopaque) void,
    run_gate_up: *const fn (*anyopaque) void,
    run_act: *const fn (*anyopaque) void,
    run_down: *const fn (*anyopaque) void,
) usize {
    const n_expert = count.len;
    var chain_i: usize = 0;
    for (0..n_expert) |e| {
        if (count[e] == 0) continue;
        chain_tasks[chain_i] = .{
            .state = &chain_states[e],
            .kind = .gather,
            .run_ctx = &gather_tasks[e],
            .run_fn = run_gather,
        };
        chain_i += 1;
    }
    const chain_initial_count = chain_i;
    for (0..n_expert) |e| {
        if (count[e] == 0) continue;
        const gate_task_start = chain_states[e].gate_start;
        const gate_task_count = chain_states[e].gate_count;
        chain_states[e].gate_start = chain_i;
        chain_states[e].gate_count = gate_task_count;
        chain_states[e].remaining_gate_up = .init(@intCast(gate_task_count));
        for (0..gate_task_count) |i| {
            chain_tasks[chain_i] = .{
                .state = &chain_states[e],
                .kind = .gate_up,
                .run_ctx = &gate_up_tasks[gate_task_start + i],
                .run_fn = run_gate_up,
            };
            chain_i += 1;
        }
    }
    for (0..n_expert) |e| {
        if (count[e] == 0) continue;
        chain_states[e].act_index = chain_i;
        chain_tasks[chain_i] = .{
            .state = &chain_states[e],
            .kind = .act,
            .run_ctx = &act_tasks[e],
            .run_fn = run_act,
        };
        chain_i += 1;
    }
    for (0..n_expert) |e| {
        if (count[e] == 0) continue;
        const down_task_start = chain_states[e].down_start;
        const down_chain_count = chain_states[e].down_count;
        chain_states[e].down_start = chain_i;
        chain_states[e].down_count = down_chain_count;
        for (0..down_chain_count) |i| {
            chain_tasks[chain_i] = .{
                .state = &chain_states[e],
                .kind = .down,
                .run_ctx = &down_tasks[down_task_start + i],
                .run_fn = run_down,
            };
            chain_i += 1;
        }
    }
    std.debug.assert(chain_i == chain_tasks.len);
    return chain_initial_count;
}

/// Expert-grouped routing plan for one batched MoE layer: counting sort of
/// the (token, expert) pairs by expert. `order[offset[e] ..][0..count[e]]`
/// lists the pair indices routed to expert `e`; `inv[p]` is pair `p`'s
/// position in that grouped order (the token-major scatter's gather index).
pub const MoeRoutePlan = struct {
    allocator: Allocator,
    storage: []usize,
    count: []usize,
    offset: []usize,
    order: []usize,
    inv: []usize,
    active_experts: usize,
    max_expert_m: usize,

    pub fn deinit(self: *MoeRoutePlan) void {
        self.allocator.free(self.storage);
        self.* = undefined;
    }

    pub fn pairCount(self: *const MoeRoutePlan) usize {
        return self.order.len;
    }

    pub fn expertCount(self: *const MoeRoutePlan) usize {
        return self.count.len;
    }
};

pub const MoeRouteBuildResult = struct {
    plan: MoeRoutePlan,
    alloc_ns: i64,
    count_sort_ns: i64,
};

pub fn buildMoeRoutePlan(
    allocator: Allocator,
    selected: []const usize,
    n_expert: usize,
    profile_enabled: bool,
    io: ?std.Io,
) !MoeRouteBuildResult {
    for (selected) |e| if (e >= n_expert) return error.IndexOutOfBounds;

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const storage = try allocator.alloc(usize, n_expert + (n_expert + 1) + n_expert + 2 * selected.len);
    errdefer allocator.free(storage);
    var storage_i: usize = 0;
    const count = storage[storage_i..][0..n_expert];
    storage_i += n_expert;
    const offset = storage[storage_i..][0 .. n_expert + 1];
    storage_i += n_expert + 1;
    const place = storage[storage_i..][0..n_expert];
    storage_i += n_expert;
    const order = storage[storage_i..][0..selected.len];
    storage_i += selected.len;
    const inv = storage[storage_i..][0..selected.len];
    const alloc_ns = if (profile_enabled) moeBatchProfileElapsed(alloc_start, io) else 0;

    const count_sort_start = moeBatchProfileStart(profile_enabled, io);
    @memset(count, 0);
    for (selected) |e| count[e] += 1;

    offset[0] = 0;
    var active_experts: usize = 0;
    var max_expert_m: usize = 0;
    for (0..n_expert) |e| {
        const c = count[e];
        if (c > 0) {
            active_experts += 1;
            max_expert_m = @max(max_expert_m, c);
        }
        offset[e + 1] = offset[e] + c;
    }

    for (0..n_expert) |e| place[e] = offset[e];
    for (selected, 0..) |e, pair| {
        const pos = place[e];
        order[pos] = pair;
        inv[pair] = pos;
        place[e] += 1;
    }
    const count_sort_ns = if (profile_enabled) moeBatchProfileElapsed(count_sort_start, io) else 0;

    return .{
        .plan = .{
            .allocator = allocator,
            .storage = storage,
            .count = count,
            .offset = offset,
            .order = order,
            .inv = inv,
            .active_experts = active_experts,
            .max_expert_m = max_expert_m,
        },
        .alloc_ns = alloc_ns,
        .count_sort_ns = count_sort_ns,
    };
}

/// Token-major routed scatter over the token range [t0, t1):
/// `out[t] = sum over t's top-k pairs of weights[p] * down_rows[inv[p]]`.
/// Every token has `top_k >= 1` pairs, so the `k == 0` write initializes the
/// row (no separate zero-fill pass). Disjoint token ranges write disjoint
/// `out` rows and each row keeps its k-accumulation order, so splitting the
/// range across workers is bit-identical to one full-range call.
pub fn scatterTokenMajor(
    out: []f32,
    down_rows: []const f32,
    weights: []const f32,
    inv: []const usize,
    hidden: usize,
    top_k: usize,
    t0: usize,
    t1: usize,
) void {
    for (t0..t1) |t| {
        const dst = out[t * hidden ..][0..hidden];
        for (0..top_k) |k| {
            const p = t * top_k + k;
            const w = weights[p];
            const src = down_rows[inv[p] * hidden ..][0..hidden];
            if (k == 0) {
                for (dst, src) |*o, s| o.* = w * s;
            } else {
                for (dst, src) |*o, s| o.* += w * s;
            }
        }
    }
}
