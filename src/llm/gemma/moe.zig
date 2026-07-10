const std = @import("std");
const fucina = @import("fucina");

const backend_mod = fucina.internal.backend_mod;
const backend_ops = backend_mod.ops;
const dtype_mod = backend_mod.dtype_info;
const gemma_moe_route = @import("moe_route.zig");
const gemma_moe_route_tensor = @import("moe_route_tensor.zig");
const tensor = fucina.internal.tensor_mod;
const thread = fucina.internal.thread_mod;

const ExecContext = fucina.ExecContext;
const MoeBatchProfile = fucina.MoeBatchProfile;
const Tensor = tensor.Tensor;
const SeqEmbedTensor = fucina.Tensor(.{ .seq, .embed });

// Shared batched-MoE scheduling scaffolding: the exec-layer leaf reached
// through the `fucina` root so the chain/task types are identical to the
// ones `thread.Pool` dispatches for the qwen MoE op.
const moe_chain = ExecContext.moe_chain;
const moeBatchProfileStart = moe_chain.moeBatchProfileStart;
const moeBatchProfileElapsed = moe_chain.moeBatchProfileElapsed;
const moeDecodeColumnSplit = moe_chain.moeDecodeColumnSplit;
const moePhaseChunkCount = moe_chain.moePhaseChunkCount;
const moePhaseChunkBounds = moe_chain.moePhaseChunkBounds;
const moePhaseColWidth = moe_chain.moePhaseColWidth;
const moeSmallMColWidth = moe_chain.moeSmallMColWidth;
const MoeBatchPhaseChainState = moe_chain.MoeBatchPhaseChainState;
const MoeBatchPhaseChainTask = moe_chain.MoeBatchPhaseChainTask;
const runMoeBatchPhaseChainTask = moe_chain.runMoeBatchPhaseChainTask;

/// Raw GGUF-layout fused-gate/up expert blocks used by Gemma-family MoE
/// inference. Per expert: `gu` = 2*out_pe rows (gate rows first, then up) of
/// hidden/256 Q6_K or Q4_K blocks; `dn_blocks` = hidden rows of out_pe/32 Q8_0
/// blocks. `device_owned` marks process-lifetime GPU storage that may use the
/// Metal wrap cache.
pub const RawExpertWeights = struct {
    gu: GuBlocks,
    dn_blocks: []const dtype_mod.BlockQ8_0,
    device_owned: bool,
    /// The blocks are borrowed straight from the still-mapped GGUF.
    borrowed: bool = false,

    pub const GuBlocks = union(enum) {
        q6_k: []const dtype_mod.BlockQ6_K,
        q4_k: []const dtype_mod.BlockQ4_K,
    };

    pub fn guBlockCount(self: *const RawExpertWeights) usize {
        return switch (self.gu) {
            inline else => |blocks| blocks.len,
        };
    }
};

fn wrapSeqEmbedTensor(ctx: *ExecContext, raw: Tensor) !SeqEmbedTensor {
    var owned = raw;
    errdefer owned.deinit();
    return SeqEmbedTensor.fromTensor(ctx, owned);
}

pub fn decodePackedTensor(
    self: *ExecContext,
    x: *const SeqEmbedTensor,
    gate: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.QuantizedMatmulRhsQ8_0x4,
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
    gate: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.QuantizedMatmulRhsQ8_0x4,
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

const GemmaMoeDecodeTask = struct {
    gate: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    up: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    down: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
    qx: []const backend_mod.quantized_matmul.BlockQ8_K,
    out_pe: usize,
    hidden: usize,
    weight: f32,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_0,
    out: []f32,
    profile_enabled: bool,
    io: ?std.Io,
    gate_up_ns: i64,
    geglu_requant_ns: i64,
    down_ns: i64,
};

fn runGemmaMoeDecodeTask(task: *const GemmaMoeDecodeTask) void {
    const qm = backend_mod.quantized_matmul;
    const task_profile = @constCast(task);

    const gate_up_start = moeBatchProfileStart(task.profile_enabled, task.io);
    qm.matmulQ6_Kx4RhsPairTile(task.gate_buf, task.up_buf, task.qx, task.gate, task.up, task.out_pe, 0, 1, 0, task.out_pe);
    if (task.profile_enabled) task_profile.gate_up_ns += moeBatchProfileElapsed(gate_up_start, task.io);

    const geglu_requant_start = moeBatchProfileStart(task.profile_enabled, task.io);
    for (task.g_buf, task.gate_buf, task.up_buf) |*g, gate_v, up_v| {
        g.* = up_v * backend_ops.geluQuantScalar(gate_v);
    }
    qm.quantizeRowQ8_0Into(task.qg, task.g_buf) catch {
        @memset(task.out, 0);
        return;
    };
    if (task.profile_enabled) task_profile.geglu_requant_ns += moeBatchProfileElapsed(geglu_requant_start, task.io);

    const down_start = moeBatchProfileStart(task.profile_enabled, task.io);
    qm.matmulQ8_0x4RhsTile(task.out, task.qg, task.down, task.hidden, 0, 1, 0, task.hidden);
    if (task.weight != 1.0) {
        for (task.out) |*o| o.* *= task.weight;
    }
    if (task.profile_enabled) task_profile.down_ns += moeBatchProfileElapsed(down_start, task.io);
}

const GemmaMoeDecodeChainState = struct {
    gate: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    up: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    down: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
    qx: []const backend_mod.quantized_matmul.BlockQ8_K,
    out_pe: usize,
    hidden: usize,
    weight: f32,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_0,
    out: []f32,
    profile_enabled: bool,
    io: ?std.Io,
    remaining_gate_up: std.atomic.Value(u32),
    geglu_requant_ns: i64,
    down_task0: usize,
};

const GemmaMoeDecodeChainTask = struct {
    state: *GemmaMoeDecodeChainState,
    kind: enum { gate_up, down },
    c0: usize,
    c1: usize,
    elapsed_ns: i64,
};

fn runGemmaMoeDecodeChainTask(task: *GemmaMoeDecodeChainTask, chain: *const thread.Chain) void {
    const qm = backend_mod.quantized_matmul;
    const state = task.state;
    switch (task.kind) {
        .gate_up => {
            const gate_up_start = moeBatchProfileStart(state.profile_enabled, state.io);
            qm.matmulQ6_Kx4RhsPairTile(state.gate_buf, state.up_buf, state.qx, state.gate, state.up, state.out_pe, 0, 1, task.c0, task.c1);
            if (state.profile_enabled) task.elapsed_ns += moeBatchProfileElapsed(gate_up_start, state.io);

            if (state.remaining_gate_up.fetchSub(1, .acq_rel) == 1) {
                const geglu_requant_start = moeBatchProfileStart(state.profile_enabled, state.io);
                for (state.g_buf, state.gate_buf, state.up_buf) |*g, gate_v, up_v| {
                    g.* = up_v * backend_ops.geluQuantScalar(gate_v);
                }
                qm.quantizeRowQ8_0Into(state.qg, state.g_buf) catch unreachable;
                if (state.profile_enabled) state.geglu_requant_ns += moeBatchProfileElapsed(geglu_requant_start, state.io);
                chain.enqueue(state.down_task0);
                chain.enqueue(state.down_task0 + 1);
            }
        },
        .down => {
            const down_start = moeBatchProfileStart(state.profile_enabled, state.io);
            qm.matmulQ8_0x4RhsTile(state.out, state.qg, state.down, state.hidden, 0, 1, task.c0, task.c1);
            if (state.weight != 1.0) {
                for (state.out[task.c0..task.c1]) |*o| o.* *= state.weight;
            }
            if (state.profile_enabled) task.elapsed_ns += moeBatchProfileElapsed(down_start, state.io);
        },
    }
}

/// Gemma 4 fused single-token MoE over the loader's widened expert RHS:
/// gate/up are Q6_Kx4, down is Q8_0x4, and the gate activation is ggml's
/// f16-LUT GELU. This mirrors Qwen's expert-parallel decode helper while
/// preserving Gemma's packed formats and numerics.
pub fn decodePacked(
    self: *ExecContext,
    x: *const Tensor,
    gate: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.QuantizedMatmulRhsQ8_0x4,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const av = try x.rankView(2);
    if (av.dim(0) != 1) return tensor.TensorError.ShapeMismatch;
    const hidden = av.dim(1);
    const top_k = selected.len;
    const n_expert = gate.len;
    const profile_enabled = profile != null;
    const total_start = moeBatchProfileStart(profile_enabled, io);

    if (top_k == 0 or weights.len != top_k) return tensor.TensorError.InvalidDataLength;
    if (up.len != n_expert or down.len != n_expert) return tensor.TensorError.ShapeMismatch;
    if (out_pe % 4 != 0 or hidden % 4 != 0) return tensor.TensorError.InvalidShape;
    for (selected) |e| {
        if (e >= n_expert) return tensor.TensorError.IndexOutOfBounds;
        if (gate[e].k != hidden or up[e].k != hidden or gate[e].n != out_pe or up[e].n != out_pe) return tensor.TensorError.ShapeMismatch;
        if (down[e].k != out_pe or down[e].n != hidden) return tensor.TensorError.ShapeMismatch;
    }

    const blocks_per_g = try qm.q8_0BlockCount(out_pe);
    const chain_task_count = 4 * top_k;
    const chain_initial_count = 2 * top_k;
    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    self.lockMoeDecodeScratch();
    defer self.unlockMoeDecodeScratch();
    const sv = try self.carveMoeDecodeChainScratch(qm.BlockQ8_0, GemmaMoeDecodeChainState, GemmaMoeDecodeChainTask, try qm.qkBlockCount(hidden), top_k, out_pe, hidden, blocks_per_g, chain_task_count);
    const gate_buf = sv.gate_buf;
    const up_buf = sv.up_buf;
    const g_buf = sv.g_buf;
    const qg = sv.qg;
    const outs = sv.outs;
    const tasks = sv.tasks;
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    const gather_quant_start = moeBatchProfileStart(profile_enabled, io);
    const qx = sv.qx;
    try qm.quantizeRowQ8_KInto(qx, try x.dataConstChecked());
    if (profile) |p| p.gather_quant_ns += moeBatchProfileElapsed(gather_quant_start, io);

    const gate_split = moeDecodeColumnSplit(out_pe, 32);
    const down_split = moeDecodeColumnSplit(hidden, 32);
    for (sv.states, 0..) |*state, j| {
        const e = selected[j];
        const down_task0 = chain_initial_count + 2 * j;
        state.* = .{
            .gate = &gate[e],
            .up = &up[e],
            .down = &down[e],
            .qx = qx,
            .out_pe = out_pe,
            .hidden = hidden,
            .weight = weights[j],
            .gate_buf = gate_buf[j * out_pe ..][0..out_pe],
            .up_buf = up_buf[j * out_pe ..][0..out_pe],
            .g_buf = g_buf[j * out_pe ..][0..out_pe],
            .qg = qg[j * blocks_per_g ..][0..blocks_per_g],
            .out = outs[j * hidden ..][0..hidden],
            .profile_enabled = profile_enabled,
            .io = io,
            .remaining_gate_up = .init(2),
            .geglu_requant_ns = 0,
            .down_task0 = down_task0,
        };
        tasks[2 * j] = .{
            .state = state,
            .kind = .gate_up,
            .c0 = 0,
            .c1 = gate_split,
            .elapsed_ns = 0,
        };
        tasks[2 * j + 1] = .{
            .state = state,
            .kind = .gate_up,
            .c0 = gate_split,
            .c1 = out_pe,
            .elapsed_ns = 0,
        };
        tasks[down_task0] = .{
            .state = state,
            .kind = .down,
            .c0 = 0,
            .c1 = down_split,
            .elapsed_ns = 0,
        };
        tasks[down_task0 + 1] = .{
            .state = state,
            .kind = .down,
            .c0 = down_split,
            .c1 = hidden,
            .elapsed_ns = 0,
        };
    }

    const expert_wall_start = moeBatchProfileStart(profile_enabled, io);
    var used_chain = false;
    if (self.workPool()) |pool| {
        used_chain = pool.parallelChained(GemmaMoeDecodeChainTask, tasks, chain_initial_count, runGemmaMoeDecodeChainTask);
    }
    if (!used_chain) {
        for (0..top_k) |j| {
            const e = selected[j];
            var t = GemmaMoeDecodeTask{
                .gate = &gate[e],
                .up = &up[e],
                .down = &down[e],
                .qx = qx,
                .out_pe = out_pe,
                .hidden = hidden,
                .weight = weights[j],
                .gate_buf = gate_buf[j * out_pe ..][0..out_pe],
                .up_buf = up_buf[j * out_pe ..][0..out_pe],
                .g_buf = g_buf[j * out_pe ..][0..out_pe],
                .qg = qg[j * blocks_per_g ..][0..blocks_per_g],
                .out = outs[j * hidden ..][0..hidden],
                .profile_enabled = profile_enabled,
                .io = io,
                .gate_up_ns = 0,
                .geglu_requant_ns = 0,
                .down_ns = 0,
            };
            runGemmaMoeDecodeTask(&t);
            if (profile) |p| {
                p.gate_up_ns += t.gate_up_ns;
                p.swiglu_requant_ns += t.geglu_requant_ns;
                p.down_ns += t.down_ns;
            }
        }
    }
    if (profile) |p| {
        p.expert_wall_ns += moeBatchProfileElapsed(expert_wall_start, io);
        if (used_chain) {
            for (tasks) |*t| switch (t.kind) {
                .gate_up => p.gate_up_ns += t.elapsed_ns,
                .down => p.down_ns += t.elapsed_ns,
            };
            for (sv.states) |*state| p.swiglu_requant_ns += state.geglu_requant_ns;
        }
        p.batches += 1;
        p.pairs += top_k;
        p.active_experts += top_k;
        p.max_expert_m = @max(p.max_expert_m, 1);
    }

    const out_alloc_start = moeBatchProfileStart(profile_enabled, io);
    var out = try self.emptyRank(2, .{ 1, hidden });
    errdefer out.deinit();
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(out_alloc_start, io);

    const scatter_start = moeBatchProfileStart(profile_enabled, io);
    const od = out.data();
    @memset(od, 0);
    for (0..top_k) |j| {
        const src = outs[j * hidden ..][0..hidden];
        for (od, src) |*o, s| o.* += s;
    }
    if (profile) |p| {
        p.scatter_ns += moeBatchProfileElapsed(scatter_start, io);
        p.total_ns += moeBatchProfileElapsed(total_start, io);
    }
    return out;
}

/// Gemma 4 batched-prefill MoE over the existing per-expert widened RHS
/// representation: gate/up are Q6_Kx4, down is Q8_0x4. This mirrors the
/// Qwen phased MoE scheduler, but keeps Gemma's loader format and GeGLU
/// activation. `weights` should already include Gemma's per-expert down scale.
/// (`-Dgpu=metal` builds don't load the x4 representation at all — they go
/// through `batchRaw`, which holds the GPU arm.)
pub fn batchPacked(
    self: *ExecContext,
    x: *const Tensor,
    gate: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.QuantizedMatmulRhsQ8_0x4,
    selected: []const usize,
    weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const a = self.allocator;
    const xv = try x.rankView(2);
    const seq = xv.dim(0);
    const hidden = xv.dim(1);
    const x_data = try x.dataConstChecked();
    const n_pairs = seq * top_k;
    const n_expert = gate.len;
    const profile_enabled = profile != null;
    const total_start = moeBatchProfileStart(profile_enabled, io);

    if (top_k == 0 or selected.len != n_pairs or weights.len != n_pairs) return tensor.TensorError.InvalidDataLength;
    if (up.len != n_expert or down.len != n_expert) return tensor.TensorError.ShapeMismatch;
    for (0..n_expert) |e| {
        if (gate[e].k != hidden or up[e].k != hidden or gate[e].n != out_pe or up[e].n != out_pe) return tensor.TensorError.ShapeMismatch;
        if (down[e].k != out_pe or down[e].n != hidden) return tensor.TensorError.ShapeMismatch;
    }

    const bpc_in = try qm.qkBlockCount(hidden);
    const bpc_g = try qm.q8_0BlockCount(out_pe);

    const route_result = try gemma_moe_route.build(a, selected, n_expert, profile_enabled, io);
    var route = route_result.plan;
    defer route.deinit();
    if (profile) |p| {
        p.alloc_ns += route_result.alloc_ns;
        p.count_sort_ns += route_result.count_sort_ns;
    }
    const count = route.count;
    const offset = route.offset;
    const order = route.order;

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const qx = try a.alloc(qm.BlockQ8_K, n_pairs * bpc_in);
    defer a.free(qx);
    const gate_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(gate_buf);
    const up_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(up_buf);
    const g_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(g_buf);
    const qg = try a.alloc(qm.BlockQ8_0, n_pairs * bpc_g);
    defer a.free(qg);
    const down_buf = try a.alloc(f32, n_pairs * hidden);
    defer a.free(down_buf);
    const gather_tasks = try a.alloc(GemmaMoeGatherTask, n_expert);
    defer a.free(gather_tasks);
    const geglu_tasks = try a.alloc(GemmaMoeGegluTask, n_expert);
    defer a.free(geglu_tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    // Small-m column chunking is a per-layer-call decision: with few active
    // experts each contributing one full-width task per projection, the
    // team starves. The width helpers keep the counting and construction
    // loops in exact agreement (the chain's enqueue contract).
    const pool = self.workPool();
    var chain_active_count: usize = 0;
    for (count) |m| {
        if (m != 0) chain_active_count += 1;
    }
    const workers = if (pool) |p| p.teamSize() else 1;
    const small_m_width = moeSmallMColWidth(chain_active_count, workers);

    var gate_up_task_count: usize = 0;
    var down_task_count: usize = 0;
    for (count) |m| {
        if (m == 0) continue;
        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        gate_up_task_count += 2 * moePhaseChunkCount(gu_width, out_pe);
        const d_width = moePhaseColWidth(m, hidden, small_m_width);
        down_task_count += moePhaseChunkCount(d_width, hidden);
    }
    const task_alloc_start = moeBatchProfileStart(profile_enabled, io);
    const gate_up_tasks = try a.alloc(GemmaMoeQ6MatmulTask, gate_up_task_count);
    defer a.free(gate_up_tasks);
    const down_tasks = try a.alloc(GemmaMoeQ8MatmulTask, down_task_count);
    defer a.free(down_tasks);
    const chain_states = try a.alloc(MoeBatchPhaseChainState, n_expert);
    defer a.free(chain_states);
    const chain_tasks = try a.alloc(MoeBatchPhaseChainTask, chain_active_count * 2 + gate_up_task_count + down_task_count);
    defer a.free(chain_tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(task_alloc_start, io);

    var gate_up_i: usize = 0;
    var down_i: usize = 0;
    for (0..n_expert) |e| {
        const m = count[e];
        const base = offset[e];
        gather_tasks[e] = .{
            .x_data = x_data,
            .order = order,
            .hidden = hidden,
            .top_k = top_k,
            .bpc_in = bpc_in,
            .row_start = base,
            .m = m,
            .qx = qx,
            .profile_enabled = profile_enabled,
            .io = io,
            .elapsed_ns = 0,
        };
        geglu_tasks[e] = .{
            .gate_buf = gate_buf,
            .up_buf = up_buf,
            .g_buf = g_buf,
            .qg = qg,
            .out_pe = out_pe,
            .bpc_g = bpc_g,
            .row_start = base,
            .m = m,
            .profile_enabled = profile_enabled,
            .io = io,
            .elapsed_ns = 0,
        };
        if (m == 0) continue;

        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        const gu_chunks = moePhaseChunkCount(gu_width, out_pe);
        const gate_task_start = gate_up_i;
        for (0..gu_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, gu_width, out_pe);
            gate_up_tasks[gate_up_i] = .{
                .rhs = &gate[e],
                .qlhs = qx,
                .bpc = bpc_in,
                .row_start = base,
                .m = m,
                .out_dim = out_pe,
                .out = gate_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            gate_up_i += 1;
            gate_up_tasks[gate_up_i] = .{
                .rhs = &up[e],
                .qlhs = qx,
                .bpc = bpc_in,
                .row_start = base,
                .m = m,
                .out_dim = out_pe,
                .out = up_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            gate_up_i += 1;
        }

        const d_width = moePhaseColWidth(m, hidden, small_m_width);
        const d_chunks = moePhaseChunkCount(d_width, hidden);
        const down_task_start = down_i;
        for (0..d_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, d_width, hidden);
            down_tasks[down_i] = .{
                .rhs = &down[e],
                .qlhs = qg,
                .bpc = bpc_g,
                .row_start = base,
                .m = m,
                .out_dim = hidden,
                .out = down_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            down_i += 1;
        }
        chain_states[e] = .{
            .gate_start = gate_task_start,
            .gate_count = gate_up_i - gate_task_start,
            .act_index = e,
            .down_start = down_task_start,
            .down_count = down_i - down_task_start,
            .remaining_gate_up = .init(0),
        };
    }

    const chain_initial_count = moe_chain.wireMoeBatchPhaseChain(
        GemmaMoeGatherTask,
        GemmaMoeQ6MatmulTask,
        GemmaMoeGegluTask,
        GemmaMoeQ8MatmulTask,
        chain_tasks,
        chain_states,
        count,
        gather_tasks,
        gate_up_tasks,
        geglu_tasks,
        down_tasks,
        runGemmaMoeGatherTaskOpaque,
        runGemmaMoeQ6MatmulTaskOpaque,
        runGemmaMoeGegluTaskOpaque,
        runGemmaMoeQ8MatmulTaskOpaque,
    );

    var expert_wall_ns: i128 = 0;
    var phase_start = moeBatchProfileStart(profile_enabled, io);
    const used_chain = if (pool) |p| p.parallelChained(MoeBatchPhaseChainTask, chain_tasks, chain_initial_count, runMoeBatchPhaseChainTask) else false;
    if (used_chain) {
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    } else {
        if (pool) |p| {
            p.parallelChunks(GemmaMoeGatherTask, gather_tasks, runGemmaMoeGatherTask);
        } else {
            for (gather_tasks) |*t| runGemmaMoeGatherTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GemmaMoeQ6MatmulTask, gate_up_tasks, runGemmaMoeQ6MatmulTask);
        } else {
            for (gate_up_tasks) |*t| runGemmaMoeQ6MatmulTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GemmaMoeGegluTask, geglu_tasks, runGemmaMoeGegluTask);
        } else {
            for (geglu_tasks) |*t| runGemmaMoeGegluTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GemmaMoeQ8MatmulTask, down_tasks, runGemmaMoeQ8MatmulTask);
        } else {
            for (down_tasks) |*t| runGemmaMoeQ8MatmulTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    }

    if (profile) |p| {
        p.expert_wall_ns += expert_wall_ns;
        for (gather_tasks) |*t| p.gather_quant_ns += t.elapsed_ns;
        for (gate_up_tasks) |*t| p.gate_up_ns += t.elapsed_ns;
        for (geglu_tasks) |*t| p.swiglu_requant_ns += t.elapsed_ns;
        for (down_tasks) |*t| p.down_ns += t.elapsed_ns;
    }

    const out = try gemma_moe_route_tensor.scatterGrouped(self, seq, hidden, top_k, &route, weights, down_buf, io, profile);
    gemma_moe_route_tensor.recordBatch(profile, total_start, io, &route);
    return out;
}

/// GPU arm of the Gemma MoE batch FFN (`-Dgpu=metal`): per layer, ONE
/// grouped dequant-in-kernel Metal GEMM over the fused gate_up experts
/// (Q6_K, gate cols [0, out_pe), up cols [out_pe, 2*out_pe)) and one over
/// the down experts (Q8_0), both straight off the raw GGUF blocks via the
/// wrap cache. The CPU keeps the cheap phases — f32 row gather into the
/// GPU staging panel (no Q8_K LHS quantization), GeGLU between the two
/// dispatches (no Q8_0 requantization), weighted scatter — an Amdahl
/// split: the grouped expert GEMMs dominate the layer's wall time, so
/// offloading only them captures nearly all of the win. Returns null
/// whenever the GPU did not run (threshold, shape, init, dispatch
/// failure): the caller falls through to the untouched CPU path,
/// never-a-loss.
fn batchRawGpu(
    self: *ExecContext,
    x_data: []const f32,
    seq: usize,
    hidden: usize,
    out_pe: usize,
    top_k: usize,
    gw: RawExpertWeights,
    route: *const gemma_moe_route.Plan,
    weights: []const f32,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
    total_start: i128,
) !?Tensor {
    const gpu = backend_mod.gpu_impl;
    const a = self.allocator;
    const count = route.count;
    const offset = route.offset;
    const order = route.order;
    const n_expert = route.expertCount();
    const n_pairs = route.pairCount();
    const gu_out = 2 * out_pe;
    const profile_enabled = profile != null;

    // whole quantized rows: Q6_K/Q4_K need hidden % 256 == 0, Q8_0 needs
    // out_pe % 32 == 0 (the kernel also wants K % 32 and n_out % 4 — both
    // implied here)
    if (hidden % 256 != 0 or out_pe % 32 != 0) return null;
    const bpr_gu = hidden / 256;
    const bpr_dn = out_pe / 32;
    if (gw.guBlockCount() != n_expert * gu_out * bpr_gu) return null;
    if (gw.dn_blocks.len != n_expert * hidden * bpr_dn) return null;
    const gu_format: backend_mod.gpu_impl.QFormat = switch (gw.gu) {
        .q6_k => .q6_k,
        .q4_k => .q4_k,
    };
    const gu_bytes: []const u8 = switch (gw.gu) {
        inline else => |gu_blocks| std.mem.sliceAsBytes(gu_blocks),
    };
    const nb01_gu = bpr_gu * @as(usize, switch (gw.gu) {
        .q6_k => @sizeOf(dtype_mod.BlockQ6_K),
        .q4_k => @sizeOf(dtype_mod.BlockQ4_K),
    });
    const nb01_dn = bpr_dn * @sizeOf(dtype_mod.BlockQ8_0);

    var n_tiles: usize = 0;
    for (count) |m| n_tiles += (m + 31) / 32;
    if (n_tiles == 0) return null;

    // Occupancy gate (measured 2026-07-03): the grouped kernel's per-tile GPU
    // cost is fill-independent (~45-53 µs/tile at 12% and at 100% fill —
    // weight dequant dominates), so GPU value scales with tile occupancy
    // n_pairs/(n_tiles*32), not with the nominal m·n·k the work gate sees.
    // Measured on gemma-4-26B Q6_K prefill: occupancy 15-30% (pp32-pp64)
    // loses 0.4-0.7x vs the raw CPU path, ~50% (pp128) is breakeven-to-loss,
    // 62-80% (pp256) wins ~2x.
    if (!gpu.qmoeFillAcceptable(n_pairs, n_tiles)) return null;

    // total m·n·k across both grouped GEMMs of this layer
    const work = @as(u64, n_pairs) * @as(u64, hidden) * @as(u64, gu_out + out_pe);
    if (!gpu.shouldUseGpuQMoe(work)) return null;

    // Accumulated locally and merged only on success: a mid-sequence GPU
    // refusal falls back to the CPU path, which records its own full
    // pass — committing partial GPU phases would double-count the batch.
    var local: MoeBatchProfile = .{};
    var expert_wall_ns: i128 = 0;

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const tiles = try a.alloc(gpu.QMMTile, n_tiles);
    defer a.free(tiles);
    const gather_tasks = try a.alloc(GemmaMoeGpuGatherTask, n_expert);
    defer a.free(gather_tasks);
    const geglu_tasks = try a.alloc(GemmaMoeGpuGegluTask, n_expert);
    defer a.free(geglu_tasks);
    if (profile_enabled) local.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    var ti: usize = 0;
    for (0..n_expert) |e| {
        const m = count[e];
        if (m == 0) continue;
        var t: usize = 0;
        while (t * 32 < m) : (t += 1) {
            tiles[ti] = .{
                .expert = @intCast(e),
                .base_row = @intCast(offset[e]),
                .m = @intCast(m),
                .tile_m = @intCast(t),
            };
            ti += 1;
        }
    }

    // The shim reuses one staging pair across calls: hold the lock for
    // the whole gather/dispatch/geglu/dispatch/scatter sequence. The in
    // panel holds the gathered rows (hidden wide) and is then reused for
    // the gated rows (out_pe wide) — size for whichever is larger.
    gpu.qmoe_lock.lock();
    defer gpu.qmoe_lock.unlock();
    const stage = gpu.qmoeStage(
        n_pairs * @max(hidden, out_pe) * @sizeOf(f32),
        n_pairs * @max(gu_out, hidden) * @sizeOf(f32),
    ) orelse return null;

    const pool = self.workPool();

    // gather the routed f32 rows into the staging panel — the kernel
    // reads f32 activations directly, so the CPU path's Q8_K LHS
    // quantization disappears here
    var phase_start = moeBatchProfileStart(profile_enabled, io);
    for (gather_tasks, 0..) |*t, e| {
        t.* = .{
            .x_data = x_data,
            .order = order,
            .hidden = hidden,
            .top_k = top_k,
            .row_start = offset[e],
            .m = count[e],
            .dst = stage.in[0 .. n_pairs * hidden],
        };
    }
    if (pool) |p| {
        p.parallelChunks(GemmaMoeGpuGatherTask, gather_tasks, runGemmaMoeGpuGatherTask);
    } else {
        for (gather_tasks) |*t| runGemmaMoeGpuGatherTask(t);
    }
    if (profile_enabled) {
        const ns = moeBatchProfileElapsed(phase_start, io);
        local.gather_quant_ns += ns;
        expert_wall_ns += ns;
    }

    const cacheable = gw.device_owned;

    phase_start = moeBatchProfileStart(profile_enabled, io);
    if (!gpu.gemmQGroupedNt(
        gu_format,
        gu_bytes,
        cacheable, // only shim-owned storage may enter the wrap cache
        nb01_gu,
        gu_out * nb01_gu,
        gu_out,
        hidden,
        tiles,
    )) return null;
    if (profile_enabled) {
        const ns = moeBatchProfileElapsed(phase_start, io);
        local.gate_up_ns += ns;
        expert_wall_ns += ns;
    }

    // GeGLU on the fused gate_up panel, written back over the staging
    // input as the down operand (f32 — no Q8_0 requantization)
    phase_start = moeBatchProfileStart(profile_enabled, io);
    for (geglu_tasks, 0..) |*t, e| {
        t.* = .{
            .src = stage.out[0 .. n_pairs * gu_out],
            .dst = stage.in[0 .. n_pairs * out_pe],
            .out_pe = out_pe,
            .row_start = offset[e],
            .m = count[e],
        };
    }
    if (pool) |p| {
        p.parallelChunks(GemmaMoeGpuGegluTask, geglu_tasks, runGemmaMoeGpuGegluTask);
    } else {
        for (geglu_tasks) |*t| runGemmaMoeGpuGegluTask(t);
    }
    if (profile_enabled) {
        const ns = moeBatchProfileElapsed(phase_start, io);
        local.swiglu_requant_ns += ns;
        expert_wall_ns += ns;
    }

    phase_start = moeBatchProfileStart(profile_enabled, io);
    if (!gpu.gemmQGroupedNt(
        .q8_0,
        std.mem.sliceAsBytes(gw.dn_blocks),
        cacheable,
        nb01_dn,
        hidden * nb01_dn,
        hidden,
        out_pe,
        tiles,
    )) return null;
    if (profile_enabled) {
        const ns = moeBatchProfileElapsed(phase_start, io);
        local.down_ns += ns;
        expert_wall_ns += ns;
    }

    const down_panel = stage.out[0 .. n_pairs * hidden];
    const scatter_profile: ?*MoeBatchProfile = if (profile_enabled) &local else null;
    const out = try gemma_moe_route_tensor.scatterGrouped(self, seq, hidden, top_k, route, weights, down_panel, io, scatter_profile);
    if (profile) |p| {
        p.alloc_ns += local.alloc_ns;
        p.gather_quant_ns += local.gather_quant_ns;
        p.gate_up_ns += local.gate_up_ns;
        p.swiglu_requant_ns += local.swiglu_requant_ns;
        p.down_ns += local.down_ns;
        p.scatter_ns += local.scatter_ns;
        p.expert_wall_ns += expert_wall_ns;
    }
    gemma_moe_route_tensor.recordBatch(profile, total_start, io, route);
    return out;
}

const GemmaMoeGpuGatherTask = struct {
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    top_k: usize,
    row_start: usize,
    m: usize,
    dst: []f32, // staging panel [n_pairs * hidden]
};

fn runGemmaMoeGpuGatherTask(task: *const GemmaMoeGpuGatherTask) void {
    const m = task.m;
    if (m == 0) return;
    const base = task.row_start;
    for (0..m) |i| {
        const token = task.order[base + i] / task.top_k;
        @memcpy(
            task.dst[(base + i) * task.hidden ..][0..task.hidden],
            task.x_data[token * task.hidden ..][0..task.hidden],
        );
    }
}

const GemmaMoeGpuGegluTask = struct {
    src: []const f32, // gate_up GEMM panel [n_pairs * 2*out_pe]
    dst: []f32, // gated rows [n_pairs * out_pe]
    out_pe: usize,
    row_start: usize,
    m: usize,
};

fn runGemmaMoeGpuGegluTask(task: *const GemmaMoeGpuGegluTask) void {
    const m = task.m;
    if (m == 0) return;
    const base = task.row_start;
    const out_pe = task.out_pe;
    for (0..m) |i| {
        const row = task.src[(base + i) * 2 * out_pe ..][0 .. 2 * out_pe];
        const g = task.dst[(base + i) * out_pe ..][0..out_pe];
        for (g, row[0..out_pe], row[out_pe..]) |*gv, gate_v, up_v| {
            gv.* = up_v * backend_ops.geluQuantScalar(gate_v);
        }
    }
}

/// Borrowed plain RHS view (one arm per gate_up dtype) over one half
/// (gate: `row_off` 0, up: `row_off` out_pe) of one expert in the raw
/// GGUF gate_up tensor (2*out_pe rows per expert, gate rows first).
const GemmaMoeRawGuRhs = union(enum) {
    q6_k: backend_mod.QuantizedMatmulRhsQ6_K,
    q4_k: backend_mod.QuantizedMatmulRhsQ4_K,
};

fn gemmaMoeRawGuView(
    gw: RawExpertWeights,
    expert: usize,
    row_off: usize,
    out_pe: usize,
    hidden: usize,
) GemmaMoeRawGuRhs {
    const bpr = hidden / 256;
    const start = (expert * 2 * out_pe + row_off) * bpr;
    return switch (gw.gu) {
        .q6_k => |gu_blocks| .{
            .q6_k = .{
                .allocator = null, // borrows the model's resident expert copies
                .blocks = gu_blocks[start..][0 .. out_pe * bpr],
                .k = hidden,
                .n = out_pe,
                .blocks_per_column = bpr,
            },
        },
        .q4_k => |gu_blocks| .{
            .q4_k = .{
                .allocator = null, // borrows the model's resident expert copies
                .blocks = gu_blocks[start..][0 .. out_pe * bpr],
                .k = hidden,
                .n = out_pe,
                .blocks_per_column = bpr,
            },
        },
    };
}

/// One gate_up matmul over a raw view: each arm forks to its compact
/// column-outer kernel at m >= 4 (the batched-prefill case — unpack each
/// weight block once per row tile) and stays on the row-outer tile below.
fn gemmaMoeRawGuMatmul(
    view: *const GemmaMoeRawGuRhs,
    out: []f32,
    qlhs: []const backend_mod.quantized_matmul.BlockQ8_K,
    out_dim: usize,
    m: usize,
    c0: usize,
    c1: usize,
) void {
    const qm = backend_mod.quantized_matmul;
    switch (view.*) {
        .q6_k => |*v| if (m >= 4) {
            qm.matmulQ6_KRhsCompactColOuter(out, qlhs, v, out_dim, 0, m, c0, c1);
        } else {
            qm.matmulQ6_KRhsTile(out, qlhs, v, out_dim, 0, m, c0, c1);
        },
        .q4_k => |*v| if (m >= 4) {
            qm.matmulQ4_KRhsCompactColOuter(out, qlhs, v, out_dim, 0, m, c0, c1);
        } else {
            qm.matmulQ4_KRhsTile(out, qlhs, v, out_dim, 0, m, c0, c1);
        },
    }
}

/// Borrowed plain-Q8_0 RHS view over one expert's raw GGUF down blocks:
/// the blocks belong to the model's resident expert copies, so the rows
/// table carries a null allocator (deinit frees nothing).
fn gemmaMoeRawQ8View(
    gw: RawExpertWeights,
    expert: usize,
    out_pe: usize,
    hidden: usize,
) backend_mod.QuantizedMatmulRhsQ8_0 {
    const bpr = out_pe / 32;
    return .{
        .rows = .{
            .allocator = null,
            .blocks = gw.dn_blocks[expert * hidden * bpr ..][0 .. hidden * bpr],
            .rows = hidden,
            .cols = out_pe,
            .blocks_per_row = bpr,
        },
        .k = out_pe,
        .n = hidden,
    };
}

const GemmaMoeRawDecodeTask = struct {
    gw: RawExpertWeights,
    qx: []const backend_mod.quantized_matmul.BlockQ8_K,
    out_pe: usize,
    hidden: usize,
    expert_index: usize,
    weight: f32,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_0,
    out: []f32,
    profile_enabled: bool,
    io: ?std.Io,
    gate_up_ns: i64,
    geglu_requant_ns: i64,
    down_ns: i64,
};

fn runGemmaMoeRawDecodeTask(task: *const GemmaMoeRawDecodeTask) void {
    const qm = backend_mod.quantized_matmul;
    const task_profile = @constCast(task);
    const e = task.expert_index;
    const out_pe = task.out_pe;

    const gate_up_start = moeBatchProfileStart(task.profile_enabled, task.io);
    const gate_view = gemmaMoeRawGuView(task.gw, e, 0, out_pe, task.hidden);
    gemmaMoeRawGuMatmul(&gate_view, task.gate_buf, task.qx, out_pe, 1, 0, out_pe);
    const up_view = gemmaMoeRawGuView(task.gw, e, out_pe, out_pe, task.hidden);
    gemmaMoeRawGuMatmul(&up_view, task.up_buf, task.qx, out_pe, 1, 0, out_pe);
    if (task.profile_enabled) task_profile.gate_up_ns += moeBatchProfileElapsed(gate_up_start, task.io);

    const geglu_requant_start = moeBatchProfileStart(task.profile_enabled, task.io);
    for (task.g_buf, task.gate_buf, task.up_buf) |*g, gate_v, up_v| {
        g.* = up_v * backend_ops.geluQuantScalar(gate_v);
    }
    qm.quantizeRowQ8_0Into(task.qg, task.g_buf) catch {
        @memset(task.out, 0);
        return;
    };
    if (task.profile_enabled) task_profile.geglu_requant_ns += moeBatchProfileElapsed(geglu_requant_start, task.io);

    const down_start = moeBatchProfileStart(task.profile_enabled, task.io);
    const down_view = gemmaMoeRawQ8View(task.gw, e, out_pe, task.hidden);
    qm.matmulQ8_0RhsTile(task.out, task.qg, &down_view, task.hidden, 0, 1, 0, task.hidden);
    if (task.weight != 1.0) {
        for (task.out) |*o| o.* *= task.weight;
    }
    if (task.profile_enabled) task_profile.down_ns += moeBatchProfileElapsed(down_start, task.io);
}

const GemmaMoeRawDecodeChainState = struct {
    gw: RawExpertWeights,
    qx: []const backend_mod.quantized_matmul.BlockQ8_K,
    out_pe: usize,
    hidden: usize,
    expert_index: usize,
    weight: f32,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_0,
    out: []f32,
    profile_enabled: bool,
    io: ?std.Io,
    remaining_gate_up: std.atomic.Value(u32),
    geglu_requant_ns: i64,
    down_task0: usize,
};

const GemmaMoeRawDecodeChainTask = struct {
    state: *GemmaMoeRawDecodeChainState,
    kind: enum { gate_up, down },
    c0: usize,
    c1: usize,
    elapsed_ns: i64,
};

fn runGemmaMoeRawDecodeChainTask(task: *GemmaMoeRawDecodeChainTask, chain: *const thread.Chain) void {
    const qm = backend_mod.quantized_matmul;
    const state = task.state;
    const e = state.expert_index;
    const out_pe = state.out_pe;
    switch (task.kind) {
        .gate_up => {
            const gate_up_start = moeBatchProfileStart(state.profile_enabled, state.io);
            const gate_view = gemmaMoeRawGuView(state.gw, e, 0, out_pe, state.hidden);
            gemmaMoeRawGuMatmul(&gate_view, state.gate_buf, state.qx, out_pe, 1, task.c0, task.c1);
            const up_view = gemmaMoeRawGuView(state.gw, e, out_pe, out_pe, state.hidden);
            gemmaMoeRawGuMatmul(&up_view, state.up_buf, state.qx, out_pe, 1, task.c0, task.c1);
            if (state.profile_enabled) task.elapsed_ns += moeBatchProfileElapsed(gate_up_start, state.io);

            if (state.remaining_gate_up.fetchSub(1, .acq_rel) == 1) {
                const geglu_requant_start = moeBatchProfileStart(state.profile_enabled, state.io);
                for (state.g_buf, state.gate_buf, state.up_buf) |*g, gate_v, up_v| {
                    g.* = up_v * backend_ops.geluQuantScalar(gate_v);
                }
                qm.quantizeRowQ8_0Into(state.qg, state.g_buf) catch unreachable;
                if (state.profile_enabled) state.geglu_requant_ns += moeBatchProfileElapsed(geglu_requant_start, state.io);
                chain.enqueue(state.down_task0);
                chain.enqueue(state.down_task0 + 1);
            }
        },
        .down => {
            const down_start = moeBatchProfileStart(state.profile_enabled, state.io);
            const down_view = gemmaMoeRawQ8View(state.gw, e, out_pe, state.hidden);
            qm.matmulQ8_0RhsTile(state.out, state.qg, &down_view, state.hidden, 0, 1, task.c0, task.c1);
            if (state.weight != 1.0) {
                for (state.out[task.c0..task.c1]) |*o| o.* *= state.weight;
            }
            if (state.profile_enabled) task.elapsed_ns += moeBatchProfileElapsed(down_start, state.io);
        },
    }
}

/// Gemma 4 fused single-token MoE over the RAW GGUF expert blocks
/// (`-Dgpu=metal` builds, which skip the x4 widening to keep a single
/// expert representation in memory, and CPU builds with Q4_K gate_up
/// experts, which have no x4 packing). Identical structure and numerics
/// to `decodePacked` — Q8_K-quantized input, ggml f16-LUT
/// GeGLU, Q8_0-requantized down input — only the weight layout differs
/// (plain row blocks straight from the mmap instead of the widened x4
/// packing).
pub fn decodeRaw(
    self: *ExecContext,
    x: *const Tensor,
    gw: RawExpertWeights,
    n_expert: usize,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const av = try x.rankView(2);
    if (av.dim(0) != 1) return tensor.TensorError.ShapeMismatch;
    const hidden = av.dim(1);
    const top_k = selected.len;
    const profile_enabled = profile != null;
    const total_start = moeBatchProfileStart(profile_enabled, io);

    if (top_k == 0 or weights.len != top_k) return tensor.TensorError.InvalidDataLength;
    if (hidden % 256 != 0 or out_pe % 32 != 0) return tensor.TensorError.InvalidShape;
    if (gw.guBlockCount() != n_expert * 2 * out_pe * (hidden / 256)) return tensor.TensorError.ShapeMismatch;
    if (gw.dn_blocks.len != n_expert * hidden * (out_pe / 32)) return tensor.TensorError.ShapeMismatch;
    for (selected) |e| if (e >= n_expert) return tensor.TensorError.IndexOutOfBounds;

    const blocks_per_g = try qm.q8_0BlockCount(out_pe);
    const chain_task_count = 4 * top_k;
    const chain_initial_count = 2 * top_k;
    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    self.lockMoeDecodeScratch();
    defer self.unlockMoeDecodeScratch();
    const sv = try self.carveMoeDecodeChainScratch(qm.BlockQ8_0, GemmaMoeRawDecodeChainState, GemmaMoeRawDecodeChainTask, try qm.qkBlockCount(hidden), top_k, out_pe, hidden, blocks_per_g, chain_task_count);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    const gather_quant_start = moeBatchProfileStart(profile_enabled, io);
    try qm.quantizeRowQ8_KInto(sv.qx, try x.dataConstChecked());
    if (profile) |p| p.gather_quant_ns += moeBatchProfileElapsed(gather_quant_start, io);

    const gate_split = moeDecodeColumnSplit(out_pe, 32);
    const down_split = moeDecodeColumnSplit(hidden, 32);
    for (sv.states, 0..) |*state, j| {
        const down_task0 = chain_initial_count + 2 * j;
        state.* = .{
            .gw = gw,
            .qx = sv.qx,
            .out_pe = out_pe,
            .hidden = hidden,
            .expert_index = selected[j],
            .weight = weights[j],
            .gate_buf = sv.gate_buf[j * out_pe ..][0..out_pe],
            .up_buf = sv.up_buf[j * out_pe ..][0..out_pe],
            .g_buf = sv.g_buf[j * out_pe ..][0..out_pe],
            .qg = sv.qg[j * blocks_per_g ..][0..blocks_per_g],
            .out = sv.outs[j * hidden ..][0..hidden],
            .profile_enabled = profile_enabled,
            .io = io,
            .remaining_gate_up = .init(2),
            .geglu_requant_ns = 0,
            .down_task0 = down_task0,
        };
        sv.tasks[2 * j] = .{
            .state = state,
            .kind = .gate_up,
            .c0 = 0,
            .c1 = gate_split,
            .elapsed_ns = 0,
        };
        sv.tasks[2 * j + 1] = .{
            .state = state,
            .kind = .gate_up,
            .c0 = gate_split,
            .c1 = out_pe,
            .elapsed_ns = 0,
        };
        sv.tasks[down_task0] = .{
            .state = state,
            .kind = .down,
            .c0 = 0,
            .c1 = down_split,
            .elapsed_ns = 0,
        };
        sv.tasks[down_task0 + 1] = .{
            .state = state,
            .kind = .down,
            .c0 = down_split,
            .c1 = hidden,
            .elapsed_ns = 0,
        };
    }

    const expert_wall_start = moeBatchProfileStart(profile_enabled, io);
    var used_chain = false;
    if (self.workPool()) |pool| {
        used_chain = pool.parallelChained(GemmaMoeRawDecodeChainTask, sv.tasks, chain_initial_count, runGemmaMoeRawDecodeChainTask);
    }
    if (!used_chain) {
        for (0..top_k) |j| {
            var t = GemmaMoeRawDecodeTask{
                .gw = gw,
                .qx = sv.qx,
                .out_pe = out_pe,
                .hidden = hidden,
                .expert_index = selected[j],
                .weight = weights[j],
                .gate_buf = sv.gate_buf[j * out_pe ..][0..out_pe],
                .up_buf = sv.up_buf[j * out_pe ..][0..out_pe],
                .g_buf = sv.g_buf[j * out_pe ..][0..out_pe],
                .qg = sv.qg[j * blocks_per_g ..][0..blocks_per_g],
                .out = sv.outs[j * hidden ..][0..hidden],
                .profile_enabled = profile_enabled,
                .io = io,
                .gate_up_ns = 0,
                .geglu_requant_ns = 0,
                .down_ns = 0,
            };
            runGemmaMoeRawDecodeTask(&t);
            if (profile) |p| {
                p.gate_up_ns += t.gate_up_ns;
                p.swiglu_requant_ns += t.geglu_requant_ns;
                p.down_ns += t.down_ns;
            }
        }
    }
    if (profile) |p| {
        p.expert_wall_ns += moeBatchProfileElapsed(expert_wall_start, io);
        if (used_chain) {
            for (sv.tasks) |*t| switch (t.kind) {
                .gate_up => p.gate_up_ns += t.elapsed_ns,
                .down => p.down_ns += t.elapsed_ns,
            };
            for (sv.states) |*state| p.swiglu_requant_ns += state.geglu_requant_ns;
        }
        p.batches += 1;
        p.pairs += top_k;
        p.active_experts += top_k;
        p.max_expert_m = @max(p.max_expert_m, 1);
    }

    const out_alloc_start = moeBatchProfileStart(profile_enabled, io);
    var out = try self.emptyRank(2, .{ 1, hidden });
    errdefer out.deinit();
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(out_alloc_start, io);

    const scatter_start = moeBatchProfileStart(profile_enabled, io);
    const od = out.data();
    @memset(od, 0);
    for (0..top_k) |j| {
        const src = sv.outs[j * hidden ..][0..hidden];
        for (od, src) |*o, s| o.* += s;
    }
    if (profile) |p| {
        p.scatter_ns += moeBatchProfileElapsed(scatter_start, io);
        p.total_ns += moeBatchProfileElapsed(total_start, io);
    }
    return out;
}

const GemmaMoeRawGuMatmulTask = struct {
    view: GemmaMoeRawGuRhs,
    qlhs: []const backend_mod.quantized_matmul.BlockQ8_K,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runGemmaMoeRawGuMatmulTask(task: *const GemmaMoeRawGuMatmulTask) void {
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    gemmaMoeRawGuMatmul(&task.view, out, q, task.out_dim, m, task.c0, task.c1);
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runGemmaMoeRawGuMatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GemmaMoeRawGuMatmulTask = @ptrCast(@alignCast(ctx));
    runGemmaMoeRawGuMatmulTask(task);
}

const GemmaMoeRawQ8MatmulTask = struct {
    view: backend_mod.QuantizedMatmulRhsQ8_0,
    qlhs: []const backend_mod.quantized_matmul.BlockQ8_0,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runGemmaMoeRawQ8MatmulTask(task: *const GemmaMoeRawQ8MatmulTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    qm.matmulQ8_0RhsTile(out, q, &task.view, task.out_dim, 0, m, task.c0, task.c1);
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runGemmaMoeRawQ8MatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GemmaMoeRawQ8MatmulTask = @ptrCast(@alignCast(ctx));
    runGemmaMoeRawQ8MatmulTask(task);
}

/// Gemma 4 batched MoE over the RAW GGUF expert blocks (`-Dgpu=metal`
/// builds, and CPU builds with Q4_K gate_up experts): tries the grouped
/// dequant-in-kernel Metal GEMM first (`batchRawGpu`, gpu
/// builds only), then falls back to a CPU path with the same phase
/// structure as `batchPacked` but plain-block kernels over
/// borrowed views (no x4 widening exists for these arms).
/// Numerics match the x4 path: Q8_K LHS, f16-LUT GeGLU, Q8_0 requant.
pub fn batchRaw(
    self: *ExecContext,
    x: *const Tensor,
    gw: RawExpertWeights,
    n_expert: usize,
    selected: []const usize,
    weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const a = self.allocator;
    const xv = try x.rankView(2);
    const seq = xv.dim(0);
    const hidden = xv.dim(1);
    const x_data = try x.dataConstChecked();
    const n_pairs = seq * top_k;
    const profile_enabled = profile != null;
    const total_start = moeBatchProfileStart(profile_enabled, io);

    if (top_k == 0 or selected.len != n_pairs or weights.len != n_pairs) return tensor.TensorError.InvalidDataLength;
    if (hidden % 256 != 0 or out_pe % 32 != 0) return tensor.TensorError.InvalidShape;
    if (gw.guBlockCount() != n_expert * 2 * out_pe * (hidden / 256)) return tensor.TensorError.ShapeMismatch;
    if (gw.dn_blocks.len != n_expert * hidden * (out_pe / 32)) return tensor.TensorError.ShapeMismatch;

    const bpc_in = try qm.qkBlockCount(hidden);
    const bpc_g = try qm.q8_0BlockCount(out_pe);

    const route_result = try gemma_moe_route.build(a, selected, n_expert, profile_enabled, io);
    var route = route_result.plan;
    defer route.deinit();
    if (profile) |p| {
        p.alloc_ns += route_result.alloc_ns;
        p.count_sort_ns += route_result.count_sort_ns;
    }
    const count = route.count;
    const offset = route.offset;
    const order = route.order;

    if (comptime backend_mod.gpu_impl.enabled) {
        if (try batchRawGpu(
            self,
            x_data,
            seq,
            hidden,
            out_pe,
            top_k,
            gw,
            &route,
            weights,
            io,
            profile,
            total_start,
        )) |out| return out;
    }

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const qx = try a.alloc(qm.BlockQ8_K, n_pairs * bpc_in);
    defer a.free(qx);
    const gate_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(gate_buf);
    const up_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(up_buf);
    const g_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(g_buf);
    const qg = try a.alloc(qm.BlockQ8_0, n_pairs * bpc_g);
    defer a.free(qg);
    const down_buf = try a.alloc(f32, n_pairs * hidden);
    defer a.free(down_buf);
    const gather_tasks = try a.alloc(GemmaMoeGatherTask, n_expert);
    defer a.free(gather_tasks);
    const geglu_tasks = try a.alloc(GemmaMoeGegluTask, n_expert);
    defer a.free(geglu_tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    // Small-m column chunking is a per-layer-call decision: with few active
    // experts each contributing one full-width task per projection, the
    // team starves. The width helpers keep the counting and construction
    // loops in exact agreement (the chain's enqueue contract).
    const pool = self.workPool();
    var chain_active_count: usize = 0;
    for (count) |m| {
        if (m != 0) chain_active_count += 1;
    }
    const workers = if (pool) |p| p.teamSize() else 1;
    const small_m_width = moeSmallMColWidth(chain_active_count, workers);

    var gate_up_task_count: usize = 0;
    var down_task_count: usize = 0;
    for (count) |m| {
        if (m == 0) continue;
        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        gate_up_task_count += 2 * moePhaseChunkCount(gu_width, out_pe);
        const d_width = moePhaseColWidth(m, hidden, small_m_width);
        down_task_count += moePhaseChunkCount(d_width, hidden);
    }
    const task_alloc_start = moeBatchProfileStart(profile_enabled, io);
    const gate_up_tasks = try a.alloc(GemmaMoeRawGuMatmulTask, gate_up_task_count);
    defer a.free(gate_up_tasks);
    const down_tasks = try a.alloc(GemmaMoeRawQ8MatmulTask, down_task_count);
    defer a.free(down_tasks);
    const chain_states = try a.alloc(MoeBatchPhaseChainState, n_expert);
    defer a.free(chain_states);
    const chain_tasks = try a.alloc(MoeBatchPhaseChainTask, chain_active_count * 2 + gate_up_task_count + down_task_count);
    defer a.free(chain_tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(task_alloc_start, io);

    var gate_up_i: usize = 0;
    var down_i: usize = 0;
    for (0..n_expert) |e| {
        const m = count[e];
        const base = offset[e];
        gather_tasks[e] = .{
            .x_data = x_data,
            .order = order,
            .hidden = hidden,
            .top_k = top_k,
            .bpc_in = bpc_in,
            .row_start = base,
            .m = m,
            .qx = qx,
            .profile_enabled = profile_enabled,
            .io = io,
            .elapsed_ns = 0,
        };
        geglu_tasks[e] = .{
            .gate_buf = gate_buf,
            .up_buf = up_buf,
            .g_buf = g_buf,
            .qg = qg,
            .out_pe = out_pe,
            .bpc_g = bpc_g,
            .row_start = base,
            .m = m,
            .profile_enabled = profile_enabled,
            .io = io,
            .elapsed_ns = 0,
        };
        if (m == 0) continue;

        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        const gu_chunks = moePhaseChunkCount(gu_width, out_pe);
        const gate_task_start = gate_up_i;
        for (0..gu_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, gu_width, out_pe);
            gate_up_tasks[gate_up_i] = .{
                .view = gemmaMoeRawGuView(gw, e, 0, out_pe, hidden),
                .qlhs = qx,
                .bpc = bpc_in,
                .row_start = base,
                .m = m,
                .out_dim = out_pe,
                .out = gate_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            gate_up_i += 1;
            gate_up_tasks[gate_up_i] = .{
                .view = gemmaMoeRawGuView(gw, e, out_pe, out_pe, hidden),
                .qlhs = qx,
                .bpc = bpc_in,
                .row_start = base,
                .m = m,
                .out_dim = out_pe,
                .out = up_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            gate_up_i += 1;
        }

        const d_width = moePhaseColWidth(m, hidden, small_m_width);
        const d_chunks = moePhaseChunkCount(d_width, hidden);
        const down_task_start = down_i;
        for (0..d_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, d_width, hidden);
            down_tasks[down_i] = .{
                .view = gemmaMoeRawQ8View(gw, e, out_pe, hidden),
                .qlhs = qg,
                .bpc = bpc_g,
                .row_start = base,
                .m = m,
                .out_dim = hidden,
                .out = down_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            down_i += 1;
        }
        chain_states[e] = .{
            .gate_start = gate_task_start,
            .gate_count = gate_up_i - gate_task_start,
            .act_index = e,
            .down_start = down_task_start,
            .down_count = down_i - down_task_start,
            .remaining_gate_up = .init(0),
        };
    }

    const chain_initial_count = moe_chain.wireMoeBatchPhaseChain(
        GemmaMoeGatherTask,
        GemmaMoeRawGuMatmulTask,
        GemmaMoeGegluTask,
        GemmaMoeRawQ8MatmulTask,
        chain_tasks,
        chain_states,
        count,
        gather_tasks,
        gate_up_tasks,
        geglu_tasks,
        down_tasks,
        runGemmaMoeGatherTaskOpaque,
        runGemmaMoeRawGuMatmulTaskOpaque,
        runGemmaMoeGegluTaskOpaque,
        runGemmaMoeRawQ8MatmulTaskOpaque,
    );

    var expert_wall_ns: i128 = 0;
    var phase_start = moeBatchProfileStart(profile_enabled, io);
    const used_chain = if (pool) |p| p.parallelChained(MoeBatchPhaseChainTask, chain_tasks, chain_initial_count, runMoeBatchPhaseChainTask) else false;
    if (used_chain) {
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    } else {
        if (pool) |p| {
            p.parallelChunks(GemmaMoeGatherTask, gather_tasks, runGemmaMoeGatherTask);
        } else {
            for (gather_tasks) |*t| runGemmaMoeGatherTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GemmaMoeRawGuMatmulTask, gate_up_tasks, runGemmaMoeRawGuMatmulTask);
        } else {
            for (gate_up_tasks) |*t| runGemmaMoeRawGuMatmulTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GemmaMoeGegluTask, geglu_tasks, runGemmaMoeGegluTask);
        } else {
            for (geglu_tasks) |*t| runGemmaMoeGegluTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GemmaMoeRawQ8MatmulTask, down_tasks, runGemmaMoeRawQ8MatmulTask);
        } else {
            for (down_tasks) |*t| runGemmaMoeRawQ8MatmulTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    }

    if (profile) |p| {
        p.expert_wall_ns += expert_wall_ns;
        for (gather_tasks) |*t| p.gather_quant_ns += t.elapsed_ns;
        for (gate_up_tasks) |*t| p.gate_up_ns += t.elapsed_ns;
        for (geglu_tasks) |*t| p.swiglu_requant_ns += t.elapsed_ns;
        for (down_tasks) |*t| p.down_ns += t.elapsed_ns;
    }

    const out = try gemma_moe_route_tensor.scatterGrouped(self, seq, hidden, top_k, &route, weights, down_buf, io, profile);
    gemma_moe_route_tensor.recordBatch(profile, total_start, io, &route);
    return out;
}

const GemmaMoeGatherTask = struct {
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    top_k: usize,
    bpc_in: usize,
    row_start: usize,
    m: usize,
    qx: []backend_mod.quantized_matmul.BlockQ8_K,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runGemmaMoeGatherTask(task: *const GemmaMoeGatherTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    for (0..m) |i| {
        const token = task.order[base + i] / task.top_k;
        const src = task.x_data[token * task.hidden ..][0..task.hidden];
        qm.quantizeRowQ8_KInto(task.qx[(base + i) * task.bpc_in ..][0..task.bpc_in], src) catch unreachable;
    }
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runGemmaMoeGatherTaskOpaque(ctx: *anyopaque) void {
    const task: *const GemmaMoeGatherTask = @ptrCast(@alignCast(ctx));
    runGemmaMoeGatherTask(task);
}

const GemmaMoeQ6MatmulTask = struct {
    rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
    qlhs: []const backend_mod.quantized_matmul.BlockQ8_K,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runGemmaMoeQ6MatmulTask(task: *const GemmaMoeQ6MatmulTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    qm.matmulQ6_Kx4RhsTile(out, q, task.rhs, task.out_dim, 0, m, task.c0, task.c1);
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runGemmaMoeQ6MatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GemmaMoeQ6MatmulTask = @ptrCast(@alignCast(ctx));
    runGemmaMoeQ6MatmulTask(task);
}

const GemmaMoeGegluTask = struct {
    gate_buf: []const f32,
    up_buf: []const f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_0,
    out_pe: usize,
    bpc_g: usize,
    row_start: usize,
    m: usize,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runGemmaMoeGegluTask(task: *const GemmaMoeGegluTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    const out_pe = task.out_pe;
    const gate_out = task.gate_buf[base * out_pe ..][0 .. m * out_pe];
    const up_out = task.up_buf[base * out_pe ..][0 .. m * out_pe];
    const g_out = task.g_buf[base * out_pe ..][0 .. m * out_pe];
    for (g_out, gate_out, up_out) |*g, gate_v, up_v| {
        g.* = up_v * backend_ops.geluQuantScalar(gate_v);
    }
    for (0..m) |i| {
        qm.quantizeRowQ8_0Into(task.qg[(base + i) * task.bpc_g ..][0..task.bpc_g], g_out[i * out_pe ..][0..out_pe]) catch unreachable;
    }
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runGemmaMoeGegluTaskOpaque(ctx: *anyopaque) void {
    const task: *const GemmaMoeGegluTask = @ptrCast(@alignCast(ctx));
    runGemmaMoeGegluTask(task);
}

const GemmaMoeQ8MatmulTask = struct {
    rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
    qlhs: []const backend_mod.quantized_matmul.BlockQ8_0,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runGemmaMoeQ8MatmulTask(task: *const GemmaMoeQ8MatmulTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    qm.matmulQ8_0x4RhsTile(out, q, task.rhs, task.out_dim, 0, m, task.c0, task.c1);
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runGemmaMoeQ8MatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GemmaMoeQ8MatmulTask = @ptrCast(@alignCast(ctx));
    runGemmaMoeQ8MatmulTask(task);
}

test {
    _ = @import("moe_tests.zig");
}
