//! Fused gate|up MoE expert kernels over the raw GGUF stack layout
//! (per expert: gate rows then up rows of Q6_K/Q4_K blocks, Q8_0 down
//! rows), plus the packed x4 arms and the GPU batch path. Gemma is the
//! one family with this weight shape, so the kernels live with it; the
//! tagged facade wrappers are in `moe.zig` next door. Decode carves the
//! runtime's decode scratch through the MoE band's view seam
//! (`fucina.moe.carveDecodeChainScratch`); batch rides the phase-chain
//! scheduling of `fucina.moe.chain` (the shared batched-MoE scaffolding
//! stays in the moe band).
const std = @import("std");

const fucina = @import("fucina");

const backend_mod = fucina.internal.backend_mod;
const offload = backend_mod.offload;
const backend_ops = backend_mod.ops;
const dtype_mod = backend_mod.dtype_info;
const tensor = fucina.internal.tensor_mod;
const thread = fucina.internal.thread_mod;
const ExecContext = fucina.ExecContext;

const MoeBatchProfile = fucina.MoeBatchProfile;
const Tensor = tensor.Tensor;

const moe_chain = fucina.moe.chain;
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

/// One decode engine for both expert-weight families. The packed x4 and
/// raw GGUF-block single-token paths differ only in how a task reaches an
/// expert's weights — exactly what the `bind` value (`PackedExpertBind` /
/// `RawExpertBind`, the same binds `guBatchBody` takes) absorbs through
/// its `decodeGateUp`/`decodeDown` methods. Every other line is shared,
/// so the two entries stay bitwise-identical by construction.
fn GuDecodeEngine(comptime Bind: type) type {
    return struct {
        const State = struct {
            bind: Bind,
            expert: usize,
            qx: []const dtype_mod.BlockQ8_K,
            out_pe: usize,
            hidden: usize,
            weight: f32,
            gate_buf: []f32,
            up_buf: []f32,
            g_buf: []f32,
            qg: []dtype_mod.BlockQ8_0,
            out: []f32,
            profiler: ?*moe_chain.MoeTaskProfiler,
            remaining_gate_up: std.atomic.Value(u32),
            down_task0: usize,
        };

        const ChainTask = struct {
            state: *State,
            kind: enum { gate_up, down },
            c0: usize,
            c1: usize,
        };

        fn runChainTask(task: *ChainTask, chain: *const thread.Chain) void {
            const state = task.state;
            switch (task.kind) {
                .gate_up => {
                    const gate_up_start = moe_chain.MoeTaskProfiler.start(state.profiler);
                    state.bind.decodeGateUp(state.expert, state.gate_buf, state.up_buf, state.qx, state.out_pe, task.c0, task.c1);
                    moe_chain.MoeTaskProfiler.add(state.profiler, .gate_up, gate_up_start);

                    if (state.remaining_gate_up.fetchSub(1, .acq_rel) == 1) {
                        const geglu_requant_start = moe_chain.MoeTaskProfiler.start(state.profiler);
                        for (state.g_buf, state.gate_buf, state.up_buf) |*g, gate_v, up_v| {
                            g.* = up_v * backend_ops.geluQuantScalar(gate_v);
                        }
                        backend_mod.kernels.quantizeRowQ8_0IntoUnchecked(state.qg, state.g_buf);
                        moe_chain.MoeTaskProfiler.add(state.profiler, .swiglu_requant, geglu_requant_start);
                        chain.enqueue(state.down_task0);
                        chain.enqueue(state.down_task0 + 1);
                    }
                },
                .down => {
                    const down_start = moe_chain.MoeTaskProfiler.start(state.profiler);
                    state.bind.decodeDown(state.expert, state.out, state.qg, state.hidden, task.c0, task.c1);
                    if (state.weight != 1.0) {
                        for (state.out[task.c0..task.c1]) |*o| o.* *= state.weight;
                    }
                    moe_chain.MoeTaskProfiler.add(state.profiler, .down, down_start);
                },
            }
        }

        /// Whole-expert serial body (no worker team): the same three
        /// phases full-width over the same carved state, with the checked
        /// requantize (the chain path pre-validates and runs unchecked).
        fn runSerial(state: *State) void {
            const gate_up_start = moe_chain.MoeTaskProfiler.start(state.profiler);
            state.bind.decodeGateUp(state.expert, state.gate_buf, state.up_buf, state.qx, state.out_pe, 0, state.out_pe);
            moe_chain.MoeTaskProfiler.add(state.profiler, .gate_up, gate_up_start);

            const geglu_requant_start = moe_chain.MoeTaskProfiler.start(state.profiler);
            for (state.g_buf, state.gate_buf, state.up_buf) |*g, gate_v, up_v| {
                g.* = up_v * backend_ops.geluQuantScalar(gate_v);
            }
            backend_mod.kernels.quantizeRowQ8_0Into(state.qg, state.g_buf) catch {
                @memset(state.out, 0);
                return;
            };
            moe_chain.MoeTaskProfiler.add(state.profiler, .swiglu_requant, geglu_requant_start);

            const down_start = moe_chain.MoeTaskProfiler.start(state.profiler);
            state.bind.decodeDown(state.expert, state.out, state.qg, state.hidden, 0, state.hidden);
            if (state.weight != 1.0) {
                for (state.out) |*o| o.* *= state.weight;
            }
            moe_chain.MoeTaskProfiler.add(state.profiler, .down, down_start);
        }
    };
}

/// The decode body both families share once the entry has validated its
/// weight-shape family: scratch carving, the one-row Q8_K quantization,
/// state/task construction, the chained dispatch with its serial
/// fallback, the profile drain, and the top_k-ordered reduce.
fn guDecodeBody(
    ctx: *ExecContext,
    x: *const Tensor,
    bind: anytype,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    hidden: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
    total_start: i128,
) !Tensor {
    const qm = backend_mod.quant;
    const Engine = GuDecodeEngine(@TypeOf(bind));
    const top_k = selected.len;
    const profile_enabled = profile != null;
    var task_prof: moe_chain.MoeTaskProfiler = undefined;
    const prof = moe_chain.MoeTaskProfiler.arm(&task_prof, profile_enabled, io);

    const blocks_per_g = try qm.types.blockCountForDType(.q8_0, out_pe);
    const chain_task_count = 4 * top_k;
    const chain_initial_count = 2 * top_k;
    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    fucina.moe.lockDecodeScratch(ctx);
    defer fucina.moe.unlockDecodeScratch(ctx);
    const sv = try fucina.moe.carveDecodeChainScratch(ctx, dtype_mod.BlockQ8_0, Engine.State, Engine.ChainTask, try qm.types.blockCountForDType(.q8_k, hidden), top_k, out_pe, hidden, blocks_per_g, chain_task_count);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    const gather_quant_start = moeBatchProfileStart(profile_enabled, io);
    try backend_mod.kernels.quantizeRowQ8_KInto(sv.qx, try x.dataConstChecked());
    if (profile) |p| p.gather_quant_ns += moeBatchProfileElapsed(gather_quant_start, io);

    const gate_split = moeDecodeColumnSplit(out_pe, 32);
    const down_split = moeDecodeColumnSplit(hidden, 32);
    for (sv.states, 0..) |*state, j| {
        const down_task0 = chain_initial_count + 2 * j;
        state.* = .{
            .bind = bind,
            .expert = selected[j],
            .qx = sv.qx,
            .out_pe = out_pe,
            .hidden = hidden,
            .weight = weights[j],
            .gate_buf = sv.gate_buf[j * out_pe ..][0..out_pe],
            .up_buf = sv.up_buf[j * out_pe ..][0..out_pe],
            .g_buf = sv.g_buf[j * out_pe ..][0..out_pe],
            .qg = sv.qg[j * blocks_per_g ..][0..blocks_per_g],
            .out = sv.outs[j * hidden ..][0..hidden],
            .profiler = prof,
            .remaining_gate_up = .init(2),
            .down_task0 = down_task0,
        };
        sv.tasks[2 * j] = .{ .state = state, .kind = .gate_up, .c0 = 0, .c1 = gate_split };
        sv.tasks[2 * j + 1] = .{ .state = state, .kind = .gate_up, .c0 = gate_split, .c1 = out_pe };
        sv.tasks[down_task0] = .{ .state = state, .kind = .down, .c0 = 0, .c1 = down_split };
        sv.tasks[down_task0 + 1] = .{ .state = state, .kind = .down, .c0 = down_split, .c1 = hidden };
    }

    const expert_wall_start = moeBatchProfileStart(profile_enabled, io);
    var used_chain = false;
    if (ctx.workPool()) |pool| {
        used_chain = pool.parallelChained(Engine.ChainTask, sv.tasks, chain_initial_count, Engine.runChainTask);
    }
    if (!used_chain) {
        for (sv.states) |*state| Engine.runSerial(state);
    }
    if (profile) |p| {
        p.expert_wall_ns += moeBatchProfileElapsed(expert_wall_start, io);
        if (prof) |tp| tp.drainInto(p);
        p.batches += 1;
        p.pairs += top_k;
        p.active_experts += top_k;
        p.max_expert_m = @max(p.max_expert_m, 1);
    }

    const out_alloc_start = moeBatchProfileStart(profile_enabled, io);
    var out = try ctx.empty(.f32, .{ 1, hidden });
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

/// Gemma 4 fused single-token MoE over the loader's widened expert RHS:
/// gate/up are Q6_Kx4, down is Q8_0x4, and the gate activation is ggml's
/// f16-LUT GELU. This mirrors Qwen's expert-parallel decode helper while
/// preserving Gemma's packed formats and numerics.
pub fn decodePacked(
    ctx: *ExecContext,
    x: *const Tensor,
    gate: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const av = try x.rankView(2);
    if (av.dim(0) != 1) return tensor.TensorError.ShapeMismatch;
    const hidden = av.dim(1);
    const top_k = selected.len;
    const n_expert = gate.len;
    const total_start = moeBatchProfileStart(profile != null, io);

    if (top_k == 0 or weights.len != top_k) return tensor.TensorError.InvalidDataLength;
    if (up.len != n_expert or down.len != n_expert) return tensor.TensorError.ShapeMismatch;
    if (out_pe % 4 != 0 or hidden % 4 != 0) return tensor.TensorError.InvalidShape;
    for (selected) |e| {
        if (e >= n_expert) return tensor.TensorError.IndexOutOfBounds;
        if (gate[e].k != hidden or up[e].k != hidden or gate[e].n != out_pe or up[e].n != out_pe) return tensor.TensorError.ShapeMismatch;
        if (down[e].k != out_pe or down[e].n != hidden) return tensor.TensorError.ShapeMismatch;
    }

    return guDecodeBody(ctx, x, PackedExpertBind{ .gate = gate, .up = up, .down = down }, selected, weights, out_pe, hidden, io, profile, total_start);
}

/// Gemma 4 batched-prefill MoE over the existing per-expert widened RHS
/// representation: gate/up are Q6_Kx4, down is Q8_0x4. This mirrors the
/// Qwen phased MoE scheduler, but keeps Gemma's loader format and GeGLU
/// activation. `weights` should already include Gemma's per-expert down scale.
/// (`-Dgpu=metal` builds don't load the x4 representation at all — they go
/// through `batchRaw`, which holds the GPU arm.)
/// Expert-weight binding for `guBatchBody`: how a gate/up or down
/// matmul task names its `rhs` — packed x4 containers by pointer.
const PackedExpertBind = struct {
    gate: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4,

    fn gu(self: PackedExpertBind, e: usize, is_up: bool) *const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4 {
        return if (is_up) &self.up[e] else &self.gate[e];
    }

    fn dn(self: PackedExpertBind, e: usize) *const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4 {
        return &self.down[e];
    }

    /// Decode gate|up over columns [c0, c1): the fused pair tile reads
    /// each Q8_K input block once for both projections.
    fn decodeGateUp(self: PackedExpertBind, e: usize, gate_buf: []f32, up_buf: []f32, qx: []const dtype_mod.BlockQ8_K, out_pe: usize, c0: usize, c1: usize) void {
        backend_mod.kernels.matmulQ6_Kx4RhsPairTile(gate_buf, up_buf, qx, self.gu(e, false), self.gu(e, true), out_pe, 0, 1, c0, c1);
    }

    /// Decode down projection over columns [c0, c1).
    fn decodeDown(self: PackedExpertBind, e: usize, out: []f32, qg: []const dtype_mod.BlockQ8_0, hidden: usize, c0: usize, c1: usize) void {
        backend_mod.kernels.matmulQ8_0x4RhsTile(out, qg, self.dn(e), hidden, 0, 1, c0, c1);
    }
};

/// `PackedExpertBind`'s raw-GGUF-block counterpart: views by value over
/// the fused gate|up stack (gate rows [0, out_pe), up rows [out_pe, 2*out_pe)).
const RawExpertBind = struct {
    gw: RawExpertWeights,
    out_pe: usize,
    hidden: usize,

    fn gu(self: RawExpertBind, e: usize, is_up: bool) GuRawGuRhs {
        return guRawGuView(self.gw, e, if (is_up) self.out_pe else 0, self.out_pe, self.hidden);
    }

    fn dn(self: RawExpertBind, e: usize) backend_mod.quant.types.QuantizedMatmulRhsQ8_0 {
        return guRawQ8View(self.gw, e, self.out_pe, self.hidden);
    }

    /// Decode gate|up over columns [c0, c1): two plain-block matmuls over
    /// borrowed views (no pair-fused kernel exists for the raw layout).
    fn decodeGateUp(self: RawExpertBind, e: usize, gate_buf: []f32, up_buf: []f32, qx: []const dtype_mod.BlockQ8_K, out_pe: usize, c0: usize, c1: usize) void {
        const gate_view = self.gu(e, false);
        guRawGuMatmul(&gate_view, gate_buf, qx, out_pe, 1, c0, c1);
        const up_view = self.gu(e, true);
        guRawGuMatmul(&up_view, up_buf, qx, out_pe, 1, c0, c1);
    }

    /// Decode down projection over columns [c0, c1).
    fn decodeDown(self: RawExpertBind, e: usize, out: []f32, qg: []const dtype_mod.BlockQ8_0, hidden: usize, c0: usize, c1: usize) void {
        const down_view = self.dn(e);
        backend_mod.kernels.matmulQ8_0RhsTile(out, qg, &down_view, hidden, 0, 1, c0, c1);
    }
};

/// The CPU batch body both expert-weight families share once the caller
/// has validated shapes and built the route plan: scratch, small-m column
/// chunking, task construction, chain wiring, the chained run with its
/// four-phase fallback, profile merge, and the weighted scatter.
/// `GuTask`/`DnTask` differ only in the type of their `rhs` field (bound by
/// `bind.gu`/`bind.dn`), so every line executes identically for both
/// families — the packed/raw paths are bitwise-equal by construction.
/// The batch geometry both families validate before calling
/// `guBatchBody`: token rows, widths, routing fan-out, and the Q8_K /
/// Q8_0 block counts of the two quantized LHS panels.
const BatchShape = struct {
    seq: usize,
    hidden: usize,
    out_pe: usize,
    top_k: usize,
    bpc_in: usize,
    bpc_g: usize,
};

fn guBatchBody(
    comptime GuTask: type,
    comptime DnTask: type,
    comptime runGu: fn (*const GuTask) void,
    comptime runDn: fn (*const DnTask) void,
    comptime runGuOpaque: fn (*anyopaque) void,
    comptime runDnOpaque: fn (*anyopaque) void,
    ctx: *ExecContext,
    x_data: []const f32,
    shape: BatchShape,
    route: *const moe_chain.MoeRoutePlan,
    weights: []const f32,
    bind: anytype,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
    total_start: i128,
) !Tensor {
    const a = ctx.allocator();
    const seq = shape.seq;
    const hidden = shape.hidden;
    const out_pe = shape.out_pe;
    const top_k = shape.top_k;
    const bpc_in = shape.bpc_in;
    const bpc_g = shape.bpc_g;
    const n_expert = route.expertCount();
    const n_pairs = route.pairCount();
    const profile_enabled = profile != null;
    var task_prof: moe_chain.MoeTaskProfiler = undefined;
    const prof = moe_chain.MoeTaskProfiler.arm(&task_prof, profile_enabled, io);
    const count = route.count;
    const offset = route.offset;
    const order = route.order;

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const qx = try a.alloc(dtype_mod.BlockQ8_K, n_pairs * bpc_in);
    defer a.free(qx);
    const gate_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(gate_buf);
    const up_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(up_buf);
    const g_buf = try a.alloc(f32, n_pairs * out_pe);
    defer a.free(g_buf);
    const qg = try a.alloc(dtype_mod.BlockQ8_0, n_pairs * bpc_g);
    defer a.free(qg);
    const down_buf = try a.alloc(f32, n_pairs * hidden);
    defer a.free(down_buf);
    const gather_tasks = try a.alloc(GuGatherTask, n_expert);
    defer a.free(gather_tasks);
    const geglu_tasks = try a.alloc(GuGegluTask, n_expert);
    defer a.free(geglu_tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    // Small-m column chunking is a per-layer-call decision: with few active
    // experts each contributing one full-width task per projection, the
    // team starves. The width helpers keep the counting and construction
    // loops in exact agreement (the chain's enqueue contract).
    const pool = ctx.workPool();
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
    const gate_up_tasks = try a.alloc(GuTask, gate_up_task_count);
    defer a.free(gate_up_tasks);
    const down_tasks = try a.alloc(DnTask, down_task_count);
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
            .profiler = prof,
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
            .profiler = prof,
        };
        if (m == 0) continue;

        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        const gu_chunks = moePhaseChunkCount(gu_width, out_pe);
        const gate_task_start = gate_up_i;
        for (0..gu_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, gu_width, out_pe);
            gate_up_tasks[gate_up_i] = .{
                .rhs = bind.gu(e, false),
                .qlhs = qx,
                .bpc = bpc_in,
                .row_start = base,
                .m = m,
                .out_dim = out_pe,
                .out = gate_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profiler = prof,
            };
            gate_up_i += 1;
            gate_up_tasks[gate_up_i] = .{
                .rhs = bind.gu(e, true),
                .qlhs = qx,
                .bpc = bpc_in,
                .row_start = base,
                .m = m,
                .out_dim = out_pe,
                .out = up_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profiler = prof,
            };
            gate_up_i += 1;
        }

        const d_width = moePhaseColWidth(m, hidden, small_m_width);
        const d_chunks = moePhaseChunkCount(d_width, hidden);
        const down_task_start = down_i;
        for (0..d_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, d_width, hidden);
            down_tasks[down_i] = .{
                .rhs = bind.dn(e),
                .qlhs = qg,
                .bpc = bpc_g,
                .row_start = base,
                .m = m,
                .out_dim = hidden,
                .out = down_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profiler = prof,
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
        GuGatherTask,
        GuTask,
        GuGegluTask,
        DnTask,
        chain_tasks,
        chain_states,
        count,
        gather_tasks,
        gate_up_tasks,
        geglu_tasks,
        down_tasks,
        runGuGatherTaskOpaque,
        runGuOpaque,
        runGuGegluTaskOpaque,
        runDnOpaque,
    );

    var expert_wall_ns: i128 = 0;
    var phase_start = moeBatchProfileStart(profile_enabled, io);
    const used_chain = if (pool) |p| p.parallelChained(MoeBatchPhaseChainTask, chain_tasks, chain_initial_count, runMoeBatchPhaseChainTask) else false;
    if (used_chain) {
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    } else {
        if (pool) |p| {
            p.parallelChunks(GuGatherTask, gather_tasks, runGuGatherTask);
        } else {
            for (gather_tasks) |*t| runGuGatherTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GuTask, gate_up_tasks, runGu);
        } else {
            for (gate_up_tasks) |*t| runGu(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(GuGegluTask, geglu_tasks, runGuGegluTask);
        } else {
            for (geglu_tasks) |*t| runGuGegluTask(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        if (pool) |p| {
            p.parallelChunks(DnTask, down_tasks, runDn);
        } else {
            for (down_tasks) |*t| runDn(t);
        }
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    }

    if (profile) |p| {
        p.expert_wall_ns += expert_wall_ns;
        if (prof) |tp| tp.drainInto(p);
    }

    const out = try scatterGrouped(ctx, seq, hidden, top_k, route, weights, down_buf, io, profile);
    recordBatch(profile, total_start, io, route);
    return out;
}

pub fn batchPacked(
    ctx: *ExecContext,
    x: *const Tensor,
    gate: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    up: []const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    down: []const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4,
    selected: []const usize,
    weights: []const f32,
    top_k: usize,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quant;
    const a = ctx.allocator();
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

    const bpc_in = try qm.types.blockCountForDType(.q8_k, hidden);
    const bpc_g = try qm.types.blockCountForDType(.q8_0, out_pe);

    const route_result = try moe_chain.buildMoeRoutePlan(a, selected, n_expert, profile_enabled, io);
    var route = route_result.plan;
    defer route.deinit();
    if (profile) |p| {
        p.alloc_ns += route_result.alloc_ns;
        p.count_sort_ns += route_result.count_sort_ns;
    }

    return guBatchBody(
        GuQ6MatmulTask,
        GuQ8MatmulTask,
        runGuQ6MatmulTask,
        runGuQ8MatmulTask,
        runGuQ6MatmulTaskOpaque,
        runGuQ8MatmulTaskOpaque,
        ctx,
        x_data,
        .{ .seq = seq, .hidden = hidden, .out_pe = out_pe, .top_k = top_k, .bpc_in = bpc_in, .bpc_g = bpc_g },
        &route,
        weights,
        PackedExpertBind{ .gate = gate, .up = up, .down = down },
        io,
        profile,
        total_start,
    );
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
    ctx: *ExecContext,
    x_data: []const f32,
    seq: usize,
    hidden: usize,
    out_pe: usize,
    top_k: usize,
    gw: RawExpertWeights,
    route: *const moe_chain.MoeRoutePlan,
    weights: []const f32,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
    total_start: i128,
) !?Tensor {
    const a = ctx.allocator();
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
    // One switch resolves the gate_up dtype's kernel tag, raw bytes and
    // block size together.
    const gu_format: offload.QuantFormat, const gu_bytes: []const u8, const gu_block_bytes: usize = switch (gw.gu) {
        inline else => |gu_blocks, tag| .{
            comptime offload.QuantFormat.fromDType(@field(dtype_mod.DType, @tagName(tag))).?,
            std.mem.sliceAsBytes(gu_blocks),
            @sizeOf(@typeInfo(@TypeOf(gu_blocks)).pointer.child),
        },
    };
    const nb01_gu = bpr_gu * gu_block_bytes;
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

    // total m·n·k across both grouped GEMMs of this layer
    const work = @as(u64, n_pairs) * @as(u64, hidden) * @as(u64, gu_out + out_pe);
    if (!offload.qmoeAccepts(n_pairs, n_tiles, work)) return null;

    // Accumulated locally and merged only on success: a mid-sequence GPU
    // refusal falls back to the CPU path, which records its own full
    // pass — committing partial GPU phases would double-count the batch.
    var local: MoeBatchProfile = .{};
    var expert_wall_ns: i128 = 0;

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const tiles = try a.alloc(offload.QMMTile, n_tiles);
    defer a.free(tiles);
    const gather_tasks = try a.alloc(GuGpuGatherTask, n_expert);
    defer a.free(gather_tasks);
    const geglu_tasks = try a.alloc(GuGpuGegluTask, n_expert);
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
    var session = offload.QMoeSession.begin(
        n_pairs * @max(hidden, out_pe) * @sizeOf(f32),
        n_pairs * @max(gu_out, hidden) * @sizeOf(f32),
    ) orelse return null;
    defer session.end();
    const stage = session.stage;

    const pool = ctx.workPool();

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
        p.parallelChunks(GuGpuGatherTask, gather_tasks, runGuGpuGatherTask);
    } else {
        for (gather_tasks) |*t| runGuGpuGatherTask(t);
    }
    if (profile_enabled) {
        const ns = moeBatchProfileElapsed(phase_start, io);
        local.gather_quant_ns += ns;
        expert_wall_ns += ns;
    }

    const cacheable = gw.device_owned;

    phase_start = moeBatchProfileStart(profile_enabled, io);
    if (!session.gemmGrouped(
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
        p.parallelChunks(GuGpuGegluTask, geglu_tasks, runGuGpuGegluTask);
    } else {
        for (geglu_tasks) |*t| runGuGpuGegluTask(t);
    }
    if (profile_enabled) {
        const ns = moeBatchProfileElapsed(phase_start, io);
        local.swiglu_requant_ns += ns;
        expert_wall_ns += ns;
    }

    phase_start = moeBatchProfileStart(profile_enabled, io);
    if (!session.gemmGrouped(
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
    const out = try scatterGrouped(ctx, seq, hidden, top_k, route, weights, down_panel, io, scatter_profile);
    if (profile) |p| {
        p.alloc_ns += local.alloc_ns;
        p.gather_quant_ns += local.gather_quant_ns;
        p.gate_up_ns += local.gate_up_ns;
        p.swiglu_requant_ns += local.swiglu_requant_ns;
        p.down_ns += local.down_ns;
        p.scatter_ns += local.scatter_ns;
        p.expert_wall_ns += expert_wall_ns;
    }
    recordBatch(profile, total_start, io, route);
    return out;
}

const GuGpuGatherTask = struct {
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    top_k: usize,
    row_start: usize,
    m: usize,
    dst: []f32, // staging panel [n_pairs * hidden]
};

fn runGuGpuGatherTask(task: *const GuGpuGatherTask) void {
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

const GuGpuGegluTask = struct {
    src: []const f32, // gate_up GEMM panel [n_pairs * 2*out_pe]
    dst: []f32, // gated rows [n_pairs * out_pe]
    out_pe: usize,
    row_start: usize,
    m: usize,
};

fn runGuGpuGegluTask(task: *const GuGpuGegluTask) void {
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
const GuRawGuRhs = union(enum) {
    q6_k: backend_mod.quant.types.QuantizedMatmulRhsQ6_K,
    q4_k: backend_mod.quant.types.QuantizedMatmulRhsQ4_K,
};

fn guRawGuView(
    gw: RawExpertWeights,
    expert: usize,
    row_off: usize,
    out_pe: usize,
    hidden: usize,
) GuRawGuRhs {
    const bpr = hidden / 256;
    const start = (expert * 2 * out_pe + row_off) * bpr;
    return switch (gw.gu) {
        inline else => |gu_blocks, tag| @unionInit(GuRawGuRhs, @tagName(tag), .{
            .allocator = null, // borrows the model's resident expert copies
            .blocks = gu_blocks[start..][0 .. out_pe * bpr],
            .k = hidden,
            .n = out_pe,
            .blocks_per_column = bpr,
        }),
    };
}

/// One gate_up matmul over a raw view: each arm forks to its compact
/// column-outer kernel at m >= 4 (the batched-prefill case — unpack each
/// weight block once per row tile) and stays on the row-outer tile below.
fn guRawGuMatmul(
    view: *const GuRawGuRhs,
    out: []f32,
    qlhs: []const dtype_mod.BlockQ8_K,
    out_dim: usize,
    m: usize,
    c0: usize,
    c1: usize,
) void {
    switch (view.*) {
        .q6_k => |*v| if (m >= 4) {
            backend_mod.kernels.matmulQ6_KRhsCompactColOuter(out, qlhs, v, out_dim, 0, m, c0, c1);
        } else {
            backend_mod.kernels.matmulQ6_KRhsTile(out, qlhs, v, out_dim, 0, m, c0, c1);
        },
        .q4_k => |*v| if (m >= 4) {
            backend_mod.kernels.matmulQ4_KRhsCompactColOuter(out, qlhs, v, out_dim, 0, m, c0, c1);
        } else {
            backend_mod.kernels.matmulQ4_KRhsTile(out, qlhs, v, out_dim, 0, m, c0, c1);
        },
    }
}

/// Borrowed plain-Q8_0 RHS view over one expert's raw GGUF down blocks:
/// the blocks belong to the model's resident expert copies, so the rows
/// table carries a null allocator (deinit frees nothing).
fn guRawQ8View(
    gw: RawExpertWeights,
    expert: usize,
    out_pe: usize,
    hidden: usize,
) backend_mod.quant.types.QuantizedMatmulRhsQ8_0 {
    const bpr = out_pe / 32;
    return .{
        .allocator = null,
        .blocks = gw.dn_blocks[expert * hidden * bpr ..][0 .. hidden * bpr],
        .blocks_per_column = bpr,
        .k = out_pe,
        .n = hidden,
    };
}

/// Gemma 4 fused single-token MoE over the RAW GGUF expert blocks
/// (`-Dgpu=metal` builds, which skip the x4 widening to keep a single
/// expert representation in memory, and CPU builds with Q4_K gate_up
/// experts, which have no x4 packing). Same body as `decodePacked`
/// (`guDecodeBody`) — Q8_K-quantized input, ggml f16-LUT GeGLU,
/// Q8_0-requantized down input — only the weight binding differs (plain
/// row-block views straight from the mmap instead of the widened x4
/// packing).
pub fn decodeRaw(
    ctx: *ExecContext,
    x: *const Tensor,
    gw: RawExpertWeights,
    n_expert: usize,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const av = try x.rankView(2);
    if (av.dim(0) != 1) return tensor.TensorError.ShapeMismatch;
    const hidden = av.dim(1);
    const top_k = selected.len;
    const total_start = moeBatchProfileStart(profile != null, io);

    if (top_k == 0 or weights.len != top_k) return tensor.TensorError.InvalidDataLength;
    if (hidden % 256 != 0 or out_pe % 32 != 0) return tensor.TensorError.InvalidShape;
    if (gw.guBlockCount() != n_expert * 2 * out_pe * (hidden / 256)) return tensor.TensorError.ShapeMismatch;
    if (gw.dn_blocks.len != n_expert * hidden * (out_pe / 32)) return tensor.TensorError.ShapeMismatch;
    for (selected) |e| if (e >= n_expert) return tensor.TensorError.IndexOutOfBounds;

    return guDecodeBody(ctx, x, RawExpertBind{ .gw = gw, .out_pe = out_pe, .hidden = hidden }, selected, weights, out_pe, hidden, io, profile, total_start);
}

const GuRawGuMatmulTask = struct {
    rhs: GuRawGuRhs,
    qlhs: []const dtype_mod.BlockQ8_K,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profiler: ?*moe_chain.MoeTaskProfiler,
};

fn runGuRawGuMatmulTask(task: *const GuRawGuMatmulTask) void {
    const m = task.m;
    if (m == 0) return;
    const start = moe_chain.MoeTaskProfiler.start(task.profiler);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    guRawGuMatmul(&task.rhs, out, q, task.out_dim, m, task.c0, task.c1);
    moe_chain.MoeTaskProfiler.add(task.profiler, .gate_up, start);
}

fn runGuRawGuMatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GuRawGuMatmulTask = @ptrCast(@alignCast(ctx));
    runGuRawGuMatmulTask(task);
}

const GuRawQ8MatmulTask = struct {
    rhs: backend_mod.quant.types.QuantizedMatmulRhsQ8_0,
    qlhs: []const dtype_mod.BlockQ8_0,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profiler: ?*moe_chain.MoeTaskProfiler,
};

fn runGuRawQ8MatmulTask(task: *const GuRawQ8MatmulTask) void {
    const m = task.m;
    if (m == 0) return;
    const start = moe_chain.MoeTaskProfiler.start(task.profiler);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    backend_mod.kernels.matmulQ8_0RhsTile(out, q, &task.rhs, task.out_dim, 0, m, task.c0, task.c1);
    moe_chain.MoeTaskProfiler.add(task.profiler, .down, start);
}

fn runGuRawQ8MatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GuRawQ8MatmulTask = @ptrCast(@alignCast(ctx));
    runGuRawQ8MatmulTask(task);
}

/// Gemma 4 batched MoE over the RAW GGUF expert blocks (`-Dgpu=metal`
/// builds, and CPU builds with Q4_K gate_up experts): tries the grouped
/// dequant-in-kernel Metal GEMM first (`batchRawGpu`, gpu
/// builds only), then falls back to a CPU path with the same phase
/// structure as `batchPacked` but plain-block kernels over
/// borrowed views (no x4 widening exists for these arms).
/// Numerics match the x4 path: Q8_K LHS, f16-LUT GeGLU, Q8_0 requant.
pub fn batchRaw(
    ctx: *ExecContext,
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
    const qm = backend_mod.quant;
    const a = ctx.allocator();
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

    const bpc_in = try qm.types.blockCountForDType(.q8_k, hidden);
    const bpc_g = try qm.types.blockCountForDType(.q8_0, out_pe);

    const route_result = try moe_chain.buildMoeRoutePlan(a, selected, n_expert, profile_enabled, io);
    var route = route_result.plan;
    defer route.deinit();
    if (profile) |p| {
        p.alloc_ns += route_result.alloc_ns;
        p.count_sort_ns += route_result.count_sort_ns;
    }

    {
        if (try batchRawGpu(
            ctx,
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

    return guBatchBody(
        GuRawGuMatmulTask,
        GuRawQ8MatmulTask,
        runGuRawGuMatmulTask,
        runGuRawQ8MatmulTask,
        runGuRawGuMatmulTaskOpaque,
        runGuRawQ8MatmulTaskOpaque,
        ctx,
        x_data,
        .{ .seq = seq, .hidden = hidden, .out_pe = out_pe, .top_k = top_k, .bpc_in = bpc_in, .bpc_g = bpc_g },
        &route,
        weights,
        RawExpertBind{ .gw = gw, .out_pe = out_pe, .hidden = hidden },
        io,
        profile,
        total_start,
    );
}

const GuGatherTask = struct {
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    top_k: usize,
    bpc_in: usize,
    row_start: usize,
    m: usize,
    qx: []dtype_mod.BlockQ8_K,
    profiler: ?*moe_chain.MoeTaskProfiler,
};

fn runGuGatherTask(task: *const GuGatherTask) void {
    const m = task.m;
    if (m == 0) return;
    const start = moe_chain.MoeTaskProfiler.start(task.profiler);
    const base = task.row_start;
    for (0..m) |i| {
        const token = task.order[base + i] / task.top_k;
        const src = task.x_data[token * task.hidden ..][0..task.hidden];
        backend_mod.kernels.quantizeRowQ8_KIntoUnchecked(task.qx[(base + i) * task.bpc_in ..][0..task.bpc_in], src);
    }
    moe_chain.MoeTaskProfiler.add(task.profiler, .gather_quant, start);
}

fn runGuGatherTaskOpaque(ctx: *anyopaque) void {
    const task: *const GuGatherTask = @ptrCast(@alignCast(ctx));
    runGuGatherTask(task);
}

const GuQ6MatmulTask = struct {
    rhs: *const backend_mod.quant.types.QuantizedMatmulRhsQ6_Kx4,
    qlhs: []const dtype_mod.BlockQ8_K,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profiler: ?*moe_chain.MoeTaskProfiler,
};

fn runGuQ6MatmulTask(task: *const GuQ6MatmulTask) void {
    const m = task.m;
    if (m == 0) return;
    const start = moe_chain.MoeTaskProfiler.start(task.profiler);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    backend_mod.kernels.matmulQ6_Kx4RhsTile(out, q, task.rhs, task.out_dim, 0, m, task.c0, task.c1);
    moe_chain.MoeTaskProfiler.add(task.profiler, .gate_up, start);
}

fn runGuQ6MatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GuQ6MatmulTask = @ptrCast(@alignCast(ctx));
    runGuQ6MatmulTask(task);
}

const GuGegluTask = struct {
    gate_buf: []const f32,
    up_buf: []const f32,
    g_buf: []f32,
    qg: []dtype_mod.BlockQ8_0,
    out_pe: usize,
    bpc_g: usize,
    row_start: usize,
    m: usize,
    profiler: ?*moe_chain.MoeTaskProfiler,
};

fn runGuGegluTask(task: *const GuGegluTask) void {
    const m = task.m;
    if (m == 0) return;
    const start = moe_chain.MoeTaskProfiler.start(task.profiler);
    const base = task.row_start;
    const out_pe = task.out_pe;
    const gate_out = task.gate_buf[base * out_pe ..][0 .. m * out_pe];
    const up_out = task.up_buf[base * out_pe ..][0 .. m * out_pe];
    const g_out = task.g_buf[base * out_pe ..][0 .. m * out_pe];
    for (g_out, gate_out, up_out) |*g, gate_v, up_v| {
        g.* = up_v * backend_ops.geluQuantScalar(gate_v);
    }
    for (0..m) |i| {
        backend_mod.kernels.quantizeRowQ8_0IntoUnchecked(task.qg[(base + i) * task.bpc_g ..][0..task.bpc_g], g_out[i * out_pe ..][0..out_pe]);
    }
    moe_chain.MoeTaskProfiler.add(task.profiler, .swiglu_requant, start);
}

fn runGuGegluTaskOpaque(ctx: *anyopaque) void {
    const task: *const GuGegluTask = @ptrCast(@alignCast(ctx));
    runGuGegluTask(task);
}

const GuQ8MatmulTask = struct {
    rhs: *const backend_mod.quant.types.QuantizedMatmulRhsQ8_0x4,
    qlhs: []const dtype_mod.BlockQ8_0,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profiler: ?*moe_chain.MoeTaskProfiler,
};

fn runGuQ8MatmulTask(task: *const GuQ8MatmulTask) void {
    const m = task.m;
    if (m == 0) return;
    const start = moe_chain.MoeTaskProfiler.start(task.profiler);
    const base = task.row_start;
    const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    backend_mod.kernels.matmulQ8_0x4RhsTile(out, q, task.rhs, task.out_dim, 0, m, task.c0, task.c1);
    moe_chain.MoeTaskProfiler.add(task.profiler, .down, start);
}

fn runGuQ8MatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const GuQ8MatmulTask = @ptrCast(@alignCast(ctx));
    runGuQ8MatmulTask(task);
}

/// Expert-major weighted scatter of the grouped down rows back into token
/// order. Serial by design: a token-parallel split needs the plan's inverse
/// mapping (`inv`) and changes each token's floating-point summation order,
/// which requires a tolerance argument against the gemma parity oracles.
fn scatterInto(
    out: []f32,
    down_rows: []const f32,
    route: *const moe_chain.MoeRoutePlan,
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

/// Grouped-rows scatter into a fresh `[seq, hidden]` output (alloc timed
/// into the profile), the batch paths' tail.
fn scatterGrouped(
    ctx: *ExecContext,
    seq: usize,
    hidden: usize,
    top_k: usize,
    route: *const moe_chain.MoeRoutePlan,
    weights: []const f32,
    down_rows: []const f32,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const profile_enabled = profile != null;
    const out_alloc_start = moeBatchProfileStart(profile_enabled, io);
    var out = try ctx.empty(.f32, .{ seq, hidden });
    errdefer out.deinit();
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(out_alloc_start, io);

    const scatter_start = moeBatchProfileStart(profile_enabled, io);
    scatterInto(out.data(), down_rows, route, weights, top_k, hidden);
    if (profile) |p| p.scatter_ns += moeBatchProfileElapsed(scatter_start, io);
    return out;
}

fn recordBatch(profile: ?*MoeBatchProfile, total_start: i128, io: ?std.Io, route: *const moe_chain.MoeRoutePlan) void {
    if (profile) |p| {
        p.total_ns += moeBatchProfileElapsed(total_start, io);
        p.batches += 1;
        p.pairs += route.pairCount();
        p.active_experts += route.active_experts;
        p.max_expert_m = @max(p.max_expert_m, route.max_expert_m);
    }
}
