const std = @import("std");
const backend_mod = @import("../backend.zig");
const Runtime = @import("runtime.zig").Runtime;
const backend_ops = backend_mod.ops;
const expert_store = @import("expert_store.zig");
const moe_chain = @import("moe_chain.zig");
const parallel = @import("../parallel.zig");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;
const GatedOp = backend_ops.GatedOp;
const Tensor = tensor.Tensor;

// Shared batched-MoE scheduling scaffolding (exec/moe_chain.zig), also
// consumed by the gemma MoE engines via `ExecContext.moe_chain`.
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

pub const MoeBatchProfile = struct {
    total_ns: i128 = 0,
    alloc_ns: i128 = 0,
    count_sort_ns: i128 = 0,
    expert_wall_ns: i128 = 0,
    gather_quant_ns: i128 = 0,
    gate_up_ns: i128 = 0,
    swiglu_requant_ns: i128 = 0,
    down_ns: i128 = 0,
    scatter_ns: i128 = 0,
    batches: usize = 0,
    pairs: usize = 0,
    active_experts: usize = 0,
    max_expert_m: usize = 0,
};

/// A Mixture-of-Experts projection: all experts of one layer's gate/up/down
/// stacked into a single RHS buffer (experts are row-contiguous, so expert
/// `e` is a zero-copy sub-view). Stored as the COMPACT raw K-quant blocks
/// (not the widened "xN" sdot packing): MoE runs everything at m=1, which is
/// memory-bandwidth-bound, so reading ~5.5 bits/weight instead of ~8 is the
/// win — and it makes load a plain memcpy instead of a repack. Only q4_k /
/// q5_k / q6_k experts (every real MoE GGUF) are supported.
pub const MoeRhs = union(enum) {
    q4_k: backend_mod.QuantizedMatmulRhsQ4_K,
    q5_k: backend_mod.QuantizedMatmulRhsQ5_K,
    q6_k: backend_mod.QuantizedMatmulRhsQ6_K,
    /// q8_0 experts (32-elem blocks): the format llama.cpp falls back to
    /// when an expert dim is not a 256 multiple (deepseek2's 1408). Pairs
    /// with Q8_0-quantized activations instead of Q8_K.
    q8_0: backend_mod.QuantizedMatmulRhsQ8_0,
    /// Ternary experts (TQ2_0, 256-elem blocks, ~2.06 bpw): the packed-trit
    /// format the ternary campaign made first-class, and the container the
    /// PTQTP plane persistence uses. Pairs with Q8_K activations like the
    /// K-quants; the sdot/vpdpbusd tile kernel covers decode and batch.
    tq2_0: backend_mod.QuantizedMatmulRhsTQ2_0,
    /// Disk-streamed expert stack (`exec/expert_store.zig`): same geometry
    /// and kernels as the resident arms, but expert blocks resolve through
    /// the store's acquire-scoped tier (pin → LRU → pread) instead of a
    /// slice into one resident buffer. The store outlives the arm (owned by
    /// the model), so `deinit` is a no-op here.
    streamed: expert_store.StreamedMoeRhs,

    pub fn deinit(self: *MoeRhs) void {
        switch (self.*) {
            .streamed => {},
            inline else => |*value| value.deinit(),
        }
        self.* = undefined;
    }

    pub fn rows(self: *const MoeRhs) usize {
        return switch (self.*) {
            .streamed => |*value| value.rows(),
            inline else => |*value| value.n,
        };
    }

    pub fn k(self: *const MoeRhs) usize {
        return switch (self.*) {
            .streamed => |*value| value.k,
            inline else => |*value| value.k,
        };
    }

    pub fn blocksPerColumn(self: *const MoeRhs) usize {
        return switch (self.*) {
            .streamed => |*value| value.blocks_per_column,
            .q8_0 => |*value| value.rows.blocks_per_row,
            .tq2_0 => |*value| value.rows.blocks_per_row,
            inline else => |*value| value.blocks_per_column,
        };
    }

    pub fn blockLen(self: *const MoeRhs) usize {
        return switch (self.*) {
            // Virtual: the streamed stack never exists in memory at once.
            .streamed => |*value| value.rows() * value.blocks_per_column,
            .q8_0 => |*value| value.rows.blocks.len,
            .tq2_0 => |*value| value.rows.blocks.len,
            inline else => |*value| value.blocks.len,
        };
    }

    /// Whether this arm multiplies against Q8_0-quantized activations
    /// (32-elem blocks) instead of the Q8_K default.
    pub fn wantsQ8_0Lhs(self: *const MoeRhs) bool {
        return switch (self.*) {
            .q8_0 => true,
            .streamed => |*value| value.quant == .q8_0,
            else => false,
        };
    }
};

fn checkedMoeProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch tensor.TensorError.InvalidDataLength;
}

fn validateMoeRhsStorage(rhs: *const MoeRhs, rows: usize, k: usize) !void {
    const qm = backend_mod.quantized_matmul;
    const expected_bpc = if (rhs.wantsQ8_0Lhs()) blk: {
        if (k == 0 or k % 32 != 0) return tensor.TensorError.InvalidShape;
        break :blk k / 32;
    } else qm.qkBlockCount(k) catch return tensor.TensorError.InvalidShape;
    const expected_blocks = std.math.mul(usize, rows, expected_bpc) catch return tensor.TensorError.ShapeMismatch;
    if (rhs.rows() != rows or rhs.k() != k) return tensor.TensorError.ShapeMismatch;
    if (rhs.blocksPerColumn() != expected_bpc or rhs.blockLen() != expected_blocks) return tensor.TensorError.ShapeMismatch;
}

fn validatePackedMoeInputs(
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    selected: []const usize,
    weights: []const f32,
    expected_route_len: usize,
    hidden: usize,
    out_pe: usize,
) !usize {
    if (expected_route_len == 0 or selected.len != expected_route_len or weights.len != expected_route_len) return tensor.TensorError.InvalidDataLength;
    if (out_pe == 0) return tensor.TensorError.InvalidShape;

    const gate_rows = gate.rows();
    if (gate_rows == 0 or gate_rows % out_pe != 0) return tensor.TensorError.ShapeMismatch;
    const n_expert = gate_rows / out_pe;
    const down_rows = std.math.mul(usize, n_expert, hidden) catch return tensor.TensorError.ShapeMismatch;

    try validateMoeRhsStorage(gate, gate_rows, hidden);
    try validateMoeRhsStorage(up, gate_rows, hidden);
    try validateMoeRhsStorage(down, down_rows, out_pe);

    for (selected) |e| if (e >= n_expert) return tensor.TensorError.IndexOutOfBounds;
    return n_expert;
}

/// Scope of one layer op's streamed-expert residency: `acquireMoeStreamed`
/// resolves the routed experts (LRU hits in place, misses read from disk)
/// and locks the store; `release` promotes the misses into the LRU and
/// unlocks. A no-op guard (resident arms) costs nothing.
const MoeStreamGuard = struct {
    store: ?*expert_store.ExpertStore = null,
    layer: usize = 0,

    fn release(self: *MoeStreamGuard) void {
        if (self.store) |s| {
            s.release(self.layer);
            self.store = null;
        }
    }
};

/// The three projections of a streamed MoE layer stream as one unit (one
/// slab per expert), so gate/up/down must all be streamed arms of the same
/// store and layer; mixing resident and streamed projections is a wiring
/// bug, not a supported configuration.
fn acquireMoeStreamed(gate: *const MoeRhs, up: *const MoeRhs, down: *const MoeRhs, selected: []const usize) !MoeStreamGuard {
    const sg = switch (gate.*) {
        .streamed => |*v| v,
        else => {
            if (up.* == .streamed or down.* == .streamed) return tensor.TensorError.ShapeMismatch;
            return .{};
        },
    };
    const su = switch (up.*) {
        .streamed => |*v| v,
        else => return tensor.TensorError.ShapeMismatch,
    };
    const sd = switch (down.*) {
        .streamed => |*v| v,
        else => return tensor.TensorError.ShapeMismatch,
    };
    if (su.store != sg.store or sd.store != sg.store) return tensor.TensorError.ShapeMismatch;
    if (su.layer != sg.layer or sd.layer != sg.layer) return tensor.TensorError.ShapeMismatch;
    try sg.store.acquire(sg.layer, selected);
    return .{ .store = sg.store, .layer = sg.layer };
}

/// out[0 .. m*out_dim] = `m` Q8_K-quantized input rows (`qlhs`) times expert
/// `e`'s contiguous row-block of `rhs`. Single threaded — one expert's GEMM,
/// run inside a pooled per-expert task. `m == 1` is the decode GEMV; `m > 1`
/// is the batched-prefill case (all rows reuse the same weights from cache).
fn moeExpertTileDotRange(rhs: *const MoeRhs, e: usize, qlhs: []const backend_mod.quantized_matmul.BlockQ8_K, qlhs8: []const backend_mod.quantized_matmul.BlockQ8_0, out: []f32, out_dim: usize, m: usize, c0: usize, c1: usize) void {
    const qm = backend_mod.quantized_matmul;
    switch (rhs.*) {
        .q8_0 => |*big| {
            const bpc = big.rows.blocks_per_row;
            const view = q8_0View(big.rows.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], big.k, out_dim, bpc);
            qm.matmulQ8_0RhsTile(out, qlhs8, &view, out_dim, 0, m, c0, c1);
        },
        .tq2_0 => |*big| {
            const bpc = big.rows.blocks_per_row;
            const view = tq2_0View(big.rows.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], big.k, out_dim, bpc);
            qm.matmulTQ2_0RhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
        },
        // Streamed expert: identical kernels over the store-resolved slab
        // (the acquire that preceded this op pinned the pointer).
        .streamed => |*s| {
            const bpc = s.blocks_per_column;
            const base = s.expertBytes(e);
            switch (s.quant) {
                .q8_0 => {
                    const blocks = @as([*]const qm.BlockQ8_0, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = q8_0View(blocks, s.k, out_dim, bpc);
                    qm.matmulQ8_0RhsTile(out, qlhs8, &view, out_dim, 0, m, c0, c1);
                },
                .tq2_0 => {
                    const blocks = @as([*]const qm.BlockTQ2_0, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = tq2_0View(blocks, s.k, out_dim, bpc);
                    qm.matmulTQ2_0RhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
                },
                .q5_k => {
                    const blocks = @as([*]const qm.BlockQ5_K, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = backend_mod.QuantizedMatmulRhsQ5_K{ .allocator = null, .blocks = blocks, .k = s.k, .n = out_dim, .blocks_per_column = bpc };
                    if (m >= 4) {
                        qm.matmulQ5_KRhsCompactColOuter(out, qlhs, &view, out_dim, 0, m, c0, c1);
                    } else {
                        qm.matmulQ5_KRhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
                    }
                },
                .q6_k => {
                    const blocks = @as([*]const qm.BlockQ6_K, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = backend_mod.QuantizedMatmulRhsQ6_K{ .allocator = null, .blocks = blocks, .k = s.k, .n = out_dim, .blocks_per_column = bpc };
                    if (m >= 4) {
                        qm.matmulQ6_KRhsCompactColOuter(out, qlhs, &view, out_dim, 0, m, c0, c1);
                    } else {
                        qm.matmulQ6_KRhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
                    }
                },
                .q4_k => {
                    const blocks = @as([*]const qm.BlockQ4_K, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = backend_mod.QuantizedMatmulRhsQ4_K{ .allocator = null, .blocks = blocks, .k = s.k, .n = out_dim, .blocks_per_column = bpc };
                    if (m >= 4) {
                        qm.matmulQ4_KRhsCompactColOuter(out, qlhs, &view, out_dim, 0, m, c0, c1);
                    } else {
                        qm.matmulQ4_KRhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
                    }
                },
            }
        },
        .q5_k => |*big| {
            const bpc = big.blocks_per_column;
            const view = backend_mod.QuantizedMatmulRhsQ5_K{ .allocator = big.allocator, .blocks = big.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], .k = big.k, .n = out_dim, .blocks_per_column = bpc };
            // Batched prefill: unpack each weight once and sdot across the row
            // tile; decode (m==1) stays on the row-outer tile.
            if (m >= 4) {
                qm.matmulQ5_KRhsCompactColOuter(out, qlhs, &view, out_dim, 0, m, c0, c1);
            } else {
                qm.matmulQ5_KRhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
            }
        },
        .q6_k => |*big| {
            const bpc = big.blocks_per_column;
            const view = backend_mod.QuantizedMatmulRhsQ6_K{ .allocator = big.allocator, .blocks = big.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], .k = big.k, .n = out_dim, .blocks_per_column = bpc };
            if (m >= 4) {
                qm.matmulQ6_KRhsCompactColOuter(out, qlhs, &view, out_dim, 0, m, c0, c1);
            } else {
                qm.matmulQ6_KRhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
            }
        },
        .q4_k => |*big| {
            const bpc = big.blocks_per_column;
            const view = backend_mod.QuantizedMatmulRhsQ4_K{ .allocator = big.allocator, .blocks = big.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], .k = big.k, .n = out_dim, .blocks_per_column = bpc };
            if (m >= 4) {
                qm.matmulQ4_KRhsCompactColOuter(out, qlhs, &view, out_dim, 0, m, c0, c1);
            } else {
                qm.matmulQ4_KRhsTile(out, qlhs, &view, out_dim, 0, m, c0, c1);
            }
        },
    }
}

fn moeExpertTileDot(rhs: *const MoeRhs, e: usize, qlhs: []const backend_mod.quantized_matmul.BlockQ8_K, qlhs8: []const backend_mod.quantized_matmul.BlockQ8_0, out: []f32, out_dim: usize, m: usize) void {
    moeExpertTileDotRange(rhs, e, qlhs, qlhs8, out, out_dim, m, 0, out_dim);
}

fn q8_0View(blocks: []const backend_mod.quantized_matmul.BlockQ8_0, k: usize, out_dim: usize, bpc: usize) backend_mod.QuantizedMatmulRhsQ8_0 {
    return .{
        .rows = .{ .allocator = null, .blocks = blocks, .rows = out_dim, .cols = k, .blocks_per_row = bpc },
        .k = k,
        .n = out_dim,
    };
}

fn tq2_0View(blocks: []const backend_mod.quantized_matmul.BlockTQ2_0, k: usize, out_dim: usize, bpc: usize) backend_mod.QuantizedMatmulRhsTQ2_0 {
    // The generic rows container carries mutable blocks; the matmul path
    // never writes them, so the @constCast borrow is sound (see the
    // stack-wrapper note in matmul2DWithQuantizedRowsTensorRhs).
    return .{
        .rows = .{ .allocator = null, .blocks = @constCast(blocks), .rows = out_dim, .cols = k, .blocks_per_row = bpc },
        .k = k,
        .n = out_dim,
    };
}

/// Whether `rhs` has a lane-packed Q8_Kx4 column-outer kernel (Q4_K / Q5_K /
/// Q6_K — i.e. every MoeRhs arm today). The MoE prefill path repacks
/// activations and routes `m >= 4` experts of these dtypes to
/// `moeExpertTileDotX4Range`; the `m < 4` tail stays on the per-row path.
fn moeRhsUsesLanePacked(rhs: *const MoeRhs) bool {
    return switch (rhs.*) {
        .q4_k, .q5_k, .q6_k => true,
        .q8_0, .tq2_0 => false,
        .streamed => |*s| s.quant != .q8_0 and s.quant != .tq2_0,
    };
}

/// Prefill dispatch over **4-row-interleaved Q8_Kx4** activations: the lane-packed
/// column-outer kernel. Only reached for `.q4_k` / `.q5_k` / `.q6_k` experts with
/// `m >= 4` (the caller gates on `moeRhsUsesLanePacked` + the packed buffer being
/// present); the `m < 4` tail stays on `moeExpertTileDotRange`. `lhs_x4`
/// holds this expert's `ceil(m/4)` Q8_Kx4 groups.
fn moeExpertTileDotX4Range(rhs: *const MoeRhs, e: usize, lhs_x4: []const backend_mod.quantized_matmul.BlockQ8_Kx4, m: usize, out: []f32, out_dim: usize, c0: usize, c1: usize) void {
    const qm = backend_mod.quantized_matmul;
    switch (rhs.*) {
        .q8_0, .tq2_0 => unreachable, // gated by moeRhsUsesLanePacked
        .streamed => |*s| {
            const bpc = s.blocks_per_column;
            const base = s.expertBytes(e);
            switch (s.quant) {
                .q8_0, .tq2_0 => unreachable, // gated by moeRhsUsesLanePacked
                .q4_k => {
                    const blocks = @as([*]const qm.BlockQ4_K, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = backend_mod.QuantizedMatmulRhsQ4_K{ .allocator = null, .blocks = blocks, .k = s.k, .n = out_dim, .blocks_per_column = bpc };
                    qm.matmulQ4_KCompactQ8_Kx4ColOuter(out, lhs_x4, &view, out_dim, m, c0, c1);
                },
                .q5_k => {
                    const blocks = @as([*]const qm.BlockQ5_K, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = backend_mod.QuantizedMatmulRhsQ5_K{ .allocator = null, .blocks = blocks, .k = s.k, .n = out_dim, .blocks_per_column = bpc };
                    qm.matmulQ5_KCompactQ8_Kx4ColOuter(out, lhs_x4, &view, out_dim, m, c0, c1);
                },
                .q6_k => {
                    const blocks = @as([*]const qm.BlockQ6_K, @ptrCast(@alignCast(base)))[0 .. out_dim * bpc];
                    const view = backend_mod.QuantizedMatmulRhsQ6_K{ .allocator = null, .blocks = blocks, .k = s.k, .n = out_dim, .blocks_per_column = bpc };
                    qm.matmulQ6_KCompactQ8_Kx4ColOuter(out, lhs_x4, &view, out_dim, m, c0, c1);
                },
            }
        },
        .q4_k => |*big| {
            const bpc = big.blocks_per_column;
            const view = backend_mod.QuantizedMatmulRhsQ4_K{ .allocator = big.allocator, .blocks = big.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], .k = big.k, .n = out_dim, .blocks_per_column = bpc };
            qm.matmulQ4_KCompactQ8_Kx4ColOuter(out, lhs_x4, &view, out_dim, m, c0, c1);
        },
        .q5_k => |*big| {
            const bpc = big.blocks_per_column;
            const view = backend_mod.QuantizedMatmulRhsQ5_K{ .allocator = big.allocator, .blocks = big.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], .k = big.k, .n = out_dim, .blocks_per_column = bpc };
            qm.matmulQ5_KCompactQ8_Kx4ColOuter(out, lhs_x4, &view, out_dim, m, c0, c1);
        },
        .q6_k => |*big| {
            const bpc = big.blocks_per_column;
            const view = backend_mod.QuantizedMatmulRhsQ6_K{ .allocator = big.allocator, .blocks = big.blocks[e * out_dim * bpc ..][0 .. out_dim * bpc], .k = big.k, .n = out_dim, .blocks_per_column = bpc };
            qm.matmulQ6_KCompactQ8_Kx4ColOuter(out, lhs_x4, &view, out_dim, m, c0, c1);
        },
    }
}

/// Grow-only scratch backing the single-row MoE decode ops: the per-token region sizes
/// are model constants, so after the first token the hot path performs no
/// allocations and takes one uncontended mutex instead of several
/// allocator/pool round-trips per layer. The mutex is held for the whole
/// op because the expert tasks write into carved slices.
pub const MoeDecodeScratch = struct {
    mutex: thread.Mutex = .{},
    words: []u64 = &.{},

    pub fn deinit(self: *MoeDecodeScratch, allocator: Allocator) void {
        if (self.words.len > 0) allocator.free(self.words);
        self.* = undefined;
    }
};

pub fn lockMoeDecodeScratch(scratch: *MoeDecodeScratch) void {
    scratch.mutex.lock();
}

pub fn unlockMoeDecodeScratch(scratch: *MoeDecodeScratch) void {
    scratch.mutex.unlock();
}

const MoeScratchCarver = struct {
    base: [*]u8,
    offset: usize = 0,

    fn carve(self: *MoeScratchCarver, comptime T: type, n: usize) ![]T {
        const start = std.mem.alignForward(usize, self.offset, @alignOf(T));
        const byte_len = std.math.mul(usize, n, @sizeOf(T)) catch return tensor.TensorError.InvalidDataLength;
        self.offset = std.math.add(usize, start, byte_len) catch return tensor.TensorError.InvalidDataLength;
        const ptr: [*]T = @ptrCast(@alignCast(self.base + start));
        return ptr[0..n];
    }

    fn need(comptime T: type, offset: usize, n: usize) !usize {
        const start = std.mem.alignForward(usize, offset, @alignOf(T));
        const byte_len = std.math.mul(usize, n, @sizeOf(T)) catch return tensor.TensorError.InvalidDataLength;
        return std.math.add(usize, start, byte_len) catch return tensor.TensorError.InvalidDataLength;
    }
};

pub fn MoeDecodeScratchView(comptime QgBlock: type, comptime Task: type) type {
    return struct {
        qx: []backend_mod.quantized_matmul.BlockQ8_K,
        gate_buf: []f32,
        up_buf: []f32,
        g_buf: []f32,
        qg: []QgBlock,
        outs: []f32,
        tasks: []Task,
    };
}

pub fn MoeDecodeChainScratchView(comptime QgBlock: type, comptime State: type, comptime Task: type) type {
    return struct {
        qx: []backend_mod.quantized_matmul.BlockQ8_K,
        gate_buf: []f32,
        up_buf: []f32,
        g_buf: []f32,
        qg: []QgBlock,
        outs: []f32,
        states: []State,
        tasks: []Task,
    };
}

/// Carve the per-token MoE decode scratch regions out of `moe_scratch`,
/// growing it once if needed. Caller must hold `moe_scratch.mutex` for the
/// lifetime of the returned view. All region types align to <= 8 (the u64
/// backing store's natural alignment).
pub fn carveMoeDecodeScratch(
    rt: *Runtime,
    scratch: *MoeDecodeScratch,
    comptime QgBlock: type,
    comptime Task: type,
    hidden_blocks: usize,
    top_k: usize,
    out_pe: usize,
    hidden: usize,
    blocks_per_g: usize,
) !MoeDecodeScratchView(QgBlock, Task) {
    const qm = backend_mod.quantized_matmul;
    comptime {
        if (@alignOf(qm.BlockQ8_K) > 8 or @alignOf(QgBlock) > 8 or @alignOf(Task) > 8) {
            @compileError("MoE scratch regions must align to <= 8");
        }
    }
    const gate_len = try checkedMoeProduct(top_k, out_pe);
    const qg_len = try checkedMoeProduct(top_k, blocks_per_g);
    const out_len = try checkedMoeProduct(top_k, hidden);

    var total: usize = 0;
    total = try MoeScratchCarver.need(qm.BlockQ8_K, total, hidden_blocks);
    total = try MoeScratchCarver.need(f32, total, gate_len);
    total = try MoeScratchCarver.need(f32, total, gate_len);
    total = try MoeScratchCarver.need(f32, total, gate_len);
    total = try MoeScratchCarver.need(QgBlock, total, qg_len);
    total = try MoeScratchCarver.need(f32, total, out_len);
    total = try MoeScratchCarver.need(Task, total, top_k);

    const rounded_total = std.math.add(usize, total, @sizeOf(u64) - 1) catch return tensor.TensorError.InvalidDataLength;
    const words_needed = rounded_total / @sizeOf(u64);
    if (scratch.words.len < words_needed) {
        const grown = try rt.allocator.alloc(u64, words_needed);
        if (scratch.words.len > 0) rt.allocator.free(scratch.words);
        scratch.words = grown;
    }
    var carver = MoeScratchCarver{ .base = @ptrCast(scratch.words.ptr) };
    return .{
        .qx = try carver.carve(qm.BlockQ8_K, hidden_blocks),
        .gate_buf = try carver.carve(f32, gate_len),
        .up_buf = try carver.carve(f32, gate_len),
        .g_buf = try carver.carve(f32, gate_len),
        .qg = try carver.carve(QgBlock, qg_len),
        .outs = try carver.carve(f32, out_len),
        .tasks = try carver.carve(Task, top_k),
    };
}

pub fn carveMoeDecodeChainScratch(
    rt: *Runtime,
    scratch: *MoeDecodeScratch,
    comptime QgBlock: type,
    comptime State: type,
    comptime Task: type,
    hidden_blocks: usize,
    top_k: usize,
    out_pe: usize,
    hidden: usize,
    blocks_per_g: usize,
    task_count: usize,
) !MoeDecodeChainScratchView(QgBlock, State, Task) {
    const qm = backend_mod.quantized_matmul;
    comptime {
        if (@alignOf(qm.BlockQ8_K) > 8 or @alignOf(QgBlock) > 8 or @alignOf(State) > 8 or @alignOf(Task) > 8) {
            @compileError("MoE scratch regions must align to <= 8");
        }
    }
    const gate_len = try checkedMoeProduct(top_k, out_pe);
    const qg_len = try checkedMoeProduct(top_k, blocks_per_g);
    const out_len = try checkedMoeProduct(top_k, hidden);

    var total: usize = 0;
    total = try MoeScratchCarver.need(qm.BlockQ8_K, total, hidden_blocks);
    total = try MoeScratchCarver.need(f32, total, gate_len);
    total = try MoeScratchCarver.need(f32, total, gate_len);
    total = try MoeScratchCarver.need(f32, total, gate_len);
    total = try MoeScratchCarver.need(QgBlock, total, qg_len);
    total = try MoeScratchCarver.need(f32, total, out_len);
    total = try MoeScratchCarver.need(State, total, top_k);
    total = try MoeScratchCarver.need(Task, total, task_count);

    const rounded_total = std.math.add(usize, total, @sizeOf(u64) - 1) catch return tensor.TensorError.InvalidDataLength;
    const words_needed = rounded_total / @sizeOf(u64);
    if (scratch.words.len < words_needed) {
        const grown = try rt.allocator.alloc(u64, words_needed);
        if (scratch.words.len > 0) rt.allocator.free(scratch.words);
        scratch.words = grown;
    }
    var carver = MoeScratchCarver{ .base = @ptrCast(scratch.words.ptr) };
    return .{
        .qx = try carver.carve(qm.BlockQ8_K, hidden_blocks),
        .gate_buf = try carver.carve(f32, gate_len),
        .up_buf = try carver.carve(f32, gate_len),
        .g_buf = try carver.carve(f32, gate_len),
        .qg = try carver.carve(QgBlock, qg_len),
        .outs = try carver.carve(f32, out_len),
        .states = try carver.carve(State, top_k),
        .tasks = try carver.carve(Task, task_count),
    };
}

const MoeExpertTask = struct {
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    qx: []const backend_mod.quantized_matmul.BlockQ8_K,
    qx8: []const backend_mod.quantized_matmul.BlockQ8_0,
    out_pe: usize,
    hidden: usize,
    expert_index: usize,
    weight: f32,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_K,
    qg8: []backend_mod.quantized_matmul.BlockQ8_0,
    out: []f32,
    gated_op: GatedOp,
    profile_enabled: bool,
    io: ?std.Io,
    gate_up_ns: i64,
    swiglu_requant_ns: i64,
    down_ns: i64,
};

fn runMoeExpertTask(task: *const MoeExpertTask) void {
    const task_profile = @constCast(task);
    const gate_up_start = moeBatchProfileStart(task.profile_enabled, task.io);
    moeExpertTileDot(task.gate, task.expert_index, task.qx, task.qx8, task.gate_buf, task.out_pe, 1);
    moeExpertTileDot(task.up, task.expert_index, task.qx, task.qx8, task.up_buf, task.out_pe, 1);
    if (task.profile_enabled) task_profile.gate_up_ns += moeBatchProfileElapsed(gate_up_start, task.io);

    // Gated activation: g = up * act(gate). `inline else` specializes the loop per op.
    const swiglu_requant_start = moeBatchProfileStart(task.profile_enabled, task.io);
    switch (task.gated_op) {
        inline else => |op| for (task.g_buf, task.gate_buf, task.up_buf) |*g, gate_v, up_v| {
            g.* = backend_ops.gatedPairScalar(op, gate_v, up_v);
        },
    }
    const requant_ok = if (task.down.wantsQ8_0Lhs())
        backend_mod.quantized_matmul.quantizeRowQ8_0Into(task.qg8, task.g_buf)
    else
        backend_mod.quantized_matmul.quantizeRowQ8_KInto(task.qg, task.g_buf);
    requant_ok catch {
        @memset(task.out, 0);
        return;
    };
    if (task.profile_enabled) task_profile.swiglu_requant_ns += moeBatchProfileElapsed(swiglu_requant_start, task.io);

    const down_start = moeBatchProfileStart(task.profile_enabled, task.io);
    moeExpertTileDot(task.down, task.expert_index, task.qg, task.qg8, task.out, task.hidden, 1);
    if (task.weight != 1.0) {
        for (task.out) |*o| o.* *= task.weight;
    }
    if (task.profile_enabled) task_profile.down_ns += moeBatchProfileElapsed(down_start, task.io);
}

const MoeDecodeChainState = struct {
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    qx: []const backend_mod.quantized_matmul.BlockQ8_K,
    qx8: []const backend_mod.quantized_matmul.BlockQ8_0,
    out_pe: usize,
    hidden: usize,
    expert_index: usize,
    weight: f32,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_K,
    qg8: []backend_mod.quantized_matmul.BlockQ8_0,
    out: []f32,
    gated_op: GatedOp,
    profile_enabled: bool,
    io: ?std.Io,
    remaining_gate_up: std.atomic.Value(u32),
    swiglu_requant_ns: i64,
    down_task0: usize,
};

const MoeDecodeChainTask = struct {
    state: *MoeDecodeChainState,
    kind: enum { gate_up, down },
    c0: usize,
    c1: usize,
    elapsed_ns: i64,
};

fn runMoeDecodeChainTask(task: *MoeDecodeChainTask, chain: *const thread.Chain) void {
    const state = task.state;
    switch (task.kind) {
        .gate_up => {
            const gate_up_start = moeBatchProfileStart(state.profile_enabled, state.io);
            moeExpertTileDotRange(state.gate, state.expert_index, state.qx, state.qx8, state.gate_buf, state.out_pe, 1, task.c0, task.c1);
            moeExpertTileDotRange(state.up, state.expert_index, state.qx, state.qx8, state.up_buf, state.out_pe, 1, task.c0, task.c1);
            if (state.profile_enabled) task.elapsed_ns += moeBatchProfileElapsed(gate_up_start, state.io);

            if (state.remaining_gate_up.fetchSub(1, .acq_rel) == 1) {
                const swiglu_requant_start = moeBatchProfileStart(state.profile_enabled, state.io);
                switch (state.gated_op) {
                    inline else => |op| for (state.g_buf, state.gate_buf, state.up_buf) |*g, gate_v, up_v| {
                        g.* = backend_ops.gatedPairScalar(op, gate_v, up_v);
                    },
                }
                if (state.down.wantsQ8_0Lhs())
                    backend_mod.quantized_matmul.quantizeRowQ8_0Into(state.qg8, state.g_buf) catch unreachable
                else
                    backend_mod.quantized_matmul.quantizeRowQ8_KInto(state.qg, state.g_buf) catch unreachable;
                if (state.profile_enabled) state.swiglu_requant_ns += moeBatchProfileElapsed(swiglu_requant_start, state.io);
                chain.enqueue(state.down_task0);
                chain.enqueue(state.down_task0 + 1);
            }
        },
        .down => {
            const down_start = moeBatchProfileStart(state.profile_enabled, state.io);
            moeExpertTileDotRange(state.down, state.expert_index, state.qg, state.qg8, state.out, state.hidden, 1, task.c0, task.c1);
            if (state.weight != 1.0) {
                for (state.out[task.c0..task.c1]) |*o| o.* *= state.weight;
            }
            if (state.profile_enabled) task.elapsed_ns += moeBatchProfileElapsed(down_start, state.io);
        },
    }
}

/// Fused MoE FFN for a single token: route-weighted sum over the selected
/// experts of down(SwiGLU(gate(x), up(x))). The whole layer's expert work runs
/// in ONE pooled dispatch — one task per selected expert, each computing its
/// full gate/up/SwiGLU/down single-threaded, so `top_k` experts map onto the
/// worker cores at active-param bandwidth instead of dispatching ~25 tiny
/// GEMV/elementwise ops per layer (which made decode ~1.6x slower, and a
/// finer 3-phase split slower still from the extra barriers). `x` is
/// (1, hidden) of K-quant experts; returns (1, hidden).
pub fn moeExpertFfn(
    rt: *Runtime,
    scratch: *MoeDecodeScratch,
    x: *const Tensor,
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    selected: []const usize,
    weights: []const f32,
    out_pe: usize,
    act: GatedOp,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const av = try x.rankView(2);
    if (av.dim(0) != 1) return tensor.TensorError.ShapeMismatch;
    const hidden = av.dim(1);
    const top_k = selected.len;
    _ = try validatePackedMoeInputs(gate, up, down, selected, weights, top_k, hidden, out_pe);
    const profile_enabled = profile != null;
    const total_start = moeBatchProfileStart(profile_enabled, io);

    // Streamed experts: resolve/read the routed set before any task spawns;
    // the guard's release (after compute) promotes misses into the LRU.
    var stream_guard = try acquireMoeStreamed(gate, up, down, selected);
    defer stream_guard.release();

    // Per-projection activation formats: K-quant arms read Q8_K rows, q8_0
    // arms (deepseek2 experts) read Q8_0 rows; both forms are produced only
    // when some arm needs them.
    const gate_up_q8 = gate.wantsQ8_0Lhs() or up.wantsQ8_0Lhs();
    const gate_up_qk = !gate.wantsQ8_0Lhs() or !up.wantsQ8_0Lhs();
    const down_q8 = down.wantsQ8_0Lhs();

    const blocks_per_g = if (down_q8) 0 else try qm.qkBlockCount(out_pe);
    const blocks_per_g8 = if (down_q8) out_pe / 32 else 0;
    const hidden_blocks_k = if (gate_up_qk) try qm.qkBlockCount(hidden) else 0;
    const chain_task_count = try checkedMoeProduct(4, top_k);
    const chain_initial_count = try checkedMoeProduct(2, top_k);

    // Q8_0-format activations live outside the carved scratch (only the
    // deepseek2-style layers pay this allocation).
    const qx8: []qm.BlockQ8_0 = if (gate_up_q8) try rt.allocator.alloc(qm.BlockQ8_0, hidden / 32) else &.{};
    defer if (qx8.len > 0) rt.allocator.free(qx8);
    const qg8_all: []qm.BlockQ8_0 = if (down_q8) try rt.allocator.alloc(qm.BlockQ8_0, try checkedMoeProduct(top_k, blocks_per_g8)) else &.{};
    defer if (qg8_all.len > 0) rt.allocator.free(qg8_all);

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    scratch.mutex.lock();
    defer scratch.mutex.unlock();
    const sv = try carveMoeDecodeChainScratch(rt, scratch, qm.BlockQ8_K, MoeDecodeChainState, MoeDecodeChainTask, hidden_blocks_k, top_k, out_pe, hidden, blocks_per_g, chain_task_count);
    const gate_buf = sv.gate_buf;
    const up_buf = sv.up_buf;
    const g_buf = sv.g_buf;
    const qg = sv.qg;
    const outs = sv.outs;
    const tasks = sv.tasks;
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    const gather_quant_start = moeBatchProfileStart(profile_enabled, io);
    const qx = sv.qx;
    const x_data = try x.dataConstChecked();
    if (gate_up_qk) try qm.quantizeRowQ8_KInto(qx, x_data);
    if (gate_up_q8) try qm.quantizeRowQ8_0Into(qx8, x_data);
    if (profile) |p| p.gather_quant_ns += moeBatchProfileElapsed(gather_quant_start, io);

    const gate_split = moeDecodeColumnSplit(out_pe, 32);
    const down_split = moeDecodeColumnSplit(hidden, 32);
    for (sv.states, 0..) |*state, j| {
        const down_task_offset = try checkedMoeProduct(2, j);
        const down_task0 = std.math.add(usize, chain_initial_count, down_task_offset) catch return tensor.TensorError.InvalidDataLength;
        state.* = .{
            .gate = gate,
            .up = up,
            .down = down,
            .qx = qx,
            .qx8 = qx8,
            .out_pe = out_pe,
            .hidden = hidden,
            .expert_index = selected[j],
            .weight = weights[j],
            .gate_buf = gate_buf[j * out_pe ..][0..out_pe],
            .up_buf = up_buf[j * out_pe ..][0..out_pe],
            .g_buf = g_buf[j * out_pe ..][0..out_pe],
            .qg = qg[j * blocks_per_g ..][0..blocks_per_g],
            .qg8 = qg8_all[j * blocks_per_g8 ..][0..blocks_per_g8],
            .out = outs[j * hidden ..][0..hidden],
            .gated_op = act,
            .profile_enabled = profile_enabled,
            .io = io,
            .remaining_gate_up = .init(2),
            .swiglu_requant_ns = 0,
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
    if (rt.workPool()) |pool| {
        used_chain = pool.parallelChained(MoeDecodeChainTask, tasks, chain_initial_count, runMoeDecodeChainTask);
    }
    if (!used_chain) {
        for (0..top_k) |j| {
            var t = MoeExpertTask{
                .gate = gate,
                .up = up,
                .down = down,
                .qx = qx,
                .qx8 = qx8,
                .out_pe = out_pe,
                .hidden = hidden,
                .expert_index = selected[j],
                .weight = weights[j],
                .gate_buf = gate_buf[j * out_pe ..][0..out_pe],
                .up_buf = up_buf[j * out_pe ..][0..out_pe],
                .g_buf = g_buf[j * out_pe ..][0..out_pe],
                .qg = qg[j * blocks_per_g ..][0..blocks_per_g],
                .qg8 = qg8_all[j * blocks_per_g8 ..][0..blocks_per_g8],
                .out = outs[j * hidden ..][0..hidden],
                .gated_op = act,
                .profile_enabled = profile_enabled,
                .io = io,
                .gate_up_ns = 0,
                .swiglu_requant_ns = 0,
                .down_ns = 0,
            };
            runMoeExpertTask(&t);
            if (profile) |p| {
                p.gate_up_ns += t.gate_up_ns;
                p.swiglu_requant_ns += t.swiglu_requant_ns;
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
            for (sv.states) |*state| p.swiglu_requant_ns += state.swiglu_requant_ns;
        }
        p.batches += 1;
        p.pairs += top_k;
        p.active_experts += top_k;
        p.max_expert_m = @max(p.max_expert_m, 1);
    }

    const out_alloc_start = moeBatchProfileStart(profile_enabled, io);
    var out = try rt.emptyRank(2, .{ 1, hidden });
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

const MoeBatchTask = struct {
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    out_pe: usize,
    top_k: usize,
    bpc_in: usize,
    blocks_per_g: usize,
    row_start: usize, // first row (in the expert-grouped order) for this expert
    m: usize, // rows routed to this expert
    expert: usize,
    qx: []backend_mod.quantized_matmul.BlockQ8_K,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_K,
    down_buf: []f32,
    gated_op: GatedOp,
    profile_enabled: bool,
    io: ?std.Io,
    gather_quant_ns: i128,
    gate_up_ns: i128,
    swiglu_requant_ns: i128,
    down_ns: i128,
};

fn runMoeBatchTask(task: *const MoeBatchTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const out_pe = task.out_pe;
    const hidden = task.hidden;
    const bpc_in = task.bpc_in;
    const bpc_g = task.blocks_per_g;
    const base = task.row_start;

    // Gather this expert's input rows and quantize them to Q8_K.
    const gather_quant_start = moeBatchProfileStart(task.profile_enabled, task.io);
    for (0..m) |i| {
        const token = task.order[base + i] / task.top_k;
        const src = task.x_data[token * hidden ..][0..hidden];
        qm.quantizeRowQ8_KInto(task.qx[(base + i) * bpc_in ..][0..bpc_in], src) catch {
            @memset(task.down_buf[base * hidden ..][0 .. m * hidden], 0);
            return;
        };
    }
    if (task.profile_enabled) task_profile.gather_quant_ns += moeBatchProfileElapsed(gather_quant_start, task.io);

    const qx = task.qx[base * bpc_in ..][0 .. m * bpc_in];
    const gate_out = task.gate_buf[base * out_pe ..][0 .. m * out_pe];
    const up_out = task.up_buf[base * out_pe ..][0 .. m * out_pe];
    const g_out = task.g_buf[base * out_pe ..][0 .. m * out_pe];

    const gate_up_start = moeBatchProfileStart(task.profile_enabled, task.io);
    const no_q8: []const backend_mod.quantized_matmul.BlockQ8_0 = &.{};
    moeExpertTileDot(task.gate, task.expert, qx, no_q8, gate_out, out_pe, m);
    moeExpertTileDot(task.up, task.expert, qx, no_q8, up_out, out_pe, m);
    if (task.profile_enabled) task_profile.gate_up_ns += moeBatchProfileElapsed(gate_up_start, task.io);

    const swiglu_requant_start = moeBatchProfileStart(task.profile_enabled, task.io);
    switch (task.gated_op) {
        inline else => |op| for (g_out, gate_out, up_out) |*g, gate_v, up_v| {
            g.* = backend_ops.gatedPairScalar(op, gate_v, up_v);
        },
    }
    for (0..m) |i| {
        qm.quantizeRowQ8_KInto(task.qg[(base + i) * bpc_g ..][0..bpc_g], g_out[i * out_pe ..][0..out_pe]) catch {
            @memset(task.down_buf[base * hidden ..][0 .. m * hidden], 0);
            return;
        };
    }
    if (task.profile_enabled) task_profile.swiglu_requant_ns += moeBatchProfileElapsed(swiglu_requant_start, task.io);

    const qg = task.qg[base * bpc_g ..][0 .. m * bpc_g];
    const down_start = moeBatchProfileStart(task.profile_enabled, task.io);
    moeExpertTileDot(task.down, task.expert, qg, no_q8, task.down_buf[base * hidden ..][0 .. m * hidden], hidden, m);
    if (task.profile_enabled) task_profile.down_ns += moeBatchProfileElapsed(down_start, task.io);
}

// Master switch for the 4-row lane-packed Q8_Kx4 column-outer kernels (Q5_K/Q6_K)
// in batched MoE prefill. Flip to false to fall back to the per-row column-outer
// path (used for A/B benchmarking; the per-row path stays bit-identical).
const moe_lane_packed_enabled = true;

const MoeBatchGatherTask = struct {
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    top_k: usize,
    bpc_in: usize,
    row_start: usize,
    m: usize,
    qx: []backend_mod.quantized_matmul.BlockQ8_K,
    // Optional 4-row-interleaved repack of `qx` for the lane-packed Q5_K kernel.
    // Empty when the gate/up experts are not q5_k. `x4_group_start` is this
    // expert's first Q8_Kx4 group index (prefix sum of ceil(m/4)).
    qx_x4: []backend_mod.quantized_matmul.BlockQ8_Kx4,
    x4_group_start: usize,
    profile_enabled: bool,
    io: ?std.Io,
    gather_quant_ns: i128,
};

fn runMoeBatchGatherTask(task: *const MoeBatchGatherTask) void {
    const qm = backend_mod.quantized_matmul;
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    for (0..m) |i| {
        const token = task.order[base + i] / task.top_k;
        const src = task.x_data[token * task.hidden ..][0..task.hidden];
        qm.quantizeRowQ8_KInto(task.qx[(base + i) * task.bpc_in ..][0..task.bpc_in], src) catch return;
    }
    // The lane-packed kernel only engages at m >= 4 (runMoeBatchMatmulTask),
    // so an m < 4 expert's packed groups are never read: skip the repack.
    // `group_offset` prefix sums still reserve the groups, so neighbors are
    // unaffected.
    if (task.qx_x4.len > 0 and m >= 4) {
        const groups = (m + 3) / 4;
        qm.packRowsQ8_Kx4PaddedInto(
            task.qx_x4[task.x4_group_start * task.bpc_in ..][0 .. groups * task.bpc_in],
            task.qx[base * task.bpc_in ..][0 .. m * task.bpc_in],
            m,
            task.bpc_in,
        );
    }
    if (task.profile_enabled) task_profile.gather_quant_ns += moeBatchProfileElapsed(start, task.io);
}

fn runMoeBatchGatherTaskOpaque(ctx: *anyopaque) void {
    const task: *const MoeBatchGatherTask = @ptrCast(@alignCast(ctx));
    runMoeBatchGatherTask(task);
}

const MoeBatchMatmulTask = struct {
    rhs: *const MoeRhs,
    qlhs: []const backend_mod.quantized_matmul.BlockQ8_K,
    // 4-row-interleaved repack of `qlhs`; empty unless `rhs` is q5_k. Used for the
    // lane-packed kernel when `m >= 4`. `x4_group_start` is the expert's first group.
    qlhs_x4: []const backend_mod.quantized_matmul.BlockQ8_Kx4,
    x4_group_start: usize,
    bpc: usize,
    row_start: usize,
    m: usize,
    out_dim: usize,
    expert: usize,
    out: []f32,
    c0: usize,
    c1: usize,
    profile_enabled: bool,
    io: ?std.Io,
    elapsed_ns: i128,
};

fn runMoeBatchMatmulTask(task: *const MoeBatchMatmulTask) void {
    const m = task.m;
    if (m == 0) return;
    const task_profile = @constCast(task);
    const start = moeBatchProfileStart(task.profile_enabled, task.io);
    const base = task.row_start;
    const out = task.out[base * task.out_dim ..][0 .. m * task.out_dim];
    if (task.qlhs_x4.len > 0 and m >= 4) {
        const groups = (m + 3) / 4;
        const lhs_x4 = task.qlhs_x4[task.x4_group_start * task.bpc ..][0 .. groups * task.bpc];
        moeExpertTileDotX4Range(task.rhs, task.expert, lhs_x4, m, out, task.out_dim, task.c0, task.c1);
    } else {
        const q = task.qlhs[base * task.bpc ..][0 .. m * task.bpc];
        const no_q8: []const backend_mod.quantized_matmul.BlockQ8_0 = &.{};
        moeExpertTileDotRange(task.rhs, task.expert, q, no_q8, out, task.out_dim, m, task.c0, task.c1);
    }
    if (task.profile_enabled) task_profile.elapsed_ns += moeBatchProfileElapsed(start, task.io);
}

fn runMoeBatchMatmulTaskOpaque(ctx: *anyopaque) void {
    const task: *const MoeBatchMatmulTask = @ptrCast(@alignCast(ctx));
    runMoeBatchMatmulTask(task);
}

const MoeBatchSwiGluTask = struct {
    gate_buf: []const f32,
    up_buf: []const f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_K,
    // Optional 4-row-interleaved repack of `qg` for the lane-packed Q5_K down-proj.
    // Empty when the down experts are not q5_k.
    qg_x4: []backend_mod.quantized_matmul.BlockQ8_Kx4,
    x4_group_start: usize,
    out_pe: usize,
    blocks_per_g: usize,
    row_start: usize,
    m: usize,
    gated_op: GatedOp,
    profile_enabled: bool,
    io: ?std.Io,
    swiglu_requant_ns: i128,
};

fn runMoeBatchSwiGluTask(task: *const MoeBatchSwiGluTask) void {
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
    switch (task.gated_op) {
        inline else => |op| for (g_out, gate_out, up_out) |*g, gate_v, up_v| {
            g.* = backend_ops.gatedPairScalar(op, gate_v, up_v);
        },
    }
    for (0..m) |i| {
        qm.quantizeRowQ8_KInto(task.qg[(base + i) * task.blocks_per_g ..][0..task.blocks_per_g], g_out[i * out_pe ..][0..out_pe]) catch return;
    }
    // Same m >= 4 gate as the gather repack: m < 4 packed groups are dead.
    if (task.qg_x4.len > 0 and m >= 4) {
        const groups = (m + 3) / 4;
        qm.packRowsQ8_Kx4PaddedInto(
            task.qg_x4[task.x4_group_start * task.blocks_per_g ..][0 .. groups * task.blocks_per_g],
            task.qg[base * task.blocks_per_g ..][0 .. m * task.blocks_per_g],
            m,
            task.blocks_per_g,
        );
    }
    if (task.profile_enabled) task_profile.swiglu_requant_ns += moeBatchProfileElapsed(start, task.io);
}

fn runMoeBatchSwiGluTaskOpaque(ctx: *anyopaque) void {
    const task: *const MoeBatchSwiGluTask = @ptrCast(@alignCast(ctx));
    runMoeBatchSwiGluTask(task);
}

fn runMoeBatchPhased(
    rt: *Runtime,
    pool: *thread.Pool,
    active_experts: usize,
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    x_data: []const f32,
    order: []const usize,
    hidden: usize,
    out_pe: usize,
    top_k: usize,
    bpc_in: usize,
    blocks_per_g: usize,
    count: []const usize,
    offset: []const usize,
    qx: []backend_mod.quantized_matmul.BlockQ8_K,
    gate_buf: []f32,
    up_buf: []f32,
    g_buf: []f32,
    qg: []backend_mod.quantized_matmul.BlockQ8_K,
    down_buf: []f32,
    group_offset: []const usize,
    qx_x4: []backend_mod.quantized_matmul.BlockQ8_Kx4,
    qg_x4: []backend_mod.quantized_matmul.BlockQ8_Kx4,
    act: GatedOp,
    profile_enabled: bool,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !void {
    const n_expert = count.len;
    // The lane-packed Q5_K kernel only applies to q5_k experts; per-projection
    // tag check selects the packed LHS (built in the gather/swiglu phases) vs the
    // per-row tile. Empty `qx_x4`/`qg_x4` (non-q5_k model) keeps everything as-is.
    const empty_x4: []const backend_mod.quantized_matmul.BlockQ8_Kx4 = &.{};
    const qx_x4_const: []const backend_mod.quantized_matmul.BlockQ8_Kx4 = qx_x4;
    const qg_x4_const: []const backend_mod.quantized_matmul.BlockQ8_Kx4 = qg_x4;
    const gate_uses_x4 = moeRhsUsesLanePacked(gate);
    const up_uses_x4 = moeRhsUsesLanePacked(up);
    const down_uses_x4 = moeRhsUsesLanePacked(down);

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const gather_tasks = try rt.allocator.alloc(MoeBatchGatherTask, n_expert);
    defer rt.allocator.free(gather_tasks);
    const swiglu_tasks = try rt.allocator.alloc(MoeBatchSwiGluTask, n_expert);
    defer rt.allocator.free(swiglu_tasks);

    // Small-m column chunking is a per-layer-call decision: with few active
    // experts each contributing one full-width task per projection, the
    // team starves; the width helpers keep the counting and construction
    // loops in exact agreement (the chain's enqueue contract).
    const workers = pool.teamSize();
    const small_m_width = moeSmallMColWidth(active_experts, workers);

    var gate_up_task_count: usize = 0;
    var down_task_count: usize = 0;
    var active_count: usize = 0;
    for (count) |m| {
        if (m == 0) continue;
        active_count = std.math.add(usize, active_count, 1) catch return tensor.TensorError.InvalidDataLength;
        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        const gate_up_chunks = try checkedMoeProduct(2, moePhaseChunkCount(gu_width, out_pe));
        gate_up_task_count = std.math.add(usize, gate_up_task_count, gate_up_chunks) catch return tensor.TensorError.InvalidDataLength;
        const d_width = moePhaseColWidth(m, hidden, small_m_width);
        down_task_count = std.math.add(usize, down_task_count, moePhaseChunkCount(d_width, hidden)) catch return tensor.TensorError.InvalidDataLength;
    }
    const gate_up_tasks = try rt.allocator.alloc(MoeBatchMatmulTask, gate_up_task_count);
    defer rt.allocator.free(gate_up_tasks);
    const down_tasks = try rt.allocator.alloc(MoeBatchMatmulTask, down_task_count);
    defer rt.allocator.free(down_tasks);
    const chain_states = try rt.allocator.alloc(MoeBatchPhaseChainState, n_expert);
    defer rt.allocator.free(chain_states);
    const active_chain_tasks = try checkedMoeProduct(active_count, 2);
    const matmul_chain_tasks = std.math.add(usize, gate_up_task_count, down_task_count) catch return tensor.TensorError.InvalidDataLength;
    const chain_task_count = std.math.add(usize, active_chain_tasks, matmul_chain_tasks) catch return tensor.TensorError.InvalidDataLength;
    const chain_tasks = try rt.allocator.alloc(MoeBatchPhaseChainTask, chain_task_count);
    defer rt.allocator.free(chain_tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    for (0..n_expert) |e| {
        gather_tasks[e] = .{
            .x_data = x_data,
            .order = order,
            .hidden = hidden,
            .top_k = top_k,
            .bpc_in = bpc_in,
            .row_start = offset[e],
            .m = count[e],
            .qx = qx,
            .qx_x4 = qx_x4,
            .x4_group_start = group_offset[e],
            .profile_enabled = profile_enabled,
            .io = io,
            .gather_quant_ns = 0,
        };
        swiglu_tasks[e] = .{
            .gate_buf = gate_buf,
            .up_buf = up_buf,
            .g_buf = g_buf,
            .qg = qg,
            .qg_x4 = qg_x4,
            .x4_group_start = group_offset[e],
            .out_pe = out_pe,
            .blocks_per_g = blocks_per_g,
            .row_start = offset[e],
            .m = count[e],
            .gated_op = act,
            .profile_enabled = profile_enabled,
            .io = io,
            .swiglu_requant_ns = 0,
        };
    }

    var gate_up_i: usize = 0;
    var down_i: usize = 0;
    for (0..n_expert) |e| {
        const m = count[e];
        if (m == 0) continue;

        const gu_width = moePhaseColWidth(m, out_pe, small_m_width);
        const gu_chunks = moePhaseChunkCount(gu_width, out_pe);
        const gate_task_start = gate_up_i;
        for (0..gu_chunks) |chunk| {
            const bounds = moePhaseChunkBounds(chunk, gu_width, out_pe);
            gate_up_tasks[gate_up_i] = .{
                .rhs = gate,
                .qlhs = qx,
                .qlhs_x4 = if (gate_uses_x4) qx_x4_const else empty_x4,
                .x4_group_start = group_offset[e],
                .bpc = bpc_in,
                .row_start = offset[e],
                .m = m,
                .out_dim = out_pe,
                .expert = e,
                .out = gate_buf,
                .c0 = bounds.c0,
                .c1 = bounds.c1,
                .profile_enabled = profile_enabled,
                .io = io,
                .elapsed_ns = 0,
            };
            gate_up_i += 1;
            gate_up_tasks[gate_up_i] = .{
                .rhs = up,
                .qlhs = qx,
                .qlhs_x4 = if (up_uses_x4) qx_x4_const else empty_x4,
                .x4_group_start = group_offset[e],
                .bpc = bpc_in,
                .row_start = offset[e],
                .m = m,
                .out_dim = out_pe,
                .expert = e,
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
                .rhs = down,
                .qlhs = qg,
                .qlhs_x4 = if (down_uses_x4) qg_x4_const else empty_x4,
                .x4_group_start = group_offset[e],
                .bpc = blocks_per_g,
                .row_start = offset[e],
                .m = m,
                .out_dim = hidden,
                .expert = e,
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
        MoeBatchGatherTask,
        MoeBatchMatmulTask,
        MoeBatchSwiGluTask,
        MoeBatchMatmulTask,
        chain_tasks,
        chain_states,
        count,
        gather_tasks,
        gate_up_tasks,
        swiglu_tasks,
        down_tasks,
        runMoeBatchGatherTaskOpaque,
        runMoeBatchMatmulTaskOpaque,
        runMoeBatchSwiGluTaskOpaque,
        runMoeBatchMatmulTaskOpaque,
    );

    var expert_wall_ns: i128 = 0;
    var phase_start = moeBatchProfileStart(profile_enabled, io);
    const used_chain = pool.parallelChained(MoeBatchPhaseChainTask, chain_tasks, chain_initial_count, runMoeBatchPhaseChainTask);
    if (used_chain) {
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    } else {
        pool.parallelChunks(MoeBatchGatherTask, gather_tasks, runMoeBatchGatherTask);
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        pool.parallelChunks(MoeBatchMatmulTask, gate_up_tasks, runMoeBatchMatmulTask);
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        pool.parallelChunks(MoeBatchSwiGluTask, swiglu_tasks, runMoeBatchSwiGluTask);
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);

        phase_start = moeBatchProfileStart(profile_enabled, io);
        pool.parallelChunks(MoeBatchMatmulTask, down_tasks, runMoeBatchMatmulTask);
        if (profile_enabled) expert_wall_ns += moeBatchProfileElapsed(phase_start, io);
    }

    if (profile) |p| {
        p.expert_wall_ns += expert_wall_ns;
        for (gather_tasks) |*t| p.gather_quant_ns += t.gather_quant_ns;
        for (gate_up_tasks) |*t| p.gate_up_ns += t.elapsed_ns;
        for (swiglu_tasks) |*t| p.swiglu_requant_ns += t.swiglu_requant_ns;
        for (down_tasks) |*t| p.down_ns += t.elapsed_ns;
    }
}

const MoeBatchScatterTask = struct {
    out: []f32,
    down_buf: []const f32,
    weights: []const f32,
    inv: []const usize,
    hidden: usize,
    top_k: usize,
    t0: usize,
    t1: usize,
};

fn runMoeBatchScatterTask(task: *const MoeBatchScatterTask) void {
    moe_chain.scatterTokenMajor(task.out, task.down_buf, task.weights, task.inv, task.hidden, task.top_k, task.t0, task.t1);
}

/// Batched-prefill MoE FFN over `seq > 1` tokens: route-weighted sum over each
/// token's top-k experts of down(SwiGLU(gate, up)). Tokens are grouped by
/// expert (counting sort) so each expert runs ONE m>1 GEMM over all its routed
/// tokens — its weights are read from RAM once (reused across rows from cache)
/// instead of re-read per token, and the whole layer is one pooled dispatch
/// over experts. `x` is (seq, hidden); `selected`/`weights` are seq*top_k
/// (row-major per token). Returns (seq, hidden).
pub fn moeExpertFfnBatch(
    rt: *Runtime,
    x: *const Tensor,
    gate: *const MoeRhs,
    up: *const MoeRhs,
    down: *const MoeRhs,
    selected: []const usize,
    weights: []const f32,
    top_k: usize,
    out_pe: usize,
    act: GatedOp,
    io: ?std.Io,
    profile: ?*MoeBatchProfile,
) !Tensor {
    const qm = backend_mod.quantized_matmul;
    const a = rt.allocator;
    const av = try x.rankView(2);
    const seq = av.dim(0);
    const hidden = av.dim(1);
    const x_data = try x.dataConstChecked();
    const n_pairs = try checkedMoeProduct(seq, top_k);
    // Batched prefill still assumes Q8_K activations end to end; q8_0
    // experts (deepseek2) decode through moeExpertFfn's dual-format path
    // and prefill sequentially until the batched path grows the same.
    if (gate.wantsQ8_0Lhs() or up.wantsQ8_0Lhs() or down.wantsQ8_0Lhs()) return tensor.TensorError.InvalidShape;
    const n_expert = try validatePackedMoeInputs(gate, up, down, selected, weights, n_pairs, hidden, out_pe);
    const profile_enabled = profile != null;
    const total_start = moeBatchProfileStart(profile_enabled, io);

    // Streamed experts: resolve the batch's whole routed union up front
    // (each missing expert read once, reused by every routed token —
    // batch-union streaming); released after compute.
    var stream_guard = try acquireMoeStreamed(gate, up, down, selected);
    defer stream_guard.release();

    const bpc_in = try qm.qkBlockCount(hidden);
    const bpc_g = try qm.qkBlockCount(out_pe);

    // Counting sort of (token,expert) pairs by expert.
    const route_result = try moe_chain.buildMoeRoutePlan(a, selected, n_expert, profile_enabled, io);
    var route = route_result.plan;
    defer route.deinit();
    if (profile) |p| {
        p.alloc_ns += route_result.alloc_ns;
        p.count_sort_ns += route_result.count_sort_ns;
    }
    const count = route.count;
    const offset = route.offset;
    const order = route.order;
    const inv = route.inv;

    const alloc_start = moeBatchProfileStart(profile_enabled, io);
    const group_offset = try a.alloc(usize, n_expert);
    defer a.free(group_offset);
    const qx = try a.alloc(qm.BlockQ8_K, try checkedMoeProduct(n_pairs, bpc_in));
    defer a.free(qx);
    const gate_up_len = try checkedMoeProduct(n_pairs, out_pe);
    const gate_buf = try a.alloc(f32, gate_up_len);
    defer a.free(gate_buf);
    const up_buf = try a.alloc(f32, gate_up_len);
    defer a.free(up_buf);
    const g_buf = try a.alloc(f32, gate_up_len);
    defer a.free(g_buf);
    const qg = try a.alloc(qm.BlockQ8_K, try checkedMoeProduct(n_pairs, bpc_g));
    defer a.free(qg);
    const down_buf = try a.alloc(f32, try checkedMoeProduct(n_pairs, hidden));
    defer a.free(down_buf);
    const tasks = try a.alloc(MoeBatchTask, n_expert);
    defer a.free(tasks);
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(alloc_start, io);

    const use_phased = n_pairs >= moe_chain.moe_batch_phase_min_pairs;

    // Lane-packed Q5_K prefill: repack each expert's Q8_K activations into
    // 4-row Q8_Kx4 groups (qx for gate/up, qg for down). `group_offset[e]` is the
    // expert's first group; buffers stay empty for non-q5_k models or small
    // (non-phased) batches, leaving the per-row path unchanged.
    const gate_up_x4 = moe_lane_packed_enabled and use_phased and (moeRhsUsesLanePacked(gate) or moeRhsUsesLanePacked(up));
    const down_x4 = moe_lane_packed_enabled and use_phased and moeRhsUsesLanePacked(down);
    var total_groups: usize = 0;
    for (0..n_expert) |e| {
        group_offset[e] = total_groups;
        total_groups += (count[e] + 3) / 4;
    }
    const qx_x4_len = if (gate_up_x4) try checkedMoeProduct(total_groups, bpc_in) else 0;
    const qx_x4 = try a.alloc(qm.BlockQ8_Kx4, qx_x4_len);
    defer a.free(qx_x4);
    const qg_x4_len = if (down_x4) try checkedMoeProduct(total_groups, bpc_g) else 0;
    const qg_x4 = try a.alloc(qm.BlockQ8_Kx4, qg_x4_len);
    defer a.free(qg_x4);
    const phased_pool = if (use_phased) rt.workPool() else null;
    if (phased_pool) |pool| {
        try runMoeBatchPhased(
            rt,
            pool,
            route.active_experts,
            gate,
            up,
            down,
            x_data,
            order,
            hidden,
            out_pe,
            top_k,
            bpc_in,
            bpc_g,
            count,
            offset,
            qx,
            gate_buf,
            up_buf,
            g_buf,
            qg,
            down_buf,
            group_offset,
            qx_x4,
            qg_x4,
            act,
            profile_enabled,
            io,
            profile,
        );
    } else {
        for (tasks, 0..) |*t, e| {
            t.* = .{
                .gate = gate,
                .up = up,
                .down = down,
                .x_data = x_data,
                .order = order,
                .hidden = hidden,
                .out_pe = out_pe,
                .top_k = top_k,
                .bpc_in = bpc_in,
                .blocks_per_g = bpc_g,
                .row_start = offset[e],
                .m = count[e],
                .expert = e,
                .qx = qx,
                .gate_buf = gate_buf,
                .up_buf = up_buf,
                .g_buf = g_buf,
                .qg = qg,
                .down_buf = down_buf,
                .gated_op = act,
                .profile_enabled = profile_enabled,
                .io = io,
                .gather_quant_ns = 0,
                .gate_up_ns = 0,
                .swiglu_requant_ns = 0,
                .down_ns = 0,
            };
        }

        const expert_wall_start = moeBatchProfileStart(profile_enabled, io);
        if (rt.workPool()) |pool| {
            pool.parallelChunks(MoeBatchTask, tasks, runMoeBatchTask);
        } else {
            for (tasks) |*t| runMoeBatchTask(t);
        }
        if (profile) |p| {
            p.expert_wall_ns += moeBatchProfileElapsed(expert_wall_start, io);
            for (tasks) |*t| {
                p.gather_quant_ns += t.gather_quant_ns;
                p.gate_up_ns += t.gate_up_ns;
                p.swiglu_requant_ns += t.swiglu_requant_ns;
                p.down_ns += t.down_ns;
            }
        }
    }
    if (profile) |p| {
        p.batches += 1;
        p.pairs += n_pairs;
        p.active_experts += route.active_experts;
        p.max_expert_m = @max(p.max_expert_m, route.max_expert_m);
    }

    // Scatter: out[token] = sum over its top-k pairs of weight * down_row.
    // Token-major split over the pool: disjoint destination rows with the
    // per-row k-order preserved, bit-identical to the serial loop.
    const out_alloc_start = moeBatchProfileStart(profile_enabled, io);
    var out = try rt.emptyRank(2, .{ seq, hidden });
    errdefer out.deinit();
    if (profile) |p| p.alloc_ns += moeBatchProfileElapsed(out_alloc_start, io);

    const scatter_start = moeBatchProfileStart(profile_enabled, io);
    const od = out.data();
    const scatter_base = MoeBatchScatterTask{
        .out = od,
        .down_buf = down_buf,
        .weights = weights,
        .inv = inv,
        .hidden = hidden,
        .top_k = top_k,
        .t0 = 0,
        .t1 = seq,
    };
    if (phased_pool) |pool| {
        const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), seq);
        var scatter_tasks: [parallel.vector_max_threads]MoeBatchScatterTask = undefined;
        for (0..task_count) |i| {
            scatter_tasks[i] = scatter_base;
            scatter_tasks[i].t0 = i * seq / task_count;
            scatter_tasks[i].t1 = (i + 1) * seq / task_count;
        }
        pool.parallelChunks(MoeBatchScatterTask, scatter_tasks[0..task_count], runMoeBatchScatterTask);
    } else {
        runMoeBatchScatterTask(&scatter_base);
    }
    if (profile) |p| {
        p.scatter_ns += moeBatchProfileElapsed(scatter_start, io);
        p.total_ns += moeBatchProfileElapsed(total_start, io);
    }
    return out;
}
