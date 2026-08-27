//! Shared substrate of the optimizers (optim/*.zig): the deterministic
//! parallel map and norm reduction, the error set, the state-buffer dtype
//! machinery (`StateBuf`, widen/narrow scalar + lane helpers), the
//! type-erased `Param` handle with its f32 master discipline, gradient
//! access and global-norm clipping, and the cross-instance grad-state
//! set. Every optimizer file imports this; nothing here knows a specific
//! optimizer (the optimizer body itself is `optimizer.Optimizer`). Public
//! surface is re-exported by `optim.zig`.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const tensor_mod = @import("../tensor.zig");
const exec_mod = @import("../exec.zig");
// The vectorized cast kernels are backend kernels (the conformed table).
const backend_kernels = @import("../backend.zig").kernels;
const ag_core = @import("../ag/core.zig");
const parallel = @import("../parallel.zig");

const Allocator = std.mem.Allocator;
pub const RawTensor = tensor_mod.Tensor;
pub const ExecContext = exec_mod.ExecContext;
pub const GradState = ag_core.GradState;

/// Elementwise update loops chunk across the worker pool above this length.
/// Every parallel loop here is an element-independent map (no reductions), so
/// the results are bitwise identical to the serial path for any thread count
/// — goldens and bit-exact resume are unaffected. Reductions (norms) run
/// through `sumSquares` below: a FIXED chunk grid with a pinned combine
/// order, so they are equally thread-count-invariant.
const parallel_map_min_len: usize = 1 << 17;

pub fn parallelMap(ctx: *ExecContext, n: usize, context: anytype, comptime runRange: fn (@TypeOf(context), usize, usize) void) void {
    ctx.parallelMap(n, parallel_map_min_len, context, runRange);
}

/// Fixed chunk length of the deterministic norm reductions (`sumSquares`):
/// the chunk grid depends only on the data length, never on the worker
/// count, so partials are a pure function of the values.
const sumsq_chunk_len: usize = 1 << 15;

/// One chunk of `sumSquares`: fixed-width f64 vector lanes, four independent
/// accumulators drained and combined in a pinned order, scalar tail — every
/// operation order is fixed by the slice length alone.
fn sumSquaresChunk(values: []const f32) f64 {
    const lanes = 4;
    const VecF = @Vector(lanes, f32);
    const VecD = @Vector(lanes, f64);
    var acc0: VecD = @splat(0);
    var acc1: VecD = @splat(0);
    var acc2: VecD = @splat(0);
    var acc3: VecD = @splat(0);
    var i: usize = 0;
    while (i + 4 * lanes <= values.len) : (i += 4 * lanes) {
        const w0: VecD = @floatCast(@as(VecF, values[i..][0..lanes].*));
        const w1: VecD = @floatCast(@as(VecF, values[i + lanes ..][0..lanes].*));
        const w2: VecD = @floatCast(@as(VecF, values[i + 2 * lanes ..][0..lanes].*));
        const w3: VecD = @floatCast(@as(VecF, values[i + 3 * lanes ..][0..lanes].*));
        acc0 += w0 * w0;
        acc1 += w1 * w1;
        acc2 += w2 * w2;
        acc3 += w3 * w3;
    }
    while (i + lanes <= values.len) : (i += lanes) {
        const w: VecD = @floatCast(@as(VecF, values[i..][0..lanes].*));
        acc0 += w * w;
    }
    const acc = (acc0 + acc1) + (acc2 + acc3);
    var total = ((acc[0] + acc[1]) + acc[2]) + acc[3];
    while (i < values.len) : (i += 1) {
        total += @as(f64, values[i]) * values[i];
    }
    return total;
}

const SumSquaresTask = struct {
    values: []const f32,
    partials: []f64,
    chunk_start: usize,
    chunk_end: usize,

    fn run(task: *const @This()) void {
        for (task.chunk_start..task.chunk_end) |chunk_i| {
            const start = chunk_i * sumsq_chunk_len;
            const end = @min(start + sumsq_chunk_len, task.values.len);
            task.partials[chunk_i] = sumSquaresChunk(task.values[start..end]);
        }
    }
};

/// Deterministic SIMD/parallel sum of squares in f64: per-chunk partials over
/// the fixed `sumsq_chunk_len` grid (workers own disjoint chunk ranges) plus
/// ONE serial sum in chunk order, so the value is a pure function of the data
/// — bitwise identical for any thread count, pool or serial. It re-associates
/// the historical single scalar chain, so the norm scalars it feeds (the
/// clip factor, Muon's Frobenius normalizer, APOLLO's growth limiter and its
/// stored prev_norm) shift at f64-roundoff scale — and with them the updates
/// they scale; the golden tests carry tolerances for exactly this. What IS
/// preserved is determinism: same data -> same bits, the property
/// checkpoint-exact resume needs. Pub for the thread-count-invariance test.
pub fn sumSquares(ctx: *ExecContext, values: []const f32) !f64 {
    if (values.len <= sumsq_chunk_len) return sumSquaresChunk(values);
    const chunk_count = (values.len + sumsq_chunk_len - 1) / sumsq_chunk_len;
    if (values.len >= parallel_map_min_len) {
        if (ctx.workPool()) |pool| {
            const partials = try ctx.allocator().alloc(f64, chunk_count);
            defer ctx.allocator().free(partials);
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), chunk_count);
            var tasks: [parallel.vector_max_threads]SumSquaresTask = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = .{
                    .values = values,
                    .partials = partials,
                    .chunk_start = task_i * chunk_count / task_count,
                    .chunk_end = (task_i + 1) * chunk_count / task_count,
                };
            }
            pool.parallelChunks(SumSquaresTask, tasks[0..task_count], SumSquaresTask.run);
            var total: f64 = 0;
            for (partials) |partial| total += partial;
            return total;
        }
    }
    var total: f64 = 0;
    var start: usize = 0;
    while (start < values.len) : (start += sumsq_chunk_len) {
        total += sumSquaresChunk(values[start..@min(start + sumsq_chunk_len, values.len)]);
    }
    return total;
}

pub const OptimError = error{
    NotAVariable,
    NonContiguousParam,
    DuplicateParam,
    GradShapeMismatch,
    CheckpointMagicMismatch,
    CheckpointShapeMismatch,
    CheckpointConfigMismatch,
    CheckpointInvalidName,
    CheckpointDuplicateName,
    CheckpointUnknownName,
    CheckpointMissingEntry,
    CheckpointDtypeMismatch,
    CheckpointUnsupportedDtype,
    CheckpointTooManyEntries,
};

/// Storage dtype for optimizer moment/momentum state. Step math is ALWAYS
/// f32: bf16 state is widened on read and narrowed (round-to-nearest-even,
/// NaN-quieting `dtype.f32ToBf16`) on write, halving that buffer's memory at
/// bf16's ~2^-8 ≈ 0.39% relative resolution. The enum values are the u8 wire
/// tags of the v4 checkpoint frames — never renumber them.
pub const StateDType = enum(u8) { f32 = 0, bf16 = 1 };

pub fn StateScalar(comptime sd: StateDType) type {
    return switch (sd) {
        .f32 => f32,
        .bf16 => u16,
    };
}

pub fn StateSlice(comptime sd: StateDType) type {
    return []StateScalar(sd);
}

/// One OWNED optimizer state buffer (moment/momentum), tagged by storage
/// dtype. Allocation, serialization, and the staged checkpoint commit treat
/// it as raw bytes; the step kernels switch ONCE per slot into a
/// comptime-instantiated arm (`StateSlice` + `stateLoad`/`stateStore`), so
/// the hot loops have zero per-element dispatch.
pub const StateBuf = union(StateDType) {
    f32: []f32,
    bf16: []u16,

    /// Allocate a zero-filled buffer (bf16 zero bits are 0.0 too).
    pub fn alloc(allocator: Allocator, sd: StateDType, n: usize) !StateBuf {
        switch (sd) {
            .f32 => {
                const s = try allocator.alloc(f32, n);
                @memset(s, 0);
                return .{ .f32 = s };
            },
            .bf16 => {
                const s = try allocator.alloc(u16, n);
                @memset(s, 0);
                return .{ .bf16 = s };
            },
        }
    }

    pub fn deinit(self: StateBuf, allocator: Allocator) void {
        switch (self) {
            .f32 => |s| allocator.free(s),
            .bf16 => |s| allocator.free(s),
        }
    }

    pub fn len(self: StateBuf) usize {
        return switch (self) {
            .f32 => |s| s.len,
            .bf16 => |s| s.len,
        };
    }

    pub fn byteLen(self: StateBuf) usize {
        return self.bytesConst().len;
    }

    /// The buffer's storage as raw (little-endian) bytes — the checkpoint
    /// wire representation and the staged-commit destination.
    pub fn bytes(self: StateBuf) []u8 {
        return switch (self) {
            .f32 => |s| std.mem.sliceAsBytes(s),
            .bf16 => |s| std.mem.sliceAsBytes(s),
        };
    }

    pub fn bytesConst(self: StateBuf) []const u8 {
        return switch (self) {
            .f32 => |s| std.mem.sliceAsBytes(s),
            .bf16 => |s| std.mem.sliceAsBytes(s),
        };
    }
};

/// Widen one stored state element to f32 (`ptr` is `*f32` / `*u16`).
pub inline fn stateLoad(comptime sd: StateDType, ptr: anytype) f32 {
    return switch (sd) {
        .f32 => ptr.*,
        .bf16 => dtype_mod.bf16ToF32(ptr.*),
    };
}

/// Narrow one just-computed f32 state element into storage. Within one
/// element's update the kernels keep using the pre-narrow f32 value; the
/// NEXT step reads the narrowed stored one.
pub inline fn stateStore(comptime sd: StateDType, ptr: anytype, value: f32) void {
    switch (sd) {
        .f32 => ptr.* = value,
        .bf16 => ptr.* = dtype_mod.f32ToBf16(value),
    }
}

/// Lane width of the hand-vectorized bf16-state kernel bodies: 8 f32 lanes =
/// two NEON registers / one AVX2 register. Hand vectorization is load-bearing:
/// LLVM does not auto-vectorize these fused sqrt/div update loops at all
/// (measured 2026-07-03 on apple-m1, ReleaseFast — even the f32 arms run
/// scalar, hidden behind memory bandwidth at 8 threads), so scalar bf16
/// conversions made the loop compute-bound and SLOWER than f32 despite the
/// smaller traffic. Every lane op below is IEEE-elementwise and lane-exact vs
/// the scalar helpers, so vector-body results are bit-identical to the scalar
/// tail for any split point — the parallelMap thread-count-invariance and the
/// exact-parity tests hold unchanged. The f32/f32 instantiations keep the
/// original scalar loops (the golden-pinned baseline) untouched.
pub const state_vec_len = 8;

pub const StateVec = @Vector(state_vec_len, f32);

/// Widen one lane group; lane-exact vs `stateLoad` (u32 shift, exact).
pub inline fn stateVecLoad(comptime sd: StateDType, src: *const [state_vec_len]StateScalar(sd)) StateVec {
    switch (sd) {
        .f32 => return src.*,
        .bf16 => {
            const Vu32 = @Vector(state_vec_len, u32);
            const widened = @as(Vu32, @intCast(@as(@Vector(state_vec_len, u16), src.*))) << @as(Vu32, @splat(16));
            return @bitCast(widened);
        },
    }
}

/// Narrow one lane group; lane-exact vs `stateStore`'s scalar
/// `dtype.f32ToBf16` INCLUDING the NaN-quieting guard (the unguarded
/// `backend/vector/primitives.f32VecToBf16` would turn NaN state into Inf).
/// The rounding add cannot wrap for non-NaN lanes (max non-NaN bits
/// 0xff80_0000 + 0x8000 < 2^32); the `+%` wrap on NaN lanes is discarded by
/// the select, which takes the quieted arm instead — exactly the scalar
/// early-return.
pub inline fn stateVecStore(comptime sd: StateDType, dst: *[state_vec_len]StateScalar(sd), values: StateVec) void {
    switch (sd) {
        .f32 => dst.* = values,
        .bf16 => {
            const Vu32 = @Vector(state_vec_len, u32);
            const bits: Vu32 = @bitCast(values);
            const shifted = bits >> @as(Vu32, @splat(16));
            const is_nan = (bits & @as(Vu32, @splat(0x7fff_ffff))) > @as(Vu32, @splat(0x7f80_0000));
            const quieted = shifted | @as(Vu32, @splat(64));
            const lsb = shifted & @as(Vu32, @splat(1));
            const rounded = (bits +% @as(Vu32, @splat(0x7fff)) +% lsb) >> @as(Vu32, @splat(16));
            const narrowed: @Vector(state_vec_len, u16) = @truncate(@select(u32, is_nan, quieted, rounded));
            dst.* = narrowed;
        },
    }
}

comptime {
    if (@import("builtin").cpu.arch.endian() != .little) {
        @compileError("optim checkpoints assume a little-endian target");
    }
}

/// Type-erased handle to one trainable parameter: a refcounted view of the
/// variable's storage (safe if the facade struct moves by value) plus its
/// heap-stable GradState. `rows`/`cols` describe the matrix view used by the
/// matrix-aware optimizers: dim 0 by the product of the remaining dims —
/// exactly Keller's conv-filter flattening `[d0, d1*d2*...]`.
///
/// f16/bf16 params carry an optimizer-owned f32 MASTER copy: every update
/// kernel steps the master (via `data()`), and `publish` narrows the master
/// back into the 16-bit storage after the step. Gradients are f32 for every
/// param dtype, so the step math is dtype-blind.
pub const Param = struct {
    value: Storage,
    grad_state: *GradState,
    rows: usize,
    cols: usize,
    raw_rank: usize,
    /// Checkpoint identity, set by `addParamNamed`. BORROWED: like the
    /// parameter tensor itself, the name must outlive the optimizer (string
    /// literals and model-struct fields qualify). Unnamed params auto-name as
    /// "param<i>" from their slot index at save time.
    name: ?[]const u8 = null,
    /// f32 master weights for 16-bit params (optimizer-owned; empty for
    /// f32 params, whose storage is stepped in place).
    master: []f32 = &.{},

    pub const Storage = union(enum) {
        f32: RawTensor,
        f16: tensor_mod.TensorOf(.f16),
        bf16: tensor_mod.TensorOf(.bf16),
    };

    /// `t` must be a pointer to an f32/f16/bf16 autograd Tensor created as
    /// a variable (16-bit variables hold f32 gradients).
    pub fn of(t: anytype) !Param {
        const P = @TypeOf(t);
        const info = @typeInfo(P);
        if (info != .pointer) @compileError("optim.Param.of expects a pointer to an f32/f16/bf16 autograd Tensor");
        const T = info.pointer.child;
        if (!@hasField(T, "grad_state")) {
            @compileError("optim.Param.of requires an f32/f16/bf16 autograd Tensor (constant/quantized tensors carry no gradients)");
        }
        const state = t.grad_state orelse return OptimError.NotAVariable;
        if (!t.value.isContiguous()) return OptimError.NonContiguousParam;
        const shape = t.value.shape.slice();
        var cols: usize = 1;
        for (shape[1..]) |dim| cols *= dim;
        var view = try t.value.cloneView();
        errdefer view.deinit();
        const ValueT = @TypeOf(t.value);
        const storage: Storage = if (ValueT == RawTensor)
            .{ .f32 = view }
        else if (ValueT == tensor_mod.TensorOf(.f16))
            .{ .f16 = view }
        else if (ValueT == tensor_mod.TensorOf(.bf16))
            .{ .bf16 = view }
        else
            @compileError("optim params must be f32, f16, or bf16 variables");
        return .{
            .value = storage,
            .grad_state = state,
            .rows = shape[0],
            .cols = cols,
            .raw_rank = shape.len,
        };
    }

    pub fn len(self: *const Param) usize {
        return self.rows * self.cols;
    }

    /// The f32 buffer the update kernels step: the param storage itself for
    /// f32 params, the master for 16-bit params.
    pub fn data(self: *Param) []f32 {
        return switch (self.value) {
            .f32 => |*t| t.data(),
            else => self.master,
        };
    }

    /// Allocate + fill the f32 master for 16-bit params (no-op for f32).
    /// Called once at registration, after which the master is authoritative
    /// between `publish` calls.
    pub fn ensureMaster(self: *Param, allocator: Allocator) !void {
        if (self.value == .f32 or self.master.len != 0) return;
        self.master = try allocator.alloc(f32, self.len());
        self.refreshMasterFromValue();
    }

    /// Re-widen the master from the current 16-bit storage (after the param
    /// VALUES were loaded externally and no checkpoint master exists).
    pub fn refreshMasterFromValue(self: *Param) void {
        switch (self.value) {
            .f32 => {},
            .f16 => |*t| backend_kernels.castF16ToF32(self.master, t.data()),
            .bf16 => |*t| backend_kernels.castBf16ToF32(self.master, t.data()),
        }
    }

    /// Narrow the stepped master back into the 16-bit param storage (no-op
    /// for f32 params).
    pub fn publish(self: *Param) void {
        switch (self.value) {
            .f32 => {},
            .f16 => |*t| backend_kernels.castF32ToF16(t.data(), self.master),
            .bf16 => |*t| backend_kernels.castF32ToBf16(t.data(), self.master),
        }
    }

    pub fn deinit(self: *Param, allocator: Allocator) void {
        switch (self.value) {
            inline else => |*t| t.deinit(),
        }
        if (self.master.len != 0) allocator.free(self.master);
        self.* = undefined;
    }
};

/// Borrow the parameter's accumulated gradient as a contiguous tensor, or null
/// if no gradient was produced (the param is then skipped, PyTorch-style).
pub fn takeGrad(ctx: *ExecContext, param: *const Param) !?RawTensor {
    var view = (try param.grad_state.gradView()) orelse return null;
    errdefer view.deinit();
    if (view.len() != param.len()) return OptimError.GradShapeMismatch;
    if (view.isContiguous()) return view;
    const out = try ctx.materialize(.f32, &view);
    view.deinit();
    return out;
}

// ---------------------------------------------------------------------------
// Gradient clipping — torch.nn.utils.clip_grad_norm_ semantics (L2).
// ---------------------------------------------------------------------------

pub fn paramGradSqNorm(ctx: *ExecContext, param: *const Param) !f64 {
    var grad = (try takeGrad(ctx, param)) orelse return 0;
    defer grad.deinit();
    return sumSquares(ctx, grad.dataConst());
}

const ScaleMapContext = struct {
    values: []f32,
    factor: f32,
};

fn scaleMapRange(c: ScaleMapContext, start: usize, end: usize) void {
    for (c.values[start..end]) |*value| value.* *= c.factor;
}

pub fn scaleParamGrad(ctx: *ExecContext, param: *const Param, factor: f32) !void {
    var view = (try param.grad_state.gradView()) orelse return;
    if (view.isContiguous()) {
        defer view.deinit();
        const values = view.data();
        parallelMap(ctx, values.len, ScaleMapContext{ .values = values, .factor = factor }, scaleMapRange);
        return;
    }
    // Rare: a non-contiguous accumulated gradient. Materialize, scale, and
    // swap it into the GradState (which owns and frees the old one).
    var owned = ctx.materialize(.f32, &view) catch |err| {
        view.deinit();
        return err;
    };
    view.deinit();
    const values = owned.data();
    parallelMap(ctx, values.len, ScaleMapContext{ .values = values, .factor = factor }, scaleMapRange);
    param.grad_state.setGrad(owned);
}

/// Shared two-phase global-norm clip: `total = sqrt(sum ||g||^2)` over every
/// registered param; if `total > max_norm`, every gradient is scaled by
/// `max_norm / (total + 1e-6)`. Returns the PRE-clip total norm (PyTorch
/// contract). Call AFTER backward() and BEFORE step().
pub fn clipByGlobalNorm(ctx: *ExecContext, opt: anytype, max_norm: f32) !f32 {
    const total: f32 = @floatCast(@sqrt(try opt.gradSquaredNorm(ctx)));
    const clip_coef = max_norm / (total + 1e-6);
    if (clip_coef < 1) try opt.scaleGradients(ctx, clip_coef);
    return total;
}

/// Set of registered `GradState` pointers (by pointer identity), used by
/// `OptimizerSet` to reject the same `Variable` being registered into two member
/// optimizers — which the per-instance `containsGradState` guard cannot see and
/// which would otherwise silently double-step that parameter.
pub const GradStateSet = std.AutoHashMapUnmanaged(*const GradState, void);

/// True if any slot's grad-state is already in `set` (a cross-member duplicate).
pub fn gradStatesCollide(set: *const GradStateSet, slots: anytype) bool {
    for (slots) |*slot| {
        if (set.contains(slot.param.grad_state)) return true;
    }
    return false;
}

pub fn insertGradStates(set: *GradStateSet, allocator: Allocator, slots: anytype) !void {
    for (slots) |*slot| try set.put(allocator, slot.param.grad_state, {});
}
