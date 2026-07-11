//! Gradient-descent optimizers over the public autograd Tensor facade.
//!
//! Optimizers, each a faithful port of its reference implementation:
//!
//! - `Adam` — PyTorch `torch.optim.Adam` single-tensor path (coupled L2
//!   weight decay: `g += weight_decay * p` before moment updates).
//! - `AdamW` — PyTorch `torch.optim.AdamW` single-tensor path (decoupled decay
//!   applied to the parameter BEFORE the Adam step; `denom = sqrt(v)/sqrt(1-b2^t) + eps`).
//! - `Muon` — Keller Jordan's reference (github.com/KellerJordan/Muon): lerp-form
//!   momentum, Newton-Schulz-5 orthogonalization with the transpose trick, and a
//!   built-in AdamW fallback for non-matrix params (biases, norms) and params the
//!   caller routes there explicitly (embeddings, output heads). Both update-scale
//!   conventions are available: Keller's spectral `sqrt(max(1, rows/cols))` and
//!   Moonlight's RMS-matching `0.2*sqrt(max(rows, cols))` (arXiv 2502.16982).
//! - `Apollo` — the official `apollo_torch` optimizer (arXiv 2412.05270): random
//!   low-rank projection of the gradient, AdamW moments in the compressed space,
//!   channel- or tensor-wise gradient scaling, the Fira norm-growth limiter, and
//!   a scaled-SGD update. APOLLO-Mini is `ApolloConfig.mini()`. Its fallback path
//!   reproduces the reference's legacy-HF AdamW (eps OUTSIDE the bias correction,
//!   decay AFTER the step) — deliberately different from `AdamW` above.
//!
//! Ownership: optimizers hold refcounted views of parameter storage plus raw
//! `*GradState` pointers, so parameter tensors may move by value but must
//! OUTLIVE the optimizer (they own the GradState). Each variable must be
//! registered with exactly one optimizer: duplicates are rejected WITHIN one
//! instance (`error.DuplicateParam`); registering the same tensor with two
//! different instances (e.g. two OptimizerSet groups) is not detectable and
//! silently double-steps — cross-instance uniqueness is the caller's
//! responsibility. Optimizer state (moments, momentum, projections) is owned
//! by the optimizer and freed by `deinit`. Newton-Schulz / projection
//! transients come from the ExecContext BufferPool.
//!
//! Checkpointing: parameter values have two formats. `saveTensors`/
//! `loadTensors` (FZT1) are positional and f32-only: the loading program must
//! list the same tensors in the same order (shapes are validated).
//! `saveStateDict`/`loadStateDict` are safetensors-backed, NAMED, and
//! dtype-aware (f32/f16/bf16, raw byte passthrough): load matches stream entries to the
//! provided list by name, so entry order is free; strict mode (the default)
//! requires an exact one-to-one match, non-strict skips unknown stream
//! entries. Each optimizer's `saveState`/`loadState` serializes moments, step
//! counts, and the structural config fields, validating them on load. When
//! every state buffer is f32 (the default) the v3 frames are written
//! byte-identically (FZAD/FZA3/FZM3/FZP3/FZS3; FZO3 for OptimizerSet), so
//! pre-bf16 builds keep reading new f32 checkpoints. When any state buffer is
//! non-f32 the v4 frames are written instead (FZD4/FZA4/FZM4/FZS4 — Adam,
//! AdamW, Muon, SGD; Apollo state stays f32, always FZP3): identical layout
//! except each state buffer is prefixed by one u8 `StateDType` tag. Loaders
//! accept both versions but require the stored dtype to match the configured
//! one EXACTLY (v3 implies f32 everywhere) — a cross-dtype load errors with
//! `CheckpointDtypeMismatch` rather than converting, because an implicit
//! f32<->bf16 conversion would silently break the bit-exact-resume contract
//! below. Optimizer slots are matched BY
//! NAME — explicit via `addParamNamed`, otherwise the auto-name "param<i>"
//! from the slot's index within its slot list — so named params may be
//! re-registered in any order within their list (Muon/Apollo route params to
//! their fallback by rank, independent of order); unnamed params must keep
//! their relative registration order to reproduce their auto-names. A failed
//! load leaves the target partially restored — treat load errors as fatal for
//! that model/optimizer instance. Resumption replays the optimizer
//! bit-exactly; end-to-end bit-exact training resume additionally requires
//! the surrounding forward/backward replay to be deterministic (true for
//! single-contribution gradients; gradients accumulated from 3+ async
//! dot-backward branches are summed in completion order). APOLLO projections
//! are not stored: P is a deterministic, repo-owned function of
//! (seed, step / update_proj_gap) and is regenerated on the next step.

const std = @import("std");
const dtype_mod = @import("dtype.zig");
const tensor_mod = @import("tensor.zig");
const exec_mod = @import("exec.zig");
const exec_convert = @import("exec/convert.zig");
const state_dict = @import("state_dict.zig");
const ag_core = @import("ag/core.zig");
const parallel = @import("parallel.zig");
// The (seed -> values) mapping of rng.gaussianFill is part of the APOLLO
// checkpoint contract (projections are regenerated from seed, not stored).
const rng = @import("rng.zig");
const gaussianFill = rng.gaussianFill;

const Allocator = std.mem.Allocator;
const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const GradState = ag_core.GradState;

/// Elementwise update loops chunk across the worker pool above this length.
/// Every parallel loop here is an element-independent map (no reductions), so
/// the results are bitwise identical to the serial path for any thread count
/// — goldens and bit-exact resume are unaffected. Reductions (norms) run
/// through `sumSquares` below: a FIXED chunk grid with a pinned combine
/// order, so they are equally thread-count-invariant.
const parallel_map_min_len: usize = 1 << 17;

fn parallelMap(ctx: *ExecContext, n: usize, context: anytype, comptime runRange: fn (@TypeOf(context), usize, usize) void) void {
    if (n >= parallel_map_min_len) {
        if (ctx.workPool()) |pool| {
            const Ctx = @TypeOf(context);
            const Task = struct {
                context: Ctx,
                start: usize,
                end: usize,
                fn run(task: *const @This()) void {
                    runRange(task.context, task.start, task.end);
                }
            };
            const task_count = @min(
                parallel.cpuThreadCount(parallel.vector_max_threads),
                1 + n / parallel_map_min_len,
            );
            if (task_count > 1) {
                var tasks: [parallel.vector_max_threads]Task = undefined;
                for (0..task_count) |i| {
                    tasks[i] = .{ .context = context, .start = i * n / task_count, .end = (i + 1) * n / task_count };
                }
                pool.parallelChunks(Task, tasks[0..task_count], Task.run);
                return;
            }
        }
    }
    runRange(context, 0, n);
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
            const partials = try ctx.allocator.alloc(f64, chunk_count);
            defer ctx.allocator.free(partials);
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

fn StateScalar(comptime sd: StateDType) type {
    return switch (sd) {
        .f32 => f32,
        .bf16 => u16,
    };
}

fn StateSlice(comptime sd: StateDType) type {
    return []StateScalar(sd);
}

/// One OWNED optimizer state buffer (moment/momentum), tagged by storage
/// dtype. Allocation, serialization, and the staged checkpoint commit treat
/// it as raw bytes; the step kernels switch ONCE per slot into a
/// comptime-instantiated arm (`StateSlice` + `stateLoad`/`stateStore`), so
/// the hot loops have zero per-element dispatch.
const StateBuf = union(StateDType) {
    f32: []f32,
    bf16: []u16,

    /// Allocate a zero-filled buffer (bf16 zero bits are 0.0 too).
    fn alloc(allocator: Allocator, sd: StateDType, n: usize) !StateBuf {
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

    fn deinit(self: StateBuf, allocator: Allocator) void {
        switch (self) {
            .f32 => |s| allocator.free(s),
            .bf16 => |s| allocator.free(s),
        }
    }

    fn len(self: StateBuf) usize {
        return switch (self) {
            .f32 => |s| s.len,
            .bf16 => |s| s.len,
        };
    }

    fn byteLen(self: StateBuf) usize {
        return self.bytesConst().len;
    }

    /// The buffer's storage as raw (little-endian) bytes — the checkpoint
    /// wire representation and the staged-commit destination.
    fn bytes(self: StateBuf) []u8 {
        return switch (self) {
            .f32 => |s| std.mem.sliceAsBytes(s),
            .bf16 => |s| std.mem.sliceAsBytes(s),
        };
    }

    fn bytesConst(self: StateBuf) []const u8 {
        return switch (self) {
            .f32 => |s| std.mem.sliceAsBytes(s),
            .bf16 => |s| std.mem.sliceAsBytes(s),
        };
    }
};

/// Widen one stored state element to f32 (`ptr` is `*f32` / `*u16`).
inline fn stateLoad(comptime sd: StateDType, ptr: anytype) f32 {
    return switch (sd) {
        .f32 => ptr.*,
        .bf16 => dtype_mod.bf16ToF32(ptr.*),
    };
}

/// Narrow one just-computed f32 state element into storage. Within one
/// element's update the kernels keep using the pre-narrow f32 value; the
/// NEXT step reads the narrowed stored one.
inline fn stateStore(comptime sd: StateDType, ptr: anytype, value: f32) void {
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
const state_vec_len = 8;

const StateVec = @Vector(state_vec_len, f32);

/// Widen one lane group; lane-exact vs `stateLoad` (u32 shift, exact).
inline fn stateVecLoad(comptime sd: StateDType, src: *const [state_vec_len]StateScalar(sd)) StateVec {
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
inline fn stateVecStore(comptime sd: StateDType, dst: *[state_vec_len]StateScalar(sd), values: StateVec) void {
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

pub const NamedTensor = state_dict.NamedTensor;
pub const NamedTensorMut = state_dict.NamedTensorMut;
pub const LoadOptions = state_dict.LoadOptions;
pub const saveStateDict = state_dict.saveStateDict;
pub const loadStateDict = state_dict.loadStateDict;

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
    fn data(self: *Param) []f32 {
        return switch (self.value) {
            .f32 => |*t| t.data(),
            else => self.master,
        };
    }

    /// Allocate + fill the f32 master for 16-bit params (no-op for f32).
    /// Called once at registration, after which the master is authoritative
    /// between `publish` calls.
    fn ensureMaster(self: *Param, allocator: Allocator) !void {
        if (self.value == .f32 or self.master.len != 0) return;
        self.master = try allocator.alloc(f32, self.len());
        self.refreshMasterFromValue();
    }

    /// Re-widen the master from the current 16-bit storage (after the param
    /// VALUES were loaded externally and no checkpoint master exists).
    fn refreshMasterFromValue(self: *Param) void {
        switch (self.value) {
            .f32 => {},
            .f16 => |*t| exec_convert.castF16ToF32(self.master, t.data()),
            .bf16 => |*t| exec_convert.castBf16ToF32(self.master, t.data()),
        }
    }

    /// Narrow the stepped master back into the 16-bit param storage (no-op
    /// for f32 params).
    fn publish(self: *Param) void {
        switch (self.value) {
            .f32 => {},
            .f16 => |*t| exec_convert.castF32ToF16(t.data(), self.master),
            .bf16 => |*t| exec_convert.castF32ToBf16(t.data(), self.master),
        }
    }

    fn deinit(self: *Param, allocator: Allocator) void {
        switch (self.value) {
            inline else => |*t| t.deinit(),
        }
        if (self.master.len != 0) allocator.free(self.master);
        self.* = undefined;
    }
};

/// Borrow the parameter's accumulated gradient as a contiguous tensor, or null
/// if no gradient was produced (the param is then skipped, PyTorch-style).
fn takeGrad(ctx: *ExecContext, param: *const Param) !?RawTensor {
    var view = (try param.grad_state.gradView()) orelse return null;
    errdefer view.deinit();
    if (view.len() != param.len()) return OptimError.GradShapeMismatch;
    if (view.isContiguous()) return view;
    const out = try ctx.materialize(&view);
    view.deinit();
    return out;
}

// ---------------------------------------------------------------------------
// AdamW — PyTorch torch.optim.AdamW single-tensor semantics.
// ---------------------------------------------------------------------------

pub const AdamWConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    /// Storage dtype of the FIRST moment (m); step math stays f32 either
    /// way. bf16 is safe for m: with beta1 = 0.9 the per-step relative
    /// change (~10%) is far above bf16's ~0.39% resolution.
    state_dtype: StateDType = .f32,
    /// Storage dtype of the SECOND moment (v) — a separate opt-in because v
    /// is precision-sensitive: with beta2 = 0.999 the per-step relative
    /// change (~0.1%) is BELOW bf16's ~2^-8 ≈ 0.39% resolution, so the EMA
    /// update can round to a no-op and stall (stale denominator →
    /// effective-LR drift). Keep .f32 unless memory forces the trade.
    second_moment_dtype: StateDType = .f32,
};

pub const AdamW = struct {
    allocator: Allocator,
    config: AdamWConfig,
    slots: std.ArrayList(Slot) = .empty,

    const Slot = struct {
        param: Param,
        m: StateBuf,
        v: StateBuf,
        step: u64 = 0,
    };

    pub fn init(allocator: Allocator, config: AdamWConfig) AdamW {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *AdamW) void {
        for (self.slots.items) |*slot| {
            slot.m.deinit(self.allocator);
            slot.v.deinit(self.allocator);
            slot.param.deinit(self.allocator);
        }
        self.slots.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addParam(self: *AdamW, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        try self.addOwnedParam(param);
    }

    /// `addParam` plus a checkpoint name (borrowed; see `Param.name`).
    pub fn addParamNamed(self: *AdamW, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        try self.addOwnedParam(param);
    }

    /// Add this optimizer's grad-states to `set` for `OptimizerSet`'s
    /// cross-member duplicate check; `DuplicateParam` if any already present
    /// (set left unchanged on collision).
    pub fn collectGradStates(self: *const AdamW, set: *GradStateSet, allocator: Allocator) !void {
        if (gradStatesCollide(set, self.slots.items)) return OptimError.DuplicateParam;
        try insertGradStates(set, allocator, self.slots.items);
    }

    fn containsGradState(self: *const AdamW, state: *const GradState) bool {
        for (self.slots.items) |*slot| {
            if (slot.param.grad_state == state) return true;
        }
        return false;
    }

    fn addOwnedParam(self: *AdamW, param: Param) !void {
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        var owned = param;
        try owned.ensureMaster(self.allocator);
        errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
        const n = owned.len();
        const m = try StateBuf.alloc(self.allocator, self.config.state_dtype, n);
        errdefer m.deinit(self.allocator);
        const v = try StateBuf.alloc(self.allocator, self.config.second_moment_dtype, n);
        errdefer v.deinit(self.allocator);
        try self.slots.append(self.allocator, .{ .param = owned, .m = m, .v = v });
    }

    pub fn step(self: *AdamW, ctx: *ExecContext) !void {
        for (self.slots.items) |*slot| {
            var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
            defer grad.deinit();
            slot.step += 1;
            adamwUpdate(ctx, self.config, slot.param.data(), grad.dataConst(), slot.m, slot.v, slot.step);
            slot.param.publish();
        }
    }

    pub fn zeroGrad(self: *AdamW) void {
        for (self.slots.items) |*slot| slot.param.grad_state.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *AdamW, ctx: *ExecContext) !f64 {
        var total: f64 = 0;
        for (self.slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
        return total;
    }

    pub fn scaleGradients(self: *AdamW, ctx: *ExecContext, factor: f32) !void {
        for (self.slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
    }

    /// L2 global-norm clip over this optimizer's params (after backward,
    /// before step). Returns the pre-clip norm.
    pub fn clipGradNorm(self: *AdamW, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    pub fn saveState(self: *const AdamW, writer: *std.Io.Writer) !void {
        try validateSlotNames(self.slots.items);
        var version = momentSlotsFrameVersion(self.slots.items);
        if (slotsCarryMasters(self.slots.items)) version = .v5;
        try writer.writeAll(switch (version) {
            .v3 => "FZA3",
            .v4 => "FZA4",
            .v5 => "FZA5",
        });
        try writer.writeInt(u32, @intCast(self.slots.items.len), .little);
        for (self.slots.items, 0..) |*slot, i| {
            try writeSlotName(writer, &slot.param, i);
            try writeSlotDims(writer, &slot.param);
            try writer.writeInt(u64, slot.step, .little);
            try writeStateSlice(writer, version, slot.m);
            try writeStateSlice(writer, version, slot.v);
            try writeSlotMaster(writer, version, &slot.param);
        }
    }

    pub fn loadState(self: *AdamW, reader: *std.Io.Reader) !void {
        const version = try expectMagicVersion(reader, "FZA3", "FZA4", "FZA5");
        const count = try reader.takeInt(u32, .little);
        var matcher = try SlotMatcher.init(self.allocator, self.slots.items.len);
        defer matcher.deinit(self.allocator);
        var staged = try std.ArrayList(StagedSlot).initCapacity(self.allocator, count);
        defer freeStaged(self.allocator, &staged);
        for (0..count) |_| {
            const idx = try matcher.match(reader, self.slots.items);
            const slot = &self.slots.items[idx];
            try expectSlotDims(reader, &slot.param);
            const step_val = try reader.takeInt(u64, .little);
            const m_bytes = slot.m.byteLen();
            const data = try self.allocator.alloc(u8, m_bytes + slot.v.byteLen());
            errdefer self.allocator.free(data);
            try readStateSlice(reader, version, slot.m, data[0..m_bytes]);
            try readStateSlice(reader, version, slot.v, data[m_bytes..]);
            const master = try readSlotMaster(self.allocator, reader, version, &slot.param);
            errdefer if (master.len != 0) self.allocator.free(master);
            try staged.append(self.allocator, .{ .idx = idx, .step = step_val, .data = data, .master = master });
        }
        try matcher.requireAllFilled();
        for (staged.items) |s| {
            const slot = &self.slots.items[s.idx];
            slot.step = s.step;
            const m_bytes = slot.m.byteLen();
            @memcpy(slot.m.bytes(), s.data[0..m_bytes]);
            @memcpy(slot.v.bytes(), s.data[m_bytes..]);
            commitSlotMaster(&slot.param, s.master);
        }
    }
};

/// The exact PyTorch `_single_tensor_adam(decoupled_weight_decay=True)` update.
/// Order matters: decay multiplies the parameter BEFORE the moment update and
/// Adam step; `eps` is added AFTER dividing sqrt(v) by sqrt(bias_correction2);
/// the 1/bias_correction1 lives in the scalar step size. Scalar prep runs in
/// f64 and is rounded once to f32 — matching torch's Python-float scalars to
/// within a few f32 ulps (bit parity is impossible with f32 config fields).
/// The three reference loops are fused into one element-independent pass
/// (same per-element op order, less memory traffic) and chunked across the
/// worker pool. All arithmetic is f32 for every state dtype: bf16 moments are
/// widened on read and narrowed on write, and the parameter update uses the
/// just-computed (pre-narrow) f32 moments; the f32/f32 instantiation is
/// op-for-op the pre-`StateDType` kernel.
const AdamWScalars = struct {
    keep: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2s: f32,
    eps: f32,
};

fn AdamWMap(comptime md: StateDType, comptime vd: StateDType) type {
    return struct {
        p: []f32,
        g: []const f32,
        m: StateSlice(md),
        v: StateSlice(vd),
        s: AdamWScalars,

        fn run(c: @This(), start: usize, end: usize) void {
            if (comptime (md == .f32 and vd == .f32)) {
                // The golden-pinned baseline stays on the original scalar loop.
                runScalar(c, start, end);
                return;
            }
            // Hand-vectorized bf16 arm (see `state_vec_len`); bit-identical
            // to runScalar per element.
            const keep: StateVec = @splat(c.s.keep);
            const beta2: StateVec = @splat(c.s.beta2);
            const one_minus_b1: StateVec = @splat(c.s.one_minus_b1);
            const one_minus_b2: StateVec = @splat(c.s.one_minus_b2);
            const step_size: StateVec = @splat(c.s.step_size);
            const bc2s: StateVec = @splat(c.s.bc2s);
            const eps: StateVec = @splat(c.s.eps);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const pv: StateVec = c.p[i..][0..state_vec_len].*;
                const gv: StateVec = c.g[i..][0..state_vec_len].*;
                const decayed = pv * keep;
                const m0 = stateVecLoad(md, c.m[i..][0..state_vec_len]);
                const v0 = stateVecLoad(vd, c.v[i..][0..state_vec_len]);
                const m1 = m0 + one_minus_b1 * (gv - m0);
                const v1 = beta2 * v0 + one_minus_b2 * gv * gv;
                stateVecStore(md, c.m[i..][0..state_vec_len], m1);
                stateVecStore(vd, c.v[i..][0..state_vec_len], v1);
                c.p[i..][0..state_vec_len].* = decayed - step_size * (m1 / (@sqrt(v1) / bc2s + eps));
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, gi, *mi, *vi| {
                const decayed = pi.* * c.s.keep;
                const m0 = stateLoad(md, mi);
                const v0 = stateLoad(vd, vi);
                const m1 = m0 + c.s.one_minus_b1 * (gi - m0);
                const v1 = c.s.beta2 * v0 + c.s.one_minus_b2 * gi * gi;
                stateStore(md, mi, m1);
                stateStore(vd, vi, v1);
                pi.* = decayed - c.s.step_size * (m1 / (@sqrt(v1) / c.s.bc2s + c.s.eps));
            }
        }
    };
}

fn adamwRun(comptime md: StateDType, comptime vd: StateDType, ctx: *ExecContext, s: AdamWScalars, p: []f32, g: []const f32, m: StateSlice(md), v: StateSlice(vd)) void {
    const Map = AdamWMap(md, vd);
    parallelMap(ctx, p.len, Map{ .p = p, .g = g, .m = m, .v = v, .s = s }, Map.run);
}

fn adamwUpdate(ctx: *ExecContext, config: AdamWConfig, p: []f32, g: []const f32, m: StateBuf, v: StateBuf, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    const bc1 = 1 - std.math.pow(f64, config.beta1, t);
    const bc2_sqrt = @sqrt(1 - std.math.pow(f64, config.beta2, t));
    const s = AdamWScalars{
        .keep = if (config.weight_decay != 0) @floatCast(1.0 - @as(f64, config.lr) * @as(f64, config.weight_decay)) else 1,
        .beta2 = config.beta2,
        .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
        .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        .step_size = @floatCast(@as(f64, config.lr) / bc1),
        .bc2s = @floatCast(bc2_sqrt),
        .eps = config.eps,
    };
    switch (m) {
        .f32 => |ms| switch (v) {
            .f32 => |vs| adamwRun(.f32, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamwRun(.f32, .bf16, ctx, s, p, g, ms, vs),
        },
        .bf16 => |ms| switch (v) {
            .f32 => |vs| adamwRun(.bf16, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamwRun(.bf16, .bf16, ctx, s, p, g, ms, vs),
        },
    }
}

// ---------------------------------------------------------------------------
// Adam — PyTorch torch.optim.Adam single-tensor semantics.
// ---------------------------------------------------------------------------

pub const AdamConfig = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0,
    /// Storage dtype of the FIRST moment (m); see `AdamWConfig.state_dtype`.
    state_dtype: StateDType = .f32,
    /// Storage dtype of the SECOND moment (v); see
    /// `AdamWConfig.second_moment_dtype` for the bf16 v-stall math.
    second_moment_dtype: StateDType = .f32,
};

pub const Adam = struct {
    allocator: Allocator,
    config: AdamConfig,
    slots: std.ArrayList(Slot) = .empty,

    const Slot = struct {
        param: Param,
        m: StateBuf,
        v: StateBuf,
        step: u64 = 0,
    };

    pub fn init(allocator: Allocator, config: AdamConfig) Adam {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *Adam) void {
        for (self.slots.items) |*slot| {
            slot.m.deinit(self.allocator);
            slot.v.deinit(self.allocator);
            slot.param.deinit(self.allocator);
        }
        self.slots.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addParam(self: *Adam, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        try self.addOwnedParam(param);
    }

    pub fn addParamNamed(self: *Adam, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        try self.addOwnedParam(param);
    }

    pub fn collectGradStates(self: *const Adam, set: *GradStateSet, allocator: Allocator) !void {
        if (gradStatesCollide(set, self.slots.items)) return OptimError.DuplicateParam;
        try insertGradStates(set, allocator, self.slots.items);
    }

    fn containsGradState(self: *const Adam, state: *const GradState) bool {
        for (self.slots.items) |*slot| {
            if (slot.param.grad_state == state) return true;
        }
        return false;
    }

    fn addOwnedParam(self: *Adam, param: Param) !void {
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        var owned = param;
        try owned.ensureMaster(self.allocator);
        errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
        const n = owned.len();
        const m = try StateBuf.alloc(self.allocator, self.config.state_dtype, n);
        errdefer m.deinit(self.allocator);
        const v = try StateBuf.alloc(self.allocator, self.config.second_moment_dtype, n);
        errdefer v.deinit(self.allocator);
        try self.slots.append(self.allocator, .{ .param = owned, .m = m, .v = v });
    }

    pub fn step(self: *Adam, ctx: *ExecContext) !void {
        for (self.slots.items) |*slot| {
            var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
            defer grad.deinit();
            slot.step += 1;
            adamUpdate(ctx, self.config, slot.param.data(), grad.dataConst(), slot.m, slot.v, slot.step);
            slot.param.publish();
        }
    }

    pub fn zeroGrad(self: *Adam) void {
        for (self.slots.items) |*slot| slot.param.grad_state.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *Adam, ctx: *ExecContext) !f64 {
        var total: f64 = 0;
        for (self.slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
        return total;
    }

    pub fn scaleGradients(self: *Adam, ctx: *ExecContext, factor: f32) !void {
        for (self.slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
    }

    pub fn clipGradNorm(self: *Adam, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    pub fn saveState(self: *const Adam, writer: *std.Io.Writer) !void {
        try validateSlotNames(self.slots.items);
        var version = momentSlotsFrameVersion(self.slots.items);
        if (slotsCarryMasters(self.slots.items)) version = .v5;
        try writer.writeAll(switch (version) {
            .v3 => "FZAD",
            .v4 => "FZD4",
            .v5 => "FZD5",
        });
        try writer.writeInt(u32, @intCast(self.slots.items.len), .little);
        for (self.slots.items, 0..) |*slot, i| {
            try writeSlotName(writer, &slot.param, i);
            try writeSlotDims(writer, &slot.param);
            try writer.writeInt(u64, slot.step, .little);
            try writeStateSlice(writer, version, slot.m);
            try writeStateSlice(writer, version, slot.v);
            try writeSlotMaster(writer, version, &slot.param);
        }
    }

    pub fn loadState(self: *Adam, reader: *std.Io.Reader) !void {
        const version = try expectMagicVersion(reader, "FZAD", "FZD4", "FZD5");
        const count = try reader.takeInt(u32, .little);
        var matcher = try SlotMatcher.init(self.allocator, self.slots.items.len);
        defer matcher.deinit(self.allocator);
        var staged = try std.ArrayList(StagedSlot).initCapacity(self.allocator, count);
        defer freeStaged(self.allocator, &staged);
        for (0..count) |_| {
            const idx = try matcher.match(reader, self.slots.items);
            const slot = &self.slots.items[idx];
            try expectSlotDims(reader, &slot.param);
            const step_val = try reader.takeInt(u64, .little);
            const m_bytes = slot.m.byteLen();
            const data = try self.allocator.alloc(u8, m_bytes + slot.v.byteLen());
            errdefer self.allocator.free(data);
            try readStateSlice(reader, version, slot.m, data[0..m_bytes]);
            try readStateSlice(reader, version, slot.v, data[m_bytes..]);
            const master = try readSlotMaster(self.allocator, reader, version, &slot.param);
            errdefer if (master.len != 0) self.allocator.free(master);
            try staged.append(self.allocator, .{ .idx = idx, .step = step_val, .data = data, .master = master });
        }
        try matcher.requireAllFilled();
        for (staged.items) |s| {
            const slot = &self.slots.items[s.idx];
            slot.step = s.step;
            const m_bytes = slot.m.byteLen();
            @memcpy(slot.m.bytes(), s.data[0..m_bytes]);
            @memcpy(slot.v.bytes(), s.data[m_bytes..]);
            commitSlotMaster(&slot.param, s.master);
        }
    }
};

/// PyTorch Adam keeps weight decay coupled to the gradient. This differs from
/// AdamW only when `weight_decay != 0`; NAM packed training uses that path.
/// State dtype handling mirrors `AdamWMap` (widen-on-read / narrow-on-write,
/// f32 math, pre-narrow values used within the element).
const AdamScalars = struct {
    weight_decay: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2s: f32,
    eps: f32,
};

fn AdamMap(comptime md: StateDType, comptime vd: StateDType) type {
    return struct {
        p: []f32,
        g: []const f32,
        m: StateSlice(md),
        v: StateSlice(vd),
        s: AdamScalars,

        fn run(c: @This(), start: usize, end: usize) void {
            if (comptime (md == .f32 and vd == .f32)) {
                // The golden-pinned baseline stays on the original scalar loop.
                runScalar(c, start, end);
                return;
            }
            // Hand-vectorized bf16 arm (see `state_vec_len`); bit-identical
            // to runScalar per element. The decay branch is loop-invariant.
            const wd: StateVec = @splat(c.s.weight_decay);
            const beta2: StateVec = @splat(c.s.beta2);
            const one_minus_b1: StateVec = @splat(c.s.one_minus_b1);
            const one_minus_b2: StateVec = @splat(c.s.one_minus_b2);
            const step_size: StateVec = @splat(c.s.step_size);
            const bc2s: StateVec = @splat(c.s.bc2s);
            const eps: StateVec = @splat(c.s.eps);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const pv: StateVec = c.p[i..][0..state_vec_len].*;
                const raw_gv: StateVec = c.g[i..][0..state_vec_len].*;
                const gv = if (c.s.weight_decay != 0) raw_gv + wd * pv else raw_gv;
                const m0 = stateVecLoad(md, c.m[i..][0..state_vec_len]);
                const v0 = stateVecLoad(vd, c.v[i..][0..state_vec_len]);
                const m1 = m0 + one_minus_b1 * (gv - m0);
                const v1 = beta2 * v0 + one_minus_b2 * gv * gv;
                stateVecStore(md, c.m[i..][0..state_vec_len], m1);
                stateVecStore(vd, c.v[i..][0..state_vec_len], v1);
                c.p[i..][0..state_vec_len].* = pv - step_size * (m1 / (@sqrt(v1) / bc2s + eps));
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, raw_gi, *mi, *vi| {
                const gi = if (c.s.weight_decay != 0) raw_gi + c.s.weight_decay * pi.* else raw_gi;
                const m0 = stateLoad(md, mi);
                const v0 = stateLoad(vd, vi);
                const m1 = m0 + c.s.one_minus_b1 * (gi - m0);
                const v1 = c.s.beta2 * v0 + c.s.one_minus_b2 * gi * gi;
                stateStore(md, mi, m1);
                stateStore(vd, vi, v1);
                pi.* -= c.s.step_size * (m1 / (@sqrt(v1) / c.s.bc2s + c.s.eps));
            }
        }
    };
}

fn adamRun(comptime md: StateDType, comptime vd: StateDType, ctx: *ExecContext, s: AdamScalars, p: []f32, g: []const f32, m: StateSlice(md), v: StateSlice(vd)) void {
    const Map = AdamMap(md, vd);
    parallelMap(ctx, p.len, Map{ .p = p, .g = g, .m = m, .v = v, .s = s }, Map.run);
}

fn adamUpdate(ctx: *ExecContext, config: AdamConfig, p: []f32, g: []const f32, m: StateBuf, v: StateBuf, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    const bc1 = 1 - std.math.pow(f64, config.beta1, t);
    const bc2_sqrt = @sqrt(1 - std.math.pow(f64, config.beta2, t));
    const s = AdamScalars{
        .weight_decay = config.weight_decay,
        .beta2 = config.beta2,
        .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
        .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        .step_size = @floatCast(@as(f64, config.lr) / bc1),
        .bc2s = @floatCast(bc2_sqrt),
        .eps = config.eps,
    };
    switch (m) {
        .f32 => |ms| switch (v) {
            .f32 => |vs| adamRun(.f32, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamRun(.f32, .bf16, ctx, s, p, g, ms, vs),
        },
        .bf16 => |ms| switch (v) {
            .f32 => |vs| adamRun(.bf16, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamRun(.bf16, .bf16, ctx, s, p, g, ms, vs),
        },
    }
}

// ---------------------------------------------------------------------------
// Muon — Keller Jordan's reference with the Moonlight scale/decay variant.
// ---------------------------------------------------------------------------

pub const MuonScale = enum {
    /// Keller's original: update *= sqrt(max(1, rows/cols)); lr stays in
    /// spectral-norm units.
    spectral,
    /// Moonlight's `0.2*sqrt(max(rows, cols))`: makes Muon reuse lr/wd tuned
    /// for AdamW (update RMS ~= 0.2).
    match_rms_adamw,
};

pub const MuonConfig = struct {
    lr: f32 = 0.02,
    momentum: f32 = 0.95,
    nesterov: bool = true,
    ns_steps: u32 = 5,
    weight_decay: f32 = 0,
    scale: MuonScale = .spectral,
    /// Storage dtype of the momentum buffer; step math stays f32. bf16 is
    /// safe here (per-step relative change ~5% at momentum 0.95; the Keller
    /// reference even runs the whole pipeline in bf16 on GPU). The embedded
    /// AdamW fallback has its own `state_dtype`/`second_moment_dtype` below.
    state_dtype: StateDType = .f32,
    /// AdamW for everything Muon must not touch: 0D/1D params route here
    /// automatically; route embeddings and output heads here explicitly via
    /// `addFallbackParam` (they are 2D but are not hidden-space maps).
    fallback: AdamWConfig = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0 },
};

pub const Muon = struct {
    allocator: Allocator,
    config: MuonConfig,
    slots: std.ArrayList(Slot) = .empty,
    fallback: AdamW,

    const Slot = struct {
        param: Param,
        momentum: StateBuf,
    };

    pub fn init(allocator: Allocator, config: MuonConfig) Muon {
        return .{
            .allocator = allocator,
            .config = config,
            .fallback = AdamW.init(allocator, config.fallback),
        };
    }

    pub fn deinit(self: *Muon) void {
        for (self.slots.items) |*slot| {
            slot.momentum.deinit(self.allocator);
            slot.param.deinit(self.allocator);
        }
        self.slots.deinit(self.allocator);
        self.fallback.deinit();
        self.* = undefined;
    }

    pub fn collectGradStates(self: *const Muon, set: *GradStateSet, allocator: Allocator) !void {
        if (gradStatesCollide(set, self.slots.items) or
            gradStatesCollide(set, self.fallback.slots.items)) return OptimError.DuplicateParam;
        try insertGradStates(set, allocator, self.slots.items);
        try insertGradStates(set, allocator, self.fallback.slots.items);
    }

    fn containsGradState(self: *const Muon, state: *const GradState) bool {
        for (self.slots.items) |*slot| {
            if (slot.param.grad_state == state) return true;
        }
        return self.fallback.containsGradState(state);
    }

    /// Matrix params (raw rank >= 2; conv filters are flattened `[d0, rest]`)
    /// get Muon; lower-rank params route to the AdamW fallback.
    pub fn addParam(self: *Muon, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        try self.addOwnedParam(param);
    }

    /// `addParam` plus a checkpoint name (borrowed; see `Param.name`).
    pub fn addParamNamed(self: *Muon, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        try self.addOwnedParam(param);
    }

    fn addOwnedParam(self: *Muon, param: Param) !void {
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        if (param.raw_rank < 2) {
            try self.fallback.addOwnedParam(param);
            return;
        }
        var owned = param;
        try owned.ensureMaster(self.allocator);
        errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
        const momentum = try StateBuf.alloc(self.allocator, self.config.state_dtype, owned.len());
        errdefer momentum.deinit(self.allocator);
        try self.slots.append(self.allocator, .{ .param = owned, .momentum = momentum });
    }

    /// Force a param onto the AdamW fallback (embeddings, lm/classifier heads).
    pub fn addFallbackParam(self: *Muon, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        try self.fallback.addOwnedParam(param);
    }

    /// `addFallbackParam` plus a checkpoint name (borrowed; see `Param.name`).
    pub fn addFallbackParamNamed(self: *Muon, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        try self.fallback.addOwnedParam(param);
    }

    pub fn step(self: *Muon, ctx: *ExecContext) !void {
        for (self.slots.items) |*slot| {
            var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
            defer grad.deinit();
            try self.muonUpdate(ctx, slot, grad.dataConst());
            slot.param.publish();
        }
        try self.fallback.step(ctx);
    }

    pub fn zeroGrad(self: *Muon) void {
        for (self.slots.items) |*slot| slot.param.grad_state.zeroGrad();
        self.fallback.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *Muon, ctx: *ExecContext) !f64 {
        var total: f64 = try self.fallback.gradSquaredNorm(ctx);
        for (self.slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
        return total;
    }

    pub fn scaleGradients(self: *Muon, ctx: *ExecContext, factor: f32) !void {
        for (self.slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
        try self.fallback.scaleGradients(ctx, factor);
    }

    /// L2 global-norm clip over Muon AND fallback params together.
    pub fn clipGradNorm(self: *Muon, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    fn MomentumMap(comptime sd: StateDType) type {
        return struct {
            m: StateSlice(sd),
            g: []const f32,
            u: []f32,
            beta: f32,
            nesterov: bool,

            fn run(c: @This(), start: usize, end: usize) void {
                if (comptime sd == .f32) {
                    // The golden-pinned baseline stays on the original scalar loop.
                    runScalar(c, start, end);
                    return;
                }
                // Hand-vectorized bf16 arm (see `state_vec_len`); bit-identical
                // to runScalar per element. The lerp-form and nesterov branches
                // are loop-invariant.
                const w = 1 - c.beta;
                const wv: StateVec = @splat(w);
                const betav: StateVec = @splat(c.beta);
                var i = start;
                while (i + state_vec_len <= end) : (i += state_vec_len) {
                    const gv: StateVec = c.g[i..][0..state_vec_len].*;
                    const m0 = stateVecLoad(sd, c.m[i..][0..state_vec_len]);
                    const m1 = if (w < 0.5) m0 + wv * (gv - m0) else gv - (gv - m0) * betav;
                    stateVecStore(sd, c.m[i..][0..state_vec_len], m1);
                    c.u[i..][0..state_vec_len].* = if (c.nesterov)
                        (if (c.beta < 0.5) gv + betav * (m1 - gv) else m1 - (m1 - gv) * wv)
                    else
                        m1;
                }
                runScalar(c, i, end);
            }

            fn runScalar(c: @This(), start: usize, end: usize) void {
                // Keller's lerp form: M <- M + (1-beta)*(g - M). The nesterov update
                // blends the raw grad with the NEW buffer: U = lerp(g, M, beta).
                // ATen's lerp kernel switches forms at |weight| = 0.5; mirror both
                // branches so the FP op sequence matches the torch-run reference
                // (beta defaults to 0.95, taking the end-anchored form). bf16
                // momentum widens on read, narrows on write; U uses the
                // just-computed (pre-narrow) f32 buffer value.
                for (c.m[start..end], c.g[start..end], c.u[start..end]) |*mi, gi, *ui| {
                    const w = 1 - c.beta;
                    const m0 = stateLoad(sd, mi);
                    const m1 = if (w < 0.5) m0 + w * (gi - m0) else gi - (gi - m0) * c.beta;
                    stateStore(sd, mi, m1);
                    ui.* = if (c.nesterov)
                        (if (c.beta < 0.5) gi + c.beta * (m1 - gi) else m1 - (m1 - gi) * w)
                    else
                        m1;
                }
            }
        };
    }

    fn momentumRun(comptime sd: StateDType, ctx: *ExecContext, m: StateSlice(sd), g: []const f32, u: []f32, beta: f32, nesterov: bool) void {
        const Map = MomentumMap(sd);
        parallelMap(ctx, g.len, Map{ .m = m, .g = g, .u = u, .beta = beta, .nesterov = nesterov }, Map.run);
    }

    const ApplyMapContext = struct {
        p: []f32,
        o: []const f32,
        keep: f32,
        lr_eff: f32,
    };

    fn applyMapRange(c: ApplyMapContext, start: usize, end: usize) void {
        // Decoupled decay with the BASE lr, then the shape-scaled update.
        for (c.p[start..end], c.o[start..end]) |*pi, oi| {
            pi.* = pi.* * c.keep - c.lr_eff * oi;
        }
    }

    fn muonUpdate(self: *Muon, ctx: *ExecContext, slot: *Slot, g: []const f32) !void {
        const config = self.config;
        const rows = slot.param.rows;
        const cols = slot.param.cols;
        var u = try ctx.emptyRank(2, .{ rows, cols });
        defer u.deinit();
        switch (slot.momentum) {
            .f32 => |ms| momentumRun(.f32, ctx, ms, g, u.data(), config.momentum, config.nesterov),
            .bf16 => |ms| momentumRun(.bf16, ctx, ms, g, u.data(), config.momentum, config.nesterov),
        }

        var ortho = try newtonSchulz5(ctx, &u, config.ns_steps);
        defer ortho.deinit();

        const rows_f: f32 = @floatFromInt(rows);
        const cols_f: f32 = @floatFromInt(cols);
        const lr_eff = switch (config.scale) {
            .spectral => config.lr * @sqrt(@max(1, rows_f / cols_f)),
            .match_rms_adamw => config.lr * 0.2 * @sqrt(@max(rows_f, cols_f)),
        };
        parallelMap(ctx, slot.param.len(), ApplyMapContext{
            .p = slot.param.data(),
            .o = ortho.dataConst(),
            .keep = if (config.weight_decay != 0) @floatCast(1.0 - @as(f64, config.lr) * @as(f64, config.weight_decay)) else 1,
            .lr_eff = lr_eff,
        }, applyMapRange);
    }

    pub fn saveState(self: *const Muon, writer: *std.Io.Writer) !void {
        try validateSlotNames(self.slots.items);
        // Version is decided by Muon's OWN momentum buffers; the fallback
        // frames its m/v independently below.
        var version: FrameVersion = .v3;
        for (self.slots.items) |*slot| {
            if (slot.momentum != .f32) version = .v4;
        }
        if (slotsCarryMasters(self.slots.items)) version = .v5;
        try writer.writeAll(switch (version) {
            .v3 => "FZM3",
            .v4 => "FZM4",
            .v5 => "FZM5",
        });
        try writer.writeInt(u8, @intFromEnum(self.config.scale), .little);
        try writer.writeInt(u8, @intFromBool(self.config.nesterov), .little);
        try writer.writeInt(u32, self.config.ns_steps, .little);
        try writer.writeInt(u32, @intCast(self.slots.items.len), .little);
        for (self.slots.items, 0..) |*slot, i| {
            try writeSlotName(writer, &slot.param, i);
            try writeSlotDims(writer, &slot.param);
            try writeStateSlice(writer, version, slot.momentum);
            try writeSlotMaster(writer, version, &slot.param);
        }
        try self.fallback.saveState(writer);
    }

    pub fn loadState(self: *Muon, reader: *std.Io.Reader) !void {
        const version = try expectMagicVersion(reader, "FZM3", "FZM4", "FZM5");
        if (try reader.takeInt(u8, .little) != @intFromEnum(self.config.scale)) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u8, .little) != @intFromBool(self.config.nesterov)) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u32, .little) != self.config.ns_steps) return OptimError.CheckpointConfigMismatch;
        const count = try reader.takeInt(u32, .little);
        var matcher = try SlotMatcher.init(self.allocator, self.slots.items.len);
        defer matcher.deinit(self.allocator);
        var staged = try std.ArrayList(StagedSlot).initCapacity(self.allocator, count);
        defer freeStaged(self.allocator, &staged);
        for (0..count) |_| {
            const idx = try matcher.match(reader, self.slots.items);
            const slot = &self.slots.items[idx];
            try expectSlotDims(reader, &slot.param);
            const data = try self.allocator.alloc(u8, slot.momentum.byteLen());
            errdefer self.allocator.free(data);
            try readStateSlice(reader, version, slot.momentum, data);
            const master = try readSlotMaster(self.allocator, reader, version, &slot.param);
            errdefer if (master.len != 0) self.allocator.free(master);
            try staged.append(self.allocator, .{ .idx = idx, .data = data, .master = master });
        }
        try matcher.requireAllFilled();
        // The fallback's own slot data follows in the stream and loads
        // transactionally too, so commit Muon's slots only after it succeeds —
        // keeping Muon + fallback atomic as a whole.
        try self.fallback.loadState(reader);
        for (staged.items) |s| {
            @memcpy(self.slots.items[s.idx].momentum.bytes(), s.data);
            commitSlotMaster(&self.slots.items[s.idx].param, s.master);
        }
    }
};

const ns_coeff_a: f32 = 3.4445;
const ns_coeff_b: f32 = -4.7750;
const ns_coeff_c: f32 = 2.0315;

/// Keller's `zeropower_via_newtonschulz5`: Frobenius-normalize so the spectral
/// norm is <= 1, then iterate the tuned quintic X <- a*X + (b*A + c*A*A)*X with
/// A = X*X^T. When rows > cols the iteration runs on X^T so the Gram matrix has
/// the small dimension. The reference runs in bf16 on GPU; f32 here is strictly
/// more accurate. The result approximates U*V^T with singular values in roughly
/// (0.5, 1.5) — by design, not a bug. `u` must be rank-2 and is never mutated.
pub fn newtonSchulz5(ctx: *ExecContext, u: *const RawTensor, steps: u32) !RawTensor {
    const rows = u.shape.at(0);
    const cols = u.shape.at(1);
    const transposed = rows > cols;
    var x = if (transposed) try transpose2D(ctx, u) else try ctx.materialize(u);
    errdefer x.deinit();

    const sumsq = try sumSquares(ctx, x.dataConst());
    const inv_norm: f32 = @floatCast(1.0 / (@sqrt(sumsq) + 1e-7));
    for (x.data()) |*value| value.* *= inv_norm;

    for (0..steps) |_| {
        var gram = try ctx.matmulTransB(&x, &x);
        defer gram.deinit();
        var quad = try ctx.matmul2D(&gram, &gram);
        defer quad.deinit();
        for (quad.data(), gram.dataConst()) |*qi, gi| qi.* = ns_coeff_b * gi + ns_coeff_c * qi.*;
        var bx = try ctx.matmul2D(&quad, &x);
        errdefer bx.deinit();
        for (bx.data(), x.dataConst()) |*oi, xi| oi.* = ns_coeff_a * xi + oi.*;
        x.deinit();
        x = bx;
    }

    if (transposed) {
        const out = try transpose2D(ctx, &x);
        x.deinit();
        return out;
    }
    return x;
}

fn transpose2D(ctx: *ExecContext, t: *const RawTensor) !RawTensor {
    const rows = t.shape.at(0);
    const cols = t.shape.at(1);
    var view = try t.viewWithStrides(&.{ cols, rows }, &.{ 1, cols });
    defer view.deinit();
    return try ctx.materialize(&view);
}

// ---------------------------------------------------------------------------
// APOLLO — official apollo_torch semantics (arXiv 2412.05270).
// ---------------------------------------------------------------------------

pub const ApolloScaleType = enum { channel, tensor };

pub const ApolloConfig = struct {
    lr: f32 = 0.01,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    /// Reference-class default (legacy HF AdamW lineage); HF Trainer overrides
    /// it to 1e-8.
    eps: f32 = 1e-6,
    weight_decay: f32 = 0,
    rank: usize = 128,
    update_proj_gap: u64 = 200,
    /// `apollo_scale`; the update multiplies sqrt(scale).
    scale: f32 = 1.0,
    scale_type: ApolloScaleType = .channel,
    correct_bias: bool = true,
    /// Apply sqrt(scale) BEFORE the norm-growth limiter (reference knob for
    /// short-warmup runs); default is after.
    scale_front: bool = false,
    disable_norm_growth_limiter: bool = false,
    seed: u64 = 0,

    /// APOLLO-Mini: rank-1 random projection with tensor-wise scaling and the
    /// sqrt(128) heuristic gradient scale.
    pub fn mini() ApolloConfig {
        return .{ .rank = 1, .scale_type = .tensor, .scale = 128 };
    }
};

pub const Apollo = struct {
    allocator: Allocator,
    config: ApolloConfig,
    slots: std.ArrayList(Slot) = .empty,
    /// Non-2D params (biases, norms) and explicitly routed params use the
    /// reference's plain-AdamW path (legacy HF order, NOT `optim.AdamW`).
    fallback_slots: std.ArrayList(FallbackSlot) = .empty,

    const Slot = struct {
        param: Param,
        m: []f32,
        v: []f32,
        /// Per-channel (or single-element, for tensor scaling) factor scratch.
        scaling: []f32,
        /// f64 accumulators for the per-channel norm sums (2 per channel),
        /// preallocated so step() stays allocation-free.
        norms: []f64,
        proj: ?RawTensor = null,
        proj_chunk: u64 = std.math.maxInt(u64),
        seed: u64,
        step: u64 = 0,
        /// Norm-growth-limiter memory; negative means "not recorded yet".
        prev_norm: f32 = -1,
    };

    const FallbackSlot = struct {
        param: Param,
        m: []f32,
        v: []f32,
        step: u64 = 0,
    };

    pub fn init(allocator: Allocator, config: ApolloConfig) Apollo {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *Apollo) void {
        for (self.slots.items) |*slot| {
            self.allocator.free(slot.m);
            self.allocator.free(slot.v);
            self.allocator.free(slot.scaling);
            self.allocator.free(slot.norms);
            if (slot.proj) |*proj| proj.deinit();
            slot.param.deinit(self.allocator);
        }
        self.slots.deinit(self.allocator);
        for (self.fallback_slots.items) |*slot| {
            self.allocator.free(slot.m);
            self.allocator.free(slot.v);
            slot.param.deinit(self.allocator);
        }
        self.fallback_slots.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn collectGradStates(self: *const Apollo, set: *GradStateSet, allocator: Allocator) !void {
        if (gradStatesCollide(set, self.slots.items) or
            gradStatesCollide(set, self.fallback_slots.items)) return OptimError.DuplicateParam;
        try insertGradStates(set, allocator, self.slots.items);
        try insertGradStates(set, allocator, self.fallback_slots.items);
    }

    fn containsGradState(self: *const Apollo, state: *const GradState) bool {
        for (self.slots.items) |*slot| {
            if (slot.param.grad_state == state) return true;
        }
        for (self.fallback_slots.items) |*slot| {
            if (slot.param.grad_state == state) return true;
        }
        return false;
    }

    /// 2D params get the APOLLO low-rank path; everything else gets the plain
    /// AdamW fallback (the reference restricts the rank path to Linear weights).
    pub fn addParam(self: *Apollo, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        try self.addOwnedParam(param);
    }

    /// `addParam` plus a checkpoint name (borrowed; see `Param.name`).
    pub fn addParamNamed(self: *Apollo, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        try self.addOwnedParam(param);
    }

    fn addOwnedParam(self: *Apollo, param: Param) !void {
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        if (param.raw_rank != 2) {
            try self.addOwnedFallback(param);
            return;
        }
        var owned = param;
        try owned.ensureMaster(self.allocator);
        errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
        const compressed = compressedLen(&owned, self.config.rank);
        const m = try self.allocator.alloc(f32, compressed);
        errdefer self.allocator.free(m);
        const v = try self.allocator.alloc(f32, compressed);
        errdefer self.allocator.free(v);
        const channels: usize = switch (self.config.scale_type) {
            .channel => if (param.rows >= param.cols) param.rows else param.cols,
            .tensor => 1,
        };
        const scaling = try self.allocator.alloc(f32, channels);
        errdefer self.allocator.free(scaling);
        const norms = try self.allocator.alloc(f64, 2 * channels);
        errdefer self.allocator.free(norms);
        @memset(m, 0);
        @memset(v, 0);
        // Distinct per-param seed (base + 1-based rank-slot index). The
        // reference enumerates every param for its torch RNG; only "distinct
        // seed, i.i.d. N(0, 1/rank) entries" is semantically required, and the
        // torch RNG stream is not reproducible here anyway.
        const seed = self.config.seed +% (self.slots.items.len + 1);
        try self.slots.append(self.allocator, .{ .param = owned, .m = m, .v = v, .scaling = scaling, .norms = norms, .seed = seed });
    }

    /// Force a param onto the AdamW fallback path (e.g. embeddings, heads).
    pub fn addFallbackParam(self: *Apollo, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        try self.addOwnedFallback(param);
    }

    /// `addFallbackParam` plus a checkpoint name (borrowed; see `Param.name`).
    pub fn addFallbackParamNamed(self: *Apollo, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        try self.addOwnedFallback(param);
    }

    fn addOwnedFallback(self: *Apollo, param: Param) !void {
        var owned = param;
        try owned.ensureMaster(self.allocator);
        errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
        const n = owned.len();
        const m = try self.allocator.alloc(f32, n);
        errdefer self.allocator.free(m);
        const v = try self.allocator.alloc(f32, n);
        errdefer self.allocator.free(v);
        @memset(m, 0);
        @memset(v, 0);
        try self.fallback_slots.append(self.allocator, .{ .param = owned, .m = m, .v = v });
    }

    fn compressedLen(param: *const Param, rank: usize) usize {
        // Tall/square (rows >= cols): R = G @ P^T is (rows, rank).
        // Wide (rows < cols): R = P^T @ G is (rank, cols).
        return if (param.rows >= param.cols) param.rows * rank else rank * param.cols;
    }

    pub fn step(self: *Apollo, ctx: *ExecContext) !void {
        for (self.slots.items) |*slot| {
            var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
            defer grad.deinit();
            try self.apolloUpdate(ctx, slot, &grad);
            slot.param.publish();
        }
        for (self.fallback_slots.items) |*slot| {
            var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
            defer grad.deinit();
            slot.step += 1;
            hfAdamwUpdate(ctx, self.config, slot.param.data(), grad.dataConst(), slot.m, slot.v, slot.step);
            slot.param.publish();
        }
    }

    pub fn zeroGrad(self: *Apollo) void {
        for (self.slots.items) |*slot| slot.param.grad_state.zeroGrad();
        for (self.fallback_slots.items) |*slot| slot.param.grad_state.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *Apollo, ctx: *ExecContext) !f64 {
        var total: f64 = 0;
        for (self.slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
        for (self.fallback_slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
        return total;
    }

    pub fn scaleGradients(self: *Apollo, ctx: *ExecContext, factor: f32) !void {
        for (self.slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
        for (self.fallback_slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
    }

    /// L2 global-norm clip over rank-path AND fallback params together. Note
    /// the APOLLO recipes disable global clipping (the norm-growth limiter
    /// replaces it) — provided for completeness and mixed setups.
    pub fn clipGradNorm(self: *Apollo, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    fn apolloUpdate(self: *Apollo, ctx: *ExecContext, slot: *Slot, grad: *RawTensor) !void {
        const config = self.config;
        const rows = slot.param.rows;
        const cols = slot.param.cols;
        const tall = rows >= cols;
        const rank = config.rank;

        // Projection regeneration uses the PRE-increment step counter:
        // chunks are [0,T), [T,2T), ... States are NOT reset on regeneration.
        const chunk = slot.step / config.update_proj_gap;
        if (slot.proj == null or slot.proj_chunk != chunk) {
            try self.regenerateProjection(ctx, slot, chunk, tall);
        }

        var r_t = if (tall)
            try ctx.matmulTransB(grad, &slot.proj.?) // (rows, rank)
        else
            try ctx.matmulTransA(&slot.proj.?, grad); // (rank, cols)
        defer r_t.deinit();
        const r_data = r_t.dataConst();

        slot.step += 1;
        const t: f64 = @floatFromInt(slot.step);

        parallelMap(ctx, r_data.len, MomentMapContext{
            .m = slot.m,
            .v = slot.v,
            .r = r_data,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
            .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        }, momentMapRange);

        var step_size: f32 = config.lr;
        if (config.correct_bias) {
            const bc1 = 1 - std.math.pow(f64, config.beta1, t);
            const bc2 = 1 - std.math.pow(f64, config.beta2, t);
            step_size = @floatCast(@as(f64, config.lr) * @sqrt(bc2) / bc1);
        }

        // Scaling factors from the UN-bias-corrected R~ = m/(sqrt(v)+eps),
        // norms taken along the rank axis; +1e-8 guards the division.
        const scaling = slot.scaling;
        switch (config.scale_type) {
            .channel => {
                const channels = scaling.len;
                const sum_opt = slot.norms[0..channels];
                const sum_raw = slot.norms[channels..];
                @memset(sum_opt, 0);
                @memset(sum_raw, 0);
                if (tall) {
                    // R is (rows=channels, rank): channel = row index.
                    for (slot.m, slot.v, r_data, 0..) |mi, vi, ri, i| {
                        const channel = i / rank;
                        const opt = mi / (@sqrt(vi) + config.eps);
                        sum_opt[channel] += @as(f64, opt) * opt;
                        sum_raw[channel] += @as(f64, ri) * ri;
                    }
                } else {
                    // R is (rank, cols=channels): channel = column index.
                    for (slot.m, slot.v, r_data, 0..) |mi, vi, ri, i| {
                        const channel = i % cols;
                        const opt = mi / (@sqrt(vi) + config.eps);
                        sum_opt[channel] += @as(f64, opt) * opt;
                        sum_raw[channel] += @as(f64, ri) * ri;
                    }
                }
                for (scaling, sum_opt, sum_raw) |*s, so, sr| {
                    s.* = @floatCast(@sqrt(so) / (@sqrt(sr) + 1e-8));
                }
            },
            .tensor => {
                var sum_opt: f64 = 0;
                var sum_raw: f64 = 0;
                for (slot.m, slot.v, r_data) |mi, vi, ri| {
                    const opt = mi / (@sqrt(vi) + config.eps);
                    sum_opt += @as(f64, opt) * opt;
                    sum_raw += @as(f64, ri) * ri;
                }
                scaling[0] = @floatCast(@sqrt(sum_opt) / (@sqrt(sum_raw) + 1e-8));
            },
        }

        // U = G (elementwise) scaled per channel (rows for tall, cols for
        // wide); the optional front sqrt(scale) is fused (same per-element op
        // order as the reference's separate pass).
        var update = try ctx.emptyRank(2, .{ rows, cols });
        defer update.deinit();
        const ud = update.data();
        const sqrt_scale = @sqrt(config.scale);
        parallelMap(ctx, ud.len, BuildMapContext{
            .u = ud,
            .g = grad.dataConst(),
            .scaling = scaling,
            .cols = cols,
            .kind = switch (config.scale_type) {
                .channel => if (tall) BuildKind.channel_tall else BuildKind.channel_wide,
                .tensor => BuildKind.tensor,
            },
            .front_scale = if (config.scale_front and config.scale != 1) sqrt_scale else 1,
        }, buildMapRange);

        // Fira norm-growth limiter (gamma = 1.01): clip the per-step growth of
        // ||U||_F; the recorded norm is the POST-limit one. The norm reduction
        // is the deterministic chunked `sumSquares`; the division and the
        // trailing sqrt(scale) fuse into the apply pass below with the same
        // per-element op order as the reference's separate passes.
        var limiter: f32 = 1;
        if (!config.disable_norm_growth_limiter) {
            const cur: f32 = @floatCast(@sqrt(try sumSquares(ctx, ud)));
            if (slot.prev_norm >= 0) {
                limiter = @max(cur / (slot.prev_norm + 1e-8), 1.01) / 1.01;
                slot.prev_norm = cur / limiter;
            } else {
                slot.prev_norm = cur;
            }
        }

        // Scaled-SGD step, then decoupled decay AFTER the step with the raw lr
        // (reference order; differs from AdamW/Muon).
        parallelMap(ctx, slot.param.len(), ApplyMapContext{
            .p = slot.param.data(),
            .u = ud,
            .limiter = limiter,
            .back_scale = if (!config.scale_front and config.scale != 1) sqrt_scale else 1,
            .step_size = step_size,
            .decay_alpha = if (config.weight_decay > 0) @floatCast(-(@as(f64, config.lr) * @as(f64, config.weight_decay))) else 0,
        }, applyMapRange);
    }

    const MomentMapContext = struct {
        m: []f32,
        v: []f32,
        r: []const f32,
        beta1: f32,
        beta2: f32,
        one_minus_b1: f32,
        one_minus_b2: f32,
    };

    fn momentMapRange(c: MomentMapContext, start: usize, end: usize) void {
        for (c.m[start..end], c.v[start..end], c.r[start..end]) |*mi, *vi, ri| {
            mi.* = c.beta1 * mi.* + c.one_minus_b1 * ri;
            vi.* = c.beta2 * vi.* + c.one_minus_b2 * ri * ri;
        }
    }

    const BuildKind = enum { channel_tall, channel_wide, tensor };

    const BuildMapContext = struct {
        u: []f32,
        g: []const f32,
        scaling: []const f32,
        cols: usize,
        kind: BuildKind,
        front_scale: f32,
    };

    fn buildMapRange(c: BuildMapContext, start: usize, end: usize) void {
        switch (c.kind) {
            .channel_tall => for (c.u[start..end], c.g[start..end], start..) |*ui, gi, i| {
                ui.* = gi * c.scaling[i / c.cols] * c.front_scale;
            },
            .channel_wide => for (c.u[start..end], c.g[start..end], start..) |*ui, gi, i| {
                ui.* = gi * c.scaling[i % c.cols] * c.front_scale;
            },
            .tensor => for (c.u[start..end], c.g[start..end]) |*ui, gi| {
                ui.* = gi * c.scaling[0] * c.front_scale;
            },
        }
    }

    const ApplyMapContext = struct {
        p: []f32,
        u: []const f32,
        limiter: f32,
        back_scale: f32,
        step_size: f32,
        decay_alpha: f32,
    };

    fn applyMapRange(c: ApplyMapContext, start: usize, end: usize) void {
        for (c.p[start..end], c.u[start..end]) |*pi, ui| {
            const adjusted = ui / c.limiter * c.back_scale;
            // Decay AFTER the step, in the reference's additive form
            // `p.add_(p, alpha=-lr*wd)`.
            const stepped = pi.* - c.step_size * adjusted;
            pi.* = stepped + c.decay_alpha * stepped;
        }
    }

    /// P entries are i.i.d. N(0, 1/rank): standard normal draws divided by
    /// sqrt(rank). Deterministic in (seed, chunk), so checkpoints don't need to
    /// store P. Tall params project the column space: P is (rank, cols); wide
    /// params project the row space: P is (rows, rank).
    fn regenerateProjection(self: *Apollo, ctx: *ExecContext, slot: *Slot, chunk: u64, tall: bool) !void {
        const config = self.config;
        const shape: [2]usize = if (tall)
            .{ config.rank, slot.param.cols }
        else
            .{ slot.param.rows, config.rank };
        if (slot.proj == null) {
            slot.proj = try ctx.emptyRank(2, shape);
        }
        const inv_sqrt_rank = 1.0 / @sqrt(@as(f32, @floatFromInt(config.rank)));
        gaussianFill(slot.seed +% chunk *% 0x9E3779B97F4A7C15, slot.proj.?.data(), inv_sqrt_rank);
        slot.proj_chunk = chunk;
    }

    pub fn saveState(self: *const Apollo, writer: *std.Io.Writer) !void {
        try validateSlotNames(self.slots.items);
        try validateSlotNames(self.fallback_slots.items);
        const version: FrameVersion = if (slotsCarryMasters(self.slots.items) or slotsCarryMasters(self.fallback_slots.items)) .v5 else .v3;
        try writer.writeAll(switch (version) {
            .v5 => "FZP5",
            else => "FZP3",
        });
        try writer.writeInt(u64, @intCast(self.config.rank), .little);
        try writer.writeInt(u64, self.config.update_proj_gap, .little);
        try writer.writeInt(u32, @bitCast(self.config.scale), .little);
        try writer.writeInt(u8, @intFromEnum(self.config.scale_type), .little);
        try writer.writeInt(u8, @intFromBool(self.config.correct_bias), .little);
        try writer.writeInt(u8, @intFromBool(self.config.scale_front), .little);
        try writer.writeInt(u8, @intFromBool(self.config.disable_norm_growth_limiter), .little);
        try writer.writeInt(u32, @intCast(self.slots.items.len), .little);
        for (self.slots.items, 0..) |*slot, i| {
            try writeSlotName(writer, &slot.param, i);
            try writeSlotDims(writer, &slot.param);
            try writer.writeInt(u64, slot.step, .little);
            try writer.writeInt(u64, slot.seed, .little);
            try writer.writeInt(u32, @bitCast(slot.prev_norm), .little);
            try writeF32Slice(writer, slot.m);
            try writeF32Slice(writer, slot.v);
            try writeSlotMaster(writer, version, &slot.param);
        }
        try writer.writeInt(u32, @intCast(self.fallback_slots.items.len), .little);
        for (self.fallback_slots.items, 0..) |*slot, i| {
            try writeSlotName(writer, &slot.param, i);
            try writeSlotDims(writer, &slot.param);
            try writer.writeInt(u64, slot.step, .little);
            try writeF32Slice(writer, slot.m);
            try writeF32Slice(writer, slot.v);
            try writeSlotMaster(writer, version, &slot.param);
        }
    }

    pub fn loadState(self: *Apollo, reader: *std.Io.Reader) !void {
        var magic: [4]u8 = undefined;
        try reader.readSliceAll(&magic);
        const version: FrameVersion = if (std.mem.eql(u8, &magic, "FZP5"))
            .v5
        else if (std.mem.eql(u8, &magic, "FZP3"))
            .v3
        else
            return OptimError.CheckpointMagicMismatch;
        if (try reader.takeInt(u64, .little) != self.config.rank) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u64, .little) != self.config.update_proj_gap) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u32, .little) != @as(u32, @bitCast(self.config.scale))) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u8, .little) != @intFromEnum(self.config.scale_type)) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u8, .little) != @intFromBool(self.config.correct_bias)) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u8, .little) != @intFromBool(self.config.scale_front)) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u8, .little) != @intFromBool(self.config.disable_norm_growth_limiter)) return OptimError.CheckpointConfigMismatch;
        const count = try reader.takeInt(u32, .little);
        var main_matcher = try SlotMatcher.init(self.allocator, self.slots.items.len);
        defer main_matcher.deinit(self.allocator);
        var staged_main = try std.ArrayList(StagedSlot).initCapacity(self.allocator, count);
        defer freeStaged(self.allocator, &staged_main);
        for (0..count) |_| {
            const idx = try main_matcher.match(reader, self.slots.items);
            const slot = &self.slots.items[idx];
            try expectSlotDims(reader, &slot.param);
            const step_val = try reader.takeInt(u64, .little);
            const seed = try reader.takeInt(u64, .little);
            const prev_norm: f32 = @bitCast(try reader.takeInt(u32, .little));
            const data = try self.allocator.alloc(u8, 4 * (slot.m.len + slot.v.len));
            errdefer self.allocator.free(data);
            try reader.readSliceAll(data);
            const master = try readSlotMaster(self.allocator, reader, version, &slot.param);
            errdefer if (master.len != 0) self.allocator.free(master);
            try staged_main.append(self.allocator, .{ .idx = idx, .step = step_val, .seed = seed, .prev_norm = prev_norm, .data = data, .master = master });
        }
        try main_matcher.requireAllFilled();

        const fallback_count = try reader.takeInt(u32, .little);
        var fb_matcher = try SlotMatcher.init(self.allocator, self.fallback_slots.items.len);
        defer fb_matcher.deinit(self.allocator);
        var staged_fb = try std.ArrayList(StagedSlot).initCapacity(self.allocator, fallback_count);
        defer freeStaged(self.allocator, &staged_fb);
        for (0..fallback_count) |_| {
            const idx = try fb_matcher.match(reader, self.fallback_slots.items);
            const slot = &self.fallback_slots.items[idx];
            try expectSlotDims(reader, &slot.param);
            const step_val = try reader.takeInt(u64, .little);
            const data = try self.allocator.alloc(u8, 4 * (slot.m.len + slot.v.len));
            errdefer self.allocator.free(data);
            try reader.readSliceAll(data);
            const master = try readSlotMaster(self.allocator, reader, version, &slot.param);
            errdefer if (master.len != 0) self.allocator.free(master);
            try staged_fb.append(self.allocator, .{ .idx = idx, .step = step_val, .data = data, .master = master });
        }
        try fb_matcher.requireAllFilled();

        // Commit — both slot sets fully validated; no failure points remain.
        for (staged_main.items) |s| {
            const slot = &self.slots.items[s.idx];
            slot.step = s.step;
            slot.seed = s.seed;
            slot.prev_norm = s.prev_norm;
            @memcpy(std.mem.sliceAsBytes(slot.m), s.data[0 .. 4 * slot.m.len]);
            @memcpy(std.mem.sliceAsBytes(slot.v), s.data[4 * slot.m.len ..]);
            commitSlotMaster(&slot.param, s.master);
            // The projection is a pure function of (seed, step/gap); force
            // regeneration on the next step.
            slot.proj_chunk = std.math.maxInt(u64);
        }
        for (staged_fb.items) |s| {
            const slot = &self.fallback_slots.items[s.idx];
            slot.step = s.step;
            @memcpy(std.mem.sliceAsBytes(slot.m), s.data[0 .. 4 * slot.m.len]);
            @memcpy(std.mem.sliceAsBytes(slot.v), s.data[4 * slot.m.len ..]);
            commitSlotMaster(&slot.param, s.master);
        }
    }
};

/// The APOLLO reference's fallback AdamW is the legacy HF formulation, NOT
/// PyTorch AdamW: `denom = sqrt(v) + eps` (eps outside the bias correction),
/// the bias correction folded into a scalar `lr*sqrt(bc2)/bc1`, and decoupled
/// decay applied AFTER the step to the already-updated parameter.
const HfAdamwMapContext = struct {
    p: []f32,
    g: []const f32,
    m: []f32,
    v: []f32,
    beta1: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    eps: f32,
    decay_alpha: f32,
};

fn hfAdamwMapRange(c: HfAdamwMapContext, start: usize, end: usize) void {
    for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, gi, *mi, *vi| {
        mi.* = c.beta1 * mi.* + c.one_minus_b1 * gi;
        vi.* = c.beta2 * vi.* + c.one_minus_b2 * gi * gi;
        // Decay AFTER the step (legacy-HF order), fused per element in the
        // reference's additive form `p.add_(p, alpha=-lr*wd)`.
        const stepped = pi.* - c.step_size * (mi.* / (@sqrt(vi.*) + c.eps));
        pi.* = stepped + c.decay_alpha * stepped;
    }
}

fn hfAdamwUpdate(ctx: *ExecContext, config: ApolloConfig, p: []f32, g: []const f32, m: []f32, v: []f32, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    var step_size: f32 = config.lr;
    if (config.correct_bias) {
        const bc1 = 1 - std.math.pow(f64, config.beta1, t);
        const bc2 = 1 - std.math.pow(f64, config.beta2, t);
        step_size = @floatCast(@as(f64, config.lr) * @sqrt(bc2) / bc1);
    }
    parallelMap(ctx, p.len, HfAdamwMapContext{
        .p = p,
        .g = g,
        .m = m,
        .v = v,
        .beta1 = config.beta1,
        .beta2 = config.beta2,
        .one_minus_b1 = @floatCast(1.0 - @as(f64, config.beta1)),
        .one_minus_b2 = @floatCast(1.0 - @as(f64, config.beta2)),
        .step_size = step_size,
        .eps = config.eps,
        .decay_alpha = if (config.weight_decay > 0) @floatCast(-(@as(f64, config.lr) * @as(f64, config.weight_decay))) else 0,
    }, hfAdamwMapRange);
}

// ---------------------------------------------------------------------------
// SGD — PyTorch torch.optim.SGD single-tensor semantics.
// ---------------------------------------------------------------------------

pub const SgdConfig = struct {
    lr: f32 = 1e-3,
    /// 0 disables the momentum buffer entirely (no state RAM).
    momentum: f32 = 0,
    dampening: f32 = 0,
    /// COUPLED L2 (g += wd*p), like PyTorch SGD — not AdamW-style decoupled.
    weight_decay: f32 = 0,
    /// Requires momentum > 0 and dampening == 0 (PyTorch constructor rule).
    nesterov: bool = false,
    /// Storage dtype of the momentum buffer; step math stays f32. bf16 is
    /// safe here (per-step relative change ~10% at momentum 0.9). Ignored
    /// when `momentum == 0` (no buffer exists). Note the first-step
    /// `buf = d_p.clone()` semantics store the NARROWED decayed gradient.
    state_dtype: StateDType = .f32,
};

pub const SGD = struct {
    allocator: Allocator,
    config: SgdConfig,
    slots: std.ArrayList(Slot) = .empty,

    const Slot = struct {
        param: Param,
        /// Momentum buffer; empty (f32-tagged, so the frame stays v3) when
        /// momentum == 0. PyTorch initializes it to a CLONE OF THE FIRST
        /// (decayed) GRADIENT, not zeros — `step` tracks whether that has
        /// happened.
        buf: StateBuf,
        step: u64 = 0,
    };

    pub fn init(allocator: Allocator, config: SgdConfig) SGD {
        // PyTorch constructor rule, enforced in every build mode (a debug
        // assert would vanish exactly where training runs: ReleaseFast).
        if (config.nesterov and (config.momentum == 0 or config.dampening != 0)) {
            @panic("SGD: nesterov requires momentum > 0 and dampening == 0");
        }
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *SGD) void {
        for (self.slots.items) |*slot| {
            slot.buf.deinit(self.allocator);
            slot.param.deinit(self.allocator);
        }
        self.slots.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn collectGradStates(self: *const SGD, set: *GradStateSet, allocator: Allocator) !void {
        if (gradStatesCollide(set, self.slots.items)) return OptimError.DuplicateParam;
        try insertGradStates(set, allocator, self.slots.items);
    }

    fn containsGradState(self: *const SGD, state: *const GradState) bool {
        for (self.slots.items) |*slot| {
            if (slot.param.grad_state == state) return true;
        }
        return false;
    }

    pub fn addParam(self: *SGD, t: anytype) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        try self.addOwnedParam(param);
    }

    /// `addParam` plus a checkpoint name (borrowed; see `Param.name`).
    pub fn addParamNamed(self: *SGD, t: anytype, name: []const u8) !void {
        var param = try Param.of(t);
        errdefer param.deinit(self.allocator);
        param.name = name;
        try self.addOwnedParam(param);
    }

    fn addOwnedParam(self: *SGD, param: Param) !void {
        if (self.containsGradState(param.grad_state)) return OptimError.DuplicateParam;
        var owned = param;
        try owned.ensureMaster(self.allocator);
        errdefer if (owned.master.len != 0) self.allocator.free(owned.master);
        const buf: StateBuf = if (self.config.momentum != 0)
            try StateBuf.alloc(self.allocator, self.config.state_dtype, owned.len())
        else
            .{ .f32 = &.{} };
        errdefer buf.deinit(self.allocator);
        try self.slots.append(self.allocator, .{ .param = owned, .buf = buf });
    }

    pub fn step(self: *SGD, ctx: *ExecContext) !void {
        for (self.slots.items) |*slot| {
            var grad = (try takeGrad(ctx, &slot.param)) orelse continue;
            defer grad.deinit();
            slot.step += 1;
            sgdUpdate(ctx, self.config, slot.param.data(), grad.dataConst(), slot.buf, slot.step == 1);
            slot.param.publish();
        }
    }

    pub fn zeroGrad(self: *SGD) void {
        for (self.slots.items) |*slot| slot.param.grad_state.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *SGD, ctx: *ExecContext) !f64 {
        var total: f64 = 0;
        for (self.slots.items) |*slot| total += try paramGradSqNorm(ctx, &slot.param);
        return total;
    }

    pub fn scaleGradients(self: *SGD, ctx: *ExecContext, factor: f32) !void {
        for (self.slots.items) |*slot| try scaleParamGrad(ctx, &slot.param, factor);
    }

    pub fn clipGradNorm(self: *SGD, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    pub fn saveState(self: *const SGD, writer: *std.Io.Writer) !void {
        try validateSlotNames(self.slots.items);
        var version: FrameVersion = .v3;
        for (self.slots.items) |*slot| {
            if (slot.buf != .f32) version = .v4;
        }
        if (slotsCarryMasters(self.slots.items)) version = .v5;
        try writer.writeAll(switch (version) {
            .v3 => "FZS3",
            .v4 => "FZS4",
            .v5 => "FZS5",
        });
        try writer.writeInt(u32, @bitCast(self.config.momentum), .little);
        try writer.writeInt(u32, @bitCast(self.config.dampening), .little);
        try writer.writeInt(u8, @intFromBool(self.config.nesterov), .little);
        try writer.writeInt(u32, @intCast(self.slots.items.len), .little);
        for (self.slots.items, 0..) |*slot, i| {
            try writeSlotName(writer, &slot.param, i);
            try writeSlotDims(writer, &slot.param);
            try writer.writeInt(u64, slot.step, .little);
            try writeStateSlice(writer, version, slot.buf);
            try writeSlotMaster(writer, version, &slot.param);
        }
    }

    pub fn loadState(self: *SGD, reader: *std.Io.Reader) !void {
        const version = try expectMagicVersion(reader, "FZS3", "FZS4", "FZS5");
        if (try reader.takeInt(u32, .little) != @as(u32, @bitCast(self.config.momentum))) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u32, .little) != @as(u32, @bitCast(self.config.dampening))) return OptimError.CheckpointConfigMismatch;
        if (try reader.takeInt(u8, .little) != @intFromBool(self.config.nesterov)) return OptimError.CheckpointConfigMismatch;
        const count = try reader.takeInt(u32, .little);
        var matcher = try SlotMatcher.init(self.allocator, self.slots.items.len);
        defer matcher.deinit(self.allocator);
        var staged = try std.ArrayList(StagedSlot).initCapacity(self.allocator, count);
        defer freeStaged(self.allocator, &staged);
        for (0..count) |_| {
            const idx = try matcher.match(reader, self.slots.items);
            const slot = &self.slots.items[idx];
            try expectSlotDims(reader, &slot.param);
            const step_val = try reader.takeInt(u64, .little);
            const data = try self.allocator.alloc(u8, slot.buf.byteLen());
            errdefer self.allocator.free(data);
            try readStateSlice(reader, version, slot.buf, data);
            const master = try readSlotMaster(self.allocator, reader, version, &slot.param);
            errdefer if (master.len != 0) self.allocator.free(master);
            try staged.append(self.allocator, .{ .idx = idx, .step = step_val, .data = data, .master = master });
        }
        try matcher.requireAllFilled();
        for (staged.items) |s| {
            const slot = &self.slots.items[s.idx];
            slot.step = s.step;
            @memcpy(slot.buf.bytes(), s.data);
            commitSlotMaster(&slot.param, s.master);
        }
    }
};

/// The exact PyTorch SGD step. With weight decay the L2 term joins the
/// gradient BEFORE the momentum buffer sees it; on the very first step the
/// buffer is initialized to that (decayed) gradient itself — not zeros —
/// matching `buf = d_p.clone()`. One fused element-independent pass, chunked
/// across the worker pool. bf16 momentum widens on read and narrows on write;
/// the nesterov blend and the parameter update use the just-computed
/// (pre-narrow) f32 buffer value.
fn SgdMap(comptime sd: StateDType) type {
    return struct {
        p: []f32,
        g: []const f32,
        buf: StateSlice(sd),
        config: SgdConfig,
        one_minus_damp: f32,
        first_step: bool,

        fn run(c: @This(), start: usize, end: usize) void {
            if (comptime sd == .f32) {
                // The golden-pinned baseline stays on the original scalar loop.
                runScalar(c, start, end);
                return;
            }
            // Hand-vectorized bf16 arm (see `state_vec_len`); bit-identical
            // to runScalar per element. A bf16 buffer only exists with
            // momentum != 0; the decay/first-step/nesterov branches are
            // loop-invariant.
            const config = c.config;
            const wdv: StateVec = @splat(config.weight_decay);
            const momentumv: StateVec = @splat(config.momentum);
            const one_minus_dampv: StateVec = @splat(c.one_minus_damp);
            const lrv: StateVec = @splat(config.lr);
            var i = start;
            while (i + state_vec_len <= end) : (i += state_vec_len) {
                const pv: StateVec = c.p[i..][0..state_vec_len].*;
                var gv: StateVec = c.g[i..][0..state_vec_len].*;
                if (config.weight_decay != 0) gv += wdv * pv;
                if (config.momentum != 0) {
                    const b1 = if (c.first_step)
                        gv
                    else
                        momentumv * stateVecLoad(sd, c.buf[i..][0..state_vec_len]) + one_minus_dampv * gv;
                    stateVecStore(sd, c.buf[i..][0..state_vec_len], b1);
                    gv = if (config.nesterov) gv + momentumv * b1 else b1;
                }
                c.p[i..][0..state_vec_len].* = pv - lrv * gv;
            }
            runScalar(c, i, end);
        }

        fn runScalar(c: @This(), start: usize, end: usize) void {
            const config = c.config;
            for (c.p[start..end], c.g[start..end], start..) |*pi, gi_raw, i| {
                var gi = gi_raw;
                if (config.weight_decay != 0) gi += config.weight_decay * pi.*;
                if (config.momentum != 0) {
                    const b1 = if (c.first_step)
                        gi
                    else
                        config.momentum * stateLoad(sd, &c.buf[i]) + c.one_minus_damp * gi;
                    stateStore(sd, &c.buf[i], b1);
                    gi = if (config.nesterov) gi + config.momentum * b1 else b1;
                }
                pi.* -= config.lr * gi;
            }
        }
    };
}

fn sgdRun(comptime sd: StateDType, ctx: *ExecContext, config: SgdConfig, p: []f32, g: []const f32, buf: StateSlice(sd), first_step: bool) void {
    const Map = SgdMap(sd);
    parallelMap(ctx, p.len, Map{
        .p = p,
        .g = g,
        .buf = buf,
        .config = config,
        .one_minus_damp = @floatCast(1.0 - @as(f64, config.dampening)),
        .first_step = first_step,
    }, Map.run);
}

fn sgdUpdate(ctx: *ExecContext, config: SgdConfig, p: []f32, g: []const f32, buf: StateBuf, first_step: bool) void {
    switch (buf) {
        .f32 => |bs| sgdRun(.f32, ctx, config, p, g, bs, first_step),
        .bf16 => |bs| sgdRun(.bf16, ctx, config, p, g, bs, first_step),
    }
}

// ---------------------------------------------------------------------------
// Gradient clipping — torch.nn.utils.clip_grad_norm_ semantics (L2).
// ---------------------------------------------------------------------------

fn paramGradSqNorm(ctx: *ExecContext, param: *const Param) !f64 {
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

fn scaleParamGrad(ctx: *ExecContext, param: *const Param, factor: f32) !void {
    var view = (try param.grad_state.gradView()) orelse return;
    if (view.isContiguous()) {
        defer view.deinit();
        const values = view.data();
        parallelMap(ctx, values.len, ScaleMapContext{ .values = values, .factor = factor }, scaleMapRange);
        return;
    }
    // Rare: a non-contiguous accumulated gradient. Materialize, scale, and
    // swap it into the GradState (which owns and frees the old one).
    var owned = ctx.materialize(&view) catch |err| {
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
fn clipByGlobalNorm(ctx: *ExecContext, opt: anytype, max_norm: f32) !f32 {
    const total: f32 = @floatCast(@sqrt(try opt.gradSquaredNorm(ctx)));
    const clip_coef = max_norm / (total + 1e-6);
    if (clip_coef < 1) try opt.scaleGradients(ctx, clip_coef);
    return total;
}

// ---------------------------------------------------------------------------
// LR schedule hook.
// ---------------------------------------------------------------------------

/// Attaches to `config.lr` fields (they are plain public f32s) and rescales
/// them all from their captured base values: `lr = base * factor`. Works
/// across optimizers and their fallbacks, e.g.:
///
///     var sched = optim.LrSchedule.init(allocator);
///     defer sched.deinit();
///     try sched.attach(&muon.config.lr);
///     try sched.attach(&muon.fallback.config.lr);
///     ...
///     sched.apply(optim.warmupCosineFactor(step, total_steps, warmup, 0.1));
///
/// Because the factor is a pure function of the step counter, re-applying it
/// while resuming from a checkpoint reproduces the schedule exactly.
pub const LrSchedule = struct {
    allocator: Allocator,
    entries: std.ArrayList(Entry) = .empty,

    const Entry = struct {
        lr: *f32,
        base: f64,
    };

    pub fn init(allocator: Allocator) LrSchedule {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LrSchedule) void {
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Captures the current value as the base lr — attach before the first
    /// `apply()`, or the factor in effect gets baked into the base. The
    /// pointee must outlive the schedule. Re-attaching a pointer refreshes
    /// its base instead of duplicating the entry.
    pub fn attach(self: *LrSchedule, lr: *f32) !void {
        for (self.entries.items) |*entry| {
            if (entry.lr == lr) {
                entry.base = lr.*;
                return;
            }
        }
        try self.entries.append(self.allocator, .{ .lr = lr, .base = lr.* });
    }

    pub fn apply(self: *const LrSchedule, factor: f64) void {
        for (self.entries.items) |entry| {
            entry.lr.* = @floatCast(entry.base * factor);
        }
    }
};

/// Linear warmup from 1/warmup_steps to 1, then cosine decay to `min_factor`
/// over the remaining steps. `step` is 0-based.
pub fn warmupCosineFactor(step: u64, total_steps: u64, warmup_steps: u64, min_factor: f64) f64 {
    if (warmup_steps > 0 and step < warmup_steps) {
        return @as(f64, @floatFromInt(step + 1)) / @as(f64, @floatFromInt(warmup_steps));
    }
    if (total_steps <= warmup_steps) return min_factor;
    const num = @as(f64, @floatFromInt(step - warmup_steps));
    const den = @as(f64, @floatFromInt(total_steps - warmup_steps));
    const progress = @min(num / den, 1.0);
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + @cos(std.math.pi * progress));
}

// ---------------------------------------------------------------------------
// Param groups: a type-erased set of optimizer instances.
//
// A "param group" is exactly {hyperparams, params, state} — which is what one
// optimizer instance already is. OptimizerSet aggregates instances behind one
// step / zeroGrad / clipGradNorm (GLOBAL norm, like PyTorch's
// clip_grad_norm_(model.parameters())) / saveState / loadState surface.
// ---------------------------------------------------------------------------

pub const AnyOptimizer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        step: *const fn (*anyopaque, *ExecContext) anyerror!void,
        zeroGrad: *const fn (*anyopaque) void,
        gradSquaredNorm: *const fn (*anyopaque, *ExecContext) anyerror!f64,
        scaleGradients: *const fn (*anyopaque, *ExecContext, f32) anyerror!void,
        saveState: *const fn (*anyopaque, *std.Io.Writer) anyerror!void,
        loadState: *const fn (*anyopaque, *std.Io.Reader) anyerror!void,
    };

    pub fn step(self: AnyOptimizer, ctx: *ExecContext) !void {
        return self.vtable.step(self.ptr, ctx);
    }

    pub fn zeroGrad(self: AnyOptimizer) void {
        self.vtable.zeroGrad(self.ptr);
    }

    pub fn gradSquaredNorm(self: AnyOptimizer, ctx: *ExecContext) !f64 {
        return self.vtable.gradSquaredNorm(self.ptr, ctx);
    }

    pub fn scaleGradients(self: AnyOptimizer, ctx: *ExecContext, factor: f32) !void {
        return self.vtable.scaleGradients(self.ptr, ctx, factor);
    }

    pub fn saveState(self: AnyOptimizer, writer: *std.Io.Writer) !void {
        return self.vtable.saveState(self.ptr, writer);
    }

    pub fn loadState(self: AnyOptimizer, reader: *std.Io.Reader) !void {
        return self.vtable.loadState(self.ptr, reader);
    }
};

/// Wrap a concrete optimizer pointer (Adam/AdamW/SGD/Muon/Apollo) as AnyOptimizer.
/// The wrapped optimizer is borrowed: the caller still owns and deinits it,
/// and it must not move (or be freed) while the AnyOptimizer/OptimizerSet is
/// in use — the raw pointer is captured.
pub fn anyOptimizer(opt: anytype) AnyOptimizer {
    const T = @typeInfo(@TypeOf(opt)).pointer.child;
    const Impl = struct {
        fn step(ptr: *anyopaque, ctx: *ExecContext) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.step(ctx);
        }
        fn zeroGrad(ptr: *anyopaque) void {
            const self: *T = @ptrCast(@alignCast(ptr));
            self.zeroGrad();
        }
        fn gradSquaredNorm(ptr: *anyopaque, ctx: *ExecContext) anyerror!f64 {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.gradSquaredNorm(ctx);
        }
        fn scaleGradients(ptr: *anyopaque, ctx: *ExecContext, factor: f32) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.scaleGradients(ctx, factor);
        }
        fn saveState(ptr: *anyopaque, writer: *std.Io.Writer) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.saveState(writer);
        }
        fn loadState(ptr: *anyopaque, reader: *std.Io.Reader) anyerror!void {
            const self: *T = @ptrCast(@alignCast(ptr));
            return self.loadState(reader);
        }
        const vtable = AnyOptimizer.VTable{
            .step = step,
            .zeroGrad = zeroGrad,
            .gradSquaredNorm = gradSquaredNorm,
            .scaleGradients = scaleGradients,
            .saveState = saveState,
            .loadState = loadState,
        };
    };
    return .{ .ptr = opt, .vtable = &Impl.vtable };
}

/// Set of registered `GradState` pointers (by pointer identity), used by
/// `OptimizerSet` to reject the same `Variable` being registered into two member
/// optimizers — which the per-instance `containsGradState` guard cannot see and
/// which would otherwise silently double-step that parameter.
const GradStateSet = std.AutoHashMapUnmanaged(*const GradState, void);

/// True if any slot's grad-state is already in `set` (a cross-member duplicate).
fn gradStatesCollide(set: *const GradStateSet, slots: anytype) bool {
    for (slots) |*slot| {
        if (set.contains(slot.param.grad_state)) return true;
    }
    return false;
}

fn insertGradStates(set: *GradStateSet, allocator: Allocator, slots: anytype) !void {
    for (slots) |*slot| try set.put(allocator, slot.param.grad_state, {});
}

pub const OptimizerSet = struct {
    allocator: Allocator,
    items: std.ArrayList(AnyOptimizer) = .empty,
    /// Every grad-state registered across ALL member optimizers; guards against
    /// cross-instance duplicate registration (silent double-step).
    grad_states: GradStateSet = .empty,

    pub fn init(allocator: Allocator) OptimizerSet {
        return .{ .allocator = allocator };
    }

    /// Frees only the set; the member optimizers stay owned by the caller.
    pub fn deinit(self: *OptimizerSet) void {
        self.grad_states.deinit(self.allocator);
        self.items.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a member optimizer. Returns `OptimError.DuplicateParam` if any of
    /// its parameters' grad-states is ALREADY registered with a previously-added
    /// member (the same `Variable` in two groups) — closing the cross-instance
    /// gap the per-optimizer guard leaves open. On a collision the set is
    /// unchanged and the member is NOT added.
    pub fn add(self: *OptimizerSet, opt: anytype) !void {
        try opt.collectGradStates(&self.grad_states, self.allocator);
        try self.items.append(self.allocator, anyOptimizer(opt));
    }

    pub fn step(self: *OptimizerSet, ctx: *ExecContext) !void {
        for (self.items.items) |opt| try opt.step(ctx);
    }

    pub fn zeroGrad(self: *OptimizerSet) void {
        for (self.items.items) |opt| opt.zeroGrad();
    }

    pub fn gradSquaredNorm(self: *OptimizerSet, ctx: *ExecContext) !f64 {
        var total: f64 = 0;
        for (self.items.items) |opt| total += try opt.gradSquaredNorm(ctx);
        return total;
    }

    pub fn scaleGradients(self: *OptimizerSet, ctx: *ExecContext, factor: f32) !void {
        for (self.items.items) |opt| try opt.scaleGradients(ctx, factor);
    }

    /// GLOBAL norm across every group — the PyTorch clip_grad_norm_ contract.
    pub fn clipGradNorm(self: *OptimizerSet, ctx: *ExecContext, max_norm: f32) !f32 {
        return clipByGlobalNorm(ctx, self, max_norm);
    }

    pub fn saveState(self: *const OptimizerSet, writer: *std.Io.Writer) !void {
        try writer.writeAll("FZO3");
        try writer.writeInt(u32, @intCast(self.items.items.len), .little);
        for (self.items.items) |opt| try opt.saveState(writer);
    }

    pub fn loadState(self: *OptimizerSet, reader: *std.Io.Reader) !void {
        try expectMagic(reader, "FZO3");
        const count = try reader.takeInt(u32, .little);
        if (count != self.items.items.len) return OptimError.CheckpointShapeMismatch;
        for (self.items.items) |opt| try opt.loadState(reader);
    }
};

// ---------------------------------------------------------------------------
// Checkpoint helpers.
// ---------------------------------------------------------------------------
//
// Transactional load contract: every `loadState` / `loadTensors` here is
// all-or-nothing. Each record is decoded into freshly-allocated scratch and the
// WHOLE stream is validated (magic, config, names, dims, lengths, slot match)
// BEFORE any live parameter/optimizer buffer is written. A truncated, short, or
// otherwise-invalid stream therefore leaves every destination byte-unchanged —
// a half-applied checkpoint can silently corrupt training, so we never produce
// one. (`OptimizerSet.loadState` is transactional per member optimizer.)

/// One decoded-but-not-yet-committed slot record for a transactional `loadState`:
/// the destination slot index, the scalar fields, and a freshly-allocated RAW-BYTE
/// scratch holding the slot's contiguous state buffers (e.g. m then v, or a lone
/// momentum/buf) in their storage dtype — dtype validation already happened at
/// read time, so the commit is a plain byte copy for every `StateDType`.
/// `seed`/`prev_norm` are only meaningful for APOLLO main slots.
const StagedSlot = struct {
    idx: usize,
    data: []u8,
    step: u64 = 0,
    seed: u64 = 0,
    prev_norm: f32 = 0,
    /// Staged v5 f32 master weights (empty when the frame carried none).
    master: []f32 = &.{},
};

fn freeStaged(allocator: Allocator, staged: *std.ArrayList(StagedSlot)) void {
    for (staged.items) |s| {
        allocator.free(s.data);
        if (s.master.len != 0) allocator.free(s.master);
    }
    staged.deinit(allocator);
}

/// Optimizer state frame revision. v3 is the pre-`StateDType` format: state
/// buffers are raw f32 bytes with no tag. v4 prefixes every state buffer with
/// one u8 `StateDType` tag. Writers emit v3 whenever every buffer is f32 (so
/// the bytes stay identical to pre-bf16 builds) and v4 otherwise; readers
/// accept both and require the stored dtype to match the live buffer's.
const FrameVersion = enum { v3, v4, v5 };

/// The frame version for the common `{ m, v }` moment-pair slot layout
/// (Adam/AdamW): v3 iff every buffer of every slot is f32.
fn momentSlotsFrameVersion(slots: anytype) FrameVersion {
    for (slots) |*slot| {
        if (slot.m != .f32 or slot.v != .f32) return .v4;
    }
    return .v3;
}

/// Read a 4-byte magic and map it to the frame version it names.
fn expectMagicVersion(reader: *std.Io.Reader, comptime v3_magic: *const [4]u8, comptime v4_magic: *const [4]u8, comptime v5_magic: *const [4]u8) !FrameVersion {
    var buf: [4]u8 = undefined;
    try reader.readSliceAll(&buf);
    if (std.mem.eql(u8, &buf, v3_magic)) return .v3;
    if (std.mem.eql(u8, &buf, v4_magic)) return .v4;
    if (std.mem.eql(u8, &buf, v5_magic)) return .v5;
    return OptimError.CheckpointMagicMismatch;
}

/// v5 frames exist to persist f32 MASTER weights for 16-bit params: resuming
/// from the narrowed values instead would re-round the master and lose the
/// sub-bf16 update accumulation. Frames without 16-bit slots keep emitting
/// v3/v4, byte-identical to before.
fn slotsCarryMasters(slots: anytype) bool {
    for (slots) |*slot| {
        if (slot.param.master.len != 0) return true;
    }
    return false;
}

/// Per-slot master record (v5 frames only): u8 presence flag + raw f32 bytes.
fn writeSlotMaster(writer: *std.Io.Writer, version: FrameVersion, param: *const Param) !void {
    if (version != .v5) return;
    const has: u8 = @intFromBool(param.master.len != 0);
    try writer.writeInt(u8, has, .little);
    if (has == 1) try writeF32Slice(writer, param.master);
}

fn readSlotMaster(allocator: Allocator, reader: *std.Io.Reader, version: FrameVersion, param: *const Param) ![]f32 {
    if (version != .v5) return &.{};
    if (try reader.takeInt(u8, .little) == 0) return &.{};
    if (param.master.len == 0) return OptimError.CheckpointDtypeMismatch;
    const buf = try allocator.alloc(f32, param.master.len);
    errdefer allocator.free(buf);
    try readF32Slice(reader, buf);
    return buf;
}

/// Commit arm for a 16-bit slot's master: install the checkpoint master and
/// narrow it into the param storage, or — when the frame carried none — re-
/// widen from the (possibly just-loaded) param values so the master never
/// goes stale. No-op for f32 params.
fn commitSlotMaster(param: *Param, staged_master: []const f32) void {
    if (param.master.len == 0) return;
    if (staged_master.len != 0) {
        @memcpy(param.master, staged_master);
        param.publish();
    } else {
        param.refreshMasterFromValue();
    }
}

/// Write one state-buffer record: v3 = raw f32 bytes (the caller guarantees
/// every buffer is f32 before choosing v3), v4 = one u8 dtype tag + the raw
/// storage bytes. The f32 v3 arm is byte-identical to the old `writeF32Slice`.
fn writeStateSlice(writer: *std.Io.Writer, version: FrameVersion, buf: StateBuf) !void {
    switch (version) {
        .v3 => std.debug.assert(buf == .f32),
        .v4, .v5 => try writer.writeInt(u8, @intFromEnum(@as(StateDType, buf)), .little),
    }
    try writer.writeAll(buf.bytesConst());
}

/// Read one state-buffer record into `dest` (raw bytes, `buf.byteLen()` long).
/// The stored dtype — implicitly f32 for v3, the u8 tag for v4 — must equal
/// the live buffer's exactly: NO implicit conversion, or a resumed run would
/// silently break the bit-exact-resume contract.
fn readStateSlice(reader: *std.Io.Reader, version: FrameVersion, buf: StateBuf, dest: []u8) !void {
    const stored: StateDType = switch (version) {
        .v3 => .f32,
        .v4, .v5 => std.enums.fromInt(StateDType, try reader.takeInt(u8, .little)) orelse
            return OptimError.CheckpointUnsupportedDtype,
    };
    if (stored != @as(StateDType, buf)) return OptimError.CheckpointDtypeMismatch;
    try reader.readSliceAll(dest);
}

/// Serialize parameter values (shapes + f32 data, little-endian). `tensors` is
/// a tuple of pointers to contiguous f32 facade tensors (variables or
/// constants). This is all an inference-time consumer needs.
pub fn saveTensors(writer: *std.Io.Writer, tensors: anytype) !void {
    try writer.writeAll("FZT1");
    try writer.writeInt(u32, @intCast(tensors.len), .little);
    inline for (tensors) |t| {
        if (!t.value.isContiguous()) return OptimError.NonContiguousParam;
        const shape = t.value.shape.slice();
        try writer.writeInt(u32, @intCast(shape.len), .little);
        for (shape) |dim| try writer.writeInt(u64, dim, .little);
        try writeF32Slice(writer, t.value.dataConst());
    }
}

/// Load parameter values saved by `saveTensors` into existing tensors of the
/// same shapes (order-based, shape-validated). Transactional (see the contract
/// above): the whole stream is staged + validated before any tensor is written,
/// so a truncated/mismatched stream leaves every tensor byte-unchanged. Needs an
/// allocator for the per-tensor scratch.
pub fn loadTensors(allocator: Allocator, reader: *std.Io.Reader, tensors: anytype) !void {
    try expectMagic(reader, "FZT1");
    const count = try reader.takeInt(u32, .little);
    if (count != tensors.len) return OptimError.CheckpointShapeMismatch;
    var staged: [tensors.len][]f32 = undefined;
    var staged_n: usize = 0;
    defer for (staged[0..staged_n]) |buf| allocator.free(buf);
    // Pass 1 — validate shapes + read every tensor into scratch.
    inline for (tensors, 0..) |t, ti| {
        if (!t.value.isContiguous()) return OptimError.NonContiguousParam;
        const shape = t.value.shape.slice();
        const rank = try reader.takeInt(u32, .little);
        if (rank != shape.len) return OptimError.CheckpointShapeMismatch;
        for (shape) |dim| {
            const stored = try reader.takeInt(u64, .little);
            if (stored != dim) return OptimError.CheckpointShapeMismatch;
        }
        const buf = try allocator.alloc(f32, t.value.data().len);
        staged[ti] = buf;
        staged_n = ti + 1;
        try readF32Slice(reader, buf);
    }
    // Pass 2 — commit (all reads succeeded).
    inline for (tensors, 0..) |t, ti| {
        @memcpy(t.value.data(), staged[ti]);
    }
}

fn writeF32Slice(writer: *std.Io.Writer, values: []const f32) !void {
    try writer.writeAll(std.mem.sliceAsBytes(values));
}

fn readF32Slice(reader: *std.Io.Reader, values: []f32) !void {
    try reader.readSliceAll(std.mem.sliceAsBytes(values));
}

/// Longest serializable name; the wire length prefix is u16.
const max_name_len: usize = std.math.maxInt(u16);
/// "param" plus at most 20 digits of a usize slot index.
const auto_name_buf_len = "param".len + 20;

/// A slot's checkpoint identity: the explicit registration name, or the
/// auto-name "param<i>" from its index within its slot list.
fn slotName(param: *const Param, index: usize, buf: *[auto_name_buf_len]u8) []const u8 {
    return param.name orelse (std.fmt.bufPrint(buf, "param{d}", .{index}) catch unreachable);
}

fn validateName(name: []const u8) !void {
    if (name.len == 0 or name.len > max_name_len) return OptimError.CheckpointInvalidName;
    if (std.mem.indexOfScalar(u8, name, 0) != null) return OptimError.CheckpointInvalidName;
    if (!std.unicode.utf8ValidateSlice(name)) return OptimError.CheckpointInvalidName;
}

/// Validate the effective names of one name-matched slot list before saving:
/// well-formed and collision-free (an explicit name can also collide with an
/// auto-name). O(n^2) compares — checkpoint-time only, allocation-free.
fn validateSlotNames(slots: anytype) !void {
    for (slots, 0..) |*a, i| {
        var buf_a: [auto_name_buf_len]u8 = undefined;
        const name_a = slotName(&a.param, i, &buf_a);
        try validateName(name_a);
        for (slots[i + 1 ..], i + 1..) |*b, j| {
            var buf_b: [auto_name_buf_len]u8 = undefined;
            if (std.mem.eql(u8, name_a, slotName(&b.param, j, &buf_b))) {
                return OptimError.CheckpointDuplicateName;
            }
        }
    }
}

fn writeSlotName(writer: *std.Io.Writer, param: *const Param, index: usize) !void {
    var buf: [auto_name_buf_len]u8 = undefined;
    const name = slotName(param, index, &buf);
    try writer.writeInt(u16, @intCast(name.len), .little);
    try writer.writeAll(name);
}

/// Name-matches v3/v4 optimizer slot records to registered slots, enforcing the
/// fill-exactly-once contract: an unknown record name, a record matching an
/// already-filled slot, and a slot left unfilled at the end all error.
const SlotMatcher = struct {
    filled: []bool,
    name_buf: []u8,

    fn init(allocator: Allocator, slot_count: usize) !SlotMatcher {
        const filled = try allocator.alloc(bool, slot_count);
        errdefer allocator.free(filled);
        @memset(filled, false);
        const name_buf = try allocator.alloc(u8, max_name_len);
        return .{ .filled = filled, .name_buf = name_buf };
    }

    fn deinit(self: *SlotMatcher, allocator: Allocator) void {
        allocator.free(self.filled);
        allocator.free(self.name_buf);
        self.* = undefined;
    }

    /// Read one record's name and resolve it to a not-yet-filled slot index.
    fn match(self: *SlotMatcher, reader: *std.Io.Reader, slots: anytype) !usize {
        const name_len = try reader.takeInt(u16, .little);
        if (name_len == 0) return OptimError.CheckpointInvalidName;
        const name = self.name_buf[0..name_len];
        try reader.readSliceAll(name);
        for (slots, 0..) |*slot, i| {
            var buf: [auto_name_buf_len]u8 = undefined;
            if (!std.mem.eql(u8, name, slotName(&slot.param, i, &buf))) continue;
            if (self.filled[i]) return OptimError.CheckpointDuplicateName;
            self.filled[i] = true;
            return i;
        }
        return OptimError.CheckpointUnknownName;
    }

    fn requireAllFilled(self: *const SlotMatcher) !void {
        for (self.filled) |was_filled| if (!was_filled) return OptimError.CheckpointMissingEntry;
    }
};

fn writeSlotDims(writer: *std.Io.Writer, param: *const Param) !void {
    try writer.writeInt(u64, param.rows, .little);
    try writer.writeInt(u64, param.cols, .little);
}

fn expectSlotDims(reader: *std.Io.Reader, param: *const Param) !void {
    if (try reader.takeInt(u64, .little) != param.rows) return OptimError.CheckpointShapeMismatch;
    if (try reader.takeInt(u64, .little) != param.cols) return OptimError.CheckpointShapeMismatch;
}

fn expectMagic(reader: *std.Io.Reader, comptime magic: *const [4]u8) !void {
    var buf: [4]u8 = undefined;
    try reader.readSliceAll(&buf);
    if (!std.mem.eql(u8, &buf, magic)) return OptimError.CheckpointMagicMismatch;
}

test {
    _ = @import("optim_tests.zig");
}
