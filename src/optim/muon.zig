//! Muon: Keller Jordan's reference (lerp-form momentum, Newton-Schulz-5
//! orthogonalization with the transpose trick) with the Moonlight
//! RMS-matching scale variant, over an embedded AdamW fallback for
//! non-matrix params. Frame magics FZM3/FZM4/FZM5.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");
const moment_pair = @import("moment_pair.zig");

const Allocator = std.mem.Allocator;
const RawTensor = common.RawTensor;
const ExecContext = common.ExecContext;
const GradState = common.GradState;
const OptimError = common.OptimError;
const StateDType = common.StateDType;
const StateSlice = common.StateSlice;
const StateBuf = common.StateBuf;
const StateVec = common.StateVec;
const state_vec_len = common.state_vec_len;
const stateLoad = common.stateLoad;
const stateStore = common.stateStore;
const stateVecLoad = common.stateVecLoad;
const stateVecStore = common.stateVecStore;
const Param = common.Param;
const parallelMap = common.parallelMap;
const sumSquares = common.sumSquares;
const takeGrad = common.takeGrad;
const paramGradSqNorm = common.paramGradSqNorm;
const scaleParamGrad = common.scaleParamGrad;
const clipByGlobalNorm = common.clipByGlobalNorm;
const GradStateSet = common.GradStateSet;
const gradStatesCollide = common.gradStatesCollide;
const insertGradStates = common.insertGradStates;
const AdamWConfig = moment_pair.AdamWConfig;
const AdamW = moment_pair.AdamW;
const StagedSlot = frame.StagedSlot;
const freeStaged = frame.freeStaged;
const SlotMatcher = frame.SlotMatcher;
const FrameVersion = frame.FrameVersion;
const slotsCarryMasters = frame.slotsCarryMasters;
const expectMagicVersion = frame.expectMagicVersion;
const validateSlotNames = frame.validateSlotNames;
const writeSlotName = frame.writeSlotName;
const writeSlotDims = frame.writeSlotDims;
const expectSlotDims = frame.expectSlotDims;
const writeStateSlice = frame.writeStateSlice;
const readStateSlice = frame.readStateSlice;
const writeSlotMaster = frame.writeSlotMaster;
const readSlotMaster = frame.readSlotMaster;
const commitSlotMaster = frame.commitSlotMaster;

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
