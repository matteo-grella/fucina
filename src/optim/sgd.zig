//! SGD: PyTorch torch.optim.SGD single-tensor semantics (coupled L2,
//! first-step buffer = decayed gradient, optional nesterov). Frame magics
//! FZS3/FZS4/FZS5.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");

const Allocator = std.mem.Allocator;
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
const takeGrad = common.takeGrad;
const paramGradSqNorm = common.paramGradSqNorm;
const scaleParamGrad = common.scaleParamGrad;
const clipByGlobalNorm = common.clipByGlobalNorm;
const GradStateSet = common.GradStateSet;
const gradStatesCollide = common.gradStatesCollide;
const insertGradStates = common.insertGradStates;
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
