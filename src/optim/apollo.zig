//! APOLLO: the official apollo_torch optimizer (arXiv 2412.05270) —
//! random low-rank gradient projection, AdamW moments in the compressed
//! space, channel/tensor scaling, the Fira norm-growth limiter, a scaled
//! SGD update, and the reference's legacy-HF AdamW fallback path (eps
//! outside the bias correction, decay after the step). Frame magics
//! FZP3/FZP5; projections are regenerated from (seed, step), never stored.

const std = @import("std");
const common = @import("common.zig");
const frame = @import("frame.zig");
// The (seed -> values) mapping of rng.gaussianFill is part of the APOLLO
// checkpoint contract (projections are regenerated from seed, not stored).
const rng = @import("../rng.zig");
const gaussianFill = rng.gaussianFill;

const Allocator = std.mem.Allocator;
const RawTensor = common.RawTensor;
const ExecContext = common.ExecContext;
const GradState = common.GradState;
const OptimError = common.OptimError;
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
const StagedSlot = frame.StagedSlot;
const freeStaged = frame.freeStaged;
const SlotMatcher = frame.SlotMatcher;
const FrameVersion = frame.FrameVersion;
const slotsCarryMasters = frame.slotsCarryMasters;
const validateSlotNames = frame.validateSlotNames;
const writeSlotName = frame.writeSlotName;
const writeSlotDims = frame.writeSlotDims;
const expectSlotDims = frame.expectSlotDims;
const writeF32Slice = frame.writeF32Slice;
const writeSlotMaster = frame.writeSlotMaster;
const readSlotMaster = frame.readSlotMaster;
const commitSlotMaster = frame.commitSlotMaster;

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
        var update = try ctx.empty(.f32, .{ rows, cols });
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
            slot.proj = try ctx.empty(.f32, shape);
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
