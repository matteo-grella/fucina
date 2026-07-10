//! nanochat MuonAdamW optimizer (karpathy/nanochat → Fucina), CPU fp32 parity
//! port. Reproduces `refs/nanochat/nanochat/optim.py`
//! `MuonAdamW.step` on the single-rank path: AdamW for embeddings/head/scalars
//! (reusing `fucina.optim.AdamW`, whose step math is bitwise nanochat's
//! `adamw_step_fused`), and a CUSTOM Muon for the transformer matrix params
//! (`muon_step_fused`: Nesterov lerp momentum, MuonEq row equilibration,
//! Polar-Express orthogonalization, Muon+ renorm, NorMuon variance reduction,
//! cautious weight-decay update). All math is f32 — the reference's bf16 cast
//! arm is skipped because COMPUTE_DTYPE=float32 on CPU.
//!
//! Muon params are grouped BY SHAPE and stacked into (K, m, n); the batched
//! Newton-Schulz GEMMs (Xᵀ·X / X·Xᵀ, A·A, X·B / B·X per-K) run through the raw
//! batched-matmul kernels (`ExecContext.bmm{,TransA,TransB}`) — the reference's
//! stacked-matmul path. Everything else (momentum lerps, norms, the row/column
//! reductions, the cautious update) is a direct f32 pass, with reductions
//! accumulated in f64 for a stable match to torch's f32 norms.
//!
//! Schedules (lr multiplier, Muon momentum, Muon weight decay) reproduce
//! `scripts/base_train.py` get_lr_multiplier / get_muon_momentum /
//! get_weight_decay with num_iterations=5000, warmup 40, warmdown 0.65,
//! final_lr_frac 0.05.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Model = model_mod.Model;
const AdamW = fucina.optim.AdamW;
const AdamWConfig = fucina.optim.AdamWConfig;

// ---------------------------------------------------------------------------
// Schedules (base_train.py, num_iterations passed explicitly). All f64 to match
// the reference's Python-float scalars; cast to f32 only when applied to a lr.
// ---------------------------------------------------------------------------

/// base_train.py schedule knobs (--warmup-steps / --warmdown-ratio /
/// --final-lr-frac). Defaults reproduce the d6 acceptance config exactly, so the
/// zero-knob helpers below are bit-identical to passing `.{}` here.
pub const ScheduleParams = struct {
    warmup_steps: f64 = 40.0,
    warmdown_ratio: f64 = 0.65,
    final_lr_frac: f64 = 0.05,
};

/// base_train.py get_lr_multiplier at the default knobs: linear warmup over
/// `warmup`(=40) steps, constant 1.0 until the warmdown window, then linear to
/// `final_lr_frac`.
pub fn lrMultiplier(step: usize, num_iters: usize) f64 {
    return lrMultiplierWith(step, num_iters, .{});
}

/// base_train.py get_lr_multiplier(it) reading args.{warmup_steps,warmdown_ratio,
/// final_lr_frac}.
pub fn lrMultiplierWith(step: usize, num_iters: usize, sp: ScheduleParams) f64 {
    const warmup = sp.warmup_steps;
    const n: f64 = @floatFromInt(num_iters);
    const it: f64 = @floatFromInt(step);
    const warmdown_iters = @round(sp.warmdown_ratio * n);
    if (it < warmup) return (it + 1.0) / warmup;
    if (it <= n - warmdown_iters) return 1.0;
    const progress = (n - it) / warmdown_iters;
    return progress * 1.0 + (1.0 - progress) * sp.final_lr_frac;
}

/// base_train.py get_muon_momentum at the default knobs: 0.85→0.97 over the first
/// 400 steps, hold at 0.97, then 0.97→0.90 across the warmdown window.
pub fn muonMomentum(step: usize, num_iters: usize) f64 {
    return muonMomentumWith(step, num_iters, .{});
}

/// base_train.py get_muon_momentum(it); the warmdown window tracks args.warmdown_ratio.
pub fn muonMomentumWith(step: usize, num_iters: usize, sp: ScheduleParams) f64 {
    const n: f64 = @floatFromInt(num_iters);
    const it: f64 = @floatFromInt(step);
    const warmdown_iters = @round(sp.warmdown_ratio * n);
    const warmdown_start = n - warmdown_iters;
    if (it < 400.0) {
        const frac = it / 400.0;
        return (1.0 - frac) * 0.85 + frac * 0.97;
    }
    if (it >= warmdown_start) {
        const progress = (it - warmdown_start) / warmdown_iters;
        return 0.97 * (1.0 - progress) + 0.90 * progress;
    }
    return 0.97;
}

/// base_train.py get_weight_decay: cosine decay of the scaled weight decay to 0
/// (independent of the warmup/warmdown/final-lr knobs).
pub fn muonWeightDecay(step: usize, num_iters: usize, wd_scaled: f64) f64 {
    const n: f64 = @floatFromInt(num_iters);
    const it: f64 = @floatFromInt(step);
    return wd_scaled * 0.5 * (1.0 + @cos(std.math.pi * it / n));
}

/// The three per-step schedule scalars applied to the optimizer before a step
/// (base_train.py sets group['lr']=initial_lr·lrm, group['momentum']=momentum,
/// group['weight_decay']=wd for the muon groups).
pub const StepSchedule = struct {
    lrm: f64,
    muon_momentum: f64,
    muon_weight_decay: f64,

    /// Default-knob schedule (d6 acceptance config).
    pub fn at(step: usize, num_iters: usize, wd_scaled: f64) StepSchedule {
        return atWith(step, num_iters, wd_scaled, .{});
    }

    /// Schedule threading the base_train.py CLI knobs (--warmup-steps /
    /// --warmdown-ratio / --final-lr-frac). At the defaults this equals `at`.
    pub fn atWith(step: usize, num_iters: usize, wd_scaled: f64, sp: ScheduleParams) StepSchedule {
        return .{
            .lrm = lrMultiplierWith(step, num_iters, sp),
            .muon_momentum = muonMomentumWith(step, num_iters, sp),
            .muon_weight_decay = muonWeightDecay(step, num_iters, wd_scaled),
        };
    }
};

// ---------------------------------------------------------------------------
// Configuration (RESOLVED per-group initial_lr from the schedule oracle — these
// bake in dmodel_lr_scale and batch_lr_scale so they differ between d2 and d6).
// betas/eps/weight_decay are architecture-independent and hardcoded below.
// ---------------------------------------------------------------------------

pub const Config = struct {
    /// initial_lr for the 6 AdamW groups, in gpt.py setup_optimizer order:
    /// 0=lm_head, 1=wte, 2=value_embeds, 3=resid_lambdas, 4=x0_lambdas, 5=smear.
    adamw_initial_lr: [6]f64,
    /// initial_lr shared by every Muon (matrix) group (= matrix_lr·batch_lr_scale).
    muon_initial_lr: f64,
};

/// Fixed AdamW group hyperparameters (gpt.py setup_optimizer; identical for d2/d6).
const AdamwHyper = struct { beta1: f32, beta2: f32, weight_decay: f32 };
const adamw_hyper = [6]AdamwHyper{
    .{ .beta1 = 0.8, .beta2 = 0.96, .weight_decay = 0.01 }, // lm_head
    .{ .beta1 = 0.8, .beta2 = 0.995, .weight_decay = 0.001 }, // wte
    .{ .beta1 = 0.8, .beta2 = 0.995, .weight_decay = 0.01 }, // value_embeds
    .{ .beta1 = 0.8, .beta2 = 0.95, .weight_decay = 0.05 }, // resid_lambdas
    .{ .beta1 = 0.96, .beta2 = 0.95, .weight_decay = 0.0 }, // x0_lambdas
    .{ .beta1 = 0.8, .beta2 = 0.95, .weight_decay = 0.0 }, // smear
};
const adamw_eps: f32 = 1e-10;

/// Muon second-moment EMA coefficient (muon_step_fused beta2_t; base_train sets 0.9).
const muon_beta2: f32 = 0.9;

/// Polar Express quintic coefficients (optim.py polar_express_coeffs, 5 iters).
const polar_coeffs = [5][3]f64{
    .{ 8.156554524902461, -22.48329292557795, 15.878769915207462 },
    .{ 4.042929935166739, -2.808917465908714, 0.5000178451051316 },
    .{ 3.8916678022926607, -2.772484153217685, 0.5060648178503393 },
    .{ 3.285753657755655, -2.3681294933425376, 0.46449024233003106 },
    .{ 2.3465413258596377, -1.7097828382687081, 0.42323551169305323 },
};

// ---------------------------------------------------------------------------
// Type-erased parameter access. The Muon groups mix tensors of different tag
// types (c_q {.qo,.d}, c_fc {.ff,.d}, …); the thunks reach their f32 storage
// and gradients uniformly through fn pointers generated per concrete type.
// ---------------------------------------------------------------------------

const ParamAccess = struct {
    ptr: *anyopaque,
    len: usize,
    readValue: *const fn (*const anyopaque, []f32) void,
    writeValue: *const fn (*anyopaque, []const f32) void,
    readGrad: *const fn (*const anyopaque, *ExecContext, []f32) anyerror!void,
    zeroGrad: *const fn (*const anyopaque) void,

    fn of(t: anytype) ParamAccess {
        const Ptr = @TypeOf(t);
        const T = @typeInfo(Ptr).pointer.child;
        const Gen = struct {
            fn readValue(p: *const anyopaque, dst: []f32) void {
                const tp: *const T = @ptrCast(@alignCast(p));
                @memcpy(dst, tp.value.dataConstChecked() catch unreachable);
            }
            fn writeValue(p: *anyopaque, src: []const f32) void {
                const tp: *T = @ptrCast(@alignCast(p));
                @memcpy(tp.value.dataChecked() catch unreachable, src);
            }
            fn readGrad(p: *const anyopaque, ctx: *ExecContext, dst: []f32) anyerror!void {
                const tp: *const T = @ptrCast(@alignCast(p));
                var gv = (try tp.gradView(ctx)) orelse return error.NoGradient;
                defer gv.deinit();
                try gv.copyTo(dst);
            }
            fn zeroGrad(p: *const anyopaque) void {
                const tp: *const T = @ptrCast(@alignCast(p));
                tp.zeroGrad();
            }
        };
        return .{
            .ptr = @ptrCast(@constCast(t)),
            .len = t.value.len(),
            .readValue = Gen.readValue,
            .writeValue = Gen.writeValue,
            .readGrad = Gen.readGrad,
            .zeroGrad = Gen.zeroGrad,
        };
    }
};

// ---------------------------------------------------------------------------
// Muon group: K params of identical (m, n) shape, stacked into (K, m, n).
// ---------------------------------------------------------------------------

const MuonGroup = struct {
    allocator: Allocator,
    /// First param name (oracle buffer key prefix, e.g. transformer.h.0.attn.c_q.weight).
    name: []const u8,
    k: usize,
    m: usize,
    n: usize,
    /// max(1, m/n)^0.5 — the per-group lr scale folded in _compute_muon.
    shape_factor: f32,
    /// m >= n → NorMuon reduces the last axis (red_dim=-1), second-moment shape
    /// (K,m,1); else reduces axis -2, shape (K,1,n).
    red_last: bool,
    /// Length of the kept axis in the second-moment buffer per K (m if red_last, else n).
    red_len: usize,
    params: []ParamAccess,
    momentum_buffer: []f32, // (K,m,n), zero-init (optim.py state["momentum_buffer"])
    second_momentum_buffer: []f32, // (K,red_len), zero-init (state["second_momentum_buffer"])

    // Per-step scratch, allocated ONCE at registration and reused every step
    // (no per-step alloc/free). Sizes are fixed by the group's (k, m, n): the
    // four (K,m,n) working buffers, the three (K,side,side) Newton-Schulz
    // intermediates (side = min(m,n)), and the two NorMuon per-axis reductions
    // (red_len). Transient only — not part of the optimizer state (save/load).
    scratch_grad: []f32, // (K,m,n)
    scratch_params: []f32, // (K,m,n)
    scratch_x: []f32, // (K,m,n)
    scratch_res: []f32, // (K,m,n)
    scratch_a: []f32, // (K,side,side)
    scratch_aa: []f32, // (K,side,side)
    scratch_b: []f32, // (K,side,side)
    scratch_v_mean: []f32, // (K,red_len) NorMuon mean(g², reduced axis), one lane per matrix
    scratch_step_size: []f32, // (K,red_len) NorMuon rsqrt step size, one lane per matrix

    fn deinit(self: *MuonGroup) void {
        self.allocator.free(self.params);
        self.allocator.free(self.momentum_buffer);
        self.allocator.free(self.second_momentum_buffer);
        self.allocator.free(self.scratch_grad);
        self.allocator.free(self.scratch_params);
        self.allocator.free(self.scratch_x);
        self.allocator.free(self.scratch_res);
        self.allocator.free(self.scratch_a);
        self.allocator.free(self.scratch_aa);
        self.allocator.free(self.scratch_b);
        self.allocator.free(self.scratch_v_mean);
        self.allocator.free(self.scratch_step_size);
        self.* = undefined;
    }
};

const BmmKind = enum { plain, trans_a, trans_b };

// ---------------------------------------------------------------------------
// MuonAdamW
// ---------------------------------------------------------------------------

pub const MuonAdamW = struct {
    allocator: Allocator,
    adamw: [6]AdamW,
    adamw_initial_lr: [6]f64,
    muon_initial_lr: f64,
    muon_groups: std.ArrayList(MuonGroup) = .empty,
    /// Heap-allocated checkpoint names (borrowed by AdamW/MuonGroup); freed here.
    owned_names: std.ArrayList([]const u8) = .empty,

    pub fn init(allocator: Allocator, cfg: Config) MuonAdamW {
        var self: MuonAdamW = .{
            .allocator = allocator,
            .adamw = undefined,
            .adamw_initial_lr = cfg.adamw_initial_lr,
            .muon_initial_lr = cfg.muon_initial_lr,
        };
        for (&self.adamw, 0..) |*a, i| {
            a.* = AdamW.init(allocator, .{
                .lr = @floatCast(cfg.adamw_initial_lr[i]),
                .beta1 = adamw_hyper[i].beta1,
                .beta2 = adamw_hyper[i].beta2,
                .eps = adamw_eps,
                .weight_decay = adamw_hyper[i].weight_decay,
            });
        }
        return self;
    }

    pub fn deinit(self: *MuonAdamW) void {
        for (&self.adamw) |*a| a.deinit();
        for (self.muon_groups.items) |*g| g.deinit();
        self.muon_groups.deinit(self.allocator);
        for (self.owned_names.items) |nm| self.allocator.free(nm);
        self.owned_names.deinit(self.allocator);
        self.* = undefined;
    }

    /// Dupe a checkpoint name and track it for freeing in `deinit`.
    fn dupeName(self: *MuonAdamW, s: []const u8) ![]const u8 {
        const owned = try self.allocator.dupe(u8, s);
        errdefer self.allocator.free(owned);
        try self.owned_names.append(self.allocator, owned);
        return owned;
    }

    /// Register every model parameter into its AdamW group or Muon shape-group,
    /// reproducing gpt.py setup_optimizer's grouping and stacking order.
    pub fn registerModel(self: *MuonAdamW, model: *Model) !void {
        // gpt.py groups matrix params by their ACTUAL shapes (sorted set); the
        // fixed four-group order below assumes MHA, where c_k/c_v share c_q's
        // (qo, d) shape. GQA would need dynamic shape grouping — reject it
        // rather than stack mismatched shapes.
        if (model.cfg.n_kv_head != model.cfg.n_head) return error.UnsupportedGqaGrouping;

        // AdamW groups (embeddings, head, scalars).
        try self.adamw[0].addParamNamed(&model.lm_head, "lm_head.weight");
        try self.adamw[1].addParamNamed(&model.wte, "transformer.wte.weight");
        {
            var buf: [48]u8 = undefined;
            for (model.value_embeds, 0..) |*ve, i| {
                if (ve.*) |*t| {
                    const nm = try std.fmt.bufPrint(&buf, "value_embeds.{d}.weight", .{i});
                    try self.adamw[2].addParamNamed(t, try self.dupeName(nm));
                }
            }
        }
        try self.adamw[3].addParamNamed(&model.resid_lambdas, "resid_lambdas");
        try self.adamw[4].addParamNamed(&model.x0_lambdas, "x0_lambdas");
        try self.adamw[5].addParamNamed(&model.smear_gate, "smear_gate.weight");
        try self.adamw[5].addParamNamed(&model.smear_lambda, "smear_lambda");
        try self.adamw[5].addParamNamed(&model.backout_lambda, "backout_lambda");

        // Muon groups, BY SHAPE, in the sorted order the reference produces:
        // (m,n) tuple ascending → ve_gate, (c_q/c_k/c_v/c_proj), (mlp c_proj),
        // (c_fc). Within a group, params are stacked in transformer.h.parameters()
        // order (layer-major; per layer c_q,c_k,c_v,c_proj[,ve_gate],c_fc,c_proj).
        const cfg = model.cfg;
        const d = cfg.n_embd;
        const hd = cfg.headDim();
        const qo = cfg.n_head * hd;
        const ff = 4 * d;

        // ve_gate (n_kv_head, 12) — layers with value embeddings.
        {
            var params: std.ArrayList(ParamAccess) = .empty;
            errdefer params.deinit(self.allocator);
            var first_name: []const u8 = "";
            var name_buf: [64]u8 = undefined;
            for (model.layers, 0..) |*l, i| {
                if (l.ve_gate) |*g| {
                    if (params.items.len == 0)
                        first_name = try self.dupeName(try std.fmt.bufPrint(&name_buf, "transformer.h.{d}.attn.ve_gate.weight", .{i}));
                    try params.append(self.allocator, ParamAccess.of(g));
                }
            }
            if (params.items.len != 0)
                try self.appendMuonGroup(first_name, cfg.n_kv_head, 12, &params);
        }

        // (c_q, c_k, c_v, c_proj) — all (qo=d, d) here (square) for these configs.
        {
            var params: std.ArrayList(ParamAccess) = .empty;
            errdefer params.deinit(self.allocator);
            for (model.layers) |*l| {
                try params.append(self.allocator, ParamAccess.of(&l.c_q));
                try params.append(self.allocator, ParamAccess.of(&l.c_k));
                try params.append(self.allocator, ParamAccess.of(&l.c_v));
                try params.append(self.allocator, ParamAccess.of(&l.c_proj));
            }
            try self.appendMuonGroup(
                try self.dupeName("transformer.h.0.attn.c_q.weight"),
                qo,
                d,
                &params,
            );
        }

        // mlp c_proj (d, ff) — wide.
        {
            var params: std.ArrayList(ParamAccess) = .empty;
            errdefer params.deinit(self.allocator);
            for (model.layers) |*l| try params.append(self.allocator, ParamAccess.of(&l.c_proj_mlp));
            try self.appendMuonGroup(
                try self.dupeName("transformer.h.0.mlp.c_proj.weight"),
                d,
                ff,
                &params,
            );
        }

        // mlp c_fc (ff, d) — tall.
        {
            var params: std.ArrayList(ParamAccess) = .empty;
            errdefer params.deinit(self.allocator);
            for (model.layers) |*l| try params.append(self.allocator, ParamAccess.of(&l.c_fc));
            try self.appendMuonGroup(
                try self.dupeName("transformer.h.0.mlp.c_fc.weight"),
                ff,
                d,
                &params,
            );
        }
    }

    /// Takes ownership of the caller's `params` items via toOwnedSlice (the
    /// caller's list is left empty, so its errdefer deinit stays a no-op — a
    /// by-value copy would alias the backing memory and double-free on error).
    fn appendMuonGroup(self: *MuonAdamW, name: []const u8, m: usize, n: usize, params: *std.ArrayList(ParamAccess)) !void {
        const k = params.items.len;
        const red_last = m >= n;
        const red_len = if (red_last) m else n;
        const rows_f: f32 = @floatFromInt(m);
        const cols_f: f32 = @floatFromInt(n);
        const mn = m * n;
        const side = @min(m, n); // Newton-Schulz intermediate matrix side
        const momentum = try self.allocator.alloc(f32, k * mn);
        errdefer self.allocator.free(momentum);
        @memset(momentum, 0);
        const second = try self.allocator.alloc(f32, k * red_len);
        errdefer self.allocator.free(second);
        @memset(second, 0);
        // Reused-every-step scratch (allocate once; sizes fixed by k/m/n).
        const scratch_grad = try self.allocator.alloc(f32, k * mn);
        errdefer self.allocator.free(scratch_grad);
        const scratch_params = try self.allocator.alloc(f32, k * mn);
        errdefer self.allocator.free(scratch_params);
        const scratch_x = try self.allocator.alloc(f32, k * mn);
        errdefer self.allocator.free(scratch_x);
        const scratch_res = try self.allocator.alloc(f32, k * mn);
        errdefer self.allocator.free(scratch_res);
        const scratch_a = try self.allocator.alloc(f32, k * side * side);
        errdefer self.allocator.free(scratch_a);
        const scratch_aa = try self.allocator.alloc(f32, k * side * side);
        errdefer self.allocator.free(scratch_aa);
        const scratch_b = try self.allocator.alloc(f32, k * side * side);
        errdefer self.allocator.free(scratch_b);
        const scratch_v_mean = try self.allocator.alloc(f32, k * red_len);
        errdefer self.allocator.free(scratch_v_mean);
        const scratch_step_size = try self.allocator.alloc(f32, k * red_len);
        errdefer self.allocator.free(scratch_step_size);
        const owned = try params.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(owned);
        try self.muon_groups.append(self.allocator, .{
            .allocator = self.allocator,
            .name = name,
            .k = k,
            .m = m,
            .n = n,
            .shape_factor = @sqrt(@max(1.0, rows_f / cols_f)),
            .red_last = red_last,
            .red_len = red_len,
            .params = owned,
            .momentum_buffer = momentum,
            .second_momentum_buffer = second,
            .scratch_grad = scratch_grad,
            .scratch_params = scratch_params,
            .scratch_x = scratch_x,
            .scratch_res = scratch_res,
            .scratch_a = scratch_a,
            .scratch_aa = scratch_aa,
            .scratch_b = scratch_b,
            .scratch_v_mean = scratch_v_mean,
            .scratch_step_size = scratch_step_size,
        });
    }

    /// One optimizer step (reads accumulated grads, updates params + state).
    /// base_train order: (schedule already chosen by caller) → step → zeroGrad.
    pub fn step(self: *MuonAdamW, ctx: *ExecContext, sched: StepSchedule) !void {
        // AdamW half: set each group's lr = initial_lr·lrm, then the fused step.
        for (&self.adamw, 0..) |*a, i| {
            a.config.lr = @floatCast(self.adamw_initial_lr[i] * sched.lrm);
            try a.step(ctx);
        }
        // Muon half.
        const muon_mom: f32 = @floatCast(sched.muon_momentum);
        const muon_wd: f32 = @floatCast(sched.muon_weight_decay);
        for (self.muon_groups.items) |*g| {
            // _compute_muon: group['lr'](=initial_lr·lrm) · max(1,m/n)^0.5.
            const lr: f32 = @floatCast(self.muon_initial_lr * sched.lrm);
            try self.muonStepGroup(ctx, g, lr * g.shape_factor, muon_wd, muon_mom);
        }
    }

    pub fn zeroGrad(self: *MuonAdamW) void {
        for (&self.adamw) |*a| a.zeroGrad();
        for (self.muon_groups.items) |*g| {
            for (g.params) |*pa| pa.zeroGrad(pa.ptr);
        }
    }

    // -- State persistence (example-local frame; reuses AdamW.saveState/
    // loadState per instance and serializes the Muon buffers directly).
    // base-train --resume needs this; the gates exercise it via a round-trip
    // check. --

    pub fn saveState(self: *MuonAdamW, writer: *std.Io.Writer) !void {
        try writer.writeAll(state_magic);
        for (&self.adamw) |*a| try a.saveState(writer);
        try writer.writeInt(u32, @intCast(self.muon_groups.items.len), .little);
        for (self.muon_groups.items) |*g| {
            try writer.writeInt(u32, @intCast(g.momentum_buffer.len), .little);
            try writer.writeAll(std.mem.sliceAsBytes(g.momentum_buffer));
            try writer.writeInt(u32, @intCast(g.second_momentum_buffer.len), .little);
            try writer.writeAll(std.mem.sliceAsBytes(g.second_momentum_buffer));
        }
    }

    pub fn loadState(self: *MuonAdamW, reader: *std.Io.Reader) !void {
        var magic: [state_magic.len]u8 = undefined;
        try reader.readSliceAll(&magic);
        if (!std.mem.eql(u8, &magic, state_magic)) return error.CheckpointMagicMismatch;
        for (&self.adamw) |*a| try a.loadState(reader);
        const count = try reader.takeInt(u32, .little);
        if (count != self.muon_groups.items.len) return error.CheckpointConfigMismatch;
        for (self.muon_groups.items) |*g| {
            const mlen = try reader.takeInt(u32, .little);
            if (mlen != g.momentum_buffer.len) return error.CheckpointShapeMismatch;
            try reader.readSliceAll(std.mem.sliceAsBytes(g.momentum_buffer));
            const slen = try reader.takeInt(u32, .little);
            if (slen != g.second_momentum_buffer.len) return error.CheckpointShapeMismatch;
            try reader.readSliceAll(std.mem.sliceAsBytes(g.second_momentum_buffer));
        }
    }

    // -- Muon step (port of muon_step_fused, all f32) ----------------------

    fn muonStepGroup(self: *MuonAdamW, ctx: *ExecContext, g: *MuonGroup, lr: f32, wd: f32, momentum: f32) !void {
        _ = self; // scratch now lives on the group (allocated once at registration)
        const k = g.k;
        const m = g.m;
        const n = g.n;
        const mn = m * n;

        // Group-owned scratch, reused every step (no per-step alloc/free).
        const grad = g.scratch_grad;
        const params = g.scratch_params;
        const x = g.scratch_x;
        const res = g.scratch_res;
        const a_buf = g.scratch_a;
        const aa_buf = g.scratch_aa;
        const b_buf = g.scratch_b;

        // Gather stacked grads and params.
        for (g.params, 0..) |*pa, ki| {
            try pa.readGrad(pa.ptr, ctx, grad[ki * mn ..][0..mn]);
            pa.readValue(pa.ptr, params[ki * mn ..][0..mn]);
        }

        // 1. Nesterov momentum (updates momentum_buffer, leaves the direction in grad).
        nesterov(ctx, g.momentum_buffer, grad, momentum);
        // 2. X = g.
        @memcpy(x, grad);
        // 3. MuonEq row equilibration.
        muonEq(ctx, x, k, m, n);
        // 4. Polar Express orthogonalization.
        try polarExpress(ctx, x, res, a_buf, aa_buf, b_buf, k, m, n);
        // 5. g = X.
        @memcpy(grad, x);
        // 6. Muon+ renormalization.
        muonPlus(ctx, grad, k, m, n);
        // 7. NorMuon variance reduction (updates second_momentum_buffer).
        norMuon(ctx, grad, g.second_momentum_buffer, g.scratch_v_mean, g.scratch_step_size, k, m, n, g.red_last, g.red_len);
        // 8. Cautious weight decay + update.
        cautiousUpdate(ctx, params, grad, lr, wd);

        // Scatter updated params back.
        for (g.params, 0..) |*pa, ki| pa.writeValue(pa.ptr, params[ki * mn ..][0..mn]);
    }
};

// ---------------------------------------------------------------------------
// Muon step kernels (f32; reductions accumulate in f64 for a stable match to
// torch's f32 norms). Cited to optim.py muon_step_fused. The loops dispatch
// disjoint ranges onto the exec worker pool ONLY where every index's work is
// independent (flat elementwise spans, or whole (m,n) matrices with their own
// serial reductions), so the split is bitwise-identical to the serial loops.
// ---------------------------------------------------------------------------

/// Elementwise spans below this stay serial (matches the library's vector
/// elementwise threshold); per-matrix loops stay serial below it too.
const par_threshold: usize = 256 * 1024;

/// Dispatch `run(context, start, end)` over disjoint ranges of [0, total) on
/// the exec worker pool; serial when the pool is absent or total < threshold
/// (threshold is in the same units as total).
fn parallelRanges(
    ctx: *ExecContext,
    total: usize,
    threshold: usize,
    context: anytype,
    comptime run: fn (@TypeOf(context), usize, usize) void,
) void {
    const Ctx = @TypeOf(context);
    const Task = struct {
        c: Ctx,
        s: usize,
        e: usize,
        fn go(t: *const @This()) void {
            run(t.c, t.s, t.e);
        }
    };
    if (total >= threshold) {
        if (ctx.workPool()) |pool| {
            const max_tasks = 16;
            const task_count: usize = @min(max_tasks, total);
            var tasks: [max_tasks]Task = undefined;
            for (0..task_count) |i| {
                tasks[i] = .{ .c = context, .s = i * total / task_count, .e = (i + 1) * total / task_count };
            }
            pool.parallelChunks(Task, tasks[0..task_count], Task.go);
            return;
        }
    }
    run(context, 0, total);
}

/// Per-matrix threshold: parallelize a 0..k loop only when each matrix carries
/// real work (k is tiny — 3..24 — so gate on the per-matrix element count).
fn kThreshold(mn: usize) usize {
    return if (mn >= par_threshold / 16) 2 else std.math.maxInt(usize);
}

/// Nesterov momentum (optim.py:132-133). buf.lerp_(g, 1-μ) then g = g.lerp_(buf, μ);
/// mirrors ATen's scalar-weight lerp form switch at |weight|=0.5 so the f32 op
/// sequence matches the torch reference (μ≈0.85..0.97 ⇒ buf uses the start form,
/// g uses the end form).
fn nesterov(ctx: *ExecContext, buf: []f32, g: []f32, mu: f32) void {
    const C = struct { buf: []f32, g: []f32, mu: f32 };
    parallelRanges(ctx, g.len, par_threshold, C{ .buf = buf, .g = g, .mu = mu }, struct {
        fn run(c: C, s: usize, e: usize) void {
            const w1 = 1.0 - c.mu; // buf weight
            for (c.buf[s..e], c.g[s..e]) |*bi, *gi| {
                const b0 = bi.*;
                const gg = gi.*;
                const b1 = if (@abs(w1) < 0.5) b0 + w1 * (gg - b0) else gg - (gg - b0) * (1.0 - w1);
                bi.* = b1;
                gi.* = if (@abs(c.mu) < 0.5) gg + c.mu * (b1 - gg) else b1 - (b1 - gg) * (1.0 - c.mu);
            }
        }
    }.run);
}

/// MuonEq (optim.py:139-141): rescale each row to target = ||X||_F/√m divided by
/// the (clamp-≥1e-6) row norm. Parallel across matrices; each matrix's norms
/// stay serially accumulated.
fn muonEq(ctx: *ExecContext, x: []f32, k: usize, m: usize, n: usize) void {
    const mn = m * n;
    const C = struct { x: []f32, m: usize, n: usize, mn: usize };
    parallelRanges(ctx, k, kThreshold(mn), C{ .x = x, .m = m, .n = n, .mn = mn }, struct {
        fn run(c: C, ks: usize, ke: usize) void {
            const inv_sqrt_m: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(c.m))));
            for (ks..ke) |ki| {
                const base = ki * c.mn;
                const fro: f32 = @floatCast(@sqrt(sumSq(c.x[base .. base + c.mn])));
                const target = fro * inv_sqrt_m;
                for (0..c.m) |i| {
                    const rb = base + i * c.n;
                    var rn: f32 = @floatCast(@sqrt(sumSq(c.x[rb .. rb + c.n])));
                    if (rn < 1e-6) rn = 1e-6;
                    const scale = target / rn;
                    for (c.x[rb .. rb + c.n]) |*v| v.* *= scale;
                }
            }
        }
    }.run);
}

/// Polar Express (optim.py:143-154): normalize by ||X||_F·1.01+1e-6, then 5
/// quintic iterations. Tall (m>n): A=Xᵀ·X, B=b·A+c·A·A, X=a·X+X·B. Wide (m≤n):
/// A=X·Xᵀ, B=b·A+c·A·A, X=a·X+B·X. The batched GEMMs run through raw bmm.
fn polarExpress(ctx: *ExecContext, x: []f32, res: []f32, a_buf: []f32, aa_buf: []f32, b_buf: []f32, k: usize, m: usize, n: usize) !void {
    const mn = m * n;
    const NormC = struct { x: []f32, mn: usize };
    parallelRanges(ctx, k, kThreshold(mn), NormC{ .x = x, .mn = mn }, struct {
        fn run(c: NormC, ks: usize, ke: usize) void {
            for (ks..ke) |ki| {
                const base = ki * c.mn;
                const fro: f32 = @floatCast(@sqrt(sumSq(c.x[base .. base + c.mn])));
                const denom = fro * 1.01 + 1e-6;
                for (c.x[base .. base + c.mn]) |*v| v.* /= denom;
            }
        }
    }.run);
    const tall = m > n;
    const side = if (tall) n else m;
    const sq = side * side;
    const AxpyC = struct { dst: []f32, a_src: []const f32, b_src: []const f32, ca: f32, cb: f32 };
    const axpy = struct {
        // dst[e] = ca·a_src[e] + cb·b_src[e] (covers both the B build and the
        // X update with cb=1 via b_src=res).
        fn run(c: AxpyC, s: usize, e: usize) void {
            for (c.dst[s..e], c.a_src[s..e], c.b_src[s..e]) |*d, av, bv| d.* = c.ca * av + c.cb * bv;
        }
    }.run;
    for (polar_coeffs) |c| {
        const a: f32 = @floatCast(c[0]);
        const b: f32 = @floatCast(c[1]);
        const cc: f32 = @floatCast(c[2]);
        if (tall) {
            try bmm3(ctx, .trans_a, x, .{ k, m, n }, x, .{ k, m, n }, a_buf, .{ k, n, n }); // A=(k,n,n)
            try bmm3(ctx, .plain, a_buf, .{ k, n, n }, a_buf, .{ k, n, n }, aa_buf, .{ k, n, n }); // A·A
            parallelRanges(ctx, k * sq, par_threshold, AxpyC{ .dst = b_buf, .a_src = a_buf, .b_src = aa_buf, .ca = b, .cb = cc }, axpy);
            try bmm3(ctx, .plain, x, .{ k, m, n }, b_buf, .{ k, n, n }, res, .{ k, m, n }); // X·B
            parallelRanges(ctx, k * mn, par_threshold, AxpyC{ .dst = x, .a_src = x, .b_src = res, .ca = a, .cb = 1.0 }, axpy);
        } else {
            try bmm3(ctx, .trans_b, x, .{ k, m, n }, x, .{ k, m, n }, a_buf, .{ k, m, m }); // A=(k,m,m)
            try bmm3(ctx, .plain, a_buf, .{ k, m, m }, a_buf, .{ k, m, m }, aa_buf, .{ k, m, m }); // A·A
            parallelRanges(ctx, k * sq, par_threshold, AxpyC{ .dst = b_buf, .a_src = a_buf, .b_src = aa_buf, .ca = b, .cb = cc }, axpy);
            try bmm3(ctx, .plain, b_buf, .{ k, m, m }, x, .{ k, m, n }, res, .{ k, m, n }); // B·X
            parallelRanges(ctx, k * mn, par_threshold, AxpyC{ .dst = x, .a_src = x, .b_src = res, .ca = a, .cb = 1.0 }, axpy);
        }
    }
}

/// Muon+ renorm (optim.py:159-161): snap ||g||_F to √min(m,n). Parallel across
/// matrices; each matrix's Frobenius norm stays serially accumulated.
fn muonPlus(ctx: *ExecContext, g: []f32, k: usize, m: usize, n: usize) void {
    const mn = m * n;
    const C = struct { g: []f32, mn: usize, target_norm: f32 };
    const target_norm: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(@min(m, n)))));
    parallelRanges(ctx, k, kThreshold(mn), C{ .g = g, .mn = mn, .target_norm = target_norm }, struct {
        fn run(c: C, ks: usize, ke: usize) void {
            for (ks..ke) |ki| {
                const base = ki * c.mn;
                var cur: f32 = @floatCast(@sqrt(sumSq(c.g[base .. base + c.mn])));
                if (cur < 1e-6) cur = 1e-6;
                const scale = c.target_norm / cur;
                for (c.g[base .. base + c.mn]) |*v| v.* *= scale;
            }
        }
    }.run);
}

/// NorMuon variance reduction (optim.py:164-174). red_last ⇒ reduce columns
/// (kept axis = rows), else reduce rows (kept axis = columns). Updates the
/// factored second-moment EMA and rescales g so the per-row/col update RMS is
/// normalized while preserving the overall Frobenius norm.
fn norMuon(ctx: *ExecContext, g: []f32, smb: []f32, v_mean: []f32, step_size: []f32, k: usize, m: usize, n: usize, red_last: bool, red_len: usize) void {
    const mn = m * n;
    // v_mean/step_size are group-owned reduction buffers, one red_len lane PER
    // MATRIX (k·red_len) so the 0..k loop can run matrix-parallel without
    // sharing scratch; every matrix's reductions stay serially accumulated.
    std.debug.assert(v_mean.len == k * red_len and step_size.len == k * red_len);

    const C = struct { g: []f32, smb: []f32, v_mean: []f32, step_size: []f32, m: usize, n: usize, mn: usize, red_last: bool, red_len: usize };
    parallelRanges(ctx, k, kThreshold(mn), C{
        .g = g,
        .smb = smb,
        .v_mean = v_mean,
        .step_size = step_size,
        .m = m,
        .n = n,
        .mn = mn,
        .red_last = red_last,
        .red_len = red_len,
    }, struct {
        fn run(c: C, ks: usize, ke: usize) void {
            // red_dim_size = size of the reduced axis (n if red_last else m).
            const red_size: usize = if (c.red_last) c.n else c.m;
            const red_size_f: f32 = @floatFromInt(red_size);
            const red_size_d: f64 = @floatFromInt(red_size);
            for (ks..ke) |ki| {
                const base = ki * c.mn;
                const sbase = ki * c.red_len;
                const vm_lane = c.v_mean[sbase .. sbase + c.red_len];
                const ss_lane = c.step_size[sbase .. sbase + c.red_len];
                // v_mean = mean(g², red_dim).
                for (0..c.red_len) |r| {
                    var s: f64 = 0;
                    if (c.red_last) {
                        const rb = base + r * c.n; // r = row i
                        for (c.g[rb .. rb + c.n]) |v| s += @as(f64, v) * v;
                    } else {
                        for (0..c.m) |i| { // r = col j
                            const v = c.g[base + i * c.n + r];
                            s += @as(f64, v) * v;
                        }
                    }
                    vm_lane[r] = @floatCast(s / red_size_d);
                }
                // v_norm = sqrt(sum(v_mean) · red_dim_size).
                var vm_sum: f64 = 0;
                for (vm_lane) |vm| vm_sum += vm;
                const v_norm: f32 = @floatCast(@sqrt(vm_sum * red_size_d));
                // second_momentum_buffer.lerp_(v_mean, 1-beta2) [weight 0.1, start
                // form]; step_size = clamp_min(smb,1e-10).rsqrt(); v_norm_new from
                // scaled sums.
                const w = 1.0 - muon_beta2;
                var vnn_acc: f64 = 0;
                for (0..c.red_len) |r| {
                    const sv = c.smb[sbase + r];
                    const nv = sv + w * (vm_lane[r] - sv);
                    c.smb[sbase + r] = nv;
                    var cl = nv;
                    if (cl < 1e-10) cl = 1e-10;
                    const ss: f32 = 1.0 / @sqrt(cl);
                    ss_lane[r] = ss;
                    const scaled = (vm_lane[r] * red_size_f) * (ss * ss);
                    vnn_acc += scaled;
                }
                var v_norm_new: f32 = @floatCast(@sqrt(vnn_acc));
                if (v_norm_new < 1e-10) v_norm_new = 1e-10;
                const ratio = v_norm / v_norm_new;
                // g *= final_scale = step_size · ratio (broadcast over the reduced axis).
                if (c.red_last) {
                    for (0..c.m) |i| {
                        const fs = ss_lane[i] * ratio;
                        const rb = base + i * c.n;
                        for (c.g[rb .. rb + c.n]) |*v| v.* *= fs;
                    }
                } else {
                    for (0..c.m) |i| {
                        const rb = base + i * c.n;
                        for (0..c.n) |j| c.g[rb + j] *= ss_lane[j] * ratio;
                    }
                }
            }
        }
    }.run);
}

/// Cautious weight decay + update (optim.py:177-180): mask = (g·p ≥ 0);
/// p -= lr·g + lr·wd·p·mask.
fn cautiousUpdate(ctx: *ExecContext, p: []f32, g: []f32, lr: f32, wd: f32) void {
    const C = struct { p: []f32, g: []f32, lr: f32, lrwd: f32 };
    parallelRanges(ctx, p.len, par_threshold, C{ .p = p, .g = g, .lr = lr, .lrwd = lr * wd }, struct {
        fn run(c: C, s: usize, e: usize) void {
            for (c.p[s..e], c.g[s..e]) |*pi, gi| {
                const mask: f32 = if (gi * pi.* >= 0) 1.0 else 0.0;
                const decay = (c.lrwd * pi.*) * mask;
                pi.* -= c.lr * gi + decay;
            }
        }
    }.run);
}

fn sumSq(s: []const f32) f64 {
    var acc: f64 = 0;
    for (s) |v| acc += @as(f64, v) * v;
    return acc;
}

/// Batched matmul over the raw kernels (stride-0 batch = leading axis). `kind`:
/// .plain a[k,M,K]·b[k,K,N]; .trans_a a[k,K,M]ᵀ·b[k,K,N]; .trans_b a[k,M,K]·b[k,N,K]ᵀ.
fn bmm3(ctx: *ExecContext, kind: BmmKind, a: []f32, ash: [3]usize, b: []f32, bsh: [3]usize, out: []f32, osh: [3]usize) !void {
    var at = try ctx.fromBorrowedSliceRank(3, ash, a);
    defer at.deinit();
    var bt = try ctx.fromBorrowedSliceRank(3, bsh, b);
    defer bt.deinit();
    var rt = switch (kind) {
        .plain => try ctx.bmm(&at, &bt),
        .trans_a => try ctx.bmmTransA(&at, &bt),
        .trans_b => try ctx.bmmTransB(&at, &bt),
    };
    defer rt.deinit();
    std.debug.assert(rt.len() == osh[0] * osh[1] * osh[2]);
    try rt.copyTo(out);
}

/// Frame magic for MuonAdamW.saveState/loadState.
const state_magic = "NCMA1";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

comptime {
    _ = @import("optim_tests.zig");
}
