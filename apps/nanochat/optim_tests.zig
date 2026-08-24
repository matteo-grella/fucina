//! Parity tests for the nanochat MuonAdamW optimizer.
//!
//! Gate 1 (SCHEDULE): the lr-multiplier / Muon-momentum / Muon-weight-decay and
//! per-group lr against optstep_d6_schedule.json (a small always-on unit check
//! runs without goldens). Gates 2-4 (OPTSTEP): load a model at init, run
//! N optimizer steps on the fixed batch (forward→backward→step→zeroGrad, the
//! reference loop order, same batch each step), and compare every parameter and
//! optimizer-state buffer to the optstep_d{2,6}_{1,10} oracle safetensors.
//!
//! All OPTSTEP gates env-gate on NANOCHAT_PARITY and skip cleanly when the env
//! var is unset OR the goldens are absent.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const optim = @import("optim.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Config = model_mod.Config;
const Model = model_mod.Model;
const safetensors = fucina.safetensors;
const MuonAdamW = optim.MuonAdamW;

const goldens_dir = "refs/nanochat-goldens";

// ---------------------------------------------------------------------------
// Golden IO helpers (mirrors model_tests.zig; kept local to the optimizer tests).
// ---------------------------------------------------------------------------

fn readFileBytes(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const nread = try file.readStreaming(io, &.{bytes[read_len..]});
        if (nread == 0) return error.EndOfStream;
        read_len += nread;
    }
    return bytes;
}

const Batch = struct {
    b: usize,
    t: usize,
    inputs: []usize,
    targets: []isize,
    allocator: Allocator,

    fn load(allocator: Allocator, io: std.Io, path: []const u8) !Batch {
        const bytes = readFileBytes(allocator, io, path) catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        defer allocator.free(bytes);
        const b = std.mem.readInt(u32, bytes[0..4], .little);
        const t = std.mem.readInt(u32, bytes[4..8], .little);
        const nbt = @as(usize, b) * @as(usize, t);
        var off: usize = 8;
        const inputs = try allocator.alloc(usize, nbt);
        errdefer allocator.free(inputs);
        for (inputs) |*v| {
            v.* = std.mem.readInt(u32, bytes[off..][0..4], .little);
            off += 4;
        }
        const targets = try allocator.alloc(isize, nbt);
        errdefer allocator.free(targets);
        for (targets) |*v| {
            v.* = std.mem.readInt(i32, bytes[off..][0..4], .little);
            off += 4;
        }
        return .{ .b = b, .t = t, .inputs = inputs, .targets = targets, .allocator = allocator };
    }

    fn deinit(self: *Batch) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
        self.* = undefined;
    }

    fn inputRow(self: *const Batch, row: usize) []const usize {
        return self.inputs[row * self.t ..][0..self.t];
    }

    fn labelRow(self: *const Batch, allocator: Allocator, row: usize) ![]usize {
        const labels = try allocator.alloc(usize, self.t);
        for (self.targets[row * self.t ..][0..self.t], labels) |tg, *l|
            l.* = if (tg < 0) model_mod.ignore_index else @intCast(tg);
        return labels;
    }

    fn nonIgnored(self: *const Batch) usize {
        var c: usize = 0;
        for (self.targets) |tg| c += @intFromBool(tg >= 0);
        return c;
    }
};

fn oracleData(allocator: Allocator, file: *const safetensors.File, name: []const u8) ![]f32 {
    const info = try file.tensor(name);
    if (info.dtype != .F32) return error.UnexpectedDtype;
    const n = info.data.len / 4;
    const out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);
    @memcpy(std.mem.sliceAsBytes(out), info.data[0 .. n * 4]);
    return out;
}

/// Worst absolute error normalized by the oracle's peak magnitude — stable near
/// individual zeros (matches model_tests.relErr).
fn relErr(got: []const f32, want: []const f32) f32 {
    std.debug.assert(got.len == want.len);
    var max_abs: f32 = 0;
    var peak: f32 = 0;
    for (got, want) |g, w| {
        const diff = @abs(g - w);
        if (diff > max_abs) max_abs = diff;
        const aw = @abs(w);
        if (aw > peak) peak = aw;
    }
    if (peak < 1e-12) peak = 1;
    return max_abs / peak;
}

fn skipUnlessParity() !void {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
}

fn loadModelOrSkip(cfg: Config, ctx: *ExecContext, allocator: Allocator, io: std.Io, path: []const u8) !Model {
    return Model.initFromSafetensors(cfg, ctx, allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
}

fn openOracleOrSkip(allocator: Allocator, io: std.Io, path: []const u8) !safetensors.File {
    return safetensors.File.load(allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
}

/// Batch mean loss (Σ per-token loss over rows / total non-ignored) under an
/// open exec scope (gpt.py cross_entropy reduction='mean').
fn buildMeanLoss(ctx: *ExecContext, model: *const Model, batch: *const Batch, allocator: Allocator) !fucina.Tensor(.{}) {
    var acc: ?fucina.Tensor(.{}) = null;
    for (0..batch.b) |row| {
        const labels = try batch.labelRow(allocator, row);
        defer allocator.free(labels);
        const ce = try model.lossSum(ctx, batch.inputRow(row), labels);
        acc = if (acc) |a| try a.add(ctx, &ce) else ce;
    }
    const total: f32 = @floatFromInt(batch.nonIgnored());
    return acc.?.scale(ctx, 1.0 / total);
}

// ---------------------------------------------------------------------------
// Schedule oracle JSON
// ---------------------------------------------------------------------------

const SchedJson = struct {
    hyperparameters: Hyper,
    groups: []Group,
    steps: []Step,

    const Hyper = struct { weight_decay_scaled: f64, num_iterations: usize };
    const Group = struct { index: usize, kind: []const u8, initial_lr: f64 };
    const Step = struct {
        step: usize,
        lrm: f64,
        muon_momentum: f64,
        muon_weight_decay: f64,
        group_lr: []f64,
    };
};

fn parseSchedule(allocator: Allocator, io: std.Io, path: []const u8) !std.json.Parsed(SchedJson) {
    const bytes = readFileBytes(allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(bytes);
    // alloc_always so parsed strings/slices are copied into the arena — `bytes`
    // is freed on return, and the default alloc_if_needed would dangle them.
    return std.json.parseFromSlice(SchedJson, allocator, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

fn configFromSchedule(sched: *const SchedJson) optim.Config {
    var cfg: optim.Config = .{ .adamw_initial_lr = undefined, .muon_initial_lr = 0 };
    var ai: usize = 0;
    for (sched.groups) |g| {
        if (std.mem.eql(u8, g.kind, "adamw")) {
            cfg.adamw_initial_lr[ai] = g.initial_lr;
            ai += 1;
        } else if (std.mem.eql(u8, g.kind, "muon") and cfg.muon_initial_lr == 0) {
            cfg.muon_initial_lr = g.initial_lr;
        }
    }
    std.debug.assert(ai == 6);
    return cfg;
}

// ---------------------------------------------------------------------------
// 1a. SCHEDULE unit check (always-on; no goldens)
// ---------------------------------------------------------------------------

test "schedule formulas match known base_train values" {
    // From optstep_d6_schedule.json step 0 and step 9.
    try std.testing.expectApproxEqAbs(@as(f64, 0.025), optim.lrMultiplier(0, 5000), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), optim.lrMultiplier(9, 5000), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.85), optim.muonMomentum(0, 5000), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8527), optim.muonMomentum(9, 5000), 1e-9);
    const wd_scaled: f64 = 0.2349029260055059;
    try std.testing.expectApproxEqAbs(wd_scaled, optim.muonWeightDecay(0, 5000, wd_scaled), 1e-15);
    // Warmup is linear (step+1)/40 through step 39, then holds at 1.0.
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), optim.lrMultiplier(39, 5000), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), optim.lrMultiplier(40, 5000), 1e-12);
}

// ---------------------------------------------------------------------------
// 1b. SCHEDULE parity vs the JSON step rows (gated on the oracle presence)
// ---------------------------------------------------------------------------

test "NANOCHAT_PARITY: schedule matches optstep_d6_schedule.json rows" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var parsed = try parseSchedule(allocator, io, goldens_dir ++ "/optstep_d6_schedule.json");
    defer parsed.deinit();
    const sched = &parsed.value;
    const cfg = configFromSchedule(sched);
    const num_iters = sched.hyperparameters.num_iterations;
    const wd_scaled = sched.hyperparameters.weight_decay_scaled;

    var worst: f64 = 0;
    for (sched.steps) |row| {
        const s = row.step;
        try expectRel(&worst, optim.lrMultiplier(s, num_iters), row.lrm);
        try expectRel(&worst, optim.muonMomentum(s, num_iters), row.muon_momentum);
        try expectRel(&worst, optim.muonWeightDecay(s, num_iters, wd_scaled), row.muon_weight_decay);
        const lrm = optim.lrMultiplier(s, num_iters);
        for (0..6) |i| try expectRel(&worst, cfg.adamw_initial_lr[i] * lrm, row.group_lr[i]);
        for (6..row.group_lr.len) |i| try expectRel(&worst, cfg.muon_initial_lr * lrm, row.group_lr[i]);
    }
    std.debug.print("schedule parity: worst rel {e:.3} over {d} steps\n", .{ worst, sched.steps.len });
}

fn expectRel(worst: *f64, got: f64, want: f64) !void {
    const rel = if (@abs(want) > 1e-30) @abs(got - want) / @abs(want) else @abs(got - want);
    if (rel > worst.*) worst.* = rel;
    try std.testing.expect(rel <= 1e-6);
}

// ---------------------------------------------------------------------------
// 2-4. OPTSTEP parity
// ---------------------------------------------------------------------------

/// Params whose gradient is ~0 at the reference init (rmsnorm scale-invariance +
/// zero projections; see the model grad gate). AdamW's update on a near-zero grad is
/// magnitude-independent and sign(noise)-driven, so the trajectory of these
/// scalars diverges from the reference by O(lr·steps) in ABSOLUTE terms — a
/// property of the init, not the optimizer. They (and their state buffers) are
/// scored by max absolute difference; everything else by relative error.
const noise_scalars = [_][]const u8{ "resid_lambdas", "x0_lambdas", "backout_lambda" };

fn baseParamName(name: []const u8) []const u8 {
    const suffixes = [_][]const u8{ ".exp_avg_sq", ".exp_avg", ".second_momentum_buffer", ".momentum_buffer" };
    for (suffixes) |s| if (std.mem.endsWith(u8, name, s)) return name[0 .. name.len - s.len];
    return name;
}

fn isNoiseProne(name: []const u8) bool {
    const b = baseParamName(name);
    for (noise_scalars) |ns| if (std.mem.eql(u8, b, ns)) return true;
    return false;
}

fn maxAbsDiff(got: []const f32, want: []const f32) f32 {
    std.debug.assert(got.len == want.len);
    var worst: f32 = 0;
    for (got, want) |g, w| {
        const d = @abs(g - w);
        if (d > worst) worst = d;
    }
    return worst;
}

const Result = struct {
    core: f32 = 0, // worst relErr over core (matrix/embedding) params + buffers
    core_name: [80]u8 = undefined,
    core_len: usize = 0,
    noise: f32 = 0, // worst absolute diff over near-zero-grad scalars + their buffers
    noise_name: [80]u8 = undefined,
    noise_len: usize = 0,
};

const Comparator = struct {
    allocator: Allocator,
    file: *const safetensors.File,
    res: Result = .{},

    fn recordSlice(self: *Comparator, name: []const u8, got: []const f32) !void {
        const orc = try oracleData(self.allocator, self.file, name);
        defer self.allocator.free(orc);
        if (isNoiseProne(name)) {
            const a = maxAbsDiff(got, orc);
            if (a > self.res.noise) {
                self.res.noise = a;
                @memcpy(self.res.noise_name[0..name.len], name);
                self.res.noise_len = name.len;
            }
        } else {
            const e = relErr(got, orc);
            if (e > self.res.core) {
                self.res.core = e;
                @memcpy(self.res.core_name[0..name.len], name);
                self.res.core_len = name.len;
            }
        }
    }

    fn recordTensor(self: *Comparator, name: []const u8, tensor: anytype) !void {
        const info = try self.file.tensor(name);
        const got = try self.allocator.alloc(f32, info.data.len / 4);
        defer self.allocator.free(got);
        try tensor.copyTo(got);
        try self.recordSlice(name, got);
    }
};

fn runOptstep(
    allocator: Allocator,
    io: std.Io,
    model_cfg: Config,
    init_path: []const u8,
    batch_path: []const u8,
    sched_path: []const u8,
    oracle_path: []const u8,
    n_steps: usize,
) !Result {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(model_cfg, &ctx, allocator, io, init_path);
    defer model.deinit();
    var batch = try Batch.load(allocator, io, batch_path);
    defer batch.deinit();
    var parsed = try parseSchedule(allocator, io, sched_path);
    defer parsed.deinit();
    const num_iters = parsed.value.hyperparameters.num_iterations;
    const wd_scaled = parsed.value.hyperparameters.weight_decay_scaled;

    var opt = MuonAdamW.init(allocator, configFromSchedule(&parsed.value));
    defer opt.deinit();
    try opt.registerModel(&model);

    for (0..n_steps) |step| {
        {
            const scope = ctx.openExecScope();
            const mean = try buildMeanLoss(&ctx, &model, &batch, allocator);
            try mean.backward(&ctx);
            ctx.closeExecScope(scope);
        }
        try opt.step(&ctx, optim.StepSchedule.at(step, num_iters, wd_scaled));
        opt.zeroGrad();
    }

    var oracle = try openOracleOrSkip(allocator, io, oracle_path);
    defer oracle.deinit();

    var cmp = Comparator{ .allocator = allocator, .file = &oracle };

    // Parameter values.
    try checkAllParamValues(&cmp, &model);
    // AdamW optimizer-state buffers (exp_avg / exp_avg_sq per registered param).
    var name_buf: [80]u8 = undefined;
    for (&opt.adamw) |*a| {
        for (a.slots.items) |*slot| {
            const pname = slot.param.name.?;
            try cmp.recordSlice(try std.fmt.bufPrint(&name_buf, "{s}.exp_avg", .{pname}), slot.m.f32);
            try cmp.recordSlice(try std.fmt.bufPrint(&name_buf, "{s}.exp_avg_sq", .{pname}), slot.v.f32);
        }
    }
    // Muon optimizer-state buffers (momentum / second_momentum per group).
    for (opt.muon_groups.items) |*g| {
        try cmp.recordSlice(try std.fmt.bufPrint(&name_buf, "{s}.momentum_buffer", .{g.name}), g.momentum_buffer);
        try cmp.recordSlice(try std.fmt.bufPrint(&name_buf, "{s}.second_momentum_buffer", .{g.name}), g.second_momentum_buffer);
    }
    return cmp.res;
}

fn checkAllParamValues(cmp: *Comparator, model: *const Model) !void {
    try cmp.recordTensor("transformer.wte.weight", &model.wte);
    try cmp.recordTensor("lm_head.weight", &model.lm_head);
    try cmp.recordTensor("resid_lambdas", &model.resid_lambdas);
    try cmp.recordTensor("x0_lambdas", &model.x0_lambdas);
    try cmp.recordTensor("smear_gate.weight", &model.smear_gate);
    try cmp.recordTensor("smear_lambda", &model.smear_lambda);
    try cmp.recordTensor("backout_lambda", &model.backout_lambda);
    var buf: [64]u8 = undefined;
    for (model.layers, 0..) |*l, i| {
        try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_q.weight", .{i}), &l.c_q);
        try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_k.weight", .{i}), &l.c_k);
        try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_v.weight", .{i}), &l.c_v);
        try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_proj.weight", .{i}), &l.c_proj);
        try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.mlp.c_fc.weight", .{i}), &l.c_fc);
        try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.mlp.c_proj.weight", .{i}), &l.c_proj_mlp);
        if (l.ve_gate) |*g|
            try cmp.recordTensor(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.ve_gate.weight", .{i}), g);
        if (model.value_embeds[i]) |*ve|
            try cmp.recordTensor(try std.fmt.bufPrint(&buf, "value_embeds.{d}.weight", .{i}), ve);
    }
}

test "NANOCHAT_PARITY: d6 optstep-1 matches oracle" {
    try skipUnlessParity();
    const res = try runOptstep(
        std.testing.allocator,
        std.testing.io,
        Config.d6,
        goldens_dir ++ "/init_d6.safetensors",
        goldens_dir ++ "/fixed_batch_d6.bin",
        goldens_dir ++ "/optstep_d6_schedule.json",
        goldens_dir ++ "/optstep_d6_1.safetensors",
        1,
    );
    reportOptstep("d6 optstep-1", res);
    try std.testing.expect(res.core <= 2e-3); // core (matrices/embeddings + buffers)
    try std.testing.expect(res.noise <= 5e-2); // near-zero-grad scalar drift (absolute)
}

test "NANOCHAT_PARITY: d6 optstep-10 matches oracle" {
    try skipUnlessParity();
    const res = try runOptstep(
        std.testing.allocator,
        std.testing.io,
        Config.d6,
        goldens_dir ++ "/init_d6.safetensors",
        goldens_dir ++ "/fixed_batch_d6.bin",
        goldens_dir ++ "/optstep_d6_schedule.json",
        goldens_dir ++ "/optstep_d6_10.safetensors",
        10,
    );
    reportOptstep("d6 optstep-10", res);
    try std.testing.expect(res.core <= 3e-2); // Polar-Express amplifies over 10 steps
    try std.testing.expect(res.noise <= 5e-1);
}

test "NANOCHAT_PARITY: d2 optstep-1 matches oracle" {
    try skipUnlessParity();
    const res = try runOptstep(
        std.testing.allocator,
        std.testing.io,
        Config.d2,
        goldens_dir ++ "/init_d2.safetensors",
        goldens_dir ++ "/fixed_batch_d2.bin",
        goldens_dir ++ "/optstep_d2_schedule.json",
        goldens_dir ++ "/optstep_d2_1.safetensors",
        1,
    );
    reportOptstep("d2 optstep-1", res);
    try std.testing.expect(res.core <= 2e-3);
    try std.testing.expect(res.noise <= 5e-2);
}

test "NANOCHAT_PARITY: d2 optstep-10 matches oracle" {
    try skipUnlessParity();
    const res = try runOptstep(
        std.testing.allocator,
        std.testing.io,
        Config.d2,
        goldens_dir ++ "/init_d2.safetensors",
        goldens_dir ++ "/fixed_batch_d2.bin",
        goldens_dir ++ "/optstep_d2_schedule.json",
        goldens_dir ++ "/optstep_d2_10.safetensors",
        10,
    );
    reportOptstep("d2 optstep-10", res);
    try std.testing.expect(res.core <= 3e-2);
    try std.testing.expect(res.noise <= 5e-1);
}

fn reportOptstep(label: []const u8, res: Result) void {
    std.debug.print("{s}: core relErr {e:.3} @ {s} | near-zero-grad scalar drift {e:.3} (abs) @ {s}\n", .{
        label,     res.core,                         res.core_name[0..res.core_len],
        res.noise, res.noise_name[0..res.noise_len],
    });
}

// ---------------------------------------------------------------------------
// 5. saveState / loadState round-trip (gated; exercises the base-train resume seam)
// ---------------------------------------------------------------------------

test "NANOCHAT_PARITY: optimizer state round-trips through save/load" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(Config.d2, &ctx, allocator, io, goldens_dir ++ "/init_d2.safetensors");
    defer model.deinit();
    var batch = try Batch.load(allocator, io, goldens_dir ++ "/fixed_batch_d2.bin");
    defer batch.deinit();
    var parsed = try parseSchedule(allocator, io, goldens_dir ++ "/optstep_d2_schedule.json");
    defer parsed.deinit();

    var opt = MuonAdamW.init(allocator, configFromSchedule(&parsed.value));
    defer opt.deinit();
    try opt.registerModel(&model);

    // One step so the state buffers are non-trivial.
    {
        const scope = ctx.openExecScope();
        const mean = try buildMeanLoss(&ctx, &model, &batch, allocator);
        try mean.backward(&ctx);
        ctx.closeExecScope(scope);
    }
    try opt.step(&ctx, optim.StepSchedule.at(0, 5000, parsed.value.hyperparameters.weight_decay_scaled));

    // Snapshot the first Muon group's buffers, serialize, corrupt, restore.
    const g = &opt.muon_groups.items[0];
    const snap_m = try allocator.dupe(f32, g.momentum_buffer);
    defer allocator.free(snap_m);
    const snap_s = try allocator.dupe(f32, g.second_momentum_buffer);
    defer allocator.free(snap_s);

    const buf = try allocator.alloc(u8, 64 * 1024 * 1024);
    defer allocator.free(buf);
    var writer = std.Io.Writer.fixed(buf);
    try opt.saveState(&writer);
    const written = writer.buffered();

    @memset(g.momentum_buffer, 0);
    @memset(g.second_momentum_buffer, 0);

    var reader = std.Io.Reader.fixed(written);
    try opt.loadState(&reader);

    try std.testing.expectEqualSlices(f32, snap_m, g.momentum_buffer);
    try std.testing.expectEqualSlices(f32, snap_s, g.second_momentum_buffer);
}
