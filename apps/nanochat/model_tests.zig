//! Numeric parity + autograd tests for the nanochat GPT port.
//!
//! Parity tests (env-gated on NANOCHAT_PARITY, skipped cleanly when the env var
//! is unset OR the goldens under refs/nanochat-goldens/ are absent) check the Zig
//! forward/backward against the Python reference oracles. The d2 finite-diff
//! gradcheck is always-on (needs only the small d2 goldens; skips if missing) and
//! proves the autograd wiring independently of the oracle.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const testlog = @import("testlog.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Config = model_mod.Config;
const Model = model_mod.Model;
const Trace = model_mod.Trace;
const safetensors = fucina.safetensors;

const goldens_dir = "refs/nanochat-goldens";

// ---------------------------------------------------------------------------
// Golden IO helpers
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
    inputs: []usize, // [b*t]
    targets: []isize, // [b*t]
    allocator: Allocator,

    fn load(allocator: Allocator, io: std.Io, path: []const u8) !Batch {
        const bytes = try readFileBytes(allocator, io, path);
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

    fn targetRow(self: *const Batch, row: usize) []const isize {
        return self.targets[row * self.t ..][0..self.t];
    }

    /// Owned usize labels with -1 → ignore sentinel; caller frees.
    fn labelRow(self: *const Batch, allocator: Allocator, row: usize) ![]usize {
        const labels = try allocator.alloc(usize, self.t);
        for (self.targetRow(row), labels) |tg, *l| l.* = if (tg < 0) model_mod.ignore_index else @intCast(tg);
        return labels;
    }

    fn nonIgnored(self: *const Batch) usize {
        var c: usize = 0;
        for (self.targets) |tg| c += @intFromBool(tg >= 0);
        return c;
    }
};

/// Copy a named f32 tensor out of an oracle safetensors file (owned; caller frees).
fn oracleData(allocator: Allocator, file: *const safetensors.File, name: []const u8) ![]f32 {
    const info = try file.tensor(name);
    if (info.dtype != .F32) return error.UnexpectedDtype;
    const n = info.data.len / 4;
    const out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);
    @memcpy(std.mem.sliceAsBytes(out), info.data[0 .. n * 4]);
    return out;
}

/// Worst absolute error normalized by the oracle slice's peak magnitude — a
/// stable "max relative error" that does not blow up near individual zeros.
/// Suitable for forward activations (all O(0.01..15), healthy peaks).
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

/// numpy.allclose-style worst violation for GRADIENTS, which span many scales
/// and include genuine zeros (e.g. resid/x0/backout lambdas at init, whose true
/// grad is 0 because rmsnorm is scale-invariant). Returns
/// max_i |g_i - w_i| / (atol + rtol·|w_i|); a value ≤ 1 means every element is
/// within tolerance. atol absorbs the ~1e-9 FP noise of the near-zero params.
fn gradViolation(got: []const f32, want: []const f32, atol: f32, rtol: f32) f32 {
    std.debug.assert(got.len == want.len);
    var worst: f32 = 0;
    for (got, want) |g, w| {
        const v = @abs(g - w) / (atol + rtol * @abs(w));
        if (v > worst) worst = v;
    }
    return worst;
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

// ---------------------------------------------------------------------------
// 1. Forward parity (d6)
// ---------------------------------------------------------------------------

test "NANOCHAT_PARITY: d6 forward intermediates + loss match oracle" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(Config.d6, &ctx, allocator, io, goldens_dir ++ "/init_d6.safetensors");
    defer model.deinit();

    var batch = Batch.load(allocator, io, goldens_dir ++ "/fixed_batch_d6.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer batch.deinit();

    var fwd = try openOracleOrSkip(allocator, io, goldens_dir ++ "/fwd_oracle_d6.safetensors");
    defer fwd.deinit();

    const fwd_tol: f32 = 1e-4;
    var worst_all: f32 = 0;
    var worst_key: [48]u8 = undefined;
    var worst_key_len: usize = 0;

    for (0..batch.b) |row| {
        var trace = Trace.init(allocator);
        defer trace.deinit();

        const scope = ctx.openExecScope();
        _ = try model.forward(&ctx, batch.inputRow(row), &trace);
        ctx.closeExecScope(scope);

        for (trace.entries.items) |e| {
            const orc = try oracleData(allocator, &fwd, e.name);
            defer allocator.free(orc);
            const row_len = e.data.len;
            const off = row * row_len;
            const err = relErr(e.data, orc[off .. off + row_len]);
            if (err > worst_all) {
                worst_all = err;
                @memcpy(worst_key[0..e.name.len], e.name);
                worst_key_len = e.name.len;
            }
            if (err > fwd_tol) {
                std.debug.print("FWD row {d} key {s}: relerr {e:.3} > tol {e:.1}\n", .{ row, e.name, err, fwd_tol });
            }
            try std.testing.expect(err <= fwd_tol);
        }
    }
    std.debug.print("d6 forward: worst relerr {e:.3} @ {s}\n", .{ worst_all, worst_key[0..worst_key_len] });

    // Combined mean loss over all B*T non-ignored targets (gpt.py reduction=mean).
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const mean = try buildMeanLoss(&ctx, &model, &batch, allocator);
    const loss_val = try mean.item();
    const orc_loss = try oracleData(allocator, &fwd, "loss");
    defer allocator.free(orc_loss);
    const loss_rel = @abs(loss_val - orc_loss[0]) / @abs(orc_loss[0]);
    std.debug.print("d6 loss: got {d:.9} want {d:.9} relerr {e:.3}\n", .{ loss_val, orc_loss[0], loss_rel });
    try std.testing.expect(loss_rel <= 1e-4);
}

/// Build the batch mean loss (Σ per-token loss over both rows / total non-ignored)
/// under an already-open exec scope. Labels are freed immediately (CE dupes them).
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
// 2. loss_none parity (d6)
// ---------------------------------------------------------------------------

test "NANOCHAT_PARITY: d6 per-token loss matches loss_none oracle" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(Config.d6, &ctx, allocator, io, goldens_dir ++ "/init_d6.safetensors");
    defer model.deinit();
    var batch = Batch.load(allocator, io, goldens_dir ++ "/fixed_batch_d6.bin") catch return error.SkipZigTest;
    defer batch.deinit();
    var fwd = try openOracleOrSkip(allocator, io, goldens_dir ++ "/fwd_oracle_d6_none.safetensors");
    defer fwd.deinit();

    const orc = try oracleData(allocator, &fwd, "loss_none"); // [B,T]
    defer allocator.free(orc);

    var worst: f32 = 0;
    for (0..batch.b) |row| {
        const scope = ctx.openExecScope();
        const per_tok = try model.lossNone(&ctx, batch.inputRow(row), batch.targetRow(row));
        const got = try allocator.alloc(f32, batch.t);
        defer allocator.free(got);
        try per_tok.copyTo(got);
        ctx.closeExecScope(scope);

        const off = row * batch.t;
        const err = relErr(got, orc[off .. off + batch.t]);
        if (err > worst) worst = err;
    }
    std.debug.print("d6 loss_none: worst relerr {e:.3}\n", .{worst});
    try std.testing.expect(worst <= 1e-4);
}

// ---------------------------------------------------------------------------
// 3. Grad parity (d6)
// ---------------------------------------------------------------------------

test "NANOCHAT_PARITY: d6 param grads match grad oracle" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(Config.d6, &ctx, allocator, io, goldens_dir ++ "/init_d6.safetensors");
    defer model.deinit();
    var batch = Batch.load(allocator, io, goldens_dir ++ "/fixed_batch_d6.bin") catch return error.SkipZigTest;
    defer batch.deinit();
    var grad_file = try openOracleOrSkip(allocator, io, goldens_dir ++ "/grad_oracle_d6.safetensors");
    defer grad_file.deinit();

    const scope = ctx.openExecScope();
    const mean = try buildMeanLoss(&ctx, &model, &batch, allocator);
    try mean.backward(&ctx);
    ctx.closeExecScope(scope);

    // Gradients are looser than the forward (f32 attention/softmax accumulation)
    // and include genuine zeros; allclose atol=1e-6, rtol=1e-3.
    var gc = GradChecker{ .ctx = &ctx, .allocator = allocator, .file = &grad_file };
    try checkAllGrads(&gc, &model);

    std.debug.print("d6 grads: worst allclose-violation {d:.3} @ {s} (≤1 passes; atol 1e-6 rtol 1e-3)\n", .{ gc.worst, gc.worst_name[0..gc.worst_len] });
    try std.testing.expect(gc.worst <= 1.0);
}

const GradChecker = struct {
    ctx: *ExecContext,
    allocator: Allocator,
    file: *const safetensors.File,
    worst: f32 = 0,
    worst_name: [64]u8 = undefined,
    worst_len: usize = 0,

    fn check(self: *GradChecker, name: []const u8, param: anytype) !void {
        var g = (try param.grad(self.ctx)) orelse return error.NoGradient;
        defer g.deinit();
        const gd = try g.dataConst();
        const orc = try oracleData(self.allocator, self.file, name);
        defer self.allocator.free(orc);
        const v = gradViolation(gd, orc, 1e-6, 1e-3);
        if (v > self.worst) {
            self.worst = v;
            @memcpy(self.worst_name[0..name.len], name);
            self.worst_len = name.len;
        }
        if (v > 1.0) std.debug.print("GRAD {s}: allclose-violation {d:.3}\n", .{ name, v });
    }
};

fn checkAllGrads(gc: *GradChecker, model: *const Model) !void {
    try gc.check("transformer.wte.weight", &model.wte);
    try gc.check("lm_head.weight", &model.lm_head);
    try gc.check("resid_lambdas", &model.resid_lambdas);
    try gc.check("x0_lambdas", &model.x0_lambdas);
    try gc.check("smear_gate.weight", &model.smear_gate);
    try gc.check("smear_lambda", &model.smear_lambda);
    try gc.check("backout_lambda", &model.backout_lambda);
    var buf: [64]u8 = undefined;
    for (model.layers, 0..) |*l, i| {
        try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_q.weight", .{i}), &l.c_q);
        try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_k.weight", .{i}), &l.c_k);
        try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_v.weight", .{i}), &l.c_v);
        try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.c_proj.weight", .{i}), &l.c_proj);
        try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.mlp.c_fc.weight", .{i}), &l.c_fc);
        try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.mlp.c_proj.weight", .{i}), &l.c_proj_mlp);
        if (l.ve_gate) |*g| {
            try gc.check(try std.fmt.bufPrint(&buf, "transformer.h.{d}.attn.ve_gate.weight", .{i}), g);
        }
        if (model.value_embeds[i]) |*ve| {
            try gc.check(try std.fmt.bufPrint(&buf, "value_embeds.{d}.weight", .{i}), ve);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Forward-with-cache == full forward (prefill + decode self-consistency)
// ---------------------------------------------------------------------------

test "NANOCHAT_PARITY: cached prefill+decode equals full forward last logits" {
    try skipUnlessParity();
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try loadModelOrSkip(Config.d6, &ctx, allocator, io, goldens_dir ++ "/init_d6.safetensors");
    defer model.deinit();
    var batch = Batch.load(allocator, io, goldens_dir ++ "/fixed_batch_d6.bin") catch return error.SkipZigTest;
    defer batch.deinit();

    // Use a short prefix of row 0.
    const seq_len: usize = 12;
    const toks = batch.inputRow(0)[0..seq_len];

    // Full forward: capture last-position logits.
    const full_last = try allocator.alloc(f32, model.cfg.vocab_size);
    defer allocator.free(full_last);
    {
        const scope = ctx.openExecScope();
        const logits = try model.forward(&ctx, toks, null);
        const last = try logits.narrow(&ctx, .seq, seq_len - 1, 1); // [1,.vocab]
        try last.copyTo(full_last);
        ctx.closeExecScope(scope);
    }

    // Cached: prefill first seq_len-1 tokens, then decode the last one.
    const cached_last = try allocator.alloc(f32, model.cfg.vocab_size);
    defer allocator.free(cached_last);
    {
        var cache = try model_mod.Cache.init(allocator, model.cfg, model.cfg.sequence_len);
        defer cache.deinit();
        {
            const scope = ctx.openExecScope();
            _ = try model.forwardStep(&ctx, &cache, toks[0 .. seq_len - 1], 0);
            ctx.closeExecScope(scope);
        }
        {
            const scope = ctx.openExecScope();
            const dec = try model.forwardStep(&ctx, &cache, toks[seq_len - 1 ..], seq_len - 1); // [1,.vocab]
            try dec.copyTo(cached_last);
            ctx.closeExecScope(scope);
        }
    }

    const err = relErr(cached_last, full_last);
    std.debug.print("cache self-consistency: worst relerr {e:.3}\n", .{err});
    try std.testing.expect(err <= 1e-4);
}

// ---------------------------------------------------------------------------
// 4. Finite-difference gradcheck (d2) — always-on, oracle-independent
// ---------------------------------------------------------------------------

test "d2 finite-difference gradcheck proves autograd wiring" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = loadModelOrSkip(Config.d2, &ctx, allocator, io, goldens_dir ++ "/init_d2.safetensors") catch |err| switch (err) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return err,
    };
    defer model.deinit();
    var batch = Batch.load(allocator, io, goldens_dir ++ "/fixed_batch_d2.bin") catch return error.SkipZigTest;
    defer batch.deinit();

    // The reference init zeros the attn/MLP output projections and uses a tiny
    // lm_head (std 1e-3); combined with rmsnorm scale-invariance that makes most
    // gradients ~0 at init — a weak wiring test. Give those weights real
    // magnitude so the gradcheck exercises the attn/MLP/rope/backout backward.
    try liveify(&model);

    // Analytic grads from one combined backward.
    {
        const scope = ctx.openExecScope();
        const mean = try buildMeanLoss(&ctx, &model, &batch, allocator);
        try mean.backward(&ctx);
        ctx.closeExecScope(scope);
    }

    // A handful of scalar entries across several params. For the matrices/
    // embeddings we probe the MAX-|grad| entry so the finite difference sees a
    // meaningful gradient (index picked from the analytic grad just computed).
    const PtrKind = enum { smear_lambda, backout_lambda, resid0, x00, cq0, cfc0, wte_used, lm_head };
    const Probe = struct { name: []const u8, param: PtrKind, idx: usize };
    const probes = [_]Probe{
        .{ .name = "smear_lambda[0]", .param = .smear_lambda, .idx = 0 },
        .{ .name = "backout_lambda[0]", .param = .backout_lambda, .idx = 0 },
        .{ .name = "resid_lambdas[0]", .param = .resid0, .idx = 0 },
        .{ .name = "x0_lambdas[0]", .param = .x00, .idx = 0 },
        .{ .name = "h0.c_q[argmax]", .param = .cq0, .idx = try argmaxAbsGrad(&ctx, &model.layers[0].c_q) },
        .{ .name = "h0.c_fc[argmax]", .param = .cfc0, .idx = try argmaxAbsGrad(&ctx, &model.layers[0].c_fc) },
        .{ .name = "wte[argmax]", .param = .wte_used, .idx = try argmaxAbsGrad(&ctx, &model.wte) },
        .{ .name = "lm_head[argmax]", .param = .lm_head, .idx = try argmaxAbsGrad(&ctx, &model.lm_head) },
    };

    const h: f32 = 5e-3;
    var worst: f32 = 0;
    for (probes) |p| {
        const store = switch (p.param) {
            .smear_lambda => try model.smear_lambda.value.dataChecked(),
            .backout_lambda => try model.backout_lambda.value.dataChecked(),
            .resid0 => try model.resid_lambdas.value.dataChecked(),
            .x00 => try model.x0_lambdas.value.dataChecked(),
            .cq0 => try model.layers[0].c_q.value.dataChecked(),
            .cfc0 => try model.layers[0].c_fc.value.dataChecked(),
            .wte_used => try model.wte.value.dataChecked(),
            .lm_head => try model.lm_head.value.dataChecked(),
        };
        const analytic = switch (p.param) {
            .smear_lambda => try gradAt(&ctx, &model.smear_lambda, p.idx),
            .backout_lambda => try gradAt(&ctx, &model.backout_lambda, p.idx),
            .resid0 => try gradAt(&ctx, &model.resid_lambdas, p.idx),
            .x00 => try gradAt(&ctx, &model.x0_lambdas, p.idx),
            .cq0 => try gradAt(&ctx, &model.layers[0].c_q, p.idx),
            .cfc0 => try gradAt(&ctx, &model.layers[0].c_fc, p.idx),
            .wte_used => try gradAt(&ctx, &model.wte, p.idx),
            .lm_head => try gradAt(&ctx, &model.lm_head, p.idx),
        };

        const orig = store[p.idx];
        store[p.idx] = orig + h;
        const lp = try fdLoss(&ctx, &model, &batch, allocator);
        store[p.idx] = orig - h;
        const lm = try fdLoss(&ctx, &model, &batch, allocator);
        store[p.idx] = orig;
        const fd = (lp - lm) / (2 * h);

        const abs_err = @abs(fd - analytic);
        const rel_err = abs_err / @max(@abs(analytic), @as(f32, 1e-6));
        if (rel_err > worst) worst = rel_err;
        testlog.print("FD {s}: analytic {d:.6} fd {d:.6} rel {e:.3} abs {e:.3}\n", .{ p.name, analytic, fd, rel_err, abs_err });
        // Central-difference at f32 has ~ULP cancellation noise in the loss
        // difference; near-zero gradients (e.g. smear_lambda at init 0) are
        // dominated by it, so accept small ABSOLUTE error too. A wiring bug
        // would miss by orders of magnitude, not by 5e-4.
        try std.testing.expect(rel_err <= 5e-2 or abs_err <= 5e-4);
    }
    testlog.print("d2 finite-diff: worst relerr {e:.3}\n", .{worst});
}

/// Overwrite the zero-initialized projections and tiny lm_head with small
/// deterministic values so the finite-diff gradcheck sees non-trivial gradients
/// through every path (attn c_proj, mlp c_proj, rope, backout, resid mix).
fn liveify(model: *Model) !void {
    var state: u64 = 0x9E3779B97F4A7C15;
    const rand = struct {
        fn next(s: *u64, scale: f32) f32 {
            s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
            const bits: u32 = @truncate(s.* >> 33);
            const u = @as(f32, @floatFromInt(bits)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
            return (u * 2 - 1) * scale;
        }
    }.next;
    for (model.layers) |*l| {
        for (try l.c_proj.value.dataChecked()) |*w| w.* = rand(&state, 0.06);
        for (try l.c_proj_mlp.value.dataChecked()) |*w| w.* = rand(&state, 0.06);
    }
    for (try model.lm_head.value.dataChecked()) |*w| w.* = rand(&state, 0.03);
    (try model.smear_lambda.value.dataChecked())[0] = 0.5;
    (try model.backout_lambda.value.dataChecked())[0] = 0.3;
}

fn gradAt(ctx: *ExecContext, param: anytype, idx: usize) !f32 {
    var g = (try param.grad(ctx)) orelse return error.NoGradient;
    defer g.deinit();
    return (try g.dataConst())[idx];
}

fn argmaxAbsGrad(ctx: *ExecContext, param: anytype) !usize {
    var g = (try param.grad(ctx)) orelse return error.NoGradient;
    defer g.deinit();
    const gd = try g.dataConst();
    var best: usize = 0;
    var best_mag: f32 = 0;
    for (gd, 0..) |v, i| {
        const m = @abs(v);
        if (m > best_mag) {
            best_mag = m;
            best = i;
        }
    }
    return best;
}

fn fdLoss(ctx: *ExecContext, model: *const Model, batch: *const Batch, allocator: Allocator) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const mean = try buildMeanLoss(ctx, model, batch, allocator);
    return mean.item();
}
