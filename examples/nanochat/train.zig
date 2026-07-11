//! nanochat base pretraining loop (karpathy/nanochat → Fucina), CPU fp32 parity
//! port. Reproduces `refs/nanochat/scripts/base_train.py`: the
//! hyperparameter derivation (dmodel/batch LR scaling, weight-decay scaling,
//! scaling-law token horizon), the training loop (grad accumulation over
//! device-batch sequences, per-step Muon/AdamW schedules, one optimizer step per
//! iteration), bits-per-byte evaluation (`nanochat/loss_eval.py evaluate_bpb`),
//! greedy sample previews, and a three-file directory checkpoint
//! (model.safetensors + optimizer.fucina + trainer_state.json).
//!
//! Loss normalization (documented for parity): base_train.py computes, per
//! micro-batch, `loss = F.cross_entropy(logits(B*T,V), targets(B*T), mean)` and
//! calls `(loss/grad_accum).backward()`. This port runs one sequence at a time
//! (the attention/CE kernels are single-sequence), so per
//! micro-batch it SUMS each sequence's summed cross-entropy over all
//! non-ignored targets, divides by the micro-batch's total non-ignored token
//! count (= the F.cross_entropy 'mean'), then backward that mean divided by
//! grad_accum. Since base pretraining has no ignore_index (-1) targets, the
//! per-micro token count is exactly B*T and this reproduces the reference mean
//! bitwise up to float summation order. The recorded/logged loss is the LAST
//! micro-batch's pre-step mean (base_train.py logs `train_loss.item()`).
//!
//! Uses only the sibling example modules + the public Fucina
//! facade (no fucina.internal). The base_train.py schedule flags --warmup-steps /
//! --warmdown-ratio / --final-lr-frac thread through optim.zig's ScheduleParams
//! into the per-step lr/momentum schedule; their defaults (40 / 0.65 / 0.05)
//! reproduce the d6 acceptance config bit-for-bit.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const optim_mod = @import("optim.zig");
const data_mod = @import("data.zig");
const tok_mod = @import("tokenizer.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const Model = model_mod.Model;
const Config = model_mod.Config;
const MuonAdamW = optim_mod.MuonAdamW;
const StepSchedule = optim_mod.StepSchedule;
const Tokenizer = tok_mod.Tokenizer;
const training_checkpoint = fucina.training_checkpoint;
const state_dict = fucina.state_dict;

const ignore_index = model_mod.ignore_index;
const ln2: f64 = 0.6931471805599453;

/// base_train.py B_REF = 2**19 (the tuned reference batch size at d12).
const b_ref: f64 = 524288.0;

// ===========================================================================
// Hyperparameter derivation (base_train.py + gpt.py setup_optimizer)
// ===========================================================================

pub const Hparams = struct {
    model_dim: usize,
    num_heads: usize,
    padded_vocab: usize,
    batch_lr_scale: f64,
    dmodel_lr_scale: f64,
    num_scaling_params: usize,
    d12_scaling_params: usize,
    target_tokens: usize,
    d_ref: f64,
    weight_decay_scaled: f64,
    grad_accum: usize,
    opt_config: optim_mod.Config,
};

/// num_scaling_params = transformer_matrices + lm_head (gpt.py num_scaling_params
/// + base_train.get_scaling_params). transformer_matrices sums, per layer, the
/// c_q/c_k/c_v/c_proj + mlp c_fc/c_proj weight counts (+ ve_gate where present);
/// lm_head uses the PADDED vocab.
fn numScalingParams(n_layer: usize, n_head: usize, n_kv_head: usize, n_embd: usize, head_dim: usize, padded_vocab: usize) usize {
    const qo = n_head * head_dim;
    const kvo = n_kv_head * head_dim;
    const ff = 4 * n_embd;
    var tm: usize = 0;
    for (0..n_layer) |i| {
        tm += qo * n_embd + kvo * n_embd + kvo * n_embd + n_embd * qo + ff * n_embd + n_embd * ff;
        if (model_mod.hasVeAt(i, n_layer)) tm += n_kv_head * 12;
    }
    return tm + padded_vocab * n_embd;
}

/// gpt.py asserts every window_pattern character is S or L
/// (_compute_window_sizes); a mistyped pattern must fail fast, not silently
/// train full-context layers.
fn validateWindowPattern(pattern: []const u8) !void {
    if (pattern.len == 0) return error.InvalidWindowPattern;
    for (pattern) |c| {
        const u = std.ascii.toUpper(c);
        if (u != 'S' and u != 'L') return error.InvalidWindowPattern;
    }
}

pub const DeriveArgs = struct {
    depth: usize,
    aspect_ratio: usize,
    head_dim: usize,
    vocab_size: usize,
    total_batch_size: usize,
    device_batch_size: usize,
    max_seq_len: usize,
    embedding_lr: f64,
    unembedding_lr: f64,
    matrix_lr: f64,
    scalar_lr: f64,
    weight_decay: f64,
};

/// Port of base_train.py's scaling-law + muP derivation with gpt.py's
/// setup_optimizer group LRs folded in. Cross-checked against
/// optstep_d6_schedule.json / loss_trace_d6.meta.json:
/// batch_lr_scale 0.17677669, dmodel_lr_scale 1.41421356,
/// weight_decay_scaled 0.23490293, num_scaling_params 23199960.
pub fn deriveHparams(a: DeriveArgs) !Hparams {
    const base_dim = a.depth * a.aspect_ratio;
    const model_dim = ((base_dim + a.head_dim - 1) / a.head_dim) * a.head_dim;
    const num_heads = model_dim / a.head_dim;
    const padded_vocab = model_mod.paddedVocab(a.vocab_size);

    // d12 reference (base_train.build_model_meta(12): same aspect_ratio/head_dim).
    const d12_base = 12 * a.aspect_ratio;
    const d12_dim = ((d12_base + a.head_dim - 1) / a.head_dim) * a.head_dim;
    const d12_heads = d12_dim / a.head_dim;

    const num_scaling = numScalingParams(a.depth, num_heads, num_heads, model_dim, a.head_dim, padded_vocab);
    const d12_scaling = numScalingParams(12, d12_heads, d12_heads, d12_dim, a.head_dim, padded_vocab);

    const target_tokens: usize = @intFromFloat(12.0 * @as(f64, @floatFromInt(num_scaling)));
    const d_ref: f64 = 12.0 * @as(f64, @floatFromInt(d12_scaling));

    const batch_lr_scale = std.math.sqrt(@as(f64, @floatFromInt(a.total_batch_size)) / b_ref);
    const dmodel_lr_scale = std.math.pow(f64, @as(f64, @floatFromInt(model_dim)) / 768.0, -0.5);
    const wd_scaled = a.weight_decay * batch_lr_scale * (d_ref / @as(f64, @floatFromInt(target_tokens)));

    const per_fwdbwd = a.device_batch_size * a.max_seq_len;
    if (per_fwdbwd == 0 or a.total_batch_size % per_fwdbwd != 0) return error.BadTotalBatchSize;
    const grad_accum = a.total_batch_size / per_fwdbwd;

    // gpt.py setup_optimizer group LRs, in optim.zig's AdamW group order
    // (0=lm_head, 1=wte, 2=value_embeds, 3=resid, 4=x0, 5=smear). base_train
    // passes each CLI lr pre-multiplied by batch_lr_scale; gpt.py additionally
    // scales the embedding/head groups by dmodel_lr_scale, resid by 0.01, and
    // hardcodes the smear group to 0.2.
    var opt_config: optim_mod.Config = .{ .adamw_initial_lr = undefined, .muon_initial_lr = 0 };
    opt_config.adamw_initial_lr[0] = a.unembedding_lr * batch_lr_scale * dmodel_lr_scale;
    opt_config.adamw_initial_lr[1] = a.embedding_lr * batch_lr_scale * dmodel_lr_scale;
    opt_config.adamw_initial_lr[2] = a.embedding_lr * batch_lr_scale * dmodel_lr_scale * 0.5;
    opt_config.adamw_initial_lr[3] = a.scalar_lr * batch_lr_scale * 0.01;
    opt_config.adamw_initial_lr[4] = a.scalar_lr * batch_lr_scale;
    opt_config.adamw_initial_lr[5] = 0.2;
    opt_config.muon_initial_lr = a.matrix_lr * batch_lr_scale;

    return .{
        .model_dim = model_dim,
        .num_heads = num_heads,
        .padded_vocab = padded_vocab,
        .batch_lr_scale = batch_lr_scale,
        .dmodel_lr_scale = dmodel_lr_scale,
        .num_scaling_params = num_scaling,
        .d12_scaling_params = d12_scaling,
        .target_tokens = target_tokens,
        .d_ref = d_ref,
        .weight_decay_scaled = wd_scaled,
        .grad_accum = grad_accum,
        .opt_config = opt_config,
    };
}

// ===========================================================================
// Training step core
// ===========================================================================

/// Process ONE micro-batch (b sequences of length t): build the mean cross-
/// entropy over the micro-batch's non-ignored targets, backward `mean/grad_accum`
/// (grads accumulate into the leaf params across micro-batches — TRAINING.md
/// §4), and return the pre-step mean loss for logging. `inputs`/`targets` are
/// row-major [b*t] i32 (targets < 0 = ignore). A per-sequence exec scope wraps
/// each row's forward+backward (see the body comment), while the ids/labels
/// buffers outlive every backward. An all-ignored micro-batch has no gradient
/// and would poison the optimizer step, so it errors instead of training on
/// nothing.
pub fn microStep(
    ctx: *ExecContext,
    model: anytype,
    inputs: []const i32,
    targets: []const i32,
    b: usize,
    t: usize,
    grad_accum: usize,
    allocator: Allocator,
) !f32 {
    const nbt = b * t;
    const ids = try allocator.alloc(usize, nbt);
    defer allocator.free(ids);
    const labels = try allocator.alloc(usize, nbt);
    defer allocator.free(labels);

    var valid: usize = 0;
    for (0..nbt) |i| {
        ids[i] = @intCast(inputs[i]);
        if (targets[i] < 0) {
            labels[i] = ignore_index;
        } else {
            labels[i] = @intCast(targets[i]);
            valid += 1;
        }
    }
    if (valid == 0) return error.NoValidTargets;

    // Per-sequence scope + backward: each row's summed CE (reduction=.sum) is
    // scaled by 1/valid (the F.cross_entropy 'mean' over the micro-batch, then
    // 1/grad_accum) and backwarded before the next row's forward, so at most
    // ONE sequence's graph is alive at a time and its buffers recycle through
    // the pool row to row. The reference's batched backward differs only in
    // LEAF gradient accumulation order (drift-budget class, covered by the
    // loss-trace gates); the logged mean is the plain f32 row-order sum of the
    // per-row CE values divided by `valid`.
    const inv_valid = 1.0 / @as(f32, @floatFromInt(valid));
    var ce_sum: f32 = 0;
    for (0..b) |row| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const ce = try model.lossSum(ctx, ids[row * t ..][0..t], labels[row * t ..][0..t]);
        ce_sum += try ce.item();
        var scaled = try ce.scale(ctx, inv_valid);
        if (grad_accum == 1) {
            try scaled.backward(ctx);
        } else {
            var scaled2 = try scaled.scale(ctx, 1.0 / @as(f32, @floatFromInt(grad_accum)));
            try scaled2.backward(ctx);
        }
    }
    return ce_sum * inv_valid;
}

// ===========================================================================
// bits-per-byte eval (loss_eval.py evaluate_bpb)
// ===========================================================================

pub const BpbAccum = struct { nats: f64 = 0, bytes: u64 = 0 };

/// Accumulate one micro-batch into the bpb running totals: for each non-ignored
/// target with token_bytes[target] > 0, add its per-token nats; add its byte
/// count to the denominator (special/0-byte tokens add 0 bytes, contribute no
/// nats). Forward-only (no backward) under a per-sequence exec scope.
pub fn accumBpb(
    ctx: *ExecContext,
    model: anytype,
    token_bytes: []const u32,
    inputs: []const i32,
    targets: []const i32,
    b: usize,
    t: usize,
    allocator: Allocator,
    acc: *BpbAccum,
) !void {
    const ids = try allocator.alloc(usize, t);
    defer allocator.free(ids);
    const tgt = try allocator.alloc(isize, t);
    defer allocator.free(tgt);
    const per_tok = try allocator.alloc(f32, t);
    defer allocator.free(per_tok);

    for (0..b) |row| {
        for (0..t) |c| {
            ids[c] = @intCast(inputs[row * t + c]);
            tgt[c] = @intCast(targets[row * t + c]);
        }
        {
            // Forward-only: skip gradient recording entirely (values identical;
            // no backward graph is built for eval sequences).
            var ng = fucina.noGrad();
            defer ng.close();
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            const loss_none = try model.lossNone(ctx, ids, tgt);
            try loss_none.copyTo(per_tok);
        }
        for (0..t) |c| {
            const target = targets[row * t + c];
            if (target < 0) continue; // ignore_index: no nats, no bytes
            const nb: u64 = if (@as(usize, @intCast(target)) < token_bytes.len) token_bytes[@intCast(target)] else 0;
            acc.bytes += nb;
            if (nb > 0) acc.nats += per_tok[c];
        }
    }
}

pub fn bpbValue(acc: BpbAccum) f64 {
    if (acc.bytes == 0) return std.math.inf(f64);
    return acc.nats / (ln2 * @as(f64, @floatFromInt(acc.bytes)));
}

// ===========================================================================
// Checkpoint (model.safetensors + optimizer.fucina + trainer_state.json)
// ===========================================================================

pub const NcState = struct {
    step: u64,
    seed: u64,
    num_iterations: u64,
    weight_decay_scaled: f64,
    pq_idx: i64,
    rg_idx: i64,
    epoch: i64,
};

fn ModelSaveCtx(comptime ModelT: type) type {
    return struct {
        model: *const ModelT,
        allocator: Allocator,

        fn write(self: @This(), w: *std.Io.Writer) !void {
            var arena_state = std.heap.ArenaAllocator.init(self.allocator);
            defer arena_state.deinit();
            const arena = arena_state.allocator();

            var entries: std.ArrayList(state_dict.NamedTensor) = .empty;
            const m = self.model;
            try entries.append(arena, try state_dict.NamedTensor.of("transformer.wte.weight", &m.wte));
            try entries.append(arena, try state_dict.NamedTensor.of("lm_head.weight", &m.lm_head));
            try entries.append(arena, try state_dict.NamedTensor.of("resid_lambdas", &m.resid_lambdas));
            try entries.append(arena, try state_dict.NamedTensor.of("x0_lambdas", &m.x0_lambdas));
            try entries.append(arena, try state_dict.NamedTensor.of("smear_gate.weight", &m.smear_gate));
            try entries.append(arena, try state_dict.NamedTensor.of("smear_lambda", &m.smear_lambda));
            try entries.append(arena, try state_dict.NamedTensor.of("backout_lambda", &m.backout_lambda));
            for (m.layers, 0..) |*l, i| {
                try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.attn.c_q.weight", .{i}), &l.c_q));
                try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.attn.c_k.weight", .{i}), &l.c_k));
                try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.attn.c_v.weight", .{i}), &l.c_v));
                try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.attn.c_proj.weight", .{i}), &l.c_proj));
                try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.mlp.c_fc.weight", .{i}), &l.c_fc));
                try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.mlp.c_proj.weight", .{i}), &l.c_proj_mlp));
                if (l.ve_gate) |*g| {
                    try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "transformer.h.{d}.attn.ve_gate.weight", .{i}), g));
                }
            }
            for (m.value_embeds, 0..) |*ve, i| {
                if (ve.*) |*t| {
                    try entries.append(arena, try state_dict.NamedTensor.of(try std.fmt.allocPrint(arena, "value_embeds.{d}.weight", .{i}), t));
                }
            }
            try state_dict.saveStateDict(self.allocator, w, entries.items);
        }
    };
}

const OptSaveCtx = struct {
    opt: *MuonAdamW,
    fn write(self: OptSaveCtx, w: *std.Io.Writer) !void {
        try self.opt.saveState(w);
    }
};

const StateSaveCtx = struct {
    state: NcState,
    fn write(self: StateSaveCtx, w: *std.Io.Writer) !void {
        const s = self.state;
        try w.print(
            "{{\n  \"format\": \"nanochat.base_train\",\n  \"version\": 1,\n  \"step\": {d},\n  \"seed\": {d},\n  \"num_iterations\": {d},\n  \"weight_decay_scaled\": {d},\n  \"pq_idx\": {d},\n  \"rg_idx\": {d},\n  \"epoch\": {d}\n}}\n",
            .{ s.step, s.seed, s.num_iterations, s.weight_decay_scaled, s.pq_idx, s.rg_idx, s.epoch },
        );
    }
};

/// Byte-stream file copy (std.Io.Dir exposes no copyFile).
fn copyFileStreaming(io: std.Io, from: []const u8, to: []const u8) !void {
    var src = try std.Io.Dir.cwd().openFile(io, from, .{});
    defer src.close(io);
    var dst = try std.Io.Dir.cwd().createFile(io, to, .{});
    defer dst.close(io);
    var wbuf: [64 * 1024]u8 = undefined;
    var writer = dst.writer(io, &wbuf);
    var rbuf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = src.readStreaming(io, &.{&rbuf}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (n == 0) break;
        try writer.interface.writeAll(rbuf[0..n]);
    }
    try writer.interface.flush();
}

/// Preserve the previous complete checkpoint as `<dir>/prev/` before a new
/// save overwrites the directory, so a crash mid-save never destroys the last
/// good state. Ordering keeps at least one complete, consistent checkpoint at
/// every instant: prev's sentinel is deleted first (invalidating stale prev),
/// payload is copied, prev's sentinel is copied LAST (prev becomes complete),
/// and only then does beginSave invalidate the main dir. (base_train.py keeps
/// per-step files instead; a rotating pair bounds disk use.)
fn rotatePrevCheckpoint(allocator: Allocator, io: std.Io, dir: []const u8) !void {
    const sentinel = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.trainer_state_file);
    defer allocator.free(sentinel);
    var probe = std.Io.Dir.cwd().openFile(io, sentinel, .{}) catch return; // nothing complete to preserve
    probe.close(io);

    const prev_dir = try std.fs.path.join(allocator, &.{ dir, "prev" });
    defer allocator.free(prev_dir);
    try std.Io.Dir.cwd().createDirPath(io, prev_dir);
    const prev_sentinel = try training_checkpoint.pathJoin(allocator, prev_dir, training_checkpoint.trainer_state_file);
    defer allocator.free(prev_sentinel);
    std.Io.Dir.cwd().deleteFile(io, prev_sentinel) catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    for ([_][]const u8{ training_checkpoint.model_state_file, training_checkpoint.optimizer_state_file }) |name| {
        const from = try training_checkpoint.pathJoin(allocator, dir, name);
        defer allocator.free(from);
        const to = try training_checkpoint.pathJoin(allocator, prev_dir, name);
        defer allocator.free(to);
        try copyFileStreaming(io, from, to);
    }
    try copyFileStreaming(io, sentinel, prev_sentinel);
}

/// Resolve which directory to resume from: the main checkpoint if its commit
/// sentinel is present, else the `prev/` rotation (a save was interrupted).
fn resumeSourceDir(allocator: Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const sentinel = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.trainer_state_file);
    defer allocator.free(sentinel);
    if (std.Io.Dir.cwd().openFile(io, sentinel, .{})) |f| {
        var file = f;
        file.close(io);
        return allocator.dupe(u8, dir);
    } else |_| {}
    return std.fs.path.join(allocator, &.{ dir, "prev" });
}

/// Write the three checkpoint files, trainer_state.json LAST as the commit
/// sentinel (training_checkpoint.beginSave deletes it first). The previous
/// complete checkpoint is rotated into `<dir>/prev/` beforehand.
pub fn saveCheckpoint(allocator: Allocator, io: std.Io, dir: []const u8, model: anytype, opt: *MuonAdamW, state: NcState) !void {
    try rotatePrevCheckpoint(allocator, io, dir);
    try training_checkpoint.beginSave(allocator, io, dir);

    const model_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.model_state_file);
    defer allocator.free(model_path);
    const SaveCtx = ModelSaveCtx(@typeInfo(@TypeOf(model)).pointer.child);
    try training_checkpoint.writeFileAtomic(io, model_path, SaveCtx{ .model = model, .allocator = allocator }, SaveCtx.write);

    const opt_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.optimizer_state_file);
    defer allocator.free(opt_path);
    try training_checkpoint.writeFileAtomic(io, opt_path, OptSaveCtx{ .opt = opt }, OptSaveCtx.write);

    const state_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.trainer_state_file);
    defer allocator.free(state_path);
    try training_checkpoint.writeFileAtomic(io, state_path, StateSaveCtx{ .state = state }, StateSaveCtx.write);
}

pub fn loadOptimizerState(allocator: Allocator, io: std.Io, dir: []const u8, opt: *MuonAdamW) !void {
    const opt_path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.optimizer_state_file);
    defer allocator.free(opt_path);
    var file = try std.Io.Dir.cwd().openFile(io, opt_path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buf);
    try opt.loadState(&reader.interface);
}

fn loadTrainerJson(allocator: Allocator, io: std.Io, dir: []const u8) !NcState {
    const path = try training_checkpoint.pathJoin(allocator, dir, training_checkpoint.trainer_state_file);
    defer allocator.free(path);
    const bytes = try readFileBytes(allocator, io, path);
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    return .{
        .step = @intCast(o.get("step").?.integer),
        .seed = @intCast(o.get("seed").?.integer),
        .num_iterations = @intCast(o.get("num_iterations").?.integer),
        .weight_decay_scaled = jsonF64(o.get("weight_decay_scaled").?),
        .pq_idx = o.get("pq_idx").?.integer,
        .rg_idx = o.get("rg_idx").?.integer,
        .epoch = o.get("epoch").?.integer,
    };
}

fn jsonF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0,
        else => 0,
    };
}

// ===========================================================================
// CLI: base-train
// ===========================================================================

const Cli = struct {
    depth: usize = 6,
    aspect_ratio: usize = 64,
    head_dim: usize = 64,
    window_pattern: []const u8 = "L",
    max_seq_len: usize = 512,
    device_batch_size: usize = 32,
    total_batch_size: usize = 16384,
    num_iterations: usize = 5000,
    embedding_lr: f64 = 0.3,
    unembedding_lr: f64 = 0.008,
    matrix_lr: f64 = 0.02,
    scalar_lr: f64 = 0.5,
    weight_decay: f64 = 0.28,
    warmup_steps: usize = 40,
    warmdown_ratio: f64 = 0.65,
    final_lr_frac: f64 = 0.05,
    eval_every: i64 = 500,
    eval_tokens: usize = 524288,
    sample_every: i64 = 1000,
    save_every: i64 = 2500,
    seed: u64 = 42,
    data: ?[]const u8 = null,
    val_data: ?[]const u8 = null,
    tokenizer: ?[]const u8 = null,
    out: ?[]const u8 = null,
    init_from: []const u8 = "",
    resume_dir: ?[]const u8 = null,
    /// Storage dtype of the trained transformer matrices (`--dtype f32|bf16`;
    /// embeddings and scalars stay f32 — see ModelOf). Must match the
    /// checkpoint when loading with --init-from/--resume.
    dtype: fucina.DType = .f32,
};

fn parseDtypeFlag(v: []const u8) !fucina.DType {
    if (std.mem.eql(u8, v, "f32")) return .f32;
    if (std.mem.eql(u8, v, "bf16")) return .bf16;
    return error.InvalidArgument;
}

const ArgWalk = struct {
    args: []const []const u8,
    i: usize = 0,

    fn value(self: *ArgWalk, name: []const u8) ?[]const u8 {
        const a = self.args[self.i];
        if (std.mem.eql(u8, a, name)) {
            if (self.i + 1 >= self.args.len) return null;
            self.i += 1;
            return self.args[self.i];
        }
        if (a.len > name.len and std.mem.startsWith(u8, a, name) and a[name.len] == '=') {
            return a[name.len + 1 ..];
        }
        return null;
    }
};

fn parseCli(args: []const []const u8) !Cli {
    var cli: Cli = .{};
    var w = ArgWalk{ .args = args };
    while (w.i < args.len) : (w.i += 1) {
        if (w.value("--depth")) |v| {
            cli.depth = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--aspect-ratio")) |v| {
            cli.aspect_ratio = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--head-dim")) |v| {
            cli.head_dim = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--window-pattern")) |v| {
            cli.window_pattern = v;
        } else if (w.value("--max-seq-len")) |v| {
            cli.max_seq_len = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--device-batch-size")) |v| {
            cli.device_batch_size = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--total-batch-size")) |v| {
            cli.total_batch_size = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--num-iterations")) |v| {
            cli.num_iterations = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--embedding-lr")) |v| {
            cli.embedding_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--unembedding-lr")) |v| {
            cli.unembedding_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--matrix-lr")) |v| {
            cli.matrix_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--scalar-lr")) |v| {
            cli.scalar_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--weight-decay")) |v| {
            cli.weight_decay = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--warmup-steps")) |v| {
            cli.warmup_steps = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--warmdown-ratio")) |v| {
            cli.warmdown_ratio = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--final-lr-frac")) |v| {
            cli.final_lr_frac = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--eval-every")) |v| {
            cli.eval_every = try std.fmt.parseInt(i64, v, 10);
        } else if (w.value("--eval-tokens")) |v| {
            cli.eval_tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--sample-every")) |v| {
            cli.sample_every = try std.fmt.parseInt(i64, v, 10);
        } else if (w.value("--save-every")) |v| {
            cli.save_every = try std.fmt.parseInt(i64, v, 10);
        } else if (w.value("--seed")) |v| {
            cli.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (w.value("--data")) |v| {
            cli.data = v;
        } else if (w.value("--val-data")) |v| {
            cli.val_data = v;
        } else if (w.value("--tokenizer")) |v| {
            cli.tokenizer = v;
        } else if (w.value("--out")) |v| {
            cli.out = v;
        } else if (w.value("--init-from")) |v| {
            cli.init_from = v;
        } else if (w.value("--resume")) |v| {
            cli.resume_dir = v;
        } else if (w.value("--dtype")) |v| {
            cli.dtype = try parseDtypeFlag(v);
        } else {
            return error.UnknownArgument;
        }
    }
    return cli;
}

fn buildConfig(cli: Cli, hp: Hparams, vocab_size: usize) Config {
    return .{
        .sequence_len = cli.max_seq_len,
        .vocab_size = vocab_size,
        .n_layer = cli.depth,
        .n_head = hp.num_heads,
        .n_kv_head = hp.num_heads,
        .n_embd = hp.model_dim,
        .window_pattern = cli.window_pattern,
    };
}

pub fn runBaseTrain(io: std.Io, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const cli = parseCli(args) catch |err| {
        try stdout.print("base-train: argument error: {s}\n", .{@errorName(err)});
        return err;
    };
    if (cli.data == null) {
        try stdout.writeAll("base-train: --data <NCDOC> is required\n");
        return error.InvalidArgument;
    }
    if (cli.tokenizer == null) {
        try stdout.writeAll("base-train: --tokenizer <tokenizer.bin> is required\n");
        return error.InvalidArgument;
    }
    if (cli.out == null) {
        try stdout.writeAll("base-train: --out <dir> is required\n");
        return error.InvalidArgument;
    }

    try validateWindowPattern(cli.window_pattern);

    // The matrix dtype is a comptime property of the model type: dispatch
    // once into the generic flow.
    switch (cli.dtype) {
        .bf16 => try baseTrainWith(model_mod.ModelOf(.bf16), io, stdout, cli),
        else => try baseTrainWith(Model, io, stdout, cli),
    }
}

fn baseTrainWith(comptime ModelT: type, io: std.Io, stdout: *std.Io.Writer, cli: Cli) !void {
    const allocator = std.heap.smp_allocator;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var tokenizer = try Tokenizer.loadBin(allocator, io, cli.tokenizer.?);
    defer tokenizer.deinit();
    const vocab_size = tokenizer.n_vocab;

    var derive_args: DeriveArgs = .{
        .depth = cli.depth,
        .aspect_ratio = cli.aspect_ratio,
        .head_dim = cli.head_dim,
        .vocab_size = vocab_size,
        .total_batch_size = cli.total_batch_size,
        .device_batch_size = cli.device_batch_size,
        .max_seq_len = cli.max_seq_len,
        .embedding_lr = cli.embedding_lr,
        .unembedding_lr = cli.unembedding_lr,
        .matrix_lr = cli.matrix_lr,
        .scalar_lr = cli.scalar_lr,
        .weight_decay = cli.weight_decay,
    };

    // Auto horizons (base_train.py's DEFAULTS, opted into here with 0):
    // --total-batch-size 0 derives the Power-Lines optimal batch
    // B_REF·(target_tokens/D_REF)^0.383 clamped to the nearest power of two
    // (base_train.py:279-284); --num-iterations 0 derives
    // target_tokens // total_batch_size (base_train.py:348-351).
    var num_iterations = cli.num_iterations;
    if (cli.total_batch_size == 0 or cli.num_iterations == 0) {
        // target_tokens/d_ref don't depend on the batch size; derive them with
        // a provisional divisible value.
        derive_args.total_batch_size = cli.device_batch_size * cli.max_seq_len;
        const pre = try deriveHparams(derive_args);
        var total_batch = cli.total_batch_size;
        if (total_batch == 0) {
            const ratio = @as(f64, @floatFromInt(pre.target_tokens)) / pre.d_ref;
            const predicted = b_ref * std.math.pow(f64, ratio, 0.383);
            const exp: u6 = @intFromFloat(@round(std.math.log2(predicted)));
            total_batch = @as(usize, 1) << exp;
            try stdout.print("base-train: auto total_batch_size={d}\n", .{total_batch});
        }
        if (num_iterations == 0) {
            num_iterations = pre.target_tokens / total_batch;
            try stdout.print("base-train: auto num_iterations={d}\n", .{num_iterations});
        }
        derive_args.total_batch_size = total_batch;
    }

    const hp = try deriveHparams(derive_args);
    const cfg = buildConfig(cli, hp, vocab_size);

    try stdout.print(
        "base-train: model_dim={d} num_heads={d} vocab={d} | batch_lr_scale={d:.6} dmodel_lr_scale={d:.6} wd_scaled={d:.6}\n",
        .{ hp.model_dim, hp.num_heads, vocab_size, hp.batch_lr_scale, hp.dmodel_lr_scale, hp.weight_decay_scaled },
    );
    try stdout.print(
        "  scaling_params={d} target_tokens={d} grad_accum={d} num_iterations={d}\n",
        .{ hp.num_scaling_params, hp.target_tokens, hp.grad_accum, num_iterations },
    );
    try stdout.flush();

    // --resume implies the checkpoint's own model.safetensors; a conflicting
    // --init-from would silently train the wrong weights against the resumed
    // optimizer state, so it is rejected.
    var resume_state: ?NcState = null;
    var resume_src: ?[]u8 = null;
    defer if (resume_src) |p| allocator.free(p);
    if (cli.resume_dir) |dir| {
        if (cli.init_from.len > 0) {
            try stdout.writeAll("base-train: --resume loads the checkpoint's model.safetensors; drop --init-from\n");
            return error.InvalidArgument;
        }
        resume_src = try resumeSourceDir(allocator, io, dir);
        resume_state = try loadTrainerJson(allocator, io, resume_src.?);
    }

    var model = if (resume_src) |src| blk: {
        const model_path = try training_checkpoint.pathJoin(allocator, src, training_checkpoint.model_state_file);
        defer allocator.free(model_path);
        break :blk try ModelT.initFromSafetensors(cfg, &ctx, allocator, io, model_path);
    } else if (cli.init_from.len > 0)
        try ModelT.initFromSafetensors(cfg, &ctx, allocator, io, cli.init_from)
    else
        try ModelT.initRandom(cfg, &ctx, allocator, cli.seed);
    defer model.deinit();

    var opt = MuonAdamW.init(allocator, hp.opt_config);
    defer opt.deinit();
    try opt.registerModel(&model);

    var start_step: usize = 0;
    if (resume_src) |src| {
        try loadOptimizerState(allocator, io, src, &opt);
        start_step = @intCast(resume_state.?.step);
        try stdout.print("resumed from {s}: step {d}\n", .{ src, start_step });
        try stdout.flush();
    }

    // Train NCDOC + BOS-bestfit loader, fast-forwarded to the checkpointed
    // stream position on resume (dataloader.py's approximate rg-level resume).
    var ncdoc = try data_mod.readNcDoc(allocator, io, cli.data.?);
    defer ncdoc.deinit();
    var loader = try data_mod.BaseLoader.init(allocator, &tokenizer, ncdoc.docs, ncdoc.docs_per_rowgroup, cli.device_batch_size, cli.max_seq_len, 1000);
    defer loader.deinit();
    if (resume_state) |st| loader.resumeAt(st.rg_idx, st.epoch);

    const token_bytes = try tokenizer.computeTokenBytes(allocator);
    defer allocator.free(token_bytes);

    // Val NCDOC read once; each eval still builds a fresh loader over it
    // (reference semantics: a fresh val stream per eval).
    var val_ncdoc: ?data_mod.NcDoc = null;
    defer if (val_ncdoc) |*vd| vd.deinit();
    if (cli.val_data) |vpath| val_ncdoc = try data_mod.readNcDoc(allocator, io, vpath);

    var last_state = data_mod.State{};
    var step: usize = start_step;
    while (true) : (step += 1) {
        const last_step = step == num_iterations;

        if (val_ncdoc) |*vd| {
            if (cli.eval_every > 0 and (last_step or step % @as(usize, @intCast(cli.eval_every)) == 0)) {
                const bpb = try evalValBpb(&ctx, &model, &tokenizer, token_bytes, allocator, vd, cli.device_batch_size, cli.max_seq_len, cli.eval_tokens);
                try stdout.print("step {d:0>5} | val bpb: {d:.6}\n", .{ step, bpb });
                try stdout.flush();
            }
        }

        if (cli.sample_every > 0 and (last_step or (step > 0 and step % @as(usize, @intCast(cli.sample_every)) == 0))) {
            try samplePreviews(&ctx, &model, &tokenizer, allocator, stdout);
            try stdout.flush();
        }

        if (last_step or (step > 0 and cli.save_every > 0 and step % @as(usize, @intCast(cli.save_every)) == 0)) {
            try saveCheckpoint(allocator, io, cli.out.?, &model, &opt, .{
                .step = step,
                .seed = cli.seed,
                .num_iterations = num_iterations,
                .weight_decay_scaled = hp.weight_decay_scaled,
                .pq_idx = last_state.pq_idx,
                .rg_idx = last_state.rg_idx,
                .epoch = last_state.epoch,
            });
            try stdout.print("step {d:0>5} | checkpoint written to {s}\n", .{ step, cli.out.? });
            try stdout.flush();
        }

        if (last_step) break;

        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        var loss: f32 = 0;
        for (0..hp.grad_accum) |_| {
            var batch = try loader.nextBatch();
            defer batch.deinit();
            last_state = batch.state;
            loss = try microStep(&ctx, &model, batch.inputs, batch.targets, batch.b, batch.t, hp.grad_accum, allocator);
        }
        const sched = StepSchedule.atWith(step, num_iterations, hp.weight_decay_scaled, .{
            .warmup_steps = @floatFromInt(cli.warmup_steps),
            .warmdown_ratio = cli.warmdown_ratio,
            .final_lr_frac = cli.final_lr_frac,
        });
        try opt.step(&ctx, sched);
        opt.zeroGrad();
        const dt_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6;
        try stdout.print("step {d:0>5}/{d:0>5} | loss: {d:.6} | lrm: {d:.4} | dt: {d:.1}ms\n", .{ step, num_iterations, loss, sched.lrm, dt_ms });
        try stdout.flush();
    }

    try stdout.writeAll("base-train: done\n");
}

/// Fresh val loader over the pre-read `ncdoc` for `eval_tokens` worth of
/// batches (base_train eval_steps = eval_tokens // (device_batch_size *
/// max_seq_len)); the loader is rebuilt per eval so every eval sees the same
/// fresh val stream (reference semantics).
fn evalValBpb(
    ctx: *ExecContext,
    model: anytype,
    tokenizer: *const Tokenizer,
    token_bytes: []const u32,
    allocator: Allocator,
    ncdoc: *const data_mod.NcDoc,
    b: usize,
    t: usize,
    eval_tokens: usize,
) !f64 {
    var loader = try data_mod.BaseLoader.init(allocator, tokenizer, ncdoc.docs, ncdoc.docs_per_rowgroup, b, t, 1000);
    defer loader.deinit();

    const eval_steps = @max(1, eval_tokens / (b * t));
    var acc = BpbAccum{};
    for (0..eval_steps) |_| {
        var batch = try loader.nextBatch();
        defer batch.deinit();
        try accumBpb(ctx, model, token_bytes, batch.inputs, batch.targets, batch.b, batch.t, allocator, &acc);
    }
    return bpbValue(acc);
}

const sample_prompts = [_][]const u8{
    "The capital of France is",
    "The chemical symbol of gold is",
    "The opposite of hot is",
};

/// Greedy (temperature 0) continuation previews via full-sequence re-forward
/// (no KV cache — fine for a short 16-token preview).
fn samplePreviews(ctx: *ExecContext, model: anytype, tokenizer: *const Tokenizer, allocator: Allocator, stdout: *std.Io.Writer) !void {
    for (sample_prompts) |prompt| {
        const prompt_ids32 = try tokenizer.encodeWithBos(allocator, prompt);
        defer allocator.free(prompt_ids32);

        var seq: std.ArrayList(usize) = .empty;
        defer seq.deinit(allocator);
        for (prompt_ids32) |id| try seq.append(allocator, id);

        for (0..16) |_| {
            var ng = fucina.noGrad();
            defer ng.close();
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            const n = seq.items.len;
            var logits = try model.forward(ctx, seq.items, null);
            var last = try logits.narrow(ctx, .seq, n - 1, 1);
            var idx = try last.argmax(ctx, .vocab);
            const next: u32 = @intCast((try idx.dataConst())[0]);
            try seq.append(allocator, next);
        }

        const ids32 = try allocator.alloc(u32, seq.items.len);
        defer allocator.free(ids32);
        for (ids32, seq.items) |*d, s| d.* = @intCast(s);
        // decode is byte-exact (round-trip tests rely on it); an early-training
        // greedy sample routinely ends mid-codepoint, so sanitize only what is
        // PRINTED (tiktoken decodes with errors='replace').
        const text = try tokenizer.decode(allocator, ids32);
        defer allocator.free(text);
        const shown = try sanitizeUtf8(allocator, text);
        defer allocator.free(shown);
        try stdout.print("  sample: {s}\n", .{shown});
    }
}

/// Replace invalid UTF-8 with U+FFFD for terminal display.
fn sanitizeUtf8(allocator: Allocator, bytes: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < bytes.len) {
        const len = std.unicode.utf8ByteSequenceLength(bytes[i]) catch {
            try out.appendSlice(allocator, "\u{FFFD}");
            i += 1;
            continue;
        };
        if (i + len > bytes.len or !std.unicode.utf8ValidateSlice(bytes[i .. i + len])) {
            try out.appendSlice(allocator, "\u{FFFD}");
            i += 1;
            continue;
        }
        try out.appendSlice(allocator, bytes[i .. i + len]);
        i += len;
    }
    return out.toOwnedSlice(allocator);
}

// ===========================================================================
// CLI: eval-bpb
// ===========================================================================

pub fn runEvalBpb(io: std.Io, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const cli = try parseCli(args);
    if (cli.data == null or cli.tokenizer == null or cli.init_from.len == 0) {
        try stdout.writeAll("eval-bpb: --init-from <safetensors>, --data <NCDOC>, --tokenizer <tokenizer.bin> required\n");
        return error.InvalidArgument;
    }

    try validateWindowPattern(cli.window_pattern);

    switch (cli.dtype) {
        .bf16 => try evalBpbWith(model_mod.ModelOf(.bf16), io, stdout, cli),
        else => try evalBpbWith(Model, io, stdout, cli),
    }
}

fn evalBpbWith(comptime ModelT: type, io: std.Io, stdout: *std.Io.Writer, cli: Cli) !void {
    const allocator = std.heap.smp_allocator;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var tokenizer = try Tokenizer.loadBin(allocator, io, cli.tokenizer.?);
    defer tokenizer.deinit();
    const vocab_size = tokenizer.n_vocab;

    const hp = try deriveHparams(.{
        .depth = cli.depth,
        .aspect_ratio = cli.aspect_ratio,
        .head_dim = cli.head_dim,
        .vocab_size = vocab_size,
        .total_batch_size = if (cli.total_batch_size == 0) cli.device_batch_size * cli.max_seq_len else cli.total_batch_size,
        .device_batch_size = cli.device_batch_size,
        .max_seq_len = cli.max_seq_len,
        .embedding_lr = cli.embedding_lr,
        .unembedding_lr = cli.unembedding_lr,
        .matrix_lr = cli.matrix_lr,
        .scalar_lr = cli.scalar_lr,
        .weight_decay = cli.weight_decay,
    });
    const cfg = buildConfig(cli, hp, vocab_size);

    var model = try ModelT.initFromSafetensors(cfg, &ctx, allocator, io, cli.init_from);
    defer model.deinit();

    const token_bytes = try tokenizer.computeTokenBytes(allocator);
    defer allocator.free(token_bytes);

    var ncdoc = try data_mod.readNcDoc(allocator, io, cli.data.?);
    defer ncdoc.deinit();
    const bpb = try evalValBpb(&ctx, &model, &tokenizer, token_bytes, allocator, &ncdoc, cli.device_batch_size, cli.max_seq_len, cli.eval_tokens);
    try stdout.print("eval-bpb: {d:.6}\n", .{bpb});
}

// ===========================================================================
// CLI: sft (chat_sft.py — supervised fine-tuning)
// ===========================================================================
//
// Extends the base-train loop with the SFT-specific pieces of chat_sft.py: a
// MuonAdamW built from the base LRs × init_lr_frac (0.8, chat_sft.py lines
// 132/157-159 — note SFT folds in dmodel_lr_scale but NOT batch_lr_scale,
// unlike base_train), weight_decay 0.0, the progress-based learning-rate
// schedule (get_lr_multiplier: warmup_ratio 0 / warmdown_ratio 0.5 /
// final_lr_frac 0) + the SFT Muon momentum ramp (0.85→0.95 over 300 steps), and
// the bestfit-pad SftLoader over the task mixture (masked targets, −1 = ignore).
// The loss normalization, grad-accum, checkpoint, and eval scaffolding are reused
// from the base loop; the only difference is that SFT has ignored (masked)
// targets — microStep/accumBpb already honor the −1 sentinel. Like chat_sft.py's
// --load-optimizer=1 default, the optimizer moments warm-start from the base
// checkpoint's optimizer.fucina next to --init-from when present (the all-Zig
// pipeline writes it with the identical group layout); LRs are config-driven
// here, so no post-load LR reset is needed.

/// chat_sft.py get_lr_multiplier: progress-based (0→1 over the run), NOT absolute
/// step counts. warmup_ratio 0 ⇒ no warmup; constant 1.0 until the warmdown window
/// (progress > 1 − warmdown_ratio), then linear to final_lr_frac.
fn sftLrMultiplier(progress: f64, warmup_ratio: f64, warmdown_ratio: f64, final_lr_frac: f64) f64 {
    if (progress < warmup_ratio) return (progress + 1e-8) / warmup_ratio;
    if (progress <= 1.0 - warmdown_ratio) return 1.0;
    const decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio;
    return (1.0 - decay) * 1.0 + decay * final_lr_frac;
}

/// chat_sft.py get_muon_momentum: 0.85→0.95 linearly over the first 300 steps,
/// then hold at 0.95.
fn sftMuonMomentum(step: usize) f64 {
    const frac = @min(@as(f64, @floatFromInt(step)) / 300.0, 1.0);
    return (1.0 - frac) * 0.85 + frac * 0.95;
}

/// Per-step SFT schedule for MuonAdamW.step. progress = step/num_iterations
/// (deterministic; chat_sft.py drives progress from the data generator, but the
/// fixed-horizon trace resolves it to step/N). weight_decay is a constant 0.0 in
/// SFT (no decay schedule).
fn sftSchedule(step: usize, num_iters: usize, warmup_ratio: f64, warmdown_ratio: f64, final_lr_frac: f64) StepSchedule {
    const progress = @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(num_iters));
    return .{
        .lrm = sftLrMultiplier(progress, warmup_ratio, warmdown_ratio, final_lr_frac),
        .muon_momentum = sftMuonMomentum(step),
        .muon_weight_decay = 0.0,
    };
}

/// gpt.py setup_optimizer group LRs for SFT (chat_sft.py). Each base CLI LR is
/// folded with dmodel_lr_scale = (model_dim/768)^-0.5 (embedding/head groups) and
/// the group-specific factors (value_embeds ×0.5, resid ×0.01, x0 ×1, smear fixed
/// 0.2), then every group is scaled by init_lr_frac (chat_sft.py sets group['lr']
/// = lr·init_lr_frac and pins it as initial_lr). scalar_lr uses setup_optimizer's
/// default 0.5 (chat_sft.py does not pass scalar_lr). batch_lr_scale is NOT applied.
fn sftOptConfig(embedding_lr: f64, unembedding_lr: f64, matrix_lr: f64, model_dim: usize, init_lr_frac: f64) optim_mod.Config {
    const scalar_lr: f64 = 0.5;
    const dmodel_lr_scale = std.math.pow(f64, @as(f64, @floatFromInt(model_dim)) / 768.0, -0.5);
    var c: optim_mod.Config = .{ .adamw_initial_lr = undefined, .muon_initial_lr = 0 };
    c.adamw_initial_lr[0] = unembedding_lr * dmodel_lr_scale * init_lr_frac;
    c.adamw_initial_lr[1] = embedding_lr * dmodel_lr_scale * init_lr_frac;
    c.adamw_initial_lr[2] = embedding_lr * dmodel_lr_scale * 0.5 * init_lr_frac;
    c.adamw_initial_lr[3] = scalar_lr * 0.01 * init_lr_frac;
    c.adamw_initial_lr[4] = scalar_lr * init_lr_frac;
    c.adamw_initial_lr[5] = 0.2 * init_lr_frac;
    c.muon_initial_lr = matrix_lr * init_lr_frac;
    return c;
}

const SftCli = struct {
    // Model geometry (must match the base checkpoint being fine-tuned; d6 defaults).
    depth: usize = 6,
    aspect_ratio: usize = 64,
    head_dim: usize = 64,
    window_pattern: []const u8 = "L",
    // Batch / horizon.
    max_seq_len: usize = 512,
    device_batch_size: usize = 2,
    total_batch_size: usize = 0, // 0 = auto (device_batch_size × max_seq_len ⇒ grad_accum 1)
    num_iterations: usize = 50,
    // Optimization (base LRs; sftOptConfig folds in dmodel_lr_scale × init_lr_frac).
    embedding_lr: f64 = 0.3,
    unembedding_lr: f64 = 0.008,
    matrix_lr: f64 = 0.02,
    init_lr_frac: f64 = 0.8,
    warmup_ratio: f64 = 0.0,
    warmdown_ratio: f64 = 0.5,
    final_lr_frac: f64 = 0.0,
    // chat_sft.py --load-optimizer default 1: warm-start moments from the base
    // checkpoint's optimizer.fucina (sibling of --init-from) when present.
    load_optimizer: bool = true,
    // Eval / IO.
    eval_every: i64 = -1,
    eval_tokens: usize = 8192,
    save_every: i64 = -1,
    seed: u64 = 42,
    mixture: ?[]const u8 = null,
    val_mixture: ?[]const u8 = null,
    tokenizer: ?[]const u8 = null,
    init_from: []const u8 = "",
    out: ?[]const u8 = null,
    resume_dir: ?[]const u8 = null,
    /// Storage dtype of the trained transformer matrices (`--dtype f32|bf16`;
    /// must match the base checkpoint being fine-tuned).
    dtype: fucina.DType = .f32,
};

fn parseSftCli(args: []const []const u8) !SftCli {
    var cli: SftCli = .{};
    var w = ArgWalk{ .args = args };
    while (w.i < args.len) : (w.i += 1) {
        if (w.value("--depth")) |v| {
            cli.depth = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--aspect-ratio")) |v| {
            cli.aspect_ratio = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--head-dim")) |v| {
            cli.head_dim = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--window-pattern")) |v| {
            cli.window_pattern = v;
        } else if (w.value("--max-seq-len")) |v| {
            cli.max_seq_len = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--device-batch-size")) |v| {
            cli.device_batch_size = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--total-batch-size")) |v| {
            cli.total_batch_size = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--num-iterations")) |v| {
            cli.num_iterations = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--embedding-lr")) |v| {
            cli.embedding_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--unembedding-lr")) |v| {
            cli.unembedding_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--matrix-lr")) |v| {
            cli.matrix_lr = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--init-lr-frac")) |v| {
            cli.init_lr_frac = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--warmup-ratio")) |v| {
            cli.warmup_ratio = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--warmdown-ratio")) |v| {
            cli.warmdown_ratio = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--final-lr-frac")) |v| {
            cli.final_lr_frac = try std.fmt.parseFloat(f64, v);
        } else if (w.value("--eval-every")) |v| {
            cli.eval_every = try std.fmt.parseInt(i64, v, 10);
        } else if (w.value("--eval-tokens")) |v| {
            cli.eval_tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (w.value("--save-every")) |v| {
            cli.save_every = try std.fmt.parseInt(i64, v, 10);
        } else if (w.value("--seed")) |v| {
            cli.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (w.value("--load-optimizer")) |v| {
            cli.load_optimizer = (try std.fmt.parseInt(u8, v, 10)) != 0;
        } else if (w.value("--model-tag")) |v| {
            _ = v; // accepted for chat_sft.py compatibility; no-op in this port
        } else if (w.value("--mixture")) |v| {
            cli.mixture = v;
        } else if (w.value("--val-mixture")) |v| {
            cli.val_mixture = v;
        } else if (w.value("--tokenizer")) |v| {
            cli.tokenizer = v;
        } else if (w.value("--init-from")) |v| {
            cli.init_from = v;
        } else if (w.value("--out")) |v| {
            cli.out = v;
        } else if (w.value("--resume")) |v| {
            cli.resume_dir = v;
        } else if (w.value("--dtype")) |v| {
            cli.dtype = try parseDtypeFlag(v);
        } else {
            return error.UnknownArgument;
        }
    }
    return cli;
}

pub fn runSft(io: std.Io, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const cli = parseSftCli(args) catch |err| {
        try stdout.print("sft: argument error: {s}\n", .{@errorName(err)});
        return err;
    };
    if (cli.init_from.len == 0 and cli.resume_dir == null) {
        try stdout.writeAll("sft: --init-from <base model.safetensors> is required\n");
        return error.InvalidArgument;
    }
    if (cli.init_from.len > 0 and cli.resume_dir != null) {
        try stdout.writeAll("sft: --resume loads the checkpoint's model.safetensors; drop --init-from\n");
        return error.InvalidArgument;
    }
    if (cli.mixture == null) {
        try stdout.writeAll("sft: --mixture <SFT JSONL> is required\n");
        return error.InvalidArgument;
    }
    if (cli.tokenizer == null) {
        try stdout.writeAll("sft: --tokenizer <tokenizer.bin> is required\n");
        return error.InvalidArgument;
    }
    if (cli.out == null) {
        try stdout.writeAll("sft: --out <dir> is required\n");
        return error.InvalidArgument;
    }
    try validateWindowPattern(cli.window_pattern);

    switch (cli.dtype) {
        .bf16 => try sftWith(model_mod.ModelOf(.bf16), io, stdout, cli),
        else => try sftWith(Model, io, stdout, cli),
    }
}

fn sftWith(comptime ModelT: type, io: std.Io, stdout: *std.Io.Writer, cli: SftCli) !void {
    const allocator = std.heap.smp_allocator;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var tokenizer = try Tokenizer.loadBin(allocator, io, cli.tokenizer.?);
    defer tokenizer.deinit();
    const vocab_size = tokenizer.n_vocab;

    // Reuse deriveHparams purely for model geometry (model_dim / num_heads) and
    // grad_accum; its base-scaled LR outputs are NOT used for SFT (see sftOptConfig).
    const total_batch = if (cli.total_batch_size == 0) cli.device_batch_size * cli.max_seq_len else cli.total_batch_size;
    const hp = try deriveHparams(.{
        .depth = cli.depth,
        .aspect_ratio = cli.aspect_ratio,
        .head_dim = cli.head_dim,
        .vocab_size = vocab_size,
        .total_batch_size = total_batch,
        .device_batch_size = cli.device_batch_size,
        .max_seq_len = cli.max_seq_len,
        .embedding_lr = cli.embedding_lr,
        .unembedding_lr = cli.unembedding_lr,
        .matrix_lr = cli.matrix_lr,
        .scalar_lr = 0.5,
        .weight_decay = 0.0,
    });
    const cfg: Config = .{
        .sequence_len = cli.max_seq_len,
        .vocab_size = vocab_size,
        .n_layer = cli.depth,
        .n_head = hp.num_heads,
        .n_kv_head = hp.num_heads,
        .n_embd = hp.model_dim,
        .window_pattern = cli.window_pattern,
    };

    const sft_cfg = sftOptConfig(cli.embedding_lr, cli.unembedding_lr, cli.matrix_lr, hp.model_dim, cli.init_lr_frac);
    try stdout.print(
        "sft: model_dim={d} num_heads={d} vocab={d} | init_lr_frac={d:.3} grad_accum={d} num_iterations={d}\n",
        .{ hp.model_dim, hp.num_heads, vocab_size, cli.init_lr_frac, hp.grad_accum, cli.num_iterations },
    );
    try stdout.print(
        "  group_lr[wte]={d:.6} lr[lm_head]={d:.6} lr[x0]={d:.6} lr[muon]={d:.6} | warmdown_ratio={d:.2} final_lr_frac={d:.2}\n",
        .{ sft_cfg.adamw_initial_lr[1], sft_cfg.adamw_initial_lr[0], sft_cfg.adamw_initial_lr[4], sft_cfg.muon_initial_lr, cli.warmdown_ratio, cli.final_lr_frac },
    );
    try stdout.flush();

    var resume_state: ?NcState = null;
    var resume_src: ?[]u8 = null;
    defer if (resume_src) |p| allocator.free(p);
    if (cli.resume_dir) |dir| {
        resume_src = try resumeSourceDir(allocator, io, dir);
        resume_state = try loadTrainerJson(allocator, io, resume_src.?);
    }

    var model = if (resume_src) |src| blk: {
        const model_path = try training_checkpoint.pathJoin(allocator, src, training_checkpoint.model_state_file);
        defer allocator.free(model_path);
        break :blk try ModelT.initFromSafetensors(cfg, &ctx, allocator, io, model_path);
    } else try ModelT.initFromSafetensors(cfg, &ctx, allocator, io, cli.init_from);
    defer model.deinit();

    var opt = MuonAdamW.init(allocator, sft_cfg);
    defer opt.deinit();
    try opt.registerModel(&model);

    var start_step: usize = 0;
    if (resume_src) |src| {
        try loadOptimizerState(allocator, io, src, &opt);
        start_step = @intCast(resume_state.?.step);
        try stdout.print("resumed from {s}: step {d}\n", .{ src, start_step });
        try stdout.flush();
    } else if (cli.load_optimizer) {
        // chat_sft.py --load-optimizer=1 default: warm-start the moments from
        // the base checkpoint's optimizer.fucina next to --init-from. Missing
        // file → fresh optimizer with a warning (matching python's fallback).
        const warm: enum { loaded, missing } = warm: {
            const base_dir = std.fs.path.dirname(cli.init_from) orelse break :warm .missing;
            loadOptimizerState(allocator, io, base_dir, &opt) catch |err| switch (err) {
                error.FileNotFound => break :warm .missing,
                else => return err,
            };
            try stdout.print("sft: warm-started optimizer from {s}\n", .{base_dir});
            break :warm .loaded;
        };
        if (warm == .missing) try stdout.writeAll("sft: base optimizer checkpoint not found; starting fresh (slightly worse)\n");
        try stdout.flush();
    }

    // Bestfit-pad SFT loader over the mixture JSONL (masked targets, −1 =
    // ignore), fast-forwarded to the checkpointed cursor on resume.
    var jc = try data_mod.readJsonlConvs(allocator, io, cli.mixture.?);
    defer jc.deinit();
    var loader = try data_mod.SftLoader.init(allocator, &tokenizer, jc.convs, cli.device_batch_size, cli.max_seq_len, 100);
    defer loader.deinit();
    if (resume_state) |st| loader.resumeAt(@intCast(st.rg_idx), st.epoch);

    // Held-out val mixture for eval; python builds a separate val TaskMixture
    // (chat_sft.py:169-173), so warn when 'val bpb' would read the train set.
    var val_jc: ?data_mod.JsonlConvs = null;
    defer if (val_jc) |*v| v.deinit();
    if (cli.eval_every > 0) {
        if (cli.val_mixture) |vm| {
            val_jc = try data_mod.readJsonlConvs(allocator, io, vm);
        } else {
            try stdout.writeAll("sft: no --val-mixture; 'val bpb' evaluates the TRAINING mixture\n");
            try stdout.flush();
        }
    }

    const token_bytes = try tokenizer.computeTokenBytes(allocator);
    defer allocator.free(token_bytes);

    // Scratch for the SftLoader's i64 targets → the i32 microStep/accumBpb take
    // (values fit: ids < vocab, −1 stays −1).
    const nbt = cli.device_batch_size * cli.max_seq_len;
    const tgt_scratch = try allocator.alloc(i32, nbt);
    defer allocator.free(tgt_scratch);

    var step: usize = start_step;
    while (true) : (step += 1) {
        const last_step = step == cli.num_iterations;

        if (cli.eval_every > 0 and (last_step or step % @as(usize, @intCast(cli.eval_every)) == 0)) {
            const eval_jc: *const data_mod.JsonlConvs = if (val_jc) |*v| v else &jc;
            const bpb = try evalSftBpb(&ctx, &model, &tokenizer, token_bytes, allocator, eval_jc, cli.device_batch_size, cli.max_seq_len, cli.eval_tokens);
            try stdout.print("step {d:0>5} | val bpb: {d:.6}\n", .{ step, bpb });
            try stdout.flush();
        }

        if (last_step or (step > 0 and cli.save_every > 0 and step % @as(usize, @intCast(cli.save_every)) == 0)) {
            try saveCheckpoint(allocator, io, cli.out.?, &model, &opt, .{
                .step = step,
                .seed = cli.seed,
                .num_iterations = cli.num_iterations,
                .weight_decay_scaled = 0.0,
                .pq_idx = 0,
                // rg_idx carries the SFT conversation cursor (the base loader's
                // rowgroup slot is unused here) for the approximate resume.
                .rg_idx = @intCast(loader.cursor),
                .epoch = loader.epoch,
            });
            try stdout.print("step {d:0>5} | checkpoint written to {s}\n", .{ step, cli.out.? });
            try stdout.flush();
        }

        if (last_step) break;

        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        var loss: f32 = 0;
        for (0..hp.grad_accum) |_| {
            var batch = try loader.nextBatch();
            defer batch.deinit();
            for (batch.targets, tgt_scratch[0..batch.targets.len]) |src, *dst| dst.* = @intCast(src);
            loss = try microStep(&ctx, &model, batch.inputs, tgt_scratch[0..batch.targets.len], batch.b, batch.t, hp.grad_accum, allocator);
        }
        const sched = sftSchedule(step, cli.num_iterations, cli.warmup_ratio, cli.warmdown_ratio, cli.final_lr_frac);
        try opt.step(&ctx, sched);
        opt.zeroGrad();
        const dt_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6;
        try stdout.print("step {d:0>5}/{d:0>5} | loss: {d:.6} | lrm: {d:.4} | mom: {d:.4} | dt: {d:.1}ms\n", .{ step, cli.num_iterations, loss, sched.lrm, sched.muon_momentum, dt_ms });
        try stdout.flush();
    }

    try stdout.writeAll("sft: done\n");
}

/// Fresh SFT val-bpb over the pre-read mixture (forward-only; masked/padding
/// targets carry no bytes and contribute no nats — loss_eval.evaluate_bpb
/// semantics); the loader is rebuilt per eval for a fresh stream.
fn evalSftBpb(
    ctx: *ExecContext,
    model: anytype,
    tokenizer: *const Tokenizer,
    token_bytes: []const u32,
    allocator: Allocator,
    jc: *const data_mod.JsonlConvs,
    b: usize,
    t: usize,
    eval_tokens: usize,
) !f64 {
    var loader = try data_mod.SftLoader.init(allocator, tokenizer, jc.convs, b, t, 100);
    defer loader.deinit();

    const tgt_scratch = try allocator.alloc(i32, b * t);
    defer allocator.free(tgt_scratch);

    const eval_steps = @max(1, eval_tokens / (b * t));
    var acc = BpbAccum{};
    for (0..eval_steps) |_| {
        var batch = try loader.nextBatch();
        defer batch.deinit();
        for (batch.targets, tgt_scratch[0..batch.targets.len]) |src, *dst| dst.* = @intCast(src);
        try accumBpb(ctx, model, token_bytes, batch.inputs, tgt_scratch[0..batch.targets.len], batch.b, batch.t, allocator, &acc);
    }
    return bpbValue(acc);
}

// ===========================================================================
// Shared helpers
// ===========================================================================

fn readFileBytes(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotAFile;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

// ===========================================================================
// Gate: SFT loss-trace parity (dump-sft-trace oracle). Env-gated on
// NANOCHAT_PARITY; skips cleanly when the goldens are absent. Same drift-budget
// philosophy as the base loss-trace gate: the per-step loss GIVEN identical weights
// matches bit-closely, then the trajectory drifts as optimizer steps accumulate
// the per-sequence-vs-batched backward float-order difference amplified by Muon
// — a benign, bounded, non-runaway drift that re-converges by the endpoint. We
// therefore gate a DRIFT BUDGET, not a bitwise trace: (a) the first few
// well-conditioned steps within ~2%, (b) a bounded chaotic envelope (no
// runaway/NaN), (c) endpoint re-convergence. The SFT trace starts from the
// TRAINED base checkpoint (step 2500) on the SFT val mixture with the FRESH SFT
// optimizer (init_lr_frac 0.2, weight_decay 0), matching the oracle exactly.
// ===========================================================================

const sft_goldens_dir = "refs/nanochat-goldens";

test "NANOCHAT_PARITY: SFT loss trace tracks reference within drift budget" {
    if (std.testing.environ.getPosix("NANOCHAT_PARITY") == null) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = Model.initFromSafetensors(model_mod.Config.d6, &ctx, allocator, io, sft_goldens_dir ++ "/base_ckpt_d6_step2500.safetensors") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer model.deinit();

    // trace_batches_sft_d6.bin: u32 n_steps, u32 B, u32 T, then per step B*T i32
    // inputs + B*T i32 targets (−1 = ignore). Same layout as the base trace.
    const bytes = readFileBytes(allocator, io, sft_goldens_dir ++ "/trace_batches_sft_d6.bin") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(bytes);
    const n_steps = std.mem.readInt(u32, bytes[0..4], .little);
    const b = std.mem.readInt(u32, bytes[4..8], .little);
    const t = std.mem.readInt(u32, bytes[8..12], .little);
    const nbt = @as(usize, b) * t;
    const inputs = try allocator.alloc(i32, @as(usize, n_steps) * nbt);
    defer allocator.free(inputs);
    const targets = try allocator.alloc(i32, @as(usize, n_steps) * nbt);
    defer allocator.free(targets);
    var off: usize = 12;
    for (0..n_steps) |s| {
        for (0..nbt) |i| {
            inputs[s * nbt + i] = @bitCast(std.mem.readInt(u32, bytes[off..][0..4], .little));
            off += 4;
        }
        for (0..nbt) |i| {
            targets[s * nbt + i] = std.mem.readInt(i32, bytes[off..][0..4], .little);
            off += 4;
        }
    }
    try std.testing.expectEqual(bytes.len, off);

    const ref_bytes = readFileBytes(allocator, io, sft_goldens_dir ++ "/loss_trace_sft_d6.json") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(ref_bytes);
    var ref_parsed = try std.json.parseFromSlice([]f64, allocator, ref_bytes, .{});
    defer ref_parsed.deinit();
    const ref = ref_parsed.value;
    try std.testing.expectEqual(@as(usize, n_steps), ref.len);

    // FRESH SFT optimizer: base LRs × init_lr_frac 0.2 (the recorded CPU-trace
    // config — chat_sft.py's 0.8 default is too hot for this tiny stress batch and
    // diverges; see sft_schedule_d6.json / nanochat_dump.py dump-sft-trace).
    const sft_cfg = sftOptConfig(0.3, 0.008, 0.02, model_mod.Config.d6.n_embd, 0.2);
    var opt = MuonAdamW.init(allocator, sft_cfg);
    defer opt.deinit();
    try opt.registerModel(&model);

    const num_iters: usize = n_steps;
    const warmup_exact: usize = 8; // early well-conditioned steps track tightly
    const envelope_max_rel: f64 = 0.15; // bounded chaotic envelope (no runaway)
    const endpoint_max_rel: f64 = 0.03; // endpoint re-convergence

    var max_abs: f64 = 0;
    var max_rel: f64 = 0;
    var breaches: usize = 0; // vs the 2% budget (reported, only asserted over warmup)
    var warmup_ok = true;
    var endpoint_rel: f64 = 0;
    for (0..n_steps) |step| {
        const loss = try microStep(&ctx, &model, inputs[step * nbt ..][0..nbt], targets[step * nbt ..][0..nbt], b, t, 1, allocator);
        try opt.step(&ctx, sftSchedule(step, num_iters, 0.0, 0.5, 0.0));
        opt.zeroGrad();

        const r = ref[step];
        const abs = @abs(@as(f64, loss) - r);
        const rel = if (@abs(r) > 1e-30) abs / @abs(r) else abs;
        if (abs > max_abs) max_abs = abs;
        if (rel > max_rel) max_rel = rel;
        const budget = @max(0.02 * @abs(r), 0.02);
        const over = abs > budget;
        if (over) breaches += 1;
        if (step < warmup_exact and over) warmup_ok = false;
        if (step == n_steps - 1) endpoint_rel = rel;
        std.debug.print("sft loss-trace step {d:>2}: zig {d:.6} ref {d:.6} abs {d:.6} rel {e:.3} budget {d:.6}{s}\n", .{ step, loss, r, abs, rel, budget, if (over) "  OVER" else "" });
    }
    std.debug.print("sft loss-trace parity: {d} steps, max abs {d:.6}, max rel {e:.3}, breaches-vs-2% {d}, warmup_ok {}, endpoint rel {e:.3}\n", .{ n_steps, max_abs, max_rel, breaches, warmup_ok, endpoint_rel });

    try std.testing.expect(warmup_ok); // (a) early steps within 2%
    try std.testing.expect(max_rel <= envelope_max_rel); // (b) bounded envelope
    try std.testing.expect(endpoint_rel <= endpoint_max_rel); // (c) endpoint re-converges
}

comptime {
    _ = @import("train_tests.zig");
}
