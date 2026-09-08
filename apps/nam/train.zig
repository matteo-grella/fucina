//! WaveNet trainer over fucina autograd — the classic released "standard"
//! recipe (upstream neural-amp-modeler full-mode configs): pure MSE, Adam
//! lr 0.004, ExponentialLR gamma 0.993 stepped per epoch, batch 16 with
//! gradient accumulation, ny 8192, deterministic seed. The model is the
//! app's one tensor WaveNet (`wavenet.zig`): the trainer records its window
//! forward, and validation streams the same model over the held-out split
//! (which doubles as a continuous check that the export weight order is
//! right) and reports the upstream full-pass ESR.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
const lstm = @import("lstm.zig");
const data = @import("data.zig");

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;
const rng = fucina.rng;

pub const ArraySpec = struct {
    input_size: usize,
    channels: usize,
    head_out: usize,
    head_bias: bool,
    kernel_size: usize,
    dilations: []const usize,
};

pub const ModelSpec = struct {
    arrays: []const ArraySpec,
    head_scale: f32,

    const classic_dilations = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

    /// The classic full-mode "standard" WaveNet
    /// (nam_full_configs/models/wavenet.json): 2 arrays, 16->8 channels,
    /// k=3, dilations 1..512, Tanh, head_scale 0.02 — 13,802 weights.
    pub const classic = ModelSpec{
        .arrays = &.{
            .{ .input_size = 1, .channels = 16, .head_out = 8, .head_bias = false, .kernel_size = 3, .dilations = &classic_dilations },
            .{ .input_size = 16, .channels = 8, .head_out = 1, .head_bias = true, .kernel_size = 3, .dilations = &classic_dilations },
        },
        .head_scale = 0.02,
    };

    const tiny_dilations = [_]usize{ 1, 2, 4, 8 };

    /// A small spec for smoke tests and quick runs.
    pub const tiny = ModelSpec{
        .arrays = &.{
            .{ .input_size = 1, .channels = 4, .head_out = 1, .head_bias = true, .kernel_size = 3, .dilations = &tiny_dilations },
        },
        .head_scale = 0.02,
    };

    pub fn receptiveField(self: *const ModelSpec) usize {
        var rf: usize = 1;
        for (self.arrays) |*a| {
            for (a.dilations) |d| rf += d * (a.kernel_size - 1);
        }
        return rf;
    }
};

pub const A2Spec = struct {
    channels: usize,
    head_scale: f32 = 0.01,

    pub const standard = A2Spec{ .channels = 8 };
    pub const nano = A2Spec{ .channels = 3 };

    pub const kernel_sizes = [_]usize{
        6, 6, 6,  6,  6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
    };
    pub const dilations = [_]usize{
        1,   3,   7, 17, 41, 101, 239, 1,  3,  7,   17,  41,
        101, 239, 1, 13, 1,  3,   7,   17, 41, 101, 239,
    };

    pub fn name(self: A2Spec) []const u8 {
        return if (self.channels == 3) "a2-nano" else "a2-standard";
    }

    pub fn receptiveField(self: *const A2Spec) usize {
        _ = self;
        var rf: usize = 1;
        for (dilations, kernel_sizes) |d, k| rf += d * (k - 1);
        return rf + 16 - 1;
    }
};

pub const PackedSpec = struct {
    pub const active = PackedSpec{};
    pub const submodel_specs = [_]A2Spec{ A2Spec.nano, A2Spec.standard };
    pub const submodel_names = [_][]const u8{ "channels_3", "channels_8" };

    pub const default_epochs: usize = 100;
    pub const default_lr: f32 = 0.004;
    pub const default_weight_decay: f32 = 3.17e-7;
    pub const default_gamma: f32 = 0.994;
    pub const default_mrstft_weight: f32 = 0.0005;

    pub fn name(_: PackedSpec) []const u8 {
        return "packed";
    }

    pub fn receptiveField(_: *const PackedSpec) usize {
        var spec = A2Spec.standard;
        return spec.receptiveField();
    }
};

pub const MrstftResolution = struct {
    fft_size: usize,
    hop_size: usize,
    win_length: usize,
};

pub const default_mrstft_resolutions = [_]MrstftResolution{
    .{ .fft_size = 1024, .hop_size = 120, .win_length = 600 },
    .{ .fft_size = 2048, .hop_size = 240, .win_length = 1200 },
    .{ .fft_size = 512, .hop_size = 50, .win_length = 240 },
};

pub const MrstftOptions = struct {
    resolutions: []const MrstftResolution = &default_mrstft_resolutions,
    eps: f32 = 1e-8,
};

pub const LossOptions = struct {
    mrstft_weight: f32 = 0,
    mrstft: MrstftOptions = .{},
};

/// The segment loss of a WaveNet on the trainer's window: MSE over the
/// last `target.len` predictions (mean over elements, the torch
/// `F.mse_loss` default), plus the optional MRSTFT term.
pub fn wavenetSegmentLoss(model: *wavenet.WaveNet, ctx: *ExecContext, window: []const f32, target: []const f32, options: LossOptions) !Tensor(.{}) {
    const pred = try model.forward(ctx, window);
    var target_tensor = try wavenet.TimeOut.fromSlice(ctx, .{ target.len, 1 }, target);
    defer target_tensor.deinit();
    const tail = try pred.narrow(ctx, .time, window.len - target.len, target.len);
    const diff = try tail.sub(ctx, &target_tensor);
    const sq = try diff.mul(ctx, &diff);
    const total = try sq.sumAll(ctx);
    var loss = try total.scale(ctx, 1.0 / @as(f32, @floatFromInt(target.len)));
    if (options.mrstft_weight > 0) {
        const pred_1d = try tail.squeeze(ctx, .out);
        var target_1d = try Tensor(.{.time}).fromSlice(ctx, .{target.len}, target);
        defer target_1d.deinit();
        const freq_loss = try mrstftLoss(ctx, &pred_1d, &target_1d, options.mrstft);
        const scaled_freq = try freq_loss.scale(ctx, options.mrstft_weight);
        loss = try loss.add(ctx, &scaled_freq);
    }
    return loss;
}

/// The model's weights plus, recursively, its condition DSP's, against the
/// template config (the snapshot borrows the template's architecture and
/// owns every weight stream).
pub fn wavenetSnapshot(model: *const wavenet.WaveNet, ctx: *ExecContext, allocator: std.mem.Allocator, template_config: *const nam_file.WaveNetConfig) anyerror!WaveNetSnapshot {
    const weights = try model.extractWeights(ctx, allocator);
    errdefer allocator.free(weights);
    var config = template_config.*;
    config.condition_dsp = try extractConditionDspSnapshot(model, ctx, allocator, template_config.condition_dsp);
    errdefer freeConditionDspSnapshot(allocator, config.condition_dsp);
    return .{ .config = config, .weights = weights };
}

fn extractConditionDspSnapshot(model: *const wavenet.WaveNet, ctx: *ExecContext, allocator: std.mem.Allocator, template: ?*const nam_file.ConditionDsp) anyerror!?*const nam_file.ConditionDsp {
    const child = model.condition_child orelse {
        if (template != null) return error.UnsupportedFeature;
        return null;
    };
    const dsp = template orelse return error.UnsupportedFeature;
    if (dsp.architecture != .wavenet or dsp.config != .wavenet) return error.UnsupportedFeature;
    const out = try allocator.create(nam_file.ConditionDsp);
    errdefer allocator.destroy(out);
    const child_snapshot = try wavenetSnapshot(child, ctx, allocator, &dsp.config.wavenet);
    out.* = .{
        .architecture = .wavenet,
        .config = .{ .wavenet = child_snapshot.config },
        .weights = child_snapshot.weights,
        .sample_rate = dsp.sample_rate,
    };
    return out;
}

fn freeConditionDspSnapshot(allocator: std.mem.Allocator, maybe_dsp: ?*const nam_file.ConditionDsp) void {
    const dsp_const = maybe_dsp orelse return;
    const dsp: *nam_file.ConditionDsp = @constCast(dsp_const);
    switch (dsp.config) {
        .wavenet => |*config| freeConditionDspSnapshot(allocator, config.condition_dsp),
        .lstm, .convnet, .linear => {},
    }
    allocator.free(dsp.weights);
    allocator.destroy(dsp);
}

pub fn mrstftLoss(ctx: *ExecContext, pred: *const Tensor(.{.time}), target: *const Tensor(.{.time}), options: MrstftOptions) !Tensor(.{}) {
    if (pred.dim(.time) != target.dim(.time)) return error.InvalidMrstftShape;
    if (options.resolutions.len == 0) return error.InvalidMrstftResolution;

    var total: Tensor(.{}) = undefined;
    var have_total = false;
    for (options.resolutions) |resolution| {
        const resolution_loss = try stftResolutionLoss(ctx, pred, target, resolution, options.eps);
        if (have_total) {
            total = try total.add(ctx, &resolution_loss);
        } else {
            total = resolution_loss;
            have_total = true;
        }
    }
    return total.scale(ctx, 1.0 / @as(f32, @floatFromInt(options.resolutions.len)));
}

fn stftResolutionLoss(
    ctx: *ExecContext,
    pred: *const Tensor(.{.time}),
    target: *const Tensor(.{.time}),
    resolution: MrstftResolution,
    eps: f32,
) !Tensor(.{}) {
    const pred_mag = try stftMagnitude(ctx, pred, resolution, eps);
    const target_mag = try stftMagnitude(ctx, target, resolution, eps);

    const diff = try pred_mag.sub(ctx, &target_mag);
    const diff_sq = try diff.mul(ctx, &diff);
    const numerator_sq = try diff_sq.sumAll(ctx);
    const numerator = try numerator_sq.sqrt(ctx);

    const target_sq = try target_mag.mul(ctx, &target_mag);
    const denominator_sq = try target_sq.sumAll(ctx);
    const denominator = try denominator_sq.sqrt(ctx);
    const spectral_convergence = try numerator.div(ctx, &denominator);

    const pred_log = try pred_mag.log(ctx);
    const target_log = try target_mag.log(ctx);
    const log_diff = try pred_log.sub(ctx, &target_log);
    const log_abs = try log_diff.abs(ctx);
    const log_sum = try log_abs.sumAll(ctx);
    const log_mean = try log_sum.scale(
        ctx,
        1.0 / @as(f32, @floatFromInt(pred_mag.dim(.frame) * pred_mag.dim(.freq))),
    );
    return spectral_convergence.add(ctx, &log_mean);
}

fn stftMagnitude(ctx: *ExecContext, signal: *const Tensor(.{.time}), resolution: MrstftResolution, eps: f32) !Tensor(.{ .frame, .freq }) {
    if (resolution.fft_size == 0 or resolution.hop_size == 0 or resolution.win_length == 0) return error.InvalidMrstftResolution;
    if (resolution.win_length > resolution.fft_size) return error.InvalidMrstftResolution;

    const seq_len = signal.dim(.time);
    const reflect_pad = resolution.fft_size / 2;
    if (seq_len <= reflect_pad) return error.InvalidMrstftShape;

    const padded_len = seq_len + 2 * reflect_pad;
    if (padded_len < resolution.fft_size) return error.InvalidMrstftShape;
    const frame_count = (padded_len - resolution.fft_size) / resolution.hop_size + 1;
    const freq_count = resolution.fft_size / 2 + 1;
    const gathered_count = frame_count * resolution.win_length;
    const window_offset = (resolution.fft_size - resolution.win_length) / 2;

    const indices = try ctx.allocator().alloc(usize, gathered_count);
    defer ctx.allocator().free(indices);
    for (0..frame_count) |frame| {
        const frame_start: i64 = @as(i64, @intCast(frame * resolution.hop_size)) - @as(i64, @intCast(reflect_pad));
        for (0..resolution.win_length) |win| {
            const source_index = frame_start + @as(i64, @intCast(window_offset + win));
            indices[frame * resolution.win_length + win] = reflectIndex(source_index, seq_len);
        }
    }

    const coeff_len = resolution.win_length * freq_count;
    const real_coeffs = try ctx.allocator().alloc(f32, coeff_len);
    defer ctx.allocator().free(real_coeffs);
    const imag_coeffs = try ctx.allocator().alloc(f32, coeff_len);
    defer ctx.allocator().free(imag_coeffs);
    fillStftCoefficients(real_coeffs, imag_coeffs, resolution, freq_count, window_offset);

    const flat = try signal.gather(ctx, .time, indices, .flat);
    const frames = try flat.split(ctx, .flat, .{ .frame, .win }, .{ frame_count, resolution.win_length });
    var real_weight = try Tensor(.{ .win, .freq }).fromSlice(ctx, .{ resolution.win_length, freq_count }, real_coeffs);
    defer real_weight.deinit();
    var imag_weight = try Tensor(.{ .win, .freq }).fromSlice(ctx, .{ resolution.win_length, freq_count }, imag_coeffs);
    defer imag_weight.deinit();

    const real = try frames.dot(ctx, &real_weight, .win);
    const imag = try frames.dot(ctx, &imag_weight, .win);
    const real_sq = try real.mul(ctx, &real);
    const imag_sq = try imag.mul(ctx, &imag);
    const power = try real_sq.add(ctx, &imag_sq);
    const clamped = try power.clamp(ctx, eps, std.math.inf(f32));
    return clamped.sqrt(ctx);
}

fn fillStftCoefficients(
    real: []f32,
    imag: []f32,
    resolution: MrstftResolution,
    freq_count: usize,
    window_offset: usize,
) void {
    const win_len_f = @as(f64, @floatFromInt(resolution.win_length));
    const fft_size_f = @as(f64, @floatFromInt(resolution.fft_size));
    for (0..resolution.win_length) |win| {
        const window_value = 0.5 - 0.5 * std.math.cos(2.0 * std.math.pi * @as(f64, @floatFromInt(win)) / win_len_f);
        const fft_sample = window_offset + win;
        for (0..freq_count) |freq| {
            const angle = -2.0 * std.math.pi * @as(f64, @floatFromInt(freq * fft_sample)) / fft_size_f;
            const idx = win * freq_count + freq;
            real[idx] = @floatCast(window_value * std.math.cos(angle));
            imag[idx] = @floatCast(window_value * std.math.sin(angle));
        }
    }
}

fn reflectIndex(index: i64, len: usize) usize {
    if (len <= 1) return 0;
    const len_i: i64 = @intCast(len);
    const period: i64 = 2 * (len_i - 1);
    var r = @mod(index, period);
    if (r >= len_i) r = period - r;
    return @intCast(r);
}

pub const PackedTrainable = struct {
    allocator: std.mem.Allocator,
    configs: []nam_file.WaveNetConfig,
    models: []wavenet.WaveNet,

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, seed: u64) !PackedTrainable {
        const count = PackedSpec.submodel_specs.len;
        const configs = try allocator.alloc(nam_file.WaveNetConfig, count);
        var configs_built: usize = 0;
        errdefer {
            for (configs[0..configs_built]) |*config| freeEngineConfig(allocator, config);
            allocator.free(configs);
        }
        const models = try allocator.alloc(wavenet.WaveNet, count);
        var models_built: usize = 0;
        errdefer {
            for (models[0..models_built]) |*model| model.deinit();
            allocator.free(models);
        }

        for (PackedSpec.submodel_specs, 0..) |sub_spec, i| {
            configs[i] = try toA2EngineConfig(allocator, &sub_spec);
            configs_built += 1;
            const sub_seed = seed +% rng.at(0x9e3779b97f4a7c15, i);
            const weights = try initWaveNetWeights(allocator, &configs[i], sub_seed);
            defer allocator.free(weights);
            models[i] = try wavenet.WaveNet.init(allocator, ctx, &configs[i], weights, .{ .trainable = true });
            models_built += 1;
        }

        return .{ .allocator = allocator, .configs = configs, .models = models };
    }

    pub fn deinit(self: *PackedTrainable) void {
        for (self.models) |*model| model.deinit();
        self.allocator.free(self.models);
        for (self.configs) |*config| freeEngineConfig(self.allocator, config);
        self.allocator.free(self.configs);
        self.* = undefined;
    }

    pub fn registerParams(self: *PackedTrainable, opt: anytype) !void {
        for (self.models) |*model| try model.registerParams(opt);
    }

    pub fn segmentLoss(self: *PackedTrainable, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        return self.segmentLossWithOptions(ctx, window, target, .{});
    }

    pub fn segmentLossWithOptions(self: *PackedTrainable, ctx: *ExecContext, window: []const f32, target: []const f32, options: LossOptions) !Tensor(.{}) {
        var total: Tensor(.{}) = undefined;
        var have_total = false;
        for (self.models) |*model| {
            const loss = try wavenetSegmentLoss(model, ctx, window, target, options);
            if (have_total) {
                total = try total.add(ctx, &loss);
            } else {
                total = loss;
                have_total = true;
            }
        }
        return total;
    }

    pub fn extractPackedSnapshot(self: *const PackedTrainable, ctx: *ExecContext, allocator: std.mem.Allocator) !PackedSnapshot {
        const submodels = try allocator.alloc(WaveNetSnapshot, self.models.len);
        var built: usize = 0;
        errdefer {
            for (submodels[0..built]) |*snapshot| snapshot.deinit(allocator);
            allocator.free(submodels);
        }
        for (self.models, self.configs, 0..) |*model, *config, i| {
            submodels[i] = try wavenetSnapshot(model, ctx, allocator, config);
            built += 1;
        }
        return .{ .submodels = submodels };
    }
};

pub const TrainingSpec = union(enum) {
    classic: ModelSpec,
    a2: A2Spec,
    packed_wavenet: PackedSpec,
    lstm: lstm.Spec,

    pub fn parse(spec_name: []const u8) !TrainingSpec {
        if (std.mem.eql(u8, spec_name, "lstm")) return .{ .lstm = lstm.Spec.standard };
        if (std.mem.eql(u8, spec_name, "tiny")) return .{ .classic = ModelSpec.tiny };
        if (std.mem.eql(u8, spec_name, "standard") or std.mem.eql(u8, spec_name, "a1") or std.mem.eql(u8, spec_name, "a1-standard")) {
            return .{ .classic = ModelSpec.classic };
        }
        if (std.mem.eql(u8, spec_name, "a2") or std.mem.eql(u8, spec_name, "a2-standard")) return .{ .a2 = A2Spec.standard };
        if (std.mem.eql(u8, spec_name, "a2-nano")) return .{ .a2 = A2Spec.nano };
        if (std.mem.eql(u8, spec_name, "packed") or std.mem.eql(u8, spec_name, "packed-a2") or std.mem.eql(u8, spec_name, "wavenet-packed")) {
            return .{ .packed_wavenet = PackedSpec.active };
        }
        return error.UnknownSpec;
    }

    pub fn name(self: *const TrainingSpec) []const u8 {
        return switch (self.*) {
            .classic => |*spec| if (spec.arrays.len == 2) "standard" else "tiny",
            .a2 => |spec| spec.name(),
            .packed_wavenet => |spec| spec.name(),
            .lstm => |spec| spec.name(),
        };
    }

    pub fn receptiveField(self: *const TrainingSpec) usize {
        return switch (self.*) {
            .classic => |*spec| spec.receptiveField(),
            .a2 => |*spec| spec.receptiveField(),
            .packed_wavenet => |*spec| spec.receptiveField(),
            .lstm => |*spec| spec.receptiveField(),
        };
    }

    /// A fresh model: WaveNet specs build their engine config, draw
    /// PyTorch's Conv1d initialization into a weight stream and load it as
    /// variables.
    pub fn initTrainable(self: *const TrainingSpec, allocator: std.mem.Allocator, ctx: *ExecContext, seed: u64) !ActiveTrainable {
        return switch (self.*) {
            .classic, .a2 => blk: {
                var config = try self.makeEngineConfig(allocator);
                defer freeEngineConfig(allocator, &config);
                const weights = try initWaveNetWeights(allocator, &config, seed);
                defer allocator.free(weights);
                break :blk .{ .wavenet = try wavenet.WaveNet.init(allocator, ctx, &config, weights, .{ .trainable = true }) };
            },
            .packed_wavenet => .{ .packed_wavenet = try PackedTrainable.init(allocator, ctx, seed) },
            .lstm => |spec| .{ .lstm = try lstm.Model.init(allocator, ctx, spec, seed) },
        };
    }

    pub fn makeEngineConfig(self: *const TrainingSpec, allocator: std.mem.Allocator) !nam_file.WaveNetConfig {
        return switch (self.*) {
            .classic => |*spec| toEngineConfig(allocator, spec),
            .a2 => |*spec| toA2EngineConfig(allocator, spec),
            .packed_wavenet, .lstm => error.UnsupportedFeature,
        };
    }

    pub fn defaultEpochs(self: *const TrainingSpec) usize {
        return switch (self.*) {
            .packed_wavenet => PackedSpec.default_epochs,
            else => 100,
        };
    }

    pub fn defaultLr(self: *const TrainingSpec) f32 {
        return switch (self.*) {
            .packed_wavenet => PackedSpec.default_lr,
            else => 0.004,
        };
    }

    pub fn defaultWeightDecay(self: *const TrainingSpec) f32 {
        return switch (self.*) {
            .packed_wavenet => PackedSpec.default_weight_decay,
            else => 0,
        };
    }

    pub fn defaultGamma(self: *const TrainingSpec) f32 {
        return switch (self.*) {
            .packed_wavenet => PackedSpec.default_gamma,
            else => 0.993,
        };
    }

    pub fn defaultMrstftWeight(self: *const TrainingSpec) f32 {
        return switch (self.*) {
            .packed_wavenet => PackedSpec.default_mrstft_weight,
            else => 0,
        };
    }
};

pub const ActiveTrainable = union(enum) {
    wavenet: wavenet.WaveNet,
    packed_wavenet: PackedTrainable,
    lstm: lstm.Model,

    pub fn deinit(self: *ActiveTrainable) void {
        switch (self.*) {
            .wavenet => |*model| model.deinit(),
            .packed_wavenet => |*model| model.deinit(),
            .lstm => |*model| model.deinit(),
        }
        self.* = undefined;
    }

    pub fn registerParams(self: *ActiveTrainable, opt: anytype) !void {
        switch (self.*) {
            .wavenet => |*model| try model.registerParams(opt),
            .packed_wavenet => |*model| try model.registerParams(opt),
            .lstm => |*model| try model.registerParams(opt),
        }
    }

    pub fn segmentLoss(self: *ActiveTrainable, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        return self.segmentLossWithOptions(ctx, window, target, .{});
    }

    pub fn segmentLossWithOptions(self: *ActiveTrainable, ctx: *ExecContext, window: []const f32, target: []const f32, options: LossOptions) !Tensor(.{}) {
        return switch (self.*) {
            .wavenet => |*model| wavenetSegmentLoss(model, ctx, window, target, options),
            .packed_wavenet => |*model| model.segmentLossWithOptions(ctx, window, target, options),
            .lstm => |*model| model.segmentLoss(ctx, window, target),
        };
    }

    pub fn extractWeights(self: *const ActiveTrainable, ctx: *ExecContext, allocator: std.mem.Allocator) ![]f32 {
        return switch (self.*) {
            .wavenet => |*model| model.extractWeights(ctx, allocator),
            .lstm => |*model| model.extractWeights(ctx, allocator),
            .packed_wavenet => error.UnsupportedFeature,
        };
    }

    pub fn extractWaveNetSnapshot(
        self: *const ActiveTrainable,
        ctx: *ExecContext,
        allocator: std.mem.Allocator,
        template_config: *const nam_file.WaveNetConfig,
    ) !WaveNetSnapshot {
        return switch (self.*) {
            .wavenet => |*model| wavenetSnapshot(model, ctx, allocator, template_config),
            .lstm => error.UnsupportedFeature,
            .packed_wavenet => error.UnsupportedFeature,
        };
    }

    pub fn extractTrainingSnapshot(
        self: *const ActiveTrainable,
        ctx: *ExecContext,
        allocator: std.mem.Allocator,
        template_config: ?*const nam_file.WaveNetConfig,
    ) !TrainingSnapshot {
        return switch (self.*) {
            .wavenet => .{ .wavenet = try self.extractWaveNetSnapshot(ctx, allocator, template_config orelse return error.UnsupportedFeature) },
            .packed_wavenet => |*model| .{ .packed_wavenet = try model.extractPackedSnapshot(ctx, allocator) },
            .lstm => |*model| .{ .lstm = .{ .config = model.spec.engineConfig(), .weights = try model.extractWeights(ctx, allocator) } },
        };
    }
};

pub const WaveNetSnapshot = struct {
    /// Borrows all architectural slices from the template config, but owns the
    /// top-level weight stream and any recursively replaced condition-DSP
    /// weight streams.
    config: nam_file.WaveNetConfig,
    weights: []f32,

    pub fn deinit(self: *WaveNetSnapshot, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        freeConditionDspSnapshot(allocator, self.config.condition_dsp);
        self.* = undefined;
    }
};

pub const PackedSnapshot = struct {
    submodels: []WaveNetSnapshot,

    pub fn deinit(self: *PackedSnapshot, allocator: std.mem.Allocator) void {
        for (self.submodels) |*snapshot| snapshot.deinit(allocator);
        allocator.free(self.submodels);
        self.* = undefined;
    }
};

pub const LstmSnapshot = struct {
    config: nam_file.LstmConfig,
    weights: []f32,

    pub fn deinit(self: *LstmSnapshot, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const TrainingSnapshot = union(enum) {
    wavenet: WaveNetSnapshot,
    packed_wavenet: PackedSnapshot,
    lstm: LstmSnapshot,

    pub fn deinit(self: *TrainingSnapshot, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .wavenet => |*snapshot| snapshot.deinit(allocator),
            .packed_wavenet => |*snapshot| snapshot.deinit(allocator),
            .lstm => |*snapshot| snapshot.deinit(allocator),
        }
        self.* = undefined;
    }
};

/// Renders `x` through the tensor LSTM from its learned initial state (the
/// validation path for an LSTM snapshot).
pub fn renderLstmConfig(allocator: std.mem.Allocator, config: *const nam_file.LstmConfig, weights: []const f32, x: []const f32, out: []f32) !void {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try lstm.Model.initFromNam(allocator, &ctx, config, weights, false, .{});
    defer model.deinit();
    var stream = try lstm.Stream.init(allocator, &ctx, &model);
    defer stream.deinit();
    try stream.process(&ctx, x, out, x.len);
}

/// Builds the engine-facing config for a spec (classic shape: ungated,
/// layer1x1 active, no head1x1, no post head, Tanh). Slices are allocated;
/// free with freeEngineConfig.
pub fn toEngineConfig(allocator: std.mem.Allocator, spec: *const ModelSpec) !nam_file.WaveNetConfig {
    const layers = try allocator.alloc(nam_file.WaveNetLayerArray, spec.arrays.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| freeLayerSlices(allocator, l);
        allocator.free(layers);
    }
    for (spec.arrays, layers) |*a, *l| {
        const n = a.dilations.len;
        const kernel_sizes = try allocator.alloc(usize, n);
        errdefer allocator.free(kernel_sizes);
        @memset(kernel_sizes, a.kernel_size);
        const activations = try allocator.alloc(nam_file.Activation, n);
        errdefer allocator.free(activations);
        @memset(activations, nam_file.Activation.tanh_default);
        const gating = try allocator.alloc(nam_file.GatingMode, n);
        errdefer allocator.free(gating);
        @memset(gating, .none);
        const secondary = try allocator.alloc(nam_file.Activation, n);
        errdefer allocator.free(secondary);
        @memset(secondary, nam_file.Activation.sigmoid_default);
        l.* = .{
            .input_size = a.input_size,
            .condition_size = 1,
            .channels = a.channels,
            .bottleneck = a.channels,
            .head_out = a.head_out,
            .head_kernel = 1,
            .head_bias = a.head_bias,
            .dilations = a.dilations,
            .kernel_sizes = kernel_sizes,
            .activations = activations,
            .gating_modes = gating,
            .secondary_activations = secondary,
            .layer1x1_active = true,
            .layer1x1_groups = 1,
            .head1x1_active = false,
            .head1x1_out = a.channels,
            .head1x1_groups = 1,
            .groups_input = 1,
            .groups_input_mixin = 1,
        };
        built += 1;
    }
    return .{ .layers = layers, .head = null, .head_scale = spec.head_scale, .in_channels = 1, .condition_dsp = null };
}

pub fn toA2EngineConfig(allocator: std.mem.Allocator, spec: *const A2Spec) !nam_file.WaveNetConfig {
    if (spec.channels != 3 and spec.channels != 8) return error.InvalidA2Channels;
    const layers = try allocator.alloc(nam_file.WaveNetLayerArray, 1);
    errdefer allocator.free(layers);

    const n = A2Spec.kernel_sizes.len;
    const kernel_sizes = try allocator.dupe(usize, &A2Spec.kernel_sizes);
    errdefer allocator.free(kernel_sizes);
    const activations = try allocator.alloc(nam_file.Activation, n);
    errdefer allocator.free(activations);
    @memset(activations, .{ .kind = .leaky_relu, .negative_slope = 0.01 });
    const gating = try allocator.alloc(nam_file.GatingMode, n);
    errdefer allocator.free(gating);
    @memset(gating, .none);
    const secondary = try allocator.alloc(nam_file.Activation, n);
    errdefer allocator.free(secondary);
    @memset(secondary, nam_file.Activation.sigmoid_default);

    layers[0] = .{
        .input_size = 1,
        .condition_size = 1,
        .channels = spec.channels,
        .bottleneck = spec.channels,
        .head_out = 1,
        .head_kernel = 16,
        .head_bias = true,
        .dilations = &A2Spec.dilations,
        .kernel_sizes = kernel_sizes,
        .activations = activations,
        .gating_modes = gating,
        .secondary_activations = secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 1,
        .head1x1_active = false,
        .head1x1_out = 1,
        .head1x1_groups = 1,
        .groups_input = 1,
        .groups_input_mixin = 1,
    };
    return .{ .layers = layers, .head = null, .head_scale = spec.head_scale, .in_channels = 1, .condition_dsp = null };
}

fn freeLayerSlices(allocator: std.mem.Allocator, l: *const nam_file.WaveNetLayerArray) void {
    allocator.free(l.kernel_sizes);
    allocator.free(l.activations);
    allocator.free(l.gating_modes);
    allocator.free(l.secondary_activations);
}

pub fn freeEngineConfig(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig) void {
    for (config.layers) |*l| freeLayerSlices(allocator, l);
    allocator.free(config.layers);
    if (config.head) |*head| allocator.free(head.kernel_sizes);
}

pub fn initWaveNetWeights(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, seed: u64) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    errdefer out.deinit(allocator);
    var seed_counter: u64 = 0;

    for (config.layers) |*array| {
        try appendRandomConvWeights(allocator, &out, array.input_size, array.channels, 1, 1, false, 1, seed, &seed_counter);
        for (0..array.layerCount()) |l| {
            const bg = array.gateWidth(l);
            try appendRandomConvWeights(allocator, &out, array.channels, bg, array.kernel_sizes[l], array.groups_input, true, array.groups_input, seed, &seed_counter);
            try appendRandomConvWeights(allocator, &out, array.condition_size, bg, 1, array.groups_input_mixin, false, array.groups_input_mixin, seed, &seed_counter);
            if (array.layer1x1_active) {
                try appendRandomConvWeights(allocator, &out, array.bottleneck, array.channels, 1, array.layer1x1_groups, true, array.layer1x1_groups, seed, &seed_counter);
            }
            if (array.head1x1_active) {
                try appendRandomConvWeights(allocator, &out, array.bottleneck, array.head1x1_out, 1, array.head1x1_groups, true, array.head1x1_groups, seed, &seed_counter);
            }
            try appendRandomFilmWeights(allocator, &out, array.condition_size, array.channels, array.conv_pre_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, bg, array.conv_post_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, array.condition_size, array.input_mixin_pre_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, bg, array.input_mixin_post_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, bg, array.activation_pre_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, array.bottleneck, array.activation_post_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, array.channels, array.layer1x1_post_film, seed, &seed_counter);
            try appendRandomFilmWeights(allocator, &out, array.condition_size, array.head1x1_out, array.head1x1_post_film, seed, &seed_counter);
        }
        const head_in = if (array.head1x1_active) array.head1x1_out else array.bottleneck;
        try appendRandomConvWeights(allocator, &out, head_in, array.head_out, array.head_kernel, 1, array.head_bias, 1, seed, &seed_counter);
    }
    if (config.head) |*head| {
        var cin = config.layers[config.layers.len - 1].head_out;
        for (head.kernel_sizes, 0..) |k, i| {
            const cout = if (i == head.kernel_sizes.len - 1) head.out_channels else head.channels;
            try appendRandomConvWeights(allocator, &out, cin, cout, k, 1, true, 1, seed, &seed_counter);
            cin = cout;
        }
    }
    try out.append(allocator, config.head_scale);
    return out.toOwnedSlice(allocator);
}

fn appendRandomFilmWeights(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(f32),
    condition_dim: usize,
    input_dim: usize,
    film: nam_file.FiLMParams,
    seed: u64,
    seed_counter: *u64,
) !void {
    if (!film.active) return;
    const out_dim = input_dim * (if (film.shift) @as(usize, 2) else 1);
    try appendRandomConvWeights(allocator, out, condition_dim, out_dim, 1, film.groups, true, film.groups, seed, seed_counter);
}

fn appendRandomConvWeights(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(f32),
    in_channels: usize,
    out_channels: usize,
    taps: usize,
    fan_groups: usize,
    has_bias: bool,
    groups: usize,
    seed: u64,
    seed_counter: *u64,
) !void {
    if (groups == 0 or fan_groups == 0) return error.InvalidConvShape;
    const in_per_group = in_channels / groups;
    const fan_in = (in_channels / fan_groups) * taps;
    const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(fan_in)));
    const weight_len = out_channels * in_per_group * taps;
    const old_len = out.items.len;
    try out.resize(allocator, old_len + weight_len);
    rng.uniformFill(rng.at(seed, seed_counter.*), out.items[old_len..][0..weight_len], -bound, bound);
    seed_counter.* += 1;
    if (has_bias) {
        const bias_old = out.items.len;
        try out.resize(allocator, bias_old + out_channels);
        rng.uniformFill(rng.at(seed, seed_counter.*), out.items[bias_old..][0..out_channels], -bound, bound);
        seed_counter.* += 1;
    }
}

/// Streams `weights` (NAM order) over `x` and returns predictions aligned
/// with x (pred[t] uses x[0..t]).
pub fn renderWeights(
    allocator: std.mem.Allocator,
    spec: *const ModelSpec,
    weights: []const f32,
    x: []const f32,
    out: []f32,
) !void {
    var config = try toEngineConfig(allocator, spec);
    defer freeEngineConfig(allocator, &config);
    try renderWaveNetConfig(allocator, &config, weights, x, out);
}

pub fn renderTrainingSpec(
    allocator: std.mem.Allocator,
    spec: *const TrainingSpec,
    weights: []const f32,
    x: []const f32,
    out: []f32,
) !void {
    var config = try spec.makeEngineConfig(allocator);
    defer freeEngineConfig(allocator, &config);
    try renderWaveNetConfig(allocator, &config, weights, x, out);
}

/// Streams `weights` (NAM order) over `x` with the tensor WaveNet in 4096-frame
/// blocks and returns predictions aligned with x (pred[t] uses x[0..t]).
pub fn renderWaveNetConfig(
    allocator: std.mem.Allocator,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
    x: []const f32,
    out: []f32,
) !void {
    const block = 4096;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try wavenet.WaveNet.init(allocator, &ctx, config, weights, .{ .chunk_hint = block });
    defer model.deinit();
    var offset: usize = 0;
    while (offset < x.len) {
        const n = @min(block, x.len - offset);
        try model.process(&ctx, x[offset..], out[offset..], n);
        offset += n;
    }
}

test {
    _ = @import("train_tests.zig");
}
