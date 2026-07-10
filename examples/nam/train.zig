//! WaveNet trainer over fucina autograd — the classic released "standard"
//! recipe (upstream neural-amp-modeler full-mode configs): pure MSE, Adam
//! lr 0.004, ExponentialLR gamma 0.993 stepped per epoch, batch 16 with
//! gradient accumulation, ny 8192, deterministic seed. Validation streams
//! the ring-buffer inference engine over the held-out split (which doubles
//! as a continuous check that the export weight order is right) and reports
//! the upstream full-pass ESR.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
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

const LayerParams = struct {
    conv_w: Tensor(.{ .tap, .ch, .bn }), // [K, C, B]
    conv_b: Tensor(.{.bn}),
    mixin_w: Tensor(.{ .bn, .cond }), // [B, 1]
    l1_w: Tensor(.{ .ch, .bn }), // [C, B]
    l1_b: Tensor(.{.ch}),
};

const ArrayParams = struct {
    rechannel_w: Tensor(.{ .chb, .ch }), // [C, in]
    layers: []LayerParams,
    head_w: Tensor(.{ .hout, .bn }), // [H, B]
    head_b: ?Tensor(.{.hout}),
};

pub const Trainable = struct {
    allocator: std.mem.Allocator,
    spec: ModelSpec,
    arrays: []ArrayParams,

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, spec: ModelSpec, seed: u64) !Trainable {
        // PyTorch Conv1d default init: weights and bias both
        // uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        var seed_counter: u64 = 0;
        var scratch: std.ArrayList(f32) = .empty;
        defer scratch.deinit(allocator);

        const arrays = try allocator.alloc(ArrayParams, spec.arrays.len);
        var built: usize = 0;
        errdefer {
            for (arrays[0..built]) |*ap| deinitArray(allocator, ap);
            allocator.free(arrays);
        }

        for (spec.arrays, arrays) |*a, *ap| {
            const c = a.channels;
            const b = a.channels; // bottleneck == channels (classic shape)
            ap.rechannel_w = try initTensor(.{ .chb, .ch }, ctx, &scratch, allocator, .{ c, a.input_size }, a.input_size, seed, &seed_counter);
            ap.head_b = null;
            ap.layers = try allocator.alloc(LayerParams, a.dilations.len);
            var layers_built: usize = 0;
            errdefer {
                ap.rechannel_w.deinit();
                for (ap.layers[0..layers_built]) |*lp| deinitLayer(lp);
                allocator.free(ap.layers);
            }
            for (ap.layers) |*lp| {
                const conv_fan = c * a.kernel_size;
                lp.conv_w = try initTensor(.{ .tap, .ch, .bn }, ctx, &scratch, allocator, .{ a.kernel_size, c, b }, conv_fan, seed, &seed_counter);
                errdefer lp.conv_w.deinit();
                lp.conv_b = try initTensor(.{.bn}, ctx, &scratch, allocator, .{b}, conv_fan, seed, &seed_counter);
                errdefer lp.conv_b.deinit();
                lp.mixin_w = try initTensor(.{ .bn, .cond }, ctx, &scratch, allocator, .{ b, 1 }, 1, seed, &seed_counter);
                errdefer lp.mixin_w.deinit();
                lp.l1_w = try initTensor(.{ .ch, .bn }, ctx, &scratch, allocator, .{ c, b }, b, seed, &seed_counter);
                errdefer lp.l1_w.deinit();
                lp.l1_b = try initTensor(.{.ch}, ctx, &scratch, allocator, .{c}, b, seed, &seed_counter);
                layers_built += 1;
            }
            ap.head_w = try initTensor(.{ .hout, .bn }, ctx, &scratch, allocator, .{ a.head_out, b }, b, seed, &seed_counter);
            errdefer ap.head_w.deinit();
            if (a.head_bias) {
                ap.head_b = try initTensor(.{.hout}, ctx, &scratch, allocator, .{a.head_out}, b, seed, &seed_counter);
            }
            built += 1;
        }

        return .{ .allocator = allocator, .spec = spec, .arrays = arrays };
    }

    fn initTensor(
        comptime tags: anytype,
        ctx: *ExecContext,
        scratch: *std.ArrayList(f32),
        allocator: std.mem.Allocator,
        shape: anytype,
        fan_in: usize,
        seed: u64,
        seed_counter: *u64,
    ) !Tensor(tags) {
        var count: usize = 1;
        inline for (shape) |dim| count *= dim;
        try scratch.resize(allocator, count);
        const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(fan_in)));
        rng.uniformFill(rng.at(seed, seed_counter.*), scratch.items, -bound, bound);
        seed_counter.* += 1;
        return Tensor(tags).variableFromSlice(ctx, shape, scratch.items);
    }

    fn deinitLayer(lp: *LayerParams) void {
        lp.conv_w.deinit();
        lp.conv_b.deinit();
        lp.mixin_w.deinit();
        lp.l1_w.deinit();
        lp.l1_b.deinit();
    }

    fn deinitArray(allocator: std.mem.Allocator, ap: *ArrayParams) void {
        ap.rechannel_w.deinit();
        ap.head_w.deinit();
        if (ap.head_b) |*t| t.deinit();
        for (ap.layers) |*lp| deinitLayer(lp);
        allocator.free(ap.layers);
    }

    pub fn deinit(self: *Trainable) void {
        for (self.arrays) |*ap| deinitArray(self.allocator, ap);
        self.allocator.free(self.arrays);
        self.* = undefined;
    }

    pub fn registerParams(self: *Trainable, opt: anytype) !void {
        for (self.arrays) |*ap| {
            try opt.addParam(&ap.rechannel_w);
            try opt.addParam(&ap.head_w);
            if (ap.head_b) |*t| try opt.addParam(t);
            for (ap.layers) |*lp| {
                try opt.addParam(&lp.conv_w);
                try opt.addParam(&lp.conv_b);
                try opt.addParam(&lp.mixin_w);
                try opt.addParam(&lp.l1_w);
                try opt.addParam(&lp.l1_b);
            }
        }
    }

    /// Full-sequence forward (zero conv state): prediction for every t of
    /// `window`. Run inside an exec scope.
    pub fn forward(self: *const Trainable, ctx: *ExecContext, window: []const f32) !Tensor(.{ .time, .hout }) {
        const frames = window.len;
        // fromSlice CREATES a caller-owned tensor (only op results are
        // scope-owned); the graph keeps refcounted views, so releasing our
        // handle here is safe.
        var cond = try Tensor(.{ .time, .cond }).fromSlice(ctx, .{ frames, 1 }, window);
        defer cond.deinit();
        var x = try cond.withTags(ctx, .{ .time, .ch });
        var head_prev: ?Tensor(.{ .time, .bn }) = null;
        var head_out: Tensor(.{ .time, .hout }) = undefined;

        for (self.arrays, self.spec.arrays) |*ap, *a| {
            const rc = try x.dot(ctx, &ap.rechannel_w, .ch);
            x = try rc.withTags(ctx, .{ .time, .ch });

            var acc = head_prev;
            for (ap.layers, 0..) |*lp, l| {
                const conv = try x.causalConv1d(ctx, .time, .ch, .tap, .bn, &lp.conv_w, a.dilations[l], null);
                const conv_b = try conv.add(ctx, &lp.conv_b);
                const mix = try cond.dot(ctx, &lp.mixin_w, .cond);
                const z = try conv_b.add(ctx, &mix);
                const activated = try z.tanh(ctx);
                acc = if (acc) |prev| try prev.add(ctx, &activated) else activated;
                const l1 = try activated.dot(ctx, &lp.l1_w, .bn);
                const l1_b = try l1.add(ctx, &lp.l1_b);
                x = try x.add(ctx, &l1_b);
            }

            var h = try acc.?.dot(ctx, &ap.head_w, .bn);
            if (ap.head_b) |*bias| h = try h.add(ctx, bias);
            head_prev = try h.withTags(ctx, .{ .time, .bn });
            head_out = h;
        }

        return head_out.scale(ctx, self.spec.head_scale);
    }

    /// MSE over the last `target.len` outputs of `window`
    /// (mean over elements, the torch F.mse_loss default).
    pub fn segmentLoss(self: *const Trainable, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        const pred = try self.forward(ctx, window);
        var target_tensor = try Tensor(.{ .time, .hout }).fromSlice(ctx, .{ target.len, 1 }, target);
        defer target_tensor.deinit();
        const tail = try pred.narrow(ctx, .time, window.len - target.len, target.len);
        const diff = try tail.sub(ctx, &target_tensor);
        const sq = try diff.mul(ctx, &diff);
        const total = try sq.sumAll(ctx);
        return total.scale(ctx, 1.0 / @as(f32, @floatFromInt(target.len)));
    }

    /// Flat weights in the canonical NAM export order (spec §6.1.2).
    pub fn extractWeights(self: *const Trainable, allocator: std.mem.Allocator) ![]f32 {
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(allocator);

        for (self.arrays, self.spec.arrays) |*ap, *a| {
            const c = a.channels;
            const b = a.channels;
            // rechannel: (out, in) row-major == our [C, in] layout.
            try out.appendSlice(allocator, try ap.rechannel_w.dataConst());
            for (ap.layers) |*lp| {
                // conv: NAM (out, in, k); ours is [k][c][b].
                const w = try lp.conv_w.dataConst();
                for (0..b) |o| {
                    for (0..c) |i| {
                        for (0..a.kernel_size) |k| {
                            try out.append(allocator, w[(k * c + i) * b + o]);
                        }
                    }
                }
                try out.appendSlice(allocator, try lp.conv_b.dataConst());
                try out.appendSlice(allocator, try lp.mixin_w.dataConst()); // (out, in) == [B, 1]
                try out.appendSlice(allocator, try lp.l1_w.dataConst()); // (out, in) == [C, B]
                try out.appendSlice(allocator, try lp.l1_b.dataConst());
            }
            try out.appendSlice(allocator, try ap.head_w.dataConst()); // (out, in) == [H, B], k=1
            if (ap.head_b) |*t| try out.appendSlice(allocator, try t.dataConst());
        }
        try out.append(allocator, self.spec.head_scale);
        return out.toOwnedSlice(allocator);
    }

    /// Loads flat NAM-order weights back into the parameter tensors
    /// (inverse of extractWeights; used to restore the best epoch).
    pub fn loadWeights(self: *Trainable, weights: []const f32) !void {
        var cursor: usize = 0;
        for (self.arrays, self.spec.arrays) |*ap, *a| {
            const c = a.channels;
            const b = a.channels;
            cursor += try copyInto(&ap.rechannel_w, weights[cursor..]);
            for (ap.layers) |*lp| {
                const w = try lp.conv_w.data();
                for (0..b) |o| {
                    for (0..c) |i| {
                        for (0..a.kernel_size) |k| {
                            w[(k * c + i) * b + o] = weights[cursor];
                            cursor += 1;
                        }
                    }
                }
                cursor += try copyInto(&lp.conv_b, weights[cursor..]);
                cursor += try copyInto(&lp.mixin_w, weights[cursor..]);
                cursor += try copyInto(&lp.l1_w, weights[cursor..]);
                cursor += try copyInto(&lp.l1_b, weights[cursor..]);
            }
            cursor += try copyInto(&ap.head_w, weights[cursor..]);
            if (ap.head_b) |*t| cursor += try copyInto(t, weights[cursor..]);
        }
        if (cursor + 1 != weights.len) return error.WeightCountMismatch;
    }

    fn copyInto(tensor: anytype, source: []const f32) !usize {
        const dst = try tensor.data();
        @memcpy(dst, source[0..dst.len]);
        return dst.len;
    }
};

const A2TimeIn = Tensor(.{ .time, .in });
const A2TimeOut = Tensor(.{ .time, .out });
const A2ConvWeight = Tensor(.{ .tap, .in_group, .out });
const A2Bias = Tensor(.{.out});

fn a2ParamFromSlice(comptime T: type, ctx: *ExecContext, shape: anytype, values: []const f32, requires_grad: bool) !T {
    return if (requires_grad)
        T.variableFromSlice(ctx, shape, values)
    else
        T.fromSlice(ctx, shape, values);
}

const A2ConvParams = struct {
    weight: A2ConvWeight,
    bias: ?A2Bias,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    taps: usize,
    dilation: usize,

    fn initFromNam(
        ctx: *ExecContext,
        allocator: std.mem.Allocator,
        scratch: *std.ArrayList(f32),
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        has_bias: bool,
        groups: usize,
        weights: []const f32,
        cursor: *usize,
        requires_grad: bool,
    ) !A2ConvParams {
        if (groups == 0 or in_channels % groups != 0 or out_channels % groups != 0) return error.InvalidConvShape;
        const in_per_group = in_channels / groups;
        const out_per_group = out_channels / groups;
        const weight_len = taps * in_per_group * out_channels;
        try scratch.resize(allocator, weight_len);
        var idx = cursor.*;
        for (0..groups) |g| {
            for (0..out_per_group) |local_o| {
                const o = g * out_per_group + local_o;
                for (0..in_per_group) |local_i| {
                    for (0..taps) |k| {
                        scratch.items[(k * in_per_group + local_i) * out_channels + o] = weights[idx];
                        idx += 1;
                    }
                }
            }
        }
        var weight = try a2ParamFromSlice(A2ConvWeight, ctx, .{ taps, in_per_group, out_channels }, scratch.items, requires_grad);
        errdefer weight.deinit();
        var bias: ?A2Bias = null;
        errdefer if (bias) |*b| b.deinit();
        if (has_bias) {
            bias = try a2ParamFromSlice(A2Bias, ctx, .{out_channels}, weights[idx..][0..out_channels], requires_grad);
            idx += out_channels;
        }
        cursor.* = idx;
        return .{
            .weight = weight,
            .bias = bias,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .groups = groups,
            .taps = taps,
            .dilation = dilation,
        };
    }

    fn deinit(self: *A2ConvParams) void {
        self.weight.deinit();
        if (self.bias) |*b| b.deinit();
        self.* = undefined;
    }

    fn registerParams(self: *A2ConvParams, opt: anytype) !void {
        try opt.addParam(&self.weight);
        if (self.bias) |*b| try opt.addParam(b);
    }

    fn forward(self: *const A2ConvParams, ctx: *ExecContext, input: *const A2TimeIn) !A2TimeOut {
        var out = try input.groupedCausalConv1d(ctx, .time, .in, .tap, .in_group, .out, &self.weight, self.dilation, self.groups, null);
        if (self.bias) |*b| out = try out.add(ctx, b);
        return out;
    }

    fn forwardNoGrad(self: *const A2ConvParams, ctx: *ExecContext, input: *const A2TimeIn) !A2TimeOut {
        var out = try input.groupedCausalConv1d(ctx, .time, .in, .tap, .in_group, .out, &self.weight, self.dilation, self.groups, null);
        errdefer out.deinit();
        if (self.bias) |*b| {
            var bias = try b.broadcastTo(ctx, .{ .time, .out }, out.shape());
            defer bias.deinit();
            return out.takeAddNoGrad(ctx, &bias);
        }
        return out;
    }

    fn requiresGrad(self: *const A2ConvParams) bool {
        if (self.weight.requiresGrad()) return true;
        if (self.bias) |*b| return b.requiresGrad();
        return false;
    }

    fn appendNamWeights(self: *const A2ConvParams, allocator: std.mem.Allocator, out: *std.ArrayList(f32)) !void {
        const w = try self.weight.dataConst();
        const in_per_group = self.in_channels / self.groups;
        const out_per_group = self.out_channels / self.groups;
        for (0..self.groups) |g| {
            for (0..out_per_group) |local_o| {
                const o = g * out_per_group + local_o;
                for (0..in_per_group) |local_i| {
                    for (0..self.taps) |k| {
                        try out.append(allocator, w[(k * in_per_group + local_i) * self.out_channels + o]);
                    }
                }
            }
        }
        if (self.bias) |*b| try out.appendSlice(allocator, try b.dataConst());
    }
};

const A2FiLMParams = struct {
    conv: A2ConvParams,
    input_dim: usize,
    shift: bool,

    fn initFromNam(
        ctx: *ExecContext,
        allocator: std.mem.Allocator,
        scratch: *std.ArrayList(f32),
        condition_dim: usize,
        input_dim: usize,
        params: nam_file.FiLMParams,
        weights: []const f32,
        cursor: *usize,
        requires_grad: bool,
    ) !?A2FiLMParams {
        if (!params.active) return null;
        const out_dim = input_dim * (if (params.shift) @as(usize, 2) else 1);
        return .{
            .conv = try A2ConvParams.initFromNam(ctx, allocator, scratch, condition_dim, out_dim, 1, 1, true, params.groups, weights, cursor, requires_grad),
            .input_dim = input_dim,
            .shift = params.shift,
        };
    }

    fn deinit(self: *A2FiLMParams) void {
        self.conv.deinit();
        self.* = undefined;
    }

    fn registerParams(self: *A2FiLMParams, opt: anytype) !void {
        try self.conv.registerParams(opt);
    }

    fn forward(self: *const A2FiLMParams, ctx: *ExecContext, condition: *const A2TimeIn, input: *const A2TimeOut) !A2TimeOut {
        const affine = try self.conv.forward(ctx, condition);
        const scale = try affine.narrow(ctx, .out, 0, self.input_dim);
        const scaled = try input.mul(ctx, &scale);
        if (!self.shift) return scaled;
        const shift = try affine.narrow(ctx, .out, self.input_dim, self.input_dim);
        return scaled.add(ctx, &shift);
    }

    fn forwardNoGrad(self: *const A2FiLMParams, ctx: *ExecContext, condition: *const A2TimeIn, input: *const A2TimeOut) !A2TimeOut {
        var affine = try self.conv.forwardNoGrad(ctx, condition);
        defer affine.deinit();
        var scale = try affine.narrow(ctx, .out, 0, self.input_dim);
        defer scale.deinit();
        var scaled = try input.mul(ctx, &scale);
        errdefer scaled.deinit();
        if (!self.shift) return scaled;
        var shift = try affine.narrow(ctx, .out, self.input_dim, self.input_dim);
        defer shift.deinit();
        return ctx.replace(scaled, scaled.add(ctx, &shift));
    }

    fn requiresGrad(self: *const A2FiLMParams) bool {
        return self.conv.requiresGrad();
    }

    fn appendNamWeights(self: *const A2FiLMParams, allocator: std.mem.Allocator, out: *std.ArrayList(f32)) !void {
        try self.conv.appendNamWeights(allocator, out);
    }
};

const A2LayerParams = struct {
    conv: A2ConvParams,
    input_mixin: A2ConvParams,
    layer1x1: ?A2ConvParams,
    head1x1: ?A2ConvParams,
    conv_pre_film: ?A2FiLMParams,
    conv_post_film: ?A2FiLMParams,
    input_mixin_pre_film: ?A2FiLMParams,
    input_mixin_post_film: ?A2FiLMParams,
    activation_pre_film: ?A2FiLMParams,
    activation_post_film: ?A2FiLMParams,
    layer1x1_post_film: ?A2FiLMParams,
    head1x1_post_film: ?A2FiLMParams,
    activation: nam_file.Activation,
    secondary_activation: nam_file.Activation,
    gating_mode: nam_file.GatingMode,
    bottleneck: usize,
};

const A2ArrayParams = struct {
    rechannel: A2ConvParams,
    layers: []A2LayerParams,
    head_rechannel: A2ConvParams,
    channels: usize,
    head_width: usize,

    fn deinit(self: *A2ArrayParams, allocator: std.mem.Allocator) void {
        self.rechannel.deinit();
        for (self.layers) |*layer| deinitA2Layer(layer);
        allocator.free(self.layers);
        self.head_rechannel.deinit();
        self.* = undefined;
    }
};

const A2PostHeadBlock = struct {
    conv: A2ConvParams,
    activation: nam_file.Activation,
};

pub const A2Trainable = struct {
    allocator: std.mem.Allocator,
    arrays: []A2ArrayParams,
    post_head: []A2PostHeadBlock,
    condition_dsp: ?*A2Trainable,
    condition_channels: usize,
    head_scale: f32,
    fast_tanh: bool = false,

    pub fn initFromWaveNet(
        allocator: std.mem.Allocator,
        ctx: *ExecContext,
        config: *const nam_file.WaveNetConfig,
        weights: []const f32,
    ) !A2Trainable {
        return initFromWaveNetMode(allocator, ctx, config, weights, true);
    }

    pub fn initConstFromWaveNet(
        allocator: std.mem.Allocator,
        ctx: *ExecContext,
        config: *const nam_file.WaveNetConfig,
        weights: []const f32,
    ) !A2Trainable {
        return initFromWaveNetMode(allocator, ctx, config, weights, false);
    }

    fn initFromWaveNetMode(
        allocator: std.mem.Allocator,
        ctx: *ExecContext,
        config: *const nam_file.WaveNetConfig,
        weights: []const f32,
        requires_grad: bool,
    ) !A2Trainable {
        var scratch: std.ArrayList(f32) = .empty;
        defer scratch.deinit(allocator);

        const arrays = try allocator.alloc(A2ArrayParams, config.layers.len);
        var arrays_built: usize = 0;
        errdefer {
            for (arrays[0..arrays_built]) |*array| array.deinit(allocator);
            allocator.free(arrays);
        }

        var cursor: usize = 0;
        var prev_head_out: usize = 0;
        for (config.layers, arrays, 0..) |*lc, *array, i| {
            array.* = try buildArray(ctx, allocator, &scratch, lc, weights, &cursor, requires_grad);
            arrays_built += 1;
            if (i > 0 and array.head_width != prev_head_out) return error.UnsupportedFeature;
            prev_head_out = array.head_rechannel.out_channels;
        }

        var post_head: []A2PostHeadBlock = &.{};
        var post_built: usize = 0;
        errdefer {
            for (post_head[0..post_built]) |*block| block.conv.deinit();
            allocator.free(post_head);
        }
        if (config.head) |*hc| {
            post_head = try allocator.alloc(A2PostHeadBlock, hc.kernel_sizes.len);
            var cin = config.layers[config.layers.len - 1].head_out;
            for (post_head, hc.kernel_sizes, 0..) |*block, k, i| {
                const cout = if (i == hc.kernel_sizes.len - 1) hc.out_channels else hc.channels;
                block.* = .{
                    .conv = try A2ConvParams.initFromNam(ctx, allocator, &scratch, cin, cout, k, 1, true, 1, weights, &cursor, requires_grad),
                    .activation = hc.activation,
                };
                post_built += 1;
                cin = cout;
            }
        }
        if (cursor + 1 != weights.len) return error.WeightCountMismatch;

        var child: ?*A2Trainable = null;
        errdefer if (child) |ptr| {
            ptr.deinit();
            allocator.destroy(ptr);
        };
        var condition_channels: usize = 1;
        if (config.condition_dsp) |dsp| {
            switch (dsp.config) {
                .wavenet => |*child_config| {
                    const ptr = try allocator.create(A2Trainable);
                    errdefer allocator.destroy(ptr);
                    ptr.* = try A2Trainable.initFromWaveNetMode(allocator, ctx, child_config, dsp.weights, requires_grad);
                    child = ptr;
                    condition_channels = ptr.outputChannels();
                },
                .lstm, .convnet, .linear => return error.UnsupportedFeature,
            }
        }
        for (config.layers) |*lc| {
            if (lc.condition_size != condition_channels) return error.UnsupportedChannels;
        }

        return .{
            .allocator = allocator,
            .arrays = arrays,
            .post_head = post_head,
            .condition_dsp = child,
            .condition_channels = condition_channels,
            .head_scale = weights[cursor],
            .fast_tanh = false,
        };
    }

    fn buildArray(
        ctx: *ExecContext,
        allocator: std.mem.Allocator,
        scratch: *std.ArrayList(f32),
        lc: *const nam_file.WaveNetLayerArray,
        weights: []const f32,
        cursor: *usize,
        requires_grad: bool,
    ) !A2ArrayParams {
        var rechannel = try A2ConvParams.initFromNam(ctx, allocator, scratch, lc.input_size, lc.channels, 1, 1, false, 1, weights, cursor, requires_grad);
        errdefer rechannel.deinit();

        const layers = try allocator.alloc(A2LayerParams, lc.layerCount());
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*layer| deinitA2Layer(layer);
            allocator.free(layers);
        }
        for (layers, 0..) |*layer, l| {
            layer.* = try buildLayer(ctx, allocator, scratch, lc, l, weights, cursor, requires_grad);
            built += 1;
        }

        const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
        var head_rechannel = try A2ConvParams.initFromNam(ctx, allocator, scratch, head_width, lc.head_out, lc.head_kernel, 1, lc.head_bias, 1, weights, cursor, requires_grad);
        errdefer head_rechannel.deinit();

        return .{
            .rechannel = rechannel,
            .layers = layers,
            .head_rechannel = head_rechannel,
            .channels = lc.channels,
            .head_width = head_width,
        };
    }

    fn buildLayer(
        ctx: *ExecContext,
        allocator: std.mem.Allocator,
        scratch: *std.ArrayList(f32),
        lc: *const nam_file.WaveNetLayerArray,
        layer_index: usize,
        weights: []const f32,
        cursor: *usize,
        requires_grad: bool,
    ) !A2LayerParams {
        const bg = lc.gateWidth(layer_index);
        var conv = try A2ConvParams.initFromNam(ctx, allocator, scratch, lc.channels, bg, lc.kernel_sizes[layer_index], lc.dilations[layer_index], true, lc.groups_input, weights, cursor, requires_grad);
        errdefer conv.deinit();
        var input_mixin = try A2ConvParams.initFromNam(ctx, allocator, scratch, lc.condition_size, bg, 1, 1, false, lc.groups_input_mixin, weights, cursor, requires_grad);
        errdefer input_mixin.deinit();

        var layer1x1: ?A2ConvParams = null;
        errdefer if (layer1x1) |*c| c.deinit();
        if (lc.layer1x1_active) {
            layer1x1 = try A2ConvParams.initFromNam(ctx, allocator, scratch, lc.bottleneck, lc.channels, 1, 1, true, lc.layer1x1_groups, weights, cursor, requires_grad);
        }
        var head1x1: ?A2ConvParams = null;
        errdefer if (head1x1) |*c| c.deinit();
        if (lc.head1x1_active) {
            head1x1 = try A2ConvParams.initFromNam(ctx, allocator, scratch, lc.bottleneck, lc.head1x1_out, 1, 1, true, lc.head1x1_groups, weights, cursor, requires_grad);
        }

        var conv_pre_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, lc.channels, lc.conv_pre_film, weights, cursor, requires_grad);
        errdefer if (conv_pre_film) |*f| f.deinit();
        var conv_post_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, bg, lc.conv_post_film, weights, cursor, requires_grad);
        errdefer if (conv_post_film) |*f| f.deinit();
        var input_mixin_pre_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, lc.condition_size, lc.input_mixin_pre_film, weights, cursor, requires_grad);
        errdefer if (input_mixin_pre_film) |*f| f.deinit();
        var input_mixin_post_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, bg, lc.input_mixin_post_film, weights, cursor, requires_grad);
        errdefer if (input_mixin_post_film) |*f| f.deinit();
        var activation_pre_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, bg, lc.activation_pre_film, weights, cursor, requires_grad);
        errdefer if (activation_pre_film) |*f| f.deinit();
        var activation_post_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, lc.bottleneck, lc.activation_post_film, weights, cursor, requires_grad);
        errdefer if (activation_post_film) |*f| f.deinit();
        var layer1x1_post_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, lc.channels, lc.layer1x1_post_film, weights, cursor, requires_grad);
        errdefer if (layer1x1_post_film) |*f| f.deinit();
        var head1x1_post_film = try A2FiLMParams.initFromNam(ctx, allocator, scratch, lc.condition_size, lc.head1x1_out, lc.head1x1_post_film, weights, cursor, requires_grad);
        errdefer if (head1x1_post_film) |*f| f.deinit();

        return .{
            .conv = conv,
            .input_mixin = input_mixin,
            .layer1x1 = layer1x1,
            .head1x1 = head1x1,
            .conv_pre_film = conv_pre_film,
            .conv_post_film = conv_post_film,
            .input_mixin_pre_film = input_mixin_pre_film,
            .input_mixin_post_film = input_mixin_post_film,
            .activation_pre_film = activation_pre_film,
            .activation_post_film = activation_post_film,
            .layer1x1_post_film = layer1x1_post_film,
            .head1x1_post_film = head1x1_post_film,
            .activation = lc.activations[layer_index],
            .secondary_activation = lc.secondary_activations[layer_index],
            .gating_mode = lc.gating_modes[layer_index],
            .bottleneck = lc.bottleneck,
        };
    }

    pub fn deinit(self: *A2Trainable) void {
        for (self.arrays) |*array| array.deinit(self.allocator);
        self.allocator.free(self.arrays);
        for (self.post_head) |*block| block.conv.deinit();
        self.allocator.free(self.post_head);
        if (self.condition_dsp) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.* = undefined;
    }

    pub fn registerParams(self: *A2Trainable, opt: anytype) !void {
        if (self.condition_dsp) |child| try child.registerParams(opt);
        for (self.arrays) |*array| {
            try array.rechannel.registerParams(opt);
            for (array.layers) |*layer| try registerA2LayerParams(layer, opt);
            try array.head_rechannel.registerParams(opt);
        }
        for (self.post_head) |*block| try block.conv.registerParams(opt);
    }

    pub fn setFastTanh(self: *A2Trainable, enabled: bool) void {
        self.fast_tanh = enabled;
        if (self.condition_dsp) |child| child.setFastTanh(enabled);
    }

    pub fn outputChannels(self: *const A2Trainable) usize {
        if (self.post_head.len > 0) return self.post_head[self.post_head.len - 1].conv.out_channels;
        return self.arrays[self.arrays.len - 1].head_rechannel.out_channels;
    }

    pub fn requiresGrad(self: *const A2Trainable) bool {
        if (self.condition_dsp) |child| {
            if (child.requiresGrad()) return true;
        }
        for (self.arrays) |*array| {
            if (array.rechannel.requiresGrad()) return true;
            for (array.layers) |*layer| {
                if (a2LayerRequiresGrad(layer)) return true;
            }
            if (array.head_rechannel.requiresGrad()) return true;
        }
        for (self.post_head) |*block| {
            if (block.conv.requiresGrad()) return true;
        }
        return false;
    }

    pub fn forward(self: *const A2Trainable, ctx: *ExecContext, window: []const f32) !A2TimeOut {
        var input = try A2TimeIn.fromSlice(ctx, .{ window.len, 1 }, window);
        defer input.deinit();
        return self.forwardTensor(ctx, &input);
    }

    pub fn renderBorrowed(self: *const A2Trainable, ctx: *ExecContext, window: []f32, out: []f32) !void {
        if (out.len != window.len * self.outputChannels()) return error.InvalidOutputLength;
        if (!ctx.execScopeActive()) return self.renderBorrowedNoGrad(ctx, window, out);

        var input = try A2TimeIn.fromBorrowedSlice(ctx, .{ window.len, 1 }, window);
        defer input.deinit();
        var pred = try self.forwardTensor(ctx, &input);
        defer pred.deinit();
        try pred.copyTo(out);
    }

    pub fn renderBorrowedNoGrad(self: *const A2Trainable, ctx: *ExecContext, window: []f32, out: []f32) !void {
        if (out.len != window.len * self.outputChannels()) return error.InvalidOutputLength;
        if (ctx.execScopeActive()) return error.ActiveExecScopeUnsupported;
        if (self.requiresGrad()) return error.RequiresGrad;
        var input = try A2TimeIn.fromBorrowedSlice(ctx, .{ window.len, 1 }, window);
        defer input.deinit();
        var pred = try self.forwardTensorNoGrad(ctx, &input);
        defer pred.deinit();
        try pred.copyTo(out);
    }

    pub fn forwardTensor(self: *const A2Trainable, ctx: *ExecContext, input: *const A2TimeIn) !A2TimeOut {
        const condition = blk: {
            if (self.condition_dsp) |child| {
                const child_out = try child.forwardTensor(ctx, input);
                break :blk try child_out.withTags(ctx, .{ .time, .in });
            }
            break :blk try input.withTags(ctx, .{ .time, .in });
        };

        var x: A2TimeIn = undefined;
        var head_prev: ?A2TimeIn = null;
        var head_out: A2TimeOut = undefined;

        for (self.arrays, 0..) |*array, array_index| {
            const array_input = if (array_index == 0) input else &x;
            const rc = try array.rechannel.forward(ctx, array_input);
            x = try rc.withTags(ctx, .{ .time, .in });

            var acc: ?A2TimeOut = null;
            if (head_prev) |prev| {
                acc = try prev.withTags(ctx, .{ .time, .out });
            }

            for (array.layers) |*layer| {
                const conv_input = blk: {
                    if (layer.conv_pre_film) |*film| {
                        const x_out = try x.withTags(ctx, .{ .time, .out });
                        const filmed = try film.forward(ctx, &condition, &x_out);
                        break :blk try filmed.withTags(ctx, .{ .time, .in });
                    }
                    break :blk x;
                };
                var z = try layer.conv.forward(ctx, &conv_input);
                if (layer.conv_post_film) |*film| {
                    z = try film.forward(ctx, &condition, &z);
                }

                const mixin_input = blk: {
                    if (layer.input_mixin_pre_film) |*film| {
                        const condition_out = try condition.withTags(ctx, .{ .time, .out });
                        const filmed = try film.forward(ctx, &condition, &condition_out);
                        break :blk try filmed.withTags(ctx, .{ .time, .in });
                    }
                    break :blk condition;
                };
                var mix = try layer.input_mixin.forward(ctx, &mixin_input);
                if (layer.input_mixin_post_film) |*film| {
                    mix = try film.forward(ctx, &condition, &mix);
                }
                z = try z.add(ctx, &mix);

                if (layer.activation_pre_film) |*film| {
                    z = try film.forward(ctx, &condition, &z);
                }

                var activated = try applyA2Gating(ctx, layer, &z, self.fast_tanh);
                if (layer.activation_post_film) |*film| {
                    activated = try film.forward(ctx, &condition, &activated);
                }

                if (layer.layer1x1) |*conv| {
                    const residual_in = try activated.withTags(ctx, .{ .time, .in });
                    var residual = try conv.forward(ctx, &residual_in);
                    if (layer.gating_mode == .blended) {
                        if (layer.layer1x1_post_film) |*film| {
                            residual = try film.forward(ctx, &condition, &residual);
                        }
                    }
                    const residual_as_in = try residual.withTags(ctx, .{ .time, .in });
                    x = try x.add(ctx, &residual_as_in);
                }

                var contribution = blk: {
                    if (layer.head1x1) |*conv| {
                        const head_in = try activated.withTags(ctx, .{ .time, .in });
                        var head = try conv.forward(ctx, &head_in);
                        if (layer.head1x1_post_film) |*film| {
                            head = try film.forward(ctx, &condition, &head);
                        }
                        break :blk head;
                    }
                    break :blk activated;
                };
                if (acc) |prev| {
                    contribution = try prev.add(ctx, &contribution);
                }
                acc = contribution;
            }

            const head_in = try acc.?.withTags(ctx, .{ .time, .in });
            head_out = try array.head_rechannel.forward(ctx, &head_in);
            head_prev = try head_out.withTags(ctx, .{ .time, .in });
        }

        var current = try head_out.scale(ctx, self.head_scale);
        for (self.post_head) |*block| {
            const activated = try applyA2Activation(ctx, &block.activation, &current, self.fast_tanh);
            const conv_in = try activated.withTags(ctx, .{ .time, .in });
            current = try block.conv.forward(ctx, &conv_in);
        }
        return current;
    }

    pub fn forwardTensorNoGrad(self: *const A2Trainable, ctx: *ExecContext, input: *const A2TimeIn) !A2TimeOut {
        if (ctx.execScopeActive()) return error.ActiveExecScopeUnsupported;
        if (input.requiresGrad() or self.requiresGrad()) return error.RequiresGrad;

        var child_out: ?A2TimeOut = null;
        defer if (child_out) |*t| t.deinit();
        var condition = blk: {
            if (self.condition_dsp) |child| {
                child_out = try child.forwardTensorNoGrad(ctx, input);
                break :blk try child_out.?.withTags(ctx, .{ .time, .in });
            }
            break :blk try input.withTags(ctx, .{ .time, .in });
        };
        defer condition.deinit();

        var x: A2TimeIn = undefined;
        var have_x = false;
        defer if (have_x) x.deinit();
        var head_prev: ?A2TimeIn = null;
        defer if (head_prev) |*t| t.deinit();
        var final_head: ?A2TimeOut = null;
        errdefer if (final_head) |*t| t.deinit();

        for (self.arrays, 0..) |*array, array_index| {
            const array_input = if (array_index == 0) input else &x;
            var rc = try array.rechannel.forwardNoGrad(ctx, array_input);
            errdefer rc.deinit();
            const next_x = try rc.withTags(ctx, .{ .time, .in });
            rc.deinit();
            if (have_x) x.deinit();
            x = next_x;
            have_x = true;

            var acc: ?A2TimeOut = null;
            errdefer if (acc) |*t| t.deinit();
            if (head_prev) |*prev| {
                acc = try prev.withTags(ctx, .{ .time, .out });
                prev.deinit();
                head_prev = null;
            }

            for (array.layers) |*layer| {
                var z: A2TimeOut = undefined;
                var have_z = false;
                errdefer if (have_z) z.deinit();
                if (layer.conv_pre_film) |*film| {
                    var x_out = try x.withTags(ctx, .{ .time, .out });
                    defer x_out.deinit();
                    var filmed = try film.forwardNoGrad(ctx, &condition, &x_out);
                    defer filmed.deinit();
                    var conv_input = try filmed.withTags(ctx, .{ .time, .in });
                    defer conv_input.deinit();
                    z = try layer.conv.forwardNoGrad(ctx, &conv_input);
                    have_z = true;
                } else {
                    z = try layer.conv.forwardNoGrad(ctx, &x);
                    have_z = true;
                }
                if (layer.conv_post_film) |*film| {
                    z = try ctx.replace(z, film.forwardNoGrad(ctx, &condition, &z));
                }

                var mix: A2TimeOut = undefined;
                var have_mix = false;
                errdefer if (have_mix) mix.deinit();
                if (layer.input_mixin_pre_film) |*film| {
                    var condition_out = try condition.withTags(ctx, .{ .time, .out });
                    defer condition_out.deinit();
                    var filmed = try film.forwardNoGrad(ctx, &condition, &condition_out);
                    defer filmed.deinit();
                    var mixin_input = try filmed.withTags(ctx, .{ .time, .in });
                    defer mixin_input.deinit();
                    mix = try layer.input_mixin.forwardNoGrad(ctx, &mixin_input);
                    have_mix = true;
                } else {
                    mix = try layer.input_mixin.forwardNoGrad(ctx, &condition);
                    have_mix = true;
                }
                if (layer.input_mixin_post_film) |*film| {
                    mix = try ctx.replace(mix, film.forwardNoGrad(ctx, &condition, &mix));
                }

                const z_sum = try z.takeAddNoGrad(ctx, &mix);
                have_z = false;
                mix.deinit();
                have_mix = false;
                z = z_sum;
                have_z = true;

                if (layer.activation_pre_film) |*film| {
                    z = try ctx.replace(z, film.forwardNoGrad(ctx, &condition, &z));
                }

                var activated = try applyA2GatingNoGrad(ctx, layer, &z, self.fast_tanh);
                z.deinit();
                have_z = false;
                var have_activated = true;
                errdefer if (have_activated) activated.deinit();
                if (layer.activation_post_film) |*film| {
                    activated = try ctx.replace(activated, film.forwardNoGrad(ctx, &condition, &activated));
                }

                if (layer.layer1x1) |*conv| {
                    var residual_in = try activated.withTags(ctx, .{ .time, .in });
                    defer residual_in.deinit();
                    var residual = try conv.forwardNoGrad(ctx, &residual_in);
                    errdefer residual.deinit();
                    if (layer.gating_mode == .blended) {
                        if (layer.layer1x1_post_film) |*film| {
                            residual = try ctx.replace(residual, film.forwardNoGrad(ctx, &condition, &residual));
                        }
                    }
                    var residual_as_in = try residual.withTags(ctx, .{ .time, .in });
                    defer residual_as_in.deinit();
                    const next_residual_x = try x.takeAddNoGrad(ctx, &residual_as_in);
                    residual.deinit();
                    x = next_residual_x;
                    have_x = true;
                }

                var contribution: A2TimeOut = undefined;
                var have_contribution = false;
                errdefer if (have_contribution) contribution.deinit();
                if (layer.head1x1) |*conv| {
                    var head_in = try activated.withTags(ctx, .{ .time, .in });
                    defer head_in.deinit();
                    contribution = try conv.forwardNoGrad(ctx, &head_in);
                    have_contribution = true;
                    if (layer.head1x1_post_film) |*film| {
                        contribution = try ctx.replace(contribution, film.forwardNoGrad(ctx, &condition, &contribution));
                    }
                    activated.deinit();
                    have_activated = false;
                } else {
                    contribution = activated;
                    have_activated = false;
                    have_contribution = true;
                }

                if (acc) |*prev| {
                    const next_acc = try prev.takeAddNoGrad(ctx, &contribution);
                    contribution.deinit();
                    have_contribution = false;
                    acc = next_acc;
                } else {
                    acc = contribution;
                    have_contribution = false;
                }
            }

            var head_in = try acc.?.withTags(ctx, .{ .time, .in });
            defer head_in.deinit();
            var new_head = try array.head_rechannel.forwardNoGrad(ctx, &head_in);
            acc.?.deinit();
            acc = null;
            if (array_index + 1 < self.arrays.len) {
                head_prev = try new_head.withTags(ctx, .{ .time, .in });
                new_head.deinit();
            } else {
                final_head = new_head;
            }
        }

        var current = try (&final_head.?).takeScaleNoGrad(ctx, self.head_scale);
        final_head = null;
        errdefer current.deinit();
        for (self.post_head) |*block| {
            var activated = try applyA2ActivationNoGrad(ctx, &block.activation, &current, self.fast_tanh);
            defer activated.deinit();
            var conv_in = try activated.withTags(ctx, .{ .time, .in });
            defer conv_in.deinit();
            const next = try block.conv.forwardNoGrad(ctx, &conv_in);
            current.deinit();
            current = next;
        }
        return current;
    }

    pub fn segmentLoss(self: *const A2Trainable, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        return self.segmentLossWithOptions(ctx, window, target, .{});
    }

    pub fn segmentLossWithOptions(self: *const A2Trainable, ctx: *ExecContext, window: []const f32, target: []const f32, options: LossOptions) !Tensor(.{}) {
        const pred = try self.forward(ctx, window);
        var target_tensor = try A2TimeOut.fromSlice(ctx, .{ target.len, 1 }, target);
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

    pub fn extractWeights(self: *const A2Trainable, allocator: std.mem.Allocator) ![]f32 {
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(allocator);
        for (self.arrays) |*array| {
            try array.rechannel.appendNamWeights(allocator, &out);
            for (array.layers) |*layer| {
                try layer.conv.appendNamWeights(allocator, &out);
                try layer.input_mixin.appendNamWeights(allocator, &out);
                if (layer.layer1x1) |*conv| try conv.appendNamWeights(allocator, &out);
                if (layer.head1x1) |*conv| try conv.appendNamWeights(allocator, &out);
                try appendOptionalFilm(allocator, &out, &layer.conv_pre_film);
                try appendOptionalFilm(allocator, &out, &layer.conv_post_film);
                try appendOptionalFilm(allocator, &out, &layer.input_mixin_pre_film);
                try appendOptionalFilm(allocator, &out, &layer.input_mixin_post_film);
                try appendOptionalFilm(allocator, &out, &layer.activation_pre_film);
                try appendOptionalFilm(allocator, &out, &layer.activation_post_film);
                try appendOptionalFilm(allocator, &out, &layer.layer1x1_post_film);
                try appendOptionalFilm(allocator, &out, &layer.head1x1_post_film);
            }
            try array.head_rechannel.appendNamWeights(allocator, &out);
        }
        for (self.post_head) |*block| try block.conv.appendNamWeights(allocator, &out);
        try out.append(allocator, self.head_scale);
        return out.toOwnedSlice(allocator);
    }

    pub fn extractWaveNetSnapshot(
        self: *const A2Trainable,
        allocator: std.mem.Allocator,
        template_config: *const nam_file.WaveNetConfig,
    ) anyerror!WaveNetSnapshot {
        const weights = try self.extractWeights(allocator);
        errdefer allocator.free(weights);
        var config = template_config.*;
        config.condition_dsp = try self.extractConditionDspSnapshot(allocator, template_config.condition_dsp);
        errdefer freeConditionDspSnapshot(allocator, config.condition_dsp);
        return .{ .config = config, .weights = weights };
    }

    fn extractConditionDspSnapshot(
        self: *const A2Trainable,
        allocator: std.mem.Allocator,
        template: ?*const nam_file.ConditionDsp,
    ) anyerror!?*const nam_file.ConditionDsp {
        const child = self.condition_dsp orelse {
            if (template != null) return error.UnsupportedFeature;
            return null;
        };
        const dsp = template orelse return error.UnsupportedFeature;
        if (dsp.architecture != .wavenet or dsp.config != .wavenet) return error.UnsupportedFeature;
        const out = try allocator.create(nam_file.ConditionDsp);
        errdefer allocator.destroy(out);
        const child_snapshot = try child.extractWaveNetSnapshot(allocator, &dsp.config.wavenet);
        out.* = .{
            .architecture = .wavenet,
            .config = .{ .wavenet = child_snapshot.config },
            .weights = child_snapshot.weights,
            .sample_rate = dsp.sample_rate,
        };
        return out;
    }
};

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

fn deinitA2Layer(layer: *A2LayerParams) void {
    layer.conv.deinit();
    layer.input_mixin.deinit();
    if (layer.layer1x1) |*conv| conv.deinit();
    if (layer.head1x1) |*conv| conv.deinit();
    if (layer.conv_pre_film) |*film| film.deinit();
    if (layer.conv_post_film) |*film| film.deinit();
    if (layer.input_mixin_pre_film) |*film| film.deinit();
    if (layer.input_mixin_post_film) |*film| film.deinit();
    if (layer.activation_pre_film) |*film| film.deinit();
    if (layer.activation_post_film) |*film| film.deinit();
    if (layer.layer1x1_post_film) |*film| film.deinit();
    if (layer.head1x1_post_film) |*film| film.deinit();
}

fn registerA2LayerParams(layer: *A2LayerParams, opt: anytype) !void {
    try layer.conv.registerParams(opt);
    try layer.input_mixin.registerParams(opt);
    if (layer.layer1x1) |*conv| try conv.registerParams(opt);
    if (layer.head1x1) |*conv| try conv.registerParams(opt);
    if (layer.conv_pre_film) |*film| try film.registerParams(opt);
    if (layer.conv_post_film) |*film| try film.registerParams(opt);
    if (layer.input_mixin_pre_film) |*film| try film.registerParams(opt);
    if (layer.input_mixin_post_film) |*film| try film.registerParams(opt);
    if (layer.activation_pre_film) |*film| try film.registerParams(opt);
    if (layer.activation_post_film) |*film| try film.registerParams(opt);
    if (layer.layer1x1_post_film) |*film| try film.registerParams(opt);
    if (layer.head1x1_post_film) |*film| try film.registerParams(opt);
}

fn a2LayerRequiresGrad(layer: *const A2LayerParams) bool {
    if (layer.conv.requiresGrad()) return true;
    if (layer.input_mixin.requiresGrad()) return true;
    if (layer.layer1x1) |*conv| {
        if (conv.requiresGrad()) return true;
    }
    if (layer.head1x1) |*conv| {
        if (conv.requiresGrad()) return true;
    }
    inline for (.{
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    }) |field| {
        if (@field(layer, field)) |*film| {
            if (film.requiresGrad()) return true;
        }
    }
    return false;
}

fn appendOptionalFilm(allocator: std.mem.Allocator, out: *std.ArrayList(f32), film: *const ?A2FiLMParams) !void {
    if (film.*) |*f| try f.appendNamWeights(allocator, out);
}

fn a2Scalar(ctx: *ExecContext, value: f32) !Tensor(.{}) {
    return Tensor(.{}).fromSlice(ctx, .{1}, &.{value});
}

fn a2AddScalar(ctx: *ExecContext, x: *const A2TimeOut, value: f32) !A2TimeOut {
    var scalar = try a2Scalar(ctx, value);
    defer scalar.deinit();
    return x.add(ctx, &scalar);
}

fn applyA2Gating(ctx: *ExecContext, layer: *const A2LayerParams, z: *const A2TimeOut, use_fast_tanh: bool) !A2TimeOut {
    const b = layer.bottleneck;
    return switch (layer.gating_mode) {
        .none => blk: {
            const body = try z.narrow(ctx, .out, 0, b);
            break :blk try applyA2Activation(ctx, &layer.activation, &body, use_fast_tanh);
        },
        .gated => blk: {
            const top = try z.narrow(ctx, .out, 0, b);
            const bottom = try z.narrow(ctx, .out, b, b);
            const primary = try applyA2Activation(ctx, &layer.activation, &top, use_fast_tanh);
            break :blk try applyA2SecondaryGate(ctx, &primary, &layer.secondary_activation, &bottom, use_fast_tanh);
        },
        .blended => blk: {
            const top = try z.narrow(ctx, .out, 0, b);
            const bottom = try z.narrow(ctx, .out, b, b);
            const primary = try applyA2Activation(ctx, &layer.activation, &top, use_fast_tanh);
            if (layer.secondary_activation.kind == .sigmoid) {
                const alpha_primary = try primary.glu(ctx, &bottom);
                const neg_bottom = try bottom.scale(ctx, -1.0);
                const passthrough = try top.glu(ctx, &neg_bottom);
                break :blk try alpha_primary.add(ctx, &passthrough);
            }
            const alpha = try applyA2Activation(ctx, &layer.secondary_activation, &bottom, use_fast_tanh);
            const alpha_primary = try alpha.mul(ctx, &primary);
            const neg_alpha = try alpha.scale(ctx, -1.0);
            const one_minus_alpha = try a2AddScalar(ctx, &neg_alpha, 1.0);
            const passthrough = try one_minus_alpha.mul(ctx, &top);
            break :blk try alpha_primary.add(ctx, &passthrough);
        },
    };
}

fn applyA2GatingNoGrad(ctx: *ExecContext, layer: *const A2LayerParams, z: *const A2TimeOut, use_fast_tanh: bool) !A2TimeOut {
    const b = layer.bottleneck;
    return switch (layer.gating_mode) {
        .none => blk: {
            var body = try z.narrow(ctx, .out, 0, b);
            defer body.deinit();
            break :blk try applyA2ActivationNoGrad(ctx, &layer.activation, &body, use_fast_tanh);
        },
        .gated => blk: {
            var top = try z.narrow(ctx, .out, 0, b);
            defer top.deinit();
            var bottom = try z.narrow(ctx, .out, b, b);
            defer bottom.deinit();
            var primary = try applyA2ActivationNoGrad(ctx, &layer.activation, &top, use_fast_tanh);
            defer primary.deinit();
            break :blk try applyA2SecondaryGateNoGrad(ctx, &primary, &layer.secondary_activation, &bottom, use_fast_tanh);
        },
        .blended => blk: {
            var top = try z.narrow(ctx, .out, 0, b);
            defer top.deinit();
            var bottom = try z.narrow(ctx, .out, b, b);
            defer bottom.deinit();
            var primary = try applyA2ActivationNoGrad(ctx, &layer.activation, &top, use_fast_tanh);
            defer primary.deinit();
            if (layer.secondary_activation.kind == .sigmoid) {
                var alpha_primary = try primary.glu(ctx, &bottom);
                defer alpha_primary.deinit();
                var neg_bottom = try bottom.scale(ctx, -1.0);
                defer neg_bottom.deinit();
                var passthrough = try top.glu(ctx, &neg_bottom);
                defer passthrough.deinit();
                break :blk try alpha_primary.add(ctx, &passthrough);
            }
            var alpha = try applyA2ActivationNoGrad(ctx, &layer.secondary_activation, &bottom, use_fast_tanh);
            defer alpha.deinit();
            var alpha_primary = try alpha.mul(ctx, &primary);
            defer alpha_primary.deinit();
            var neg_alpha = try alpha.scale(ctx, -1.0);
            defer neg_alpha.deinit();
            var one_minus_alpha = try a2AddScalar(ctx, &neg_alpha, 1.0);
            defer one_minus_alpha.deinit();
            var passthrough = try one_minus_alpha.mul(ctx, &top);
            defer passthrough.deinit();
            break :blk try alpha_primary.add(ctx, &passthrough);
        },
    };
}

fn applyA2SecondaryGate(ctx: *ExecContext, primary: *const A2TimeOut, secondary: *const nam_file.Activation, bottom: *const A2TimeOut, use_fast_tanh: bool) !A2TimeOut {
    return switch (secondary.kind) {
        .sigmoid => primary.glu(ctx, bottom),
        .silu => primary.swiglu(ctx, bottom),
        else => blk: {
            const secondary_value = try applyA2Activation(ctx, secondary, bottom, use_fast_tanh);
            break :blk try primary.mul(ctx, &secondary_value);
        },
    };
}

fn applyA2SecondaryGateNoGrad(ctx: *ExecContext, primary: *const A2TimeOut, secondary: *const nam_file.Activation, bottom: *const A2TimeOut, use_fast_tanh: bool) !A2TimeOut {
    return switch (secondary.kind) {
        .sigmoid => primary.glu(ctx, bottom),
        .silu => primary.swiglu(ctx, bottom),
        else => blk: {
            var secondary_value = try applyA2ActivationNoGrad(ctx, secondary, bottom, use_fast_tanh);
            defer secondary_value.deinit();
            break :blk try primary.mul(ctx, &secondary_value);
        },
    };
}

fn applyA2Activation(ctx: *ExecContext, act: *const nam_file.Activation, x: *const A2TimeOut, use_fast_tanh: bool) !A2TimeOut {
    return switch (act.kind) {
        .tanh => if (use_fast_tanh) x.fastTanh(ctx) else x.tanh(ctx),
        .hardtanh => x.clamp(ctx, -1.0, 1.0),
        .fasttanh => x.fastTanh(ctx),
        .relu => x.relu(ctx),
        .leaky_relu => x.leakyRelu(ctx, act.negative_slope),
        .prelu => applyA2PRelu(ctx, act, x),
        .sigmoid => x.sigmoid(ctx),
        .silu => x.silu(ctx),
        .hardswish => blk: {
            const shifted = try a2AddScalar(ctx, x, 3.0);
            const clipped = try shifted.clamp(ctx, 0.0, 6.0);
            const product = try x.mul(ctx, &clipped);
            break :blk try product.scale(ctx, 1.0 / 6.0);
        },
        .leaky_hardtanh => blk: {
            const middle = try x.clamp(ctx, act.min_val, act.max_val);
            const below_arg = try a2AddScalar(ctx, x, -act.min_val);
            const below = try below_arg.clamp(ctx, -std.math.inf(f32), 0.0);
            const below_scaled = try below.scale(ctx, act.min_slope);
            const with_below = try middle.add(ctx, &below_scaled);
            const above_arg = try a2AddScalar(ctx, x, -act.max_val);
            const above = try above_arg.clamp(ctx, 0.0, std.math.inf(f32));
            const above_scaled = try above.scale(ctx, act.max_slope);
            break :blk try with_below.add(ctx, &above_scaled);
        },
        .softsign => blk: {
            const abs_x = try x.abs(ctx);
            const denom = try a2AddScalar(ctx, &abs_x, 1.0);
            break :blk try x.div(ctx, &denom);
        },
    };
}

fn applyA2ActivationNoGrad(ctx: *ExecContext, act: *const nam_file.Activation, x: *const A2TimeOut, use_fast_tanh: bool) !A2TimeOut {
    return switch (act.kind) {
        .tanh => if (use_fast_tanh) x.fastTanh(ctx) else x.tanh(ctx),
        .hardtanh => x.clamp(ctx, -1.0, 1.0),
        .fasttanh => x.fastTanh(ctx),
        .relu => x.relu(ctx),
        .leaky_relu => x.leakyRelu(ctx, act.negative_slope),
        .prelu => applyA2PReluNoGrad(ctx, act, x),
        .sigmoid => x.sigmoid(ctx),
        .silu => x.silu(ctx),
        .hardswish => blk: {
            var shifted = try a2AddScalar(ctx, x, 3.0);
            defer shifted.deinit();
            var clipped = try shifted.clamp(ctx, 0.0, 6.0);
            defer clipped.deinit();
            var product = try x.mul(ctx, &clipped);
            defer product.deinit();
            break :blk try product.scale(ctx, 1.0 / 6.0);
        },
        .leaky_hardtanh => blk: {
            var middle = try x.clamp(ctx, act.min_val, act.max_val);
            defer middle.deinit();
            var below_arg = try a2AddScalar(ctx, x, -act.min_val);
            defer below_arg.deinit();
            var below = try below_arg.clamp(ctx, -std.math.inf(f32), 0.0);
            defer below.deinit();
            var below_scaled = try below.scale(ctx, act.min_slope);
            defer below_scaled.deinit();
            var with_below = try middle.add(ctx, &below_scaled);
            defer with_below.deinit();
            var above_arg = try a2AddScalar(ctx, x, -act.max_val);
            defer above_arg.deinit();
            var above = try above_arg.clamp(ctx, 0.0, std.math.inf(f32));
            defer above.deinit();
            var above_scaled = try above.scale(ctx, act.max_slope);
            defer above_scaled.deinit();
            break :blk try with_below.add(ctx, &above_scaled);
        },
        .softsign => blk: {
            var abs_x = try x.abs(ctx);
            defer abs_x.deinit();
            var denom = try a2AddScalar(ctx, &abs_x, 1.0);
            defer denom.deinit();
            break :blk try x.div(ctx, &denom);
        },
    };
}

fn a2FastTanh(ctx: *ExecContext, x: *const A2TimeOut) !A2TimeOut {
    const abs_x = try x.abs(ctx);
    const x2 = try x.mul(ctx, x);

    const ax = try abs_x.scale(ctx, 2.45550750702956);
    const first = try a2AddScalar(ctx, &ax, 2.45550750702956);
    const cx = try abs_x.scale(ctx, 0.821226666969744);
    const second_base = try a2AddScalar(ctx, &cx, 0.893229853513558);
    const second = try second_base.mul(ctx, &x2);
    const numerator_base = try first.add(ctx, &second);
    const numerator = try x.mul(ctx, &numerator_base);

    const x_abs = try x.mul(ctx, &abs_x);
    const scaled_x_abs = try x_abs.scale(ctx, 0.814642734961073);
    const denom_abs_arg = try x.add(ctx, &scaled_x_abs);
    const denom_abs = try denom_abs_arg.abs(ctx);
    const d_plus_x2 = try a2AddScalar(ctx, &x2, 2.44506634652299);
    const denom_tail = try d_plus_x2.mul(ctx, &denom_abs);
    const denom = try a2AddScalar(ctx, &denom_tail, 2.44506634652299);
    return numerator.div(ctx, &denom);
}

fn a2FastTanhNoGrad(ctx: *ExecContext, x: *const A2TimeOut) !A2TimeOut {
    var abs_x = try x.abs(ctx);
    defer abs_x.deinit();
    var x2 = try x.mul(ctx, x);
    defer x2.deinit();

    var ax = try abs_x.scale(ctx, 2.45550750702956);
    defer ax.deinit();
    var first = try a2AddScalar(ctx, &ax, 2.45550750702956);
    defer first.deinit();
    var cx = try abs_x.scale(ctx, 0.821226666969744);
    defer cx.deinit();
    var second_base = try a2AddScalar(ctx, &cx, 0.893229853513558);
    defer second_base.deinit();
    var second = try second_base.mul(ctx, &x2);
    defer second.deinit();
    var numerator_base = try first.add(ctx, &second);
    defer numerator_base.deinit();
    var numerator = try x.mul(ctx, &numerator_base);
    defer numerator.deinit();

    var x_abs = try x.mul(ctx, &abs_x);
    defer x_abs.deinit();
    var scaled_x_abs = try x_abs.scale(ctx, 0.814642734961073);
    defer scaled_x_abs.deinit();
    var denom_abs_arg = try x.add(ctx, &scaled_x_abs);
    defer denom_abs_arg.deinit();
    var denom_abs = try denom_abs_arg.abs(ctx);
    defer denom_abs.deinit();
    var d_plus_x2 = try a2AddScalar(ctx, &x2, 2.44506634652299);
    defer d_plus_x2.deinit();
    var denom_tail = try d_plus_x2.mul(ctx, &denom_abs);
    defer denom_tail.deinit();
    var denom = try a2AddScalar(ctx, &denom_tail, 2.44506634652299);
    defer denom.deinit();
    return numerator.div(ctx, &denom);
}

fn applyA2PRelu(ctx: *ExecContext, act: *const nam_file.Activation, x: *const A2TimeOut) !A2TimeOut {
    const positive = try x.relu(ctx);
    const negative = try x.clamp(ctx, -std.math.inf(f32), 0.0);
    const width = x.dim(.out);
    if (act.negative_slopes.len == 0) {
        const scaled_negative = try negative.scale(ctx, act.negative_slope);
        return positive.add(ctx, &scaled_negative);
    }

    const slopes = try ctx.allocator.alloc(f32, width);
    defer ctx.allocator.free(slopes);
    for (slopes, 0..) |*dst, i| dst.* = act.negative_slopes[i % act.negative_slopes.len];
    var slope_tensor = try A2Bias.fromSlice(ctx, .{width}, slopes);
    defer slope_tensor.deinit();
    const scaled_negative = try negative.mul(ctx, &slope_tensor);
    return positive.add(ctx, &scaled_negative);
}

fn applyA2PReluNoGrad(ctx: *ExecContext, act: *const nam_file.Activation, x: *const A2TimeOut) !A2TimeOut {
    var positive = try x.relu(ctx);
    defer positive.deinit();
    var negative = try x.clamp(ctx, -std.math.inf(f32), 0.0);
    defer negative.deinit();
    const width = x.dim(.out);
    if (act.negative_slopes.len == 0) {
        var scaled_negative = try negative.scale(ctx, act.negative_slope);
        defer scaled_negative.deinit();
        return positive.add(ctx, &scaled_negative);
    }

    const slopes = try ctx.allocator.alloc(f32, width);
    defer ctx.allocator.free(slopes);
    for (slopes, 0..) |*dst, i| dst.* = act.negative_slopes[i % act.negative_slopes.len];
    var slope_tensor = try A2Bias.fromSlice(ctx, .{width}, slopes);
    defer slope_tensor.deinit();
    var scaled_negative = try negative.mul(ctx, &slope_tensor);
    defer scaled_negative.deinit();
    return positive.add(ctx, &scaled_negative);
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

// --- Long-term average spectrum (LTAS) tone matching -------------------------
// The MagicMatch-style objective: match the *average* magnitude spectrum (the
// tonal/"formant" envelope) of the model output to a reference, with no frame
// alignment — so the model input and the reference can be different musical
// content (unpaired). This is what makes it tone *matching*, not NAM's paired
// transfer-function capture.

/// STFT resolution used for LTAS tone matching.
pub const ltas_resolution = MrstftResolution{ .fft_size = 2048, .hop_size = 512, .win_length = 2048 };
pub const ltas_eps: f32 = 1e-7;

// Perceptual (mel/ERB-like) band projection. Matching raw FFT bins over-weights
// inaudible fine detail and under-weights perceptual bands; projecting the
// envelope onto ~48 mel bands before the log-spectral loss is the cheapest
// reliable spectral-realism win. 48 kHz pipeline assumed.
pub const n_mels: usize = 48;
pub const mel_sample_rate: f64 = 48000.0;

fn hzToMel(f: f64) f64 {
    return 2595.0 * std.math.log10(1.0 + f / 700.0);
}
fn melToHz(m: f64) f64 {
    return 700.0 * (std.math.pow(f64, 10.0, m / 2595.0) - 1.0);
}

/// Triangular mel filterbank as [n_freq * n_mels] row-major ([freq][mel]).
fn melMatrix(allocator: std.mem.Allocator, n_freq: usize) ![]f32 {
    const fft_size = (n_freq - 1) * 2;
    const pts = n_mels + 2;
    const centers = try allocator.alloc(f64, pts);
    defer allocator.free(centers);
    const mmin = hzToMel(20.0);
    const mmax = hzToMel(mel_sample_rate / 2.0);
    for (centers, 0..) |*v, i| {
        v.* = melToHz(mmin + (mmax - mmin) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(pts - 1)));
    }
    const mat = try allocator.alloc(f32, n_freq * n_mels);
    @memset(mat, 0);
    for (0..n_freq) |k| {
        const f = @as(f64, @floatFromInt(k)) * mel_sample_rate / @as(f64, @floatFromInt(fft_size));
        for (0..n_mels) |mi| {
            const left = centers[mi];
            const center = centers[mi + 1];
            const right = centers[mi + 2];
            var w: f64 = 0;
            if (f >= left and f <= center and center > left) {
                w = (f - left) / (center - left);
            } else if (f > center and f <= right and right > center) {
                w = (right - f) / (right - center);
            }
            mat[k * n_mels + mi] = @floatCast(w);
        }
    }
    return mat;
}

/// Project a freq envelope onto mel bands (constant matrix; differentiable dot).
fn melProject(ctx: *ExecContext, env: *const Tensor(.{.freq})) !Tensor(.{.mel}) {
    const n_freq = env.dim(.freq);
    const m = try melMatrix(ctx.allocator, n_freq);
    defer ctx.allocator.free(m);
    var mat = try Tensor(.{ .freq, .mel }).fromSlice(ctx, .{ n_freq, n_mels }, m);
    defer mat.deinit();
    return env.dot(ctx, &mat, .freq);
}

/// Long-term average MEL spectrum (mean over STFT frames, projected to mel bands).
/// Differentiable; must run inside an exec scope.
pub fn ltasEnvelope(ctx: *ExecContext, signal: *const Tensor(.{.time}), resolution: MrstftResolution, eps: f32) !Tensor(.{.mel}) {
    const mag = try stftMagnitude(ctx, signal, resolution, eps);
    const e = try mag.mean(ctx, .frame);
    return melProject(ctx, &e);
}

/// LTAS envelope of a raw signal as an owned []f32 (no grad — opens its own scope).
pub fn ltasOfSignal(ctx: *ExecContext, allocator: std.mem.Allocator, signal: []const f32, resolution: MrstftResolution, eps: f32) ![]f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var sig = try Tensor(.{.time}).fromSlice(ctx, .{signal.len}, signal);
    defer sig.deinit();
    const env = try ltasEnvelope(ctx, &sig, resolution, eps);
    return allocator.dupe(f32, try env.dataConst());
}

/// Unpaired tone-match loss: mean-squared log-magnitude difference between
/// `pred`'s LTAS and a fixed target LTAS envelope (the standard log-spectral
/// distance — operating in the log domain keeps it shape-focused and stable,
/// unlike a raw-magnitude term that chases overall level). Differentiable in
/// `pred`; `target_ltas` is treated as a constant.
pub fn ltasMatchLoss(ctx: *ExecContext, pred: *const Tensor(.{.time}), target_ltas: []const f32, resolution: MrstftResolution, eps: f32) !Tensor(.{}) {
    const pred_ltas = try ltasEnvelope(ctx, pred, resolution, eps);
    if (pred_ltas.dim(.mel) != target_ltas.len) return error.InvalidMrstftShape;
    var target_tensor = try Tensor(.{.mel}).fromSlice(ctx, .{target_ltas.len}, target_ltas);
    defer target_tensor.deinit();

    const pred_log = try pred_ltas.log(ctx);
    const tgt_log = try target_tensor.log(ctx);
    const log_diff = try pred_log.sub(ctx, &tgt_log);
    const log_sq = try log_diff.mul(ctx, &log_diff);
    const log_sum = try log_sq.sumAll(ctx);
    return log_sum.scale(ctx, 1.0 / @as(f32, @floatFromInt(pred_ltas.dim(.mel))));
}

// --- Richer tone-match features (capture nonlinearity, not just EQ) -----------
// A single LTAS is a global EQ and cannot tell clean from distorted. We add:
//  E1, an ENERGY-WEIGHTED envelope (emphasises loud frames) — together with the
//      flat LTAS (E0) its difference encodes how the spectrum fans out with drive
//      (the saturation/level-dependence signature); and
//  a sample-amplitude SOFT HISTOGRAM — the static waveshaping fingerprint
//      (clipping piles mass at the rails), independent of which notes are played.
// All differentiable, all on existing ops. Design drawn from the amp-modeling /
// unpaired-timbre-transfer literature.

pub const hist_bins: usize = 33;
pub const hist_half_range: f32 = 4.0; // +/- 4 RMS-normalised units
pub const hist_sigma: f32 = 0.30; // KDE bandwidth in the same units

/// Energy-weighted (loud-emphasis) average spectrum: sum_f (pow_f * mag_f) / sum_f pow_f.
pub fn energyWeightedEnvelope(ctx: *ExecContext, mag: *const Tensor(.{ .frame, .freq })) !Tensor(.{.mel}) {
    const power = try mag.mul(ctx, mag);
    const frame_pow = try power.sum(ctx, .freq); // {frame}
    const weighted = try mag.mul(ctx, &frame_pow); // {frame,freq} (frame_pow broadcast over freq)
    const wsum = try weighted.sum(ctx, .frame); // {freq}
    const norm = try frame_pow.sum(ctx, .frame); // {}
    const env = try wsum.div(ctx, &norm); // {freq}
    return melProject(ctx, &env); // {mel}
}

/// Energy-weighted envelope of a raw signal as owned []f32 (no grad).
pub fn e1OfSignal(ctx: *ExecContext, allocator: std.mem.Allocator, signal: []const f32, resolution: MrstftResolution, eps: f32) ![]f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var sig = try Tensor(.{.time}).fromSlice(ctx, .{signal.len}, signal);
    defer sig.deinit();
    const mag = try stftMagnitude(ctx, &sig, resolution, eps);
    const env = try energyWeightedEnvelope(ctx, &mag);
    return allocator.dupe(f32, try env.dataConst());
}

/// Level-invariant log-spectral SHAPE loss (mean-removed log-magnitude L2).
pub fn logSpecShapeLoss(ctx: *ExecContext, pred_env: *const Tensor(.{.mel}), ref_env: *const Tensor(.{.mel}), eps: f32) !Tensor(.{}) {
    const pl = try (try pred_env.clamp(ctx, eps, std.math.inf(f32))).log(ctx);
    const rl = try (try ref_env.clamp(ctx, eps, std.math.inf(f32))).log(ctx);
    const d = try pl.sub(ctx, &rl); // {mel}
    const dm = try d.mean(ctx, .mel); // {}
    const dc = try d.sub(ctx, &dm); // {mel} mean-removed (broadcast)
    const dc2 = try dc.mul(ctx, &dc);
    return dc2.mean(ctx, .mel); // {}
}

/// Fixed KDE bin centers in RMS-normalised units, as owned []f32.
pub fn histCenters(allocator: std.mem.Allocator) ![]f32 {
    const c = try allocator.alloc(f32, hist_bins);
    for (c, 0..) |*v, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(hist_bins - 1)); // 0..1
        v.* = -hist_half_range + 2.0 * hist_half_range * t;
    }
    return c;
}

/// Soft (KDE) histogram of RMS-normalised sample amplitudes, summing to 1.
pub fn softHistogram(ctx: *ExecContext, signal: *const Tensor(.{.time}), centers: *const Tensor(.{.bin}), eps: f32) !Tensor(.{.bin}) {
    const sq = try signal.mul(ctx, signal);
    const ms = try sq.mean(ctx, .time); // {} mean square
    const rms = try (try ms.clamp(ctx, eps, std.math.inf(f32))).sqrt(ctx); // {}
    const xn = try signal.div(ctx, &rms); // {time} normalised (broadcast scalar)
    const diff = try xn.sub(ctx, centers); // {time,bin} outer broadcast
    const d2 = try diff.mul(ctx, &diff);
    const inv = -1.0 / (2.0 * hist_sigma * hist_sigma);
    const k = try (try d2.scale(ctx, inv)).exp(ctx); // {time,bin}
    const hist = try k.mean(ctx, .time); // {bin}
    const total = try hist.sum(ctx, .bin); // {}
    return hist.div(ctx, &total); // {bin}
}

/// Soft histogram of a raw signal as owned []f32 (no grad).
pub fn histOfSignal(ctx: *ExecContext, allocator: std.mem.Allocator, signal: []const f32, eps: f32) ![]f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const centers_slice = try histCenters(ctx.allocator);
    defer ctx.allocator.free(centers_slice);
    var sig = try Tensor(.{.time}).fromSlice(ctx, .{signal.len}, signal);
    defer sig.deinit();
    var centers = try Tensor(.{.bin}).fromSlice(ctx, .{hist_bins}, centers_slice);
    defer centers.deinit();
    const hist = try softHistogram(ctx, &sig, &centers, eps);
    return allocator.dupe(f32, try hist.dataConst());
}

/// L2 between two soft histograms (scaled up — histogram entries are tiny).
pub fn histLoss(ctx: *ExecContext, pred_hist: *const Tensor(.{.bin}), ref_hist: *const Tensor(.{.bin})) !Tensor(.{}) {
    const d = try pred_hist.sub(ctx, ref_hist);
    const d2 = try d.mul(ctx, &d);
    const m = try d2.mean(ctx, .bin);
    return m.scale(ctx, @as(f32, @floatFromInt(hist_bins * hist_bins)));
}

// Dynamics / "feel": the distribution of the short-time loudness ENVELOPE. A
// compressed amp squashes the envelope into a narrow band; a dynamic one spreads
// it. Matching the (mean-removed log) envelope histogram pushes the model toward
// the reference's compression/sustain behaviour — the part spectral averages miss.
pub const env_half_range: f32 = 2.5; // natural-log units (~±21 dB)
pub const env_sigma: f32 = 0.30;

pub fn envHistCenters(allocator: std.mem.Allocator) ![]f32 {
    const c = try allocator.alloc(f32, hist_bins);
    for (c, 0..) |*v, i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(hist_bins - 1));
        v.* = -env_half_range + 2.0 * env_half_range * t;
    }
    return c;
}

/// Soft histogram of the mean-removed log loudness-envelope (per STFT frame).
pub fn envHistogram(ctx: *ExecContext, mag: *const Tensor(.{ .frame, .freq }), centers: *const Tensor(.{.bin}), eps: f32) !Tensor(.{.bin}) {
    const power = try mag.mul(ctx, mag);
    const frame_pow = try power.sum(ctx, .freq); // {frame}
    const env = try (try frame_pow.clamp(ctx, eps, std.math.inf(f32))).sqrt(ctx); // {frame}
    const logenv = try (try env.clamp(ctx, eps, std.math.inf(f32))).log(ctx); // {frame}
    const m = try logenv.mean(ctx, .frame); // {}
    const centered = try logenv.sub(ctx, &m); // {frame} broadcast
    const diff = try centered.sub(ctx, centers); // {frame,bin}
    const d2 = try diff.mul(ctx, &diff);
    const inv = -1.0 / (2.0 * env_sigma * env_sigma);
    const k = try (try d2.scale(ctx, inv)).exp(ctx); // {frame,bin}
    const hist = try k.mean(ctx, .frame); // {bin}
    const total = try hist.sum(ctx, .bin); // {}
    return hist.div(ctx, &total);
}

/// Envelope histogram of a raw signal as owned []f32 (no grad).
pub fn envHistOfSignal(ctx: *ExecContext, allocator: std.mem.Allocator, signal: []const f32, resolution: MrstftResolution, eps: f32) ![]f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const centers_slice = try envHistCenters(ctx.allocator);
    defer ctx.allocator.free(centers_slice);
    var sig = try Tensor(.{.time}).fromSlice(ctx, .{signal.len}, signal);
    defer sig.deinit();
    var centers = try Tensor(.{.bin}).fromSlice(ctx, .{hist_bins}, centers_slice);
    defer centers.deinit();
    const mag = try stftMagnitude(ctx, &sig, resolution, eps);
    const h = try envHistogram(ctx, &mag, &centers, eps);
    return allocator.dupe(f32, try h.dataConst());
}

/// Precomputed (no-grad) target statistics for the composite tone-match loss.
pub const ToneRefs = struct {
    e0: []const f32, // flat LTAS envelope (EQ / cab voicing)
    e1: []const f32, // energy-weighted envelope (saturation vs level)
    hist: []const f32, // sample-amplitude soft histogram (waveshaping shape)
    env_hist: []const f32, // loudness-envelope histogram (compression / dynamics)
};

/// Composite unpaired tone-match loss: level-conditioned log-spectral shape
/// (E0 flat + E1 loud-emphasis) + sample-amplitude histogram. Differentiable in
/// `pred`; refs are constants. This is the composite objective that captures
/// the amp's nonlinear/saturation character, not just its average EQ.
pub fn toneMatchLoss(
    ctx: *ExecContext,
    pred: *const Tensor(.{.time}),
    refs: ToneRefs,
    resolution: MrstftResolution,
    eps: f32,
    w_e1: f32,
    w_hist: f32,
    w_env: f32,
) !Tensor(.{}) {
    const mag = try stftMagnitude(ctx, pred, resolution, eps);
    const e0_freq = try mag.mean(ctx, .frame); // {freq}
    const e0 = try melProject(ctx, &e0_freq); // {mel}
    const e1 = try energyWeightedEnvelope(ctx, &mag); // {mel}
    var e0_ref = try Tensor(.{.mel}).fromSlice(ctx, .{refs.e0.len}, refs.e0);
    defer e0_ref.deinit();
    var e1_ref = try Tensor(.{.mel}).fromSlice(ctx, .{refs.e1.len}, refs.e1);
    defer e1_ref.deinit();
    const l0 = try logSpecShapeLoss(ctx, &e0, &e0_ref, eps);
    const l1 = try logSpecShapeLoss(ctx, &e1, &e1_ref, eps);
    const l1w = try l1.scale(ctx, w_e1);
    var loss = try l0.add(ctx, &l1w);

    if (w_hist > 0) {
        const centers_slice = try histCenters(ctx.allocator);
        defer ctx.allocator.free(centers_slice);
        var centers = try Tensor(.{.bin}).fromSlice(ctx, .{hist_bins}, centers_slice);
        defer centers.deinit();
        const ph = try softHistogram(ctx, pred, &centers, eps);
        var rh = try Tensor(.{.bin}).fromSlice(ctx, .{refs.hist.len}, refs.hist);
        defer rh.deinit();
        const lh = try histLoss(ctx, &ph, &rh);
        const lhw = try lh.scale(ctx, w_hist);
        loss = try loss.add(ctx, &lhw);
    }
    if (w_env > 0) {
        const centers_slice = try envHistCenters(ctx.allocator);
        defer ctx.allocator.free(centers_slice);
        var centers = try Tensor(.{.bin}).fromSlice(ctx, .{hist_bins}, centers_slice);
        defer centers.deinit();
        const pe = try envHistogram(ctx, &mag, &centers, eps);
        var re = try Tensor(.{.bin}).fromSlice(ctx, .{refs.env_hist.len}, refs.env_hist);
        defer re.deinit();
        const le = try histLoss(ctx, &pe, &re);
        const lew = try le.scale(ctx, w_env);
        loss = try loss.add(ctx, &lew);
    }
    return loss;
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

    const indices = try ctx.allocator.alloc(usize, gathered_count);
    defer ctx.allocator.free(indices);
    for (0..frame_count) |frame| {
        const frame_start: i64 = @as(i64, @intCast(frame * resolution.hop_size)) - @as(i64, @intCast(reflect_pad));
        for (0..resolution.win_length) |win| {
            const source_index = frame_start + @as(i64, @intCast(window_offset + win));
            indices[frame * resolution.win_length + win] = reflectIndex(source_index, seq_len);
        }
    }

    const coeff_len = resolution.win_length * freq_count;
    const real_coeffs = try ctx.allocator.alloc(f32, coeff_len);
    defer ctx.allocator.free(real_coeffs);
    const imag_coeffs = try ctx.allocator.alloc(f32, coeff_len);
    defer ctx.allocator.free(imag_coeffs);
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
    models: []A2Trainable,

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, seed: u64) !PackedTrainable {
        const count = PackedSpec.submodel_specs.len;
        const configs = try allocator.alloc(nam_file.WaveNetConfig, count);
        var configs_built: usize = 0;
        errdefer {
            for (configs[0..configs_built]) |*config| freeEngineConfig(allocator, config);
            allocator.free(configs);
        }
        const models = try allocator.alloc(A2Trainable, count);
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
            models[i] = try A2Trainable.initFromWaveNet(allocator, ctx, &configs[i], weights);
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

    pub fn segmentLoss(self: *const PackedTrainable, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        return self.segmentLossWithOptions(ctx, window, target, .{});
    }

    pub fn segmentLossWithOptions(self: *const PackedTrainable, ctx: *ExecContext, window: []const f32, target: []const f32, options: LossOptions) !Tensor(.{}) {
        var total: Tensor(.{}) = undefined;
        var have_total = false;
        for (self.models) |*model| {
            const loss = try model.segmentLossWithOptions(ctx, window, target, options);
            if (have_total) {
                total = try total.add(ctx, &loss);
            } else {
                total = loss;
                have_total = true;
            }
        }
        return total;
    }

    pub fn extractPackedSnapshot(self: *const PackedTrainable, allocator: std.mem.Allocator) !PackedSnapshot {
        const submodels = try allocator.alloc(WaveNetSnapshot, self.models.len);
        var built: usize = 0;
        errdefer {
            for (submodels[0..built]) |*snapshot| snapshot.deinit(allocator);
            allocator.free(submodels);
        }
        for (self.models, self.configs, 0..) |*model, *config, i| {
            submodels[i] = try model.extractWaveNetSnapshot(allocator, config);
            built += 1;
        }
        return .{ .submodels = submodels };
    }
};

pub const TrainingSpec = union(enum) {
    classic: ModelSpec,
    a2: A2Spec,
    packed_wavenet: PackedSpec,

    pub fn parse(spec_name: []const u8) !TrainingSpec {
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
        };
    }

    pub fn receptiveField(self: *const TrainingSpec) usize {
        return switch (self.*) {
            .classic => |*spec| spec.receptiveField(),
            .a2 => |*spec| spec.receptiveField(),
            .packed_wavenet => |*spec| spec.receptiveField(),
        };
    }

    pub fn initTrainable(self: *const TrainingSpec, allocator: std.mem.Allocator, ctx: *ExecContext, seed: u64) !ActiveTrainable {
        return switch (self.*) {
            .classic => |spec| .{ .classic = try Trainable.init(allocator, ctx, spec, seed) },
            .a2 => |spec| blk: {
                var config = try toA2EngineConfig(allocator, &spec);
                defer freeEngineConfig(allocator, &config);
                const weights = try initWaveNetWeights(allocator, &config, seed);
                defer allocator.free(weights);
                break :blk .{ .a2 = try A2Trainable.initFromWaveNet(allocator, ctx, &config, weights) };
            },
            .packed_wavenet => .{ .packed_wavenet = try PackedTrainable.init(allocator, ctx, seed) },
        };
    }

    pub fn makeEngineConfig(self: *const TrainingSpec, allocator: std.mem.Allocator) !nam_file.WaveNetConfig {
        return switch (self.*) {
            .classic => |*spec| toEngineConfig(allocator, spec),
            .a2 => |*spec| toA2EngineConfig(allocator, spec),
            .packed_wavenet => error.UnsupportedFeature,
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
    classic: Trainable,
    a2: A2Trainable,
    packed_wavenet: PackedTrainable,

    pub fn deinit(self: *ActiveTrainable) void {
        switch (self.*) {
            .classic => |*model| model.deinit(),
            .a2 => |*model| model.deinit(),
            .packed_wavenet => |*model| model.deinit(),
        }
        self.* = undefined;
    }

    pub fn registerParams(self: *ActiveTrainable, opt: anytype) !void {
        switch (self.*) {
            .classic => |*model| try model.registerParams(opt),
            .a2 => |*model| try model.registerParams(opt),
            .packed_wavenet => |*model| try model.registerParams(opt),
        }
    }

    pub fn segmentLoss(self: *const ActiveTrainable, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        return self.segmentLossWithOptions(ctx, window, target, .{});
    }

    pub fn segmentLossWithOptions(self: *const ActiveTrainable, ctx: *ExecContext, window: []const f32, target: []const f32, options: LossOptions) !Tensor(.{}) {
        return switch (self.*) {
            .classic => |*model| model.segmentLoss(ctx, window, target),
            .a2 => |*model| model.segmentLossWithOptions(ctx, window, target, options),
            .packed_wavenet => |*model| model.segmentLossWithOptions(ctx, window, target, options),
        };
    }

    pub fn extractWeights(self: *const ActiveTrainable, allocator: std.mem.Allocator) ![]f32 {
        return switch (self.*) {
            .classic => |*model| model.extractWeights(allocator),
            .a2 => |*model| model.extractWeights(allocator),
            .packed_wavenet => error.UnsupportedFeature,
        };
    }

    pub fn extractWaveNetSnapshot(
        self: *const ActiveTrainable,
        allocator: std.mem.Allocator,
        template_config: *const nam_file.WaveNetConfig,
    ) !WaveNetSnapshot {
        return switch (self.*) {
            .classic => |*model| .{
                .config = template_config.*,
                .weights = try model.extractWeights(allocator),
            },
            .a2 => |*model| model.extractWaveNetSnapshot(allocator, template_config),
            .packed_wavenet => error.UnsupportedFeature,
        };
    }

    pub fn extractTrainingSnapshot(
        self: *const ActiveTrainable,
        allocator: std.mem.Allocator,
        template_config: ?*const nam_file.WaveNetConfig,
    ) !TrainingSnapshot {
        return switch (self.*) {
            .classic, .a2 => .{ .wavenet = try self.extractWaveNetSnapshot(allocator, template_config orelse return error.UnsupportedFeature) },
            .packed_wavenet => |*model| .{ .packed_wavenet = try model.extractPackedSnapshot(allocator) },
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

pub const TrainingSnapshot = union(enum) {
    wavenet: WaveNetSnapshot,
    packed_wavenet: PackedSnapshot,

    pub fn deinit(self: *TrainingSnapshot, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .wavenet => |*snapshot| snapshot.deinit(allocator),
            .packed_wavenet => |*snapshot| snapshot.deinit(allocator),
        }
        self.* = undefined;
    }
};

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

/// Streams `weights` (NAM order) over `x` with the ring-buffer inference
/// engine and returns predictions aligned with x (pred[t] uses x[0..t]).
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

pub fn renderWaveNetConfig(
    allocator: std.mem.Allocator,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
    x: []const f32,
    out: []f32,
) !void {
    var engine = try wavenet.WaveNetEngine.init(allocator, config, weights);
    defer engine.deinit();
    const block = 4096;
    try engine.reset(block);
    var offset: usize = 0;
    while (offset < x.len) {
        const n = @min(block, x.len - offset);
        engine.process(x[offset..], out[offset..], n);
        offset += n;
    }
}

test "A2 trainable backward matches finite difference through grouped conv and FiLM branches" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    const dilations = [_]usize{1};
    const kernels = [_]usize{2};
    const activations = [_]nam_file.Activation{.{ .kind = .fasttanh }};
    const secondary = [_]nam_file.Activation{.{ .kind = .sigmoid }};
    const gating = [_]nam_file.GatingMode{.blended};
    const layers = [_]nam_file.WaveNetLayerArray{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 2,
        .bottleneck = 2,
        .head_out = 1,
        .head_kernel = 1,
        .head_bias = true,
        .dilations = &dilations,
        .kernel_sizes = &kernels,
        .activations = &activations,
        .gating_modes = &gating,
        .secondary_activations = &secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 2,
        .head1x1_active = true,
        .head1x1_out = 1,
        .head1x1_groups = 1,
        .groups_input = 2,
        .groups_input_mixin = 1,
        .conv_post_film = .{ .active = true, .shift = true },
        .activation_post_film = .{ .active = true, .shift = true },
        .layer1x1_post_film = .{ .active = true, .shift = true },
        .head1x1_post_film = .{ .active = true, .shift = true },
    }};
    const config = nam_file.WaveNetConfig{
        .layers = &layers,
        .head = null,
        .head_scale = 1.0,
        .in_channels = 1,
        .condition_dsp = null,
    };
    const file_config = nam_file.Config{ .wavenet = config };
    const weight_count = nam_file.expectedWeightCount(&file_config);
    var weights = try allocator.alloc(f32, weight_count);
    defer allocator.free(weights);
    for (weights, 0..) |*v, i| {
        v.* = 0.17 * @sin(@as(f32, @floatFromInt(i)) * 0.37) + 0.03 * @cos(@as(f32, @floatFromInt(i)) * 0.11);
    }
    weights[weights.len - 1] = 1.0;

    const input = [_]f32{ -0.4, 0.2, 0.7, -0.1, 0.5 };

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try A2Trainable.initFromWaveNet(allocator, &ctx, &config, weights);
    defer model.deinit();
    const roundtrip = try model.extractWeights(allocator);
    defer allocator.free(roundtrip);
    try std.testing.expectEqualSlices(f32, weights, roundtrip);

    var analytic: f32 = undefined;
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const pred = try model.forward(&ctx, &input);
        const loss = try pred.sumAll(&ctx);
        try loss.backward(&ctx);

        var conv_grad = (try model.arrays[0].layers[0].conv.weight.grad(&ctx)).?;
        defer conv_grad.deinit();
        analytic = (try conv_grad.dataConst())[0];
        try std.testing.expect(std.math.isFinite(analytic));

        var film_grad_sum: f32 = 0;
        try addA2FilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].conv_post_film, &ctx);
        try addA2FilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].activation_post_film, &ctx);
        try addA2FilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].layer1x1_post_film, &ctx);
        try addA2FilmGradAbs(&film_grad_sum, &model.arrays[0].layers[0].head1x1_post_film, &ctx);
        try std.testing.expect(film_grad_sum > 1e-6);
    }

    // Flat NAM index 2 is the first grouped dilated-conv weight after the
    // two rechannel weights; internally that is conv.weight[0].
    const selected_flat_index: usize = 2;
    const eps: f32 = 1e-3;
    var plus = try allocator.dupe(f32, weights);
    defer allocator.free(plus);
    var minus = try allocator.dupe(f32, weights);
    defer allocator.free(minus);
    plus[selected_flat_index] += eps;
    minus[selected_flat_index] -= eps;
    const plus_loss = try a2SumLossForWeights(allocator, &config, plus, &input);
    const minus_loss = try a2SumLossForWeights(allocator, &config, minus, &input);
    const numeric = (plus_loss - minus_loss) / (2.0 * eps);
    try std.testing.expect(std.math.isFinite(numeric));
    try std.testing.expectApproxEqAbs(numeric, analytic, 2e-2);
}

fn a2SumLossForWeights(
    allocator: std.mem.Allocator,
    config: *const nam_file.WaveNetConfig,
    weights: []const f32,
    input: []const f32,
) !f32 {
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var model = try A2Trainable.initFromWaveNet(allocator, &ctx, config, weights);
    defer model.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const pred = try model.forward(&ctx, input);
    const loss = try pred.sumAll(&ctx);
    return loss.item();
}

fn addA2FilmGradAbs(sum: *f32, film: *?A2FiLMParams, ctx: *ExecContext) !void {
    if (film.*) |*f| {
        var weight_grad = try f.conv.weight.grad(ctx);
        if (weight_grad) |*g| {
            defer g.deinit();
            for (try g.dataConst()) |v| sum.* += @abs(v);
        }
        if (f.conv.bias) |*bias| {
            var bias_grad = try bias.grad(ctx);
            if (bias_grad) |*g| {
                defer g.deinit();
                for (try g.dataConst()) |v| sum.* += @abs(v);
            }
        }
    }
}

test {
    _ = @import("train_tests.zig");
}
