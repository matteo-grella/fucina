//! The NAM WaveNet as fucina tensors: one model that trains and streams.
//!
//! Semantics follow NeuralAmpModelerCore's wavenet runtime
//! (NAM/wavenet/model.cpp, detail.h). Per block: condition = the raw input
//! (or the nested condition DSP's output); for each layer array (array 0
//! takes the condition as layer input and starts its head accumulator from
//! the first layer; array i>0 takes the previous array's residual outputs
//! and seeds its accumulator with the previous head output):
//!   x = rechannel(input)                         [Conv1x1, no bias]
//!   per layer: z = dilated_conv(x) + input_mixin(condition)
//!              a = activation(z)                 [gated: act(top)*act2(bottom);
//!                                                 blended: top + alpha*(act(top) - top)]
//!              head_acc += head1x1(a) or a
//!              x = x + layer1x1(a)               [or x unchanged if inactive]
//!   head_out = head_rechannel(head_acc)          [causal conv, has memory when k>1]
//! Output = head_scale * head_out of the last array, optionally through the
//! post-stack head (activation before each conv). FiLM slots
//! (`scale[, shift]` from the condition) sit where the A2 config puts them.
//!
//! One code path serves both regimes. Every causal conv is a
//! `groupedCausalConv1dStreaming` over its own `CausalState` ring: without
//! gradients the op is one fused kernel call (conv, bias, state carry) and
//! the block runs allocation-free once the runtime's pool is warm; when the
//! weights are variables the same op records the differentiable conv over
//! the ring's rows, so `forward` on a training window (from a reset state,
//! under the caller's exec scope) builds the graph the trainer needs. The
//! two accumulations in the layer (`x += layer1x1(a)`, `head_acc += ...`)
//! are in place when nothing records and out of place otherwise. Streaming
//! (`process`) runs under no-grad inside a per-block scope, so a trainable
//! model renders validation audio through the identical path.
//!
//! Weights are the `.nam` stream in the spec's order (§6.1.2): each conv's
//! `(out, in_per_group, tap)` rows are read as a permuted view and copied
//! out once into the core's `[tap, in_per_group, out]` layout; export is
//! the inverse view.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;
const Activation = nam_file.Activation;
const CausalState = fucina.streamconv.CausalState;

pub const TimeIn = Tensor(.{ .time, .in });
pub const TimeOut = Tensor(.{ .time, .out });
pub const GroupedWeight = Tensor(.{ .tap, .in_group, .out });
pub const Bias = Tensor(.{.out});

pub const Error = error{ UnsupportedFeature, WeightCountMismatch, InvalidConvShape, ExecScopeRequired, PreluSlopesMissing, PreluWidthMismatch };

pub const Options = struct {
    /// Weights as variables (gradients, optimizer registration).
    trainable: bool = false,
    /// The block length the context rings are sized for (any block length
    /// stays correct).
    chunk_hint: usize = 64,
};

fn asIn(ctx: *ExecContext, t: *const TimeOut) !TimeIn {
    return t.withTags(ctx, .{ .time, .in });
}

fn asOut(ctx: *ExecContext, t: *const TimeIn) !TimeOut {
    return t.withTags(ctx, .{ .time, .out });
}

/// `acc += x`: in place when nothing records, a new tensor otherwise (the
/// graph needs the operands as they were).
fn accumulate(ctx: *ExecContext, acc: *TimeOut, x: *const TimeOut) !void {
    if (acc.requiresGrad() or x.requiresGrad()) {
        const next = try acc.add(ctx, x);
        acc.deinit();
        acc.* = next;
        return;
    }
    try acc.addScaledInPlace(ctx, x, 1.0);
}

fn accumulateIn(ctx: *ExecContext, acc: *TimeIn, x: *const TimeIn) !void {
    if (acc.requiresGrad() or x.requiresGrad()) {
        const next = try acc.add(ctx, x);
        acc.deinit();
        acc.* = next;
        return;
    }
    try acc.addScaledInPlace(ctx, x, 1.0);
}

/// A causal conv with its context ring: the NAM stream's `(out, in_per_group,
/// tap)` rows permuted once into the core's `[tap, in_per_group, out]`.
pub const Conv = struct {
    weight: GroupedWeight,
    bias: ?Bias,
    state: CausalState,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    taps: usize,
    dilation: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *ExecContext,
        in_channels: usize,
        out_channels: usize,
        taps: usize,
        dilation: usize,
        has_bias: bool,
        groups: usize,
        options: Options,
        stream: []const f32,
        cursor: *usize,
    ) !Conv {
        if (taps < 1 or dilation < 1 or groups == 0) return Error.InvalidConvShape;
        if (in_channels % groups != 0 or out_channels % groups != 0) return Error.InvalidConvShape;
        const in_per_group = in_channels / groups;
        const out_per_group = out_channels / groups;
        const bias_len: usize = if (has_bias) out_channels else 0;
        const weight_len = taps * in_per_group * out_channels;
        var idx = cursor.*;
        if (idx + weight_len + bias_len > stream.len) return Error.WeightCountMismatch;
        // The stream holds each group as (out_per_group, in_per_group, tap):
        // a rank-4 view of it, permuted to [tap, in_per_group, group,
        // out_per_group], copied once into the packed layout the conv owns.
        var stream_view = try Tensor(.{ .grp, .opg, .in_group, .tap }).fromBorrowedConstSlice(ctx, .{ groups, out_per_group, in_per_group, taps }, stream[idx..][0..weight_len]);
        defer stream_view.deinit();
        var permuted = try stream_view.permuteTo(ctx, .{ .tap, .in_group, .grp, .opg });
        defer permuted.deinit();
        var packed_weight = try permuted.merge(ctx, .out, .{ .grp, .opg });
        defer packed_weight.deinit();
        var weight = try if (options.trainable) packed_weight.copyAsVariable(ctx) else packed_weight.copy(ctx);
        errdefer weight.deinit();
        idx += weight_len;
        var bias: ?Bias = null;
        errdefer if (bias) |*b| b.deinit();
        if (has_bias) {
            var bias_view = try Bias.fromBorrowedConstSlice(ctx, .{out_channels}, stream[idx..][0..out_channels]);
            defer bias_view.deinit();
            bias = try if (options.trainable) bias_view.copyAsVariable(ctx) else bias_view.copy(ctx);
            idx += out_channels;
        }
        const conv = try fromTensors(allocator, weight, bias, in_channels, out_channels, groups, taps, dilation, options.chunk_hint);
        cursor.* = idx;
        return conv;
    }

    /// A conv over tensors the caller built (taken over, `weight`
    /// `[tap, in_per_group, out]`, `bias` `[out]` or null), with a fresh
    /// context ring for `chunk_hint`-frame blocks.
    pub fn fromTensors(allocator: std.mem.Allocator, weight: GroupedWeight, bias: ?Bias, in_channels: usize, out_channels: usize, groups: usize, taps: usize, dilation: usize, chunk_hint: usize) !Conv {
        if (taps < 1 or dilation < 1 or groups == 0) return Error.InvalidConvShape;
        if (in_channels % groups != 0 or out_channels % groups != 0) return Error.InvalidConvShape;
        const shape = weight.shape();
        if (shape[0] != taps or shape[1] != in_channels / groups or shape[2] != out_channels) return Error.InvalidConvShape;
        if (bias) |*b| {
            if (b.shape()[0] != out_channels) return Error.InvalidConvShape;
        }
        const state = try CausalState.init(allocator, in_channels, taps, dilation, chunk_hint);
        return .{
            .weight = weight,
            .bias = bias,
            .state = state,
            .in_channels = in_channels,
            .out_channels = out_channels,
            .groups = groups,
            .taps = taps,
            .dilation = dilation,
        };
    }

    pub fn deinit(self: *Conv) void {
        self.weight.deinit();
        if (self.bias) |*b| b.deinit();
        self.state.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Conv) void {
        self.state.reset();
    }

    pub fn registerParams(self: *Conv, opt: anytype) !void {
        try opt.addParam(&self.weight);
        if (self.bias) |*b| try opt.addParam(b);
    }

    pub fn requiresGrad(self: *const Conv) bool {
        if (self.weight.requiresGrad()) return true;
        if (self.bias) |*b| return b.requiresGrad();
        return false;
    }

    /// `conv(x) + bias` over the stream: one op, the state carried inside,
    /// recorded when the weights are variables.
    pub fn forward(self: *Conv, ctx: *ExecContext, x: *const TimeIn) !TimeOut {
        if (self.bias) |*b| return x.groupedCausalConv1dStreaming(ctx, .time, .in, .tap, .in_group, .out, &self.weight, b, self.dilation, self.groups, &self.state);
        return x.groupedCausalConv1dStreaming(ctx, .time, .in, .tap, .in_group, .out, &self.weight, null, self.dilation, self.groups, &self.state);
    }

    /// The NAM stream order (the inverse of `init`): our `[tap, in_per_group,
    /// out]` split into groups, permuted, copied out stride-aware.
    pub fn appendNamWeights(self: *const Conv, ctx: *ExecContext, allocator: std.mem.Allocator, out: *std.ArrayList(f32)) !void {
        var grouped = try self.weight.split(ctx, .out, .{ .grp, .opg }, .{ self.groups, self.out_channels / self.groups });
        defer grouped.deinit();
        var nam_order = try grouped.permuteTo(ctx, .{ .grp, .opg, .in_group, .tap });
        defer nam_order.deinit();
        try appendView(allocator, out, &nam_order);
        if (self.bias) |*b| try out.appendSlice(allocator, try b.dataConst());
    }
};

fn appendView(allocator: std.mem.Allocator, out: *std.ArrayList(f32), view: anytype) !void {
    var count: usize = 1;
    for (view.shape()) |dim| count *= dim;
    const start = out.items.len;
    try out.resize(allocator, start + count);
    try view.copyTo(out.items[start..]);
}

pub const Film = struct {
    conv: Conv,
    input_dim: usize,
    shift: bool,

    fn forward(self: *Film, ctx: *ExecContext, condition: *const TimeIn, input: *const TimeOut) !TimeOut {
        var affine = try self.conv.forward(ctx, condition);
        defer affine.deinit();
        var scale = try affine.narrow(ctx, .out, 0, self.input_dim);
        defer scale.deinit();
        var scaled = try input.mul(ctx, &scale);
        if (!self.shift) return scaled;
        errdefer scaled.deinit();
        var shift = try affine.narrow(ctx, .out, self.input_dim, self.input_dim);
        defer shift.deinit();
        try accumulate(ctx, &scaled, &shift);
        return scaled;
    }
};

/// One activation over `[time, out]`. `fast_tanh` swaps the exact tanh for
/// the rational approximation, as the trainer's option does; `slopes` is
/// the per-channel PReLU slope tensor built once by `preluSlopes` (null
/// for every other kind).
pub fn activate(ctx: *ExecContext, act: *const Activation, x: *const TimeOut, fast_tanh: bool, slopes: ?*const Bias) !TimeOut {
    return switch (act.kind) {
        .tanh => if (fast_tanh) x.fastTanh(ctx) else x.tanh(ctx),
        .fasttanh => x.fastTanh(ctx),
        .hardtanh => x.clamp(ctx, -1.0, 1.0),
        .relu => x.relu(ctx),
        .leaky_relu => x.leakyRelu(ctx, act.negative_slope),
        .sigmoid => x.sigmoid(ctx),
        .silu => x.silu(ctx),
        .prelu => prelu(ctx, act, x, slopes),
        .hardswish => blk: {
            // x · clamp(x + 3, 0, 6) / 6
            var shifted = try x.addScalar(ctx, 3.0);
            defer shifted.deinit();
            var clipped = try shifted.clamp(ctx, 0.0, 6.0);
            defer clipped.deinit();
            var product = try x.mul(ctx, &clipped);
            defer product.deinit();
            break :blk try product.scale(ctx, 1.0 / 6.0);
        },
        .leaky_hardtanh => blk: {
            // clamp(x, min, max) + min_slope·min(x − min, 0) + max_slope·max(x − max, 0)
            var middle = try x.clamp(ctx, act.min_val, act.max_val);
            defer middle.deinit();
            var below_arg = try x.addScalar(ctx, -act.min_val);
            defer below_arg.deinit();
            var below = try below_arg.clamp(ctx, -std.math.inf(f32), 0.0);
            defer below.deinit();
            var below_scaled = try below.scale(ctx, act.min_slope);
            defer below_scaled.deinit();
            var with_below = try middle.add(ctx, &below_scaled);
            defer with_below.deinit();
            var above_arg = try x.addScalar(ctx, -act.max_val);
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
            var denom = try abs_x.addScalar(ctx, 1.0);
            defer denom.deinit();
            break :blk try x.div(ctx, &denom);
        },
    };
}

/// The per-channel PReLU slopes of `act` as a `[width]` tensor (the
/// upstream list cycled over the channels), built once at load; null when
/// `act` is not a per-channel PReLU.
pub fn preluSlopes(ctx: *ExecContext, act: *const Activation, width: usize) !?Bias {
    if (act.kind != .prelu or act.negative_slopes.len == 0) return null;
    const slopes = try ctx.allocator().alloc(f32, width);
    defer ctx.allocator().free(slopes);
    for (slopes, 0..) |*dst, i| dst.* = act.negative_slopes[i % act.negative_slopes.len];
    return try Bias.fromSlice(ctx, .{width}, slopes);
}

/// `max(x, 0) + slope·min(x, 0)`, the slope per channel from the tensor
/// `preluSlopes` built for this activation (refused when it is missing or
/// sized for another width) or the single scalar.
fn prelu(ctx: *ExecContext, act: *const Activation, x: *const TimeOut, slopes: ?*const Bias) !TimeOut {
    var positive = try x.relu(ctx);
    defer positive.deinit();
    var negative = try x.clamp(ctx, -std.math.inf(f32), 0.0);
    defer negative.deinit();
    var scaled_negative = if (act.negative_slopes.len != 0) blk: {
        const slope_tensor = slopes orelse return Error.PreluSlopesMissing;
        if (slope_tensor.dim(.out) != x.dim(.out)) return Error.PreluWidthMismatch;
        break :blk try negative.mul(ctx, slope_tensor);
    } else try negative.scale(ctx, act.negative_slope);
    defer scaled_negative.deinit();
    return positive.add(ctx, &scaled_negative);
}

pub const Layer = struct {
    conv: Conv,
    input_mixin: Conv,
    layer1x1: ?Conv,
    head1x1: ?Conv,
    conv_pre_film: ?Film,
    conv_post_film: ?Film,
    input_mixin_pre_film: ?Film,
    input_mixin_post_film: ?Film,
    activation_pre_film: ?Film,
    activation_post_film: ?Film,
    layer1x1_post_film: ?Film,
    head1x1_post_film: ?Film,
    activation: Activation,
    secondary_activation: Activation,
    /// Per-channel PReLU slopes of the two activations, built once (null
    /// for every other kind).
    activation_slopes: ?Bias,
    secondary_slopes: ?Bias,
    gating_mode: nam_file.GatingMode,
    bottleneck: usize,
};

pub const LayerArray = struct {
    rechannel: Conv,
    layers: []Layer,
    head_rechannel: Conv,
};

pub const PostHeadBlock = struct {
    conv: Conv,
    activation: Activation,
    activation_slopes: ?Bias,
};

/// A persistent owned `[max_frames, 1]` input tensor: a block is copied
/// into it and forwarded as a `narrow` view, so the per-block path creates
/// no storage header.
pub const InputSlot = struct {
    tensor: ?TimeIn = null,
    max_frames: usize = 0,

    pub fn deinit(self: *InputSlot) void {
        if (self.tensor) |*t| t.deinit();
        self.* = .{};
    }

    /// The block as a `[frames, 1]` view of the slot (grown on demand).
    pub fn view(self: *InputSlot, ctx: *ExecContext, input: []const f32, frames: usize) !TimeIn {
        if (frames > self.max_frames) {
            if (self.tensor) |*t| t.deinit();
            self.tensor = null;
            self.tensor = try TimeIn.zeros(ctx, .{ frames, 1 });
            self.max_frames = frames;
        }
        var block = try self.tensor.?.narrow(ctx, .time, 0, frames);
        try block.copyFrom(input[0..frames]);
        return block;
    }
};

pub const WaveNet = struct {
    allocator: std.mem.Allocator,
    arrays: []LayerArray,
    post_head: []PostHeadBlock,
    condition_child: ?*WaveNet,
    condition_channels: usize,
    head_scale: f32,
    /// 1 + the sum of the arrays' receptive fields (+ the post head's), the
    /// zero samples upstream runs before audio (model.cpp:615-620).
    prewarm_samples: usize,
    fast_tanh: bool = false,
    input_slot: InputSlot = .{},

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.WaveNetConfig, weights: []const f32, options: Options) anyerror!WaveNet {
        if (config.layers.len == 0) return Error.UnsupportedFeature;
        const arrays = try allocator.alloc(LayerArray, config.layers.len);
        errdefer allocator.free(arrays);
        var arrays_built: usize = 0;
        errdefer for (arrays[0..arrays_built]) |*array| deinitLayerArray(allocator, array);

        var cursor: usize = 0;
        var prev_head_out: usize = 0;
        for (config.layers, arrays, 0..) |*lc, *array, i| {
            if (lc.layerCount() == 0) return Error.UnsupportedFeature;
            array.* = try buildLayerArray(allocator, ctx, lc, options, weights, &cursor);
            arrays_built += 1;
            const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
            if (i > 0 and head_width != prev_head_out) return Error.UnsupportedFeature;
            prev_head_out = array.head_rechannel.out_channels;
        }

        var post_head: []PostHeadBlock = &.{};
        errdefer allocator.free(post_head);
        var post_built: usize = 0;
        errdefer for (post_head[0..post_built]) |*block| deinitPostHead(block);
        if (config.head) |*hc| {
            post_head = try allocator.alloc(PostHeadBlock, hc.kernel_sizes.len);
            var cin = config.layers[config.layers.len - 1].head_out;
            for (post_head, hc.kernel_sizes, 0..) |*block, k, i| {
                const cout = if (i == hc.kernel_sizes.len - 1) hc.out_channels else hc.channels;
                block.activation = hc.activation;
                block.activation_slopes = try preluSlopes(ctx, &hc.activation, cin);
                errdefer if (block.activation_slopes) |*t| t.deinit();
                block.conv = try Conv.init(allocator, ctx, cin, cout, k, 1, true, 1, options, weights, &cursor);
                post_built += 1;
                cin = cout;
            }
        }
        if (cursor + 1 != weights.len) return Error.WeightCountMismatch;

        var condition_child: ?*WaveNet = null;
        errdefer if (condition_child) |child| {
            child.deinit();
            allocator.destroy(child);
        };
        var condition_channels: usize = 1;
        if (config.condition_dsp) |dsp| {
            switch (dsp.config) {
                .wavenet => |*c| {
                    const child = try allocator.create(WaveNet);
                    errdefer allocator.destroy(child);
                    child.* = try WaveNet.init(allocator, ctx, c, dsp.weights, options);
                    condition_child = child;
                    condition_channels = child.outputChannels();
                },
                .lstm, .convnet, .linear => return Error.UnsupportedFeature,
            }
        }
        for (config.layers) |*lc| {
            if (lc.condition_size != condition_channels) return Error.UnsupportedFeature;
        }

        var prewarm: usize = if (condition_child) |child| child.prewarmSamples() else 1;
        for (config.layers) |*lc| prewarm += lc.receptiveField() - 1;
        if (config.head) |*hc| {
            for (hc.kernel_sizes) |k| prewarm += k - 1;
        }

        return .{
            .allocator = allocator,
            .arrays = arrays,
            .post_head = post_head,
            .condition_child = condition_child,
            .condition_channels = condition_channels,
            .head_scale = weights[cursor],
            .prewarm_samples = prewarm,
        };
    }

    fn buildLayerArray(allocator: std.mem.Allocator, ctx: *ExecContext, lc: *const nam_file.WaveNetLayerArray, options: Options, weights: []const f32, cursor: *usize) !LayerArray {
        var rechannel = try Conv.init(allocator, ctx, lc.input_size, lc.channels, 1, 1, false, 1, options, weights, cursor);
        errdefer rechannel.deinit();

        const layers = try allocator.alloc(Layer, lc.layerCount());
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*layer| deinitLayer(layer);
        for (layers, 0..) |*layer, l| {
            layer.* = try buildLayer(allocator, ctx, lc, l, options, weights, cursor);
            built += 1;
        }

        const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
        const head_rechannel = try Conv.init(allocator, ctx, head_width, lc.head_out, lc.head_kernel, 1, lc.head_bias, 1, options, weights, cursor);
        return .{ .rechannel = rechannel, .layers = layers, .head_rechannel = head_rechannel };
    }

    fn buildLayer(allocator: std.mem.Allocator, ctx: *ExecContext, lc: *const nam_file.WaveNetLayerArray, l: usize, options: Options, weights: []const f32, cursor: *usize) !Layer {
        const bg = lc.gateWidth(l);
        var conv = try Conv.init(allocator, ctx, lc.channels, bg, lc.kernel_sizes[l], lc.dilations[l], true, lc.groups_input, options, weights, cursor);
        errdefer conv.deinit();
        var input_mixin = try Conv.init(allocator, ctx, lc.condition_size, bg, 1, 1, false, lc.groups_input_mixin, options, weights, cursor);
        errdefer input_mixin.deinit();

        var layer1x1: ?Conv = null;
        errdefer if (layer1x1) |*c| c.deinit();
        if (lc.layer1x1_active) layer1x1 = try Conv.init(allocator, ctx, lc.bottleneck, lc.channels, 1, 1, true, lc.layer1x1_groups, options, weights, cursor);
        var head1x1: ?Conv = null;
        errdefer if (head1x1) |*c| c.deinit();
        if (lc.head1x1_active) head1x1 = try Conv.init(allocator, ctx, lc.bottleneck, lc.head1x1_out, 1, 1, true, lc.head1x1_groups, options, weights, cursor);

        var conv_pre_film = try buildFilm(allocator, ctx, lc, lc.channels, lc.conv_pre_film, options, weights, cursor);
        errdefer if (conv_pre_film) |*f| f.conv.deinit();
        var conv_post_film = try buildFilm(allocator, ctx, lc, bg, lc.conv_post_film, options, weights, cursor);
        errdefer if (conv_post_film) |*f| f.conv.deinit();
        var input_mixin_pre_film = try buildFilm(allocator, ctx, lc, lc.condition_size, lc.input_mixin_pre_film, options, weights, cursor);
        errdefer if (input_mixin_pre_film) |*f| f.conv.deinit();
        var input_mixin_post_film = try buildFilm(allocator, ctx, lc, bg, lc.input_mixin_post_film, options, weights, cursor);
        errdefer if (input_mixin_post_film) |*f| f.conv.deinit();
        var activation_pre_film = try buildFilm(allocator, ctx, lc, bg, lc.activation_pre_film, options, weights, cursor);
        errdefer if (activation_pre_film) |*f| f.conv.deinit();
        var activation_post_film = try buildFilm(allocator, ctx, lc, lc.bottleneck, lc.activation_post_film, options, weights, cursor);
        errdefer if (activation_post_film) |*f| f.conv.deinit();
        var layer1x1_post_film = try buildFilm(allocator, ctx, lc, lc.channels, lc.layer1x1_post_film, options, weights, cursor);
        errdefer if (layer1x1_post_film) |*f| f.conv.deinit();
        var head1x1_post_film = try buildFilm(allocator, ctx, lc, lc.head1x1_out, lc.head1x1_post_film, options, weights, cursor);
        errdefer if (head1x1_post_film) |*f| f.conv.deinit();
        var activation_slopes = try preluSlopes(ctx, &lc.activations[l], lc.bottleneck);
        errdefer if (activation_slopes) |*t| t.deinit();
        const secondary_slopes = try preluSlopes(ctx, &lc.secondary_activations[l], lc.bottleneck);

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
            .activation = lc.activations[l],
            .secondary_activation = lc.secondary_activations[l],
            .activation_slopes = activation_slopes,
            .secondary_slopes = secondary_slopes,
            .gating_mode = lc.gating_modes[l],
            .bottleneck = lc.bottleneck,
        };
    }

    fn buildFilm(
        allocator: std.mem.Allocator,
        ctx: *ExecContext,
        lc: *const nam_file.WaveNetLayerArray,
        input_dim: usize,
        params: nam_file.FiLMParams,
        options: Options,
        weights: []const f32,
        cursor: *usize,
    ) !?Film {
        if (!params.active) return null;
        const out_dim = input_dim * (if (params.shift) @as(usize, 2) else 1);
        const conv = try Conv.init(allocator, ctx, lc.condition_size, out_dim, 1, 1, true, params.groups, options, weights, cursor);
        return .{ .conv = conv, .input_dim = input_dim, .shift = params.shift };
    }

    fn deinitFilm(film: *?Film) void {
        if (film.*) |*f| f.conv.deinit();
    }

    fn deinitPostHead(block: *PostHeadBlock) void {
        block.conv.deinit();
        if (block.activation_slopes) |*t| t.deinit();
    }

    fn deinitLayer(layer: *Layer) void {
        layer.conv.deinit();
        if (layer.activation_slopes) |*t| t.deinit();
        if (layer.secondary_slopes) |*t| t.deinit();
        layer.input_mixin.deinit();
        if (layer.layer1x1) |*c| c.deinit();
        if (layer.head1x1) |*c| c.deinit();
        deinitFilm(&layer.conv_pre_film);
        deinitFilm(&layer.conv_post_film);
        deinitFilm(&layer.input_mixin_pre_film);
        deinitFilm(&layer.input_mixin_post_film);
        deinitFilm(&layer.activation_pre_film);
        deinitFilm(&layer.activation_post_film);
        deinitFilm(&layer.layer1x1_post_film);
        deinitFilm(&layer.head1x1_post_film);
    }

    fn deinitLayerArray(allocator: std.mem.Allocator, array: *LayerArray) void {
        array.rechannel.deinit();
        array.head_rechannel.deinit();
        for (array.layers) |*layer| deinitLayer(layer);
        allocator.free(array.layers);
    }

    pub fn deinit(self: *WaveNet) void {
        for (self.arrays) |*array| deinitLayerArray(self.allocator, array);
        self.allocator.free(self.arrays);
        for (self.post_head) |*block| deinitPostHead(block);
        self.allocator.free(self.post_head);
        if (self.condition_child) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.input_slot.deinit();
        self.* = undefined;
    }

    /// Every FiLM of `layer`, for the per-layer walks below.
    fn films(layer: *Layer) [8]*?Film {
        return .{
            &layer.conv_pre_film,        &layer.conv_post_film,
            &layer.input_mixin_pre_film, &layer.input_mixin_post_film,
            &layer.activation_pre_film,  &layer.activation_post_film,
            &layer.layer1x1_post_film,   &layer.head1x1_post_film,
        };
    }

    pub fn registerParams(self: *WaveNet, opt: anytype) !void {
        if (self.condition_child) |child| try child.registerParams(opt);
        for (self.arrays) |*array| {
            try array.rechannel.registerParams(opt);
            for (array.layers) |*layer| {
                try layer.conv.registerParams(opt);
                try layer.input_mixin.registerParams(opt);
                if (layer.layer1x1) |*c| try c.registerParams(opt);
                if (layer.head1x1) |*c| try c.registerParams(opt);
                for (films(layer)) |film| {
                    if (film.*) |*f| try f.conv.registerParams(opt);
                }
            }
            try array.head_rechannel.registerParams(opt);
        }
        for (self.post_head) |*block| try block.conv.registerParams(opt);
    }

    pub fn requiresGrad(self: *const WaveNet) bool {
        return self.arrays[0].rechannel.requiresGrad();
    }

    pub fn setFastTanh(self: *WaveNet, enabled: bool) void {
        self.fast_tanh = enabled;
        if (self.condition_child) |child| child.setFastTanh(enabled);
    }

    pub fn outputChannels(self: *const WaveNet) usize {
        if (self.post_head.len > 0) return self.post_head[self.post_head.len - 1].conv.out_channels;
        return self.arrays[self.arrays.len - 1].head_rechannel.out_channels;
    }

    pub fn prewarmSamples(self: *const WaveNet) usize {
        return self.prewarm_samples;
    }

    /// Zeroes every conv history (the caller drives prewarm, as upstream's
    /// wrapper does).
    pub fn reset(self: *WaveNet) void {
        if (self.condition_child) |child| child.reset();
        for (self.arrays) |*array| {
            array.rechannel.reset();
            array.head_rechannel.reset();
            for (array.layers) |*layer| {
                layer.conv.reset();
                layer.input_mixin.reset();
                if (layer.layer1x1) |*c| c.reset();
                if (layer.head1x1) |*c| c.reset();
                for (films(layer)) |film| {
                    if (film.*) |*f| f.conv.reset();
                }
            }
        }
        for (self.post_head) |*block| block.conv.reset();
    }

    /// Streaming: mono in -> mono out over one block, continuing the
    /// stream. No gradients whatever the weights are; every transient is
    /// owned by the scope opened here.
    pub fn process(self: *WaveNet, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        var no_grad = fucina.noGrad();
        defer no_grad.close();
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        const input_t = try self.input_slot.view(ctx, input, frames);
        const out = try self.forwardTensor(ctx, &input_t);
        try out.copyTo(output[0..frames]);
    }

    /// Training: the whole window from a reset stream, recorded under the
    /// caller's exec scope; `[T, out]` predictions for every t.
    pub fn forward(self: *WaveNet, ctx: *ExecContext, window: []const f32) !TimeOut {
        if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
        self.reset();
        var input = try TimeIn.fromSlice(ctx, .{ window.len, 1 }, window);
        defer input.deinit();
        return self.forwardTensor(ctx, &input);
    }

    /// The block forward inside the caller's exec scope (shared by
    /// `process`, `forward`, and a parent's condition path).
    pub fn forwardTensor(self: *WaveNet, ctx: *ExecContext, input: *const TimeIn) anyerror!TimeOut {
        var condition_t: TimeIn = undefined;
        var condition: *const TimeIn = input;
        if (self.condition_child) |child| {
            var child_out = try child.forwardTensor(ctx, input);
            defer child_out.deinit();
            condition_t = try asIn(ctx, &child_out);
            condition = &condition_t;
        }
        defer if (self.condition_child != null) condition_t.deinit();

        var x: TimeIn = undefined;
        var have_x = false;
        defer if (have_x) x.deinit();
        var head_prev: ?TimeOut = null;
        defer if (head_prev) |*h| h.deinit();
        for (self.arrays, 0..) |*array, array_index| {
            const array_input: *const TimeIn = if (array_index == 0) input else &x;
            var rc = try array.rechannel.forward(ctx, array_input);
            defer rc.deinit();
            const next_x = try asIn(ctx, &rc);
            if (have_x) x.deinit();
            x = next_x;
            have_x = true;
            // Head accumulator: array 0 starts from the first layer's
            // contribution; later arrays are seeded by the previous head
            // output.
            var acc: ?TimeOut = head_prev;
            head_prev = null;
            defer if (acc) |*a| a.deinit();
            for (array.layers) |*layer| try self.forwardLayer(ctx, layer, &x, condition, &acc);
            var head_in = try asIn(ctx, &acc.?);
            defer head_in.deinit();
            head_prev = try array.head_rechannel.forward(ctx, &head_in);
        }

        var current = try head_prev.?.scale(ctx, self.head_scale);
        errdefer current.deinit();
        for (self.post_head) |*block| {
            var activated = try activate(ctx, &block.activation, &current, self.fast_tanh, if (block.activation_slopes) |*t| t else null);
            defer activated.deinit();
            var conv_in = try asIn(ctx, &activated);
            defer conv_in.deinit();
            const next = try block.conv.forward(ctx, &conv_in);
            current.deinit();
            current = next;
        }
        return current;
    }

    fn forwardLayer(self: *const WaveNet, ctx: *ExecContext, layer: *Layer, x: *TimeIn, condition: *const TimeIn, acc: *?TimeOut) !void {
        const b = layer.bottleneck;

        // z = conv(x) + bias + input_mixin(condition)
        var conv_input: *const TimeIn = x;
        var conv_filmed: TimeIn = undefined;
        var have_conv_filmed = false;
        defer if (have_conv_filmed) conv_filmed.deinit();
        if (layer.conv_pre_film) |*film| {
            var x_out = try asOut(ctx, x);
            defer x_out.deinit();
            var filmed = try film.forward(ctx, condition, &x_out);
            defer filmed.deinit();
            conv_filmed = try asIn(ctx, &filmed);
            have_conv_filmed = true;
            conv_input = &conv_filmed;
        }
        var z = try layer.conv.forward(ctx, conv_input);
        defer z.deinit();
        if (layer.conv_post_film) |*film| {
            const filmed = try film.forward(ctx, condition, &z);
            z.deinit();
            z = filmed;
        }

        var mixin_input: *const TimeIn = condition;
        var mixin_filmed: TimeIn = undefined;
        var have_mixin_filmed = false;
        defer if (have_mixin_filmed) mixin_filmed.deinit();
        if (layer.input_mixin_pre_film) |*film| {
            var condition_out = try asOut(ctx, condition);
            defer condition_out.deinit();
            var filmed = try film.forward(ctx, condition, &condition_out);
            defer filmed.deinit();
            mixin_filmed = try asIn(ctx, &filmed);
            have_mixin_filmed = true;
            mixin_input = &mixin_filmed;
        }
        var mix = try layer.input_mixin.forward(ctx, mixin_input);
        defer mix.deinit();
        if (layer.input_mixin_post_film) |*film| {
            const filmed = try film.forward(ctx, condition, &mix);
            mix.deinit();
            mix = filmed;
        }
        try accumulate(ctx, &z, &mix);
        if (layer.activation_pre_film) |*film| {
            const filmed = try film.forward(ctx, condition, &z);
            z.deinit();
            z = filmed;
        }

        // activation (gated: act(top) * act2(bottom); blended: top + alpha*(act(top) - top))
        var activated = try self.gate(ctx, layer, &z, b);
        defer activated.deinit();
        if (layer.activation_post_film) |*film| {
            const filmed = try film.forward(ctx, condition, &activated);
            activated.deinit();
            activated = filmed;
        }

        // residual (layer1x1 before head1x1, as in the reference)
        if (layer.layer1x1) |*conv| {
            var residual_in = try asIn(ctx, &activated);
            defer residual_in.deinit();
            var residual = try conv.forward(ctx, &residual_in);
            defer residual.deinit();
            if (layer.gating_mode == .blended) {
                if (layer.layer1x1_post_film) |*film| {
                    const filmed = try film.forward(ctx, condition, &residual);
                    residual.deinit();
                    residual = filmed;
                }
            }
            var residual_as_in = try asIn(ctx, &residual);
            defer residual_as_in.deinit();
            try accumulateIn(ctx, x, &residual_as_in);
        }

        // head contribution
        var contribution: TimeOut = undefined;
        var own_contribution = false;
        defer if (own_contribution) contribution.deinit();
        if (layer.head1x1) |*conv| {
            var head_in = try asIn(ctx, &activated);
            defer head_in.deinit();
            contribution = try conv.forward(ctx, &head_in);
            own_contribution = true;
            if (layer.head1x1_post_film) |*film| {
                const filmed = try film.forward(ctx, condition, &contribution);
                contribution.deinit();
                contribution = filmed;
            }
        } else {
            contribution = activated;
        }
        if (acc.*) |*a| {
            try accumulate(ctx, a, &contribution);
        } else if (own_contribution) {
            acc.* = contribution;
            own_contribution = false;
        } else {
            // The activation itself seeds the accumulator: keep it alive
            // past this layer as the accumulator's own handle.
            acc.* = try activated.withTags(ctx, .{ .time, .out });
        }
    }

    fn gate(self: *const WaveNet, ctx: *ExecContext, layer: *const Layer, z: *const TimeOut, b: usize) !TimeOut {
        switch (layer.gating_mode) {
            .none => return activate(ctx, &layer.activation, z, self.fast_tanh, if (layer.activation_slopes) |*t| t else null),
            .gated => {
                var top = try z.narrow(ctx, .out, 0, b);
                defer top.deinit();
                var bottom = try z.narrow(ctx, .out, b, b);
                defer bottom.deinit();
                var primary = try activate(ctx, &layer.activation, &top, self.fast_tanh, if (layer.activation_slopes) |*t| t else null);
                defer primary.deinit();
                return switch (layer.secondary_activation.kind) {
                    .sigmoid => primary.glu(ctx, &bottom),
                    .silu => primary.swiglu(ctx, &bottom),
                    else => blk: {
                        var gate_value = try activate(ctx, &layer.secondary_activation, &bottom, self.fast_tanh, if (layer.secondary_slopes) |*t| t else null);
                        defer gate_value.deinit();
                        break :blk try primary.mul(ctx, &gate_value);
                    },
                };
            },
            .blended => {
                var top = try z.narrow(ctx, .out, 0, b);
                defer top.deinit();
                var bottom = try z.narrow(ctx, .out, b, b);
                defer bottom.deinit();
                var primary = try activate(ctx, &layer.activation, &top, self.fast_tanh, if (layer.activation_slopes) |*t| t else null);
                defer primary.deinit();
                var alpha = try activate(ctx, &layer.secondary_activation, &bottom, self.fast_tanh, if (layer.secondary_slopes) |*t| t else null);
                defer alpha.deinit();
                var diff = try primary.sub(ctx, &top);
                defer diff.deinit();
                var blended = try alpha.mul(ctx, &diff);
                errdefer blended.deinit();
                try accumulate(ctx, &blended, &top);
                return blended;
            },
        }
    }

    /// The NAM weight stream (the inverse of `init`).
    pub fn extractWeights(self: *const WaveNet, ctx: *ExecContext, allocator: std.mem.Allocator) ![]f32 {
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(allocator);
        for (self.arrays) |*array| {
            try array.rechannel.appendNamWeights(ctx, allocator, &out);
            for (array.layers) |*layer| {
                try layer.conv.appendNamWeights(ctx, allocator, &out);
                try layer.input_mixin.appendNamWeights(ctx, allocator, &out);
                if (layer.layer1x1) |*conv| try conv.appendNamWeights(ctx, allocator, &out);
                if (layer.head1x1) |*conv| try conv.appendNamWeights(ctx, allocator, &out);
                for (films(layer)) |film| {
                    if (film.*) |*f| try f.conv.appendNamWeights(ctx, allocator, &out);
                }
            }
            try array.head_rechannel.appendNamWeights(ctx, allocator, &out);
        }
        for (self.post_head) |*block| try block.conv.appendNamWeights(ctx, allocator, &out);
        try out.append(allocator, self.head_scale);
        return out.toOwnedSlice(allocator);
    }
};

/// Allocation counter around a backing allocator (the steady-state
/// allocation-free claim is measured, not assumed).
pub const CountingAllocator = struct {
    backing: std.mem.Allocator,
    allocs: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    frees: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    bytes: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = allocImpl,
        .resize = resizeImpl,
        .remap = remapImpl,
        .free = freeImpl,
    };

    fn allocImpl(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        _ = self.allocs.fetchAdd(1, .monotonic);
        _ = self.bytes.fetchAdd(len, .monotonic);
        return self.backing.vtable.alloc(self.backing.ptr, len, alignment, ret_addr);
    }

    fn resizeImpl(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.vtable.resize(self.backing.ptr, memory, alignment, new_len, ret_addr);
    }

    fn remapImpl(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        return self.backing.vtable.remap(self.backing.ptr, memory, alignment, new_len, ret_addr);
    }

    fn freeImpl(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        _ = self.frees.fetchAdd(1, .monotonic);
        self.backing.vtable.free(self.backing.ptr, memory, alignment, ret_addr);
    }
};

// ---------------------------------------------------------------------------
// The classic "standard WaveNet" (the bulk of Tone3000 profiles): two arrays,
// 16 then 8 channels, kernel 3, dilations 1..512, tanh, no gating.
// ---------------------------------------------------------------------------

const standard_dilations = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
const standard_kernels = [_]usize{3} ** 10;
const standard_activations = [_]Activation{.{ .kind = .tanh }} ** 10;
const standard_secondary = [_]Activation{.{ .kind = .sigmoid }} ** 10;
const standard_gating = [_]nam_file.GatingMode{.none} ** 10;

pub const standard_layers = [_]nam_file.WaveNetLayerArray{
    .{
        .input_size = 1,
        .condition_size = 1,
        .channels = 16,
        .bottleneck = 16,
        .head_out = 8,
        .head_kernel = 1,
        .head_bias = false,
        .dilations = &standard_dilations,
        .kernel_sizes = &standard_kernels,
        .activations = &standard_activations,
        .gating_modes = &standard_gating,
        .secondary_activations = &standard_secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 1,
        .head1x1_active = false,
        .head1x1_out = 16,
        .head1x1_groups = 1,
        .groups_input = 1,
        .groups_input_mixin = 1,
    },
    .{
        .input_size = 16,
        .condition_size = 1,
        .channels = 8,
        .bottleneck = 8,
        .head_out = 1,
        .head_kernel = 1,
        .head_bias = true,
        .dilations = &standard_dilations,
        .kernel_sizes = &standard_kernels,
        .activations = &standard_activations,
        .gating_modes = &standard_gating,
        .secondary_activations = &standard_secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 1,
        .head1x1_active = false,
        .head1x1_out = 8,
        .head1x1_groups = 1,
        .groups_input = 1,
        .groups_input_mixin = 1,
    },
};

pub const standard_config = nam_file.WaveNetConfig{
    .layers = &standard_layers,
    .head = null,
    .head_scale = 0.02,
    .in_channels = 1,
    .condition_dsp = null,
};

/// Deterministic weights for `config`: uniform in [-amp, amp], the final
/// head_scale float pinned to 1. Caller frees.
pub fn syntheticWeights(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, seed: u64, amp: f32) ![]f32 {
    const file_config = nam_file.Config{ .wavenet = config.* };
    const count = nam_file.expectedWeightCount(&file_config);
    const weights = try allocator.alloc(f32, count);
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (weights) |*w| w.* = (random.float(f32) * 2 - 1) * amp;
    weights[count - 1] = 1.0;
    return weights;
}

/// Deterministic test signal: a guitar-ish sum of partials plus noise,
/// peak about 0.6.
pub fn fillSignal(buf: []f32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (buf, 0..) |*v, i| {
        const t = @as(f32, @floatFromInt(i));
        v.* = 0.3 * @sin(t * 0.0217) + 0.15 * @sin(t * 0.0651 + 0.3) + 0.08 * @sin(t * 0.1302) + 0.05 * (random.float(f32) * 2 - 1);
    }
}

test {
    _ = @import("wavenet_tests.zig");
}
