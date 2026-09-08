//! Assessment scaffold (branch assess/nam-facade): the NAM streaming engines
//! re-expressed over the fucina Tensor facade, to measure feasibility,
//! parity and speed against the hand-rolled kernels in `stream_conv.zig`,
//! `wavenet.zig`, `ir_cab.zig` and `models.zig`. Not part of the shipped
//! app; the findings live in the assessment document.
//!
//! Streaming discipline: each causal conv owns a core `streamconv.CausalState`
//! (the ring of context rows) and runs as one `groupedCausalConv1dStreaming`
//! call per block: conv, bias epilogue and state carry in the op. All
//! per-block transients live in one exec scope (`openExecScope` /
//! `closeExecScope`), so a block is allocation-free once the runtime's buffer
//! pool has warmed (the assessment tests measure that claim). The block
//! enters through a persistent input slot (`copyFrom`) rather than a
//! borrowed tensor per block: same numbers, no storage header per block.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const models = @import("models.zig");

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;
const Activation = nam_file.Activation;
const CausalState = fucina.streamconv.CausalState;

pub const TimeIn = Tensor(.{ .time, .in });
pub const TimeOut = Tensor(.{ .time, .out });
pub const GroupedWeight = Tensor(.{ .tap, .in_group, .out });
pub const DenseWeight = Tensor(.{ .tap, .in, .out });

pub const Error = error{ UnsupportedFeature, WeightCountMismatch, InvalidConvShape };

fn asIn(ctx: *ExecContext, t: *const TimeOut) !TimeIn {
    return t.withTags(ctx, .{ .time, .in });
}

fn asOut(ctx: *ExecContext, t: *const TimeIn) !TimeOut {
    return t.withTags(ctx, .{ .time, .out });
}

/// A streaming causal conv over the facade: the counterpart of
/// `stream_conv.StreamConv`. Same NAM weight permutation (`(out, in, k)`
/// stream order -> `[tap, in_per_group, out]`); the context rows live in a
/// core `CausalState`.
pub const Conv = struct {
    allocator: std.mem.Allocator,
    weight: GroupedWeight,
    bias: []f32,
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
        chunk_hint: usize,
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
        // The NAM stream holds each group as (out_per_group, in_per_group, tap):
        // read it as a rank-4 view, permute to [tap, in_per_group, group,
        // out_per_group], materialize once, and merge (group, out_per_group)
        // into `out`. No index arithmetic, one copy.
        var stream_view = try Tensor(.{ .grp, .opg, .in_group, .tap }).fromBorrowedConstSlice(ctx, .{ groups, out_per_group, in_per_group, taps }, stream[idx..][0..weight_len]);
        defer stream_view.deinit();
        var permuted = try stream_view.permuteTo(ctx, .{ .tap, .in_group, .grp, .opg });
        defer permuted.deinit();
        var packed_weight = try permuted.materialize(ctx);
        defer packed_weight.deinit();
        var weight = try packed_weight.merge(ctx, .out, .{ .grp, .opg });
        errdefer weight.deinit();
        idx += weight_len;
        const bias = try allocator.dupe(f32, stream[idx..][0..bias_len]);
        errdefer allocator.free(bias);
        idx += bias_len;
        const state = try CausalState.init(allocator, in_channels, taps, dilation, chunk_hint);
        cursor.* = idx;
        return .{
            .allocator = allocator,
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
        self.allocator.free(self.bias);
        self.state.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Conv) void {
        self.state.reset();
    }

    pub fn biasOrNull(self: *const Conv) ?[]const f32 {
        return if (self.bias.len > 0) self.bias else null;
    }

    /// `conv(x) + bias` over the stream: one op, state carried inside.
    pub fn forward(self: *Conv, ctx: *ExecContext, x: *const TimeIn) !TimeOut {
        return x.groupedCausalConv1dStreaming(ctx, .time, .in, .tap, .in_group, .out, &self.weight, self.biasOrNull(), self.dilation, self.groups, &self.state);
    }
};

const Film = struct {
    conv: Conv,
    input_dim: usize,
    shift: bool,

    fn forward(self: *Film, ctx: *ExecContext, condition: *const TimeIn, input: *const TimeOut) !TimeOut {
        const affine = try self.conv.forward(ctx, condition);
        const scale = try affine.narrow(ctx, .out, 0, self.input_dim);
        var scaled = try input.mul(ctx, &scale);
        if (!self.shift) return scaled;
        const shift = try affine.narrow(ctx, .out, self.input_dim, self.input_dim);
        try scaled.addScaledInPlace(ctx, &shift, 1.0);
        return scaled;
    }
};

fn activate(ctx: *ExecContext, act: *const Activation, x: *const TimeOut) !TimeOut {
    return switch (act.kind) {
        .tanh => x.tanh(ctx),
        .fasttanh => x.fastTanh(ctx),
        .hardtanh => x.clamp(ctx, -1.0, 1.0),
        .relu => x.relu(ctx),
        .leaky_relu => x.leakyRelu(ctx, act.negative_slope),
        .sigmoid => x.sigmoid(ctx),
        .silu => x.silu(ctx),
        .prelu, .hardswish, .leaky_hardtanh, .softsign => Error.UnsupportedFeature,
    };
}

const Layer = struct {
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
    gating_mode: nam_file.GatingMode,
    bottleneck: usize,
};

const LayerArray = struct {
    rechannel: Conv,
    layers: []Layer,
    head_rechannel: Conv,
};

const PostHeadBlock = struct {
    conv: Conv,
    activation: Activation,
};

/// A persistent owned `[max_frames, 1]` input tensor: a block is copied
/// into it and forwarded as a `narrow` view, so the per-block path creates
/// no storage header (a `fromBorrowedSlice` per block would be one
/// allocation per block).
const InputSlot = struct {
    tensor: ?TimeIn = null,
    max_frames: usize = 0,

    fn deinit(self: *InputSlot) void {
        if (self.tensor) |*t| t.deinit();
        self.* = .{};
    }

    /// The block as a `[frames, 1]` view of the slot (grown on demand).
    fn view(self: *InputSlot, ctx: *ExecContext, input: []const f32, frames: usize) !TimeIn {
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
    input_slot: InputSlot = .{},

    /// `chunk_hint` sizes every conv's context ring for the block length
    /// the engine will be fed (any block length stays correct).
    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.WaveNetConfig, weights: []const f32, chunk_hint: usize) anyerror!WaveNet {
        const arrays = try allocator.alloc(LayerArray, config.layers.len);
        errdefer allocator.free(arrays);
        var arrays_built: usize = 0;
        errdefer for (arrays[0..arrays_built]) |*array| deinitLayerArray(allocator, array);

        var cursor: usize = 0;
        var prev_head_out: usize = 0;
        for (config.layers, arrays, 0..) |*lc, *array, i| {
            if (lc.layerCount() == 0) return Error.UnsupportedFeature;
            array.* = try buildLayerArray(allocator, ctx, lc, chunk_hint, weights, &cursor);
            arrays_built += 1;
            const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
            if (i > 0 and head_width != prev_head_out) return Error.UnsupportedFeature;
            prev_head_out = array.head_rechannel.out_channels;
        }

        var post_head: []PostHeadBlock = &.{};
        errdefer allocator.free(post_head);
        var post_built: usize = 0;
        errdefer for (post_head[0..post_built]) |*block| block.conv.deinit();
        if (config.head) |*hc| {
            post_head = try allocator.alloc(PostHeadBlock, hc.kernel_sizes.len);
            var cin = config.layers[config.layers.len - 1].head_out;
            for (post_head, hc.kernel_sizes, 0..) |*block, k, i| {
                const cout = if (i == hc.kernel_sizes.len - 1) hc.out_channels else hc.channels;
                block.activation = hc.activation;
                block.conv = try Conv.init(allocator, ctx, cin, cout, k, 1, true, 1, chunk_hint, weights, &cursor);
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
                    child.* = try WaveNet.init(allocator, ctx, c, dsp.weights, chunk_hint);
                    condition_child = child;
                    condition_channels = child.outputChannels();
                },
                else => return Error.UnsupportedFeature,
            }
        }
        for (config.layers) |*lc| {
            if (lc.condition_size != condition_channels) return Error.UnsupportedFeature;
        }

        return .{
            .allocator = allocator,
            .arrays = arrays,
            .post_head = post_head,
            .condition_child = condition_child,
            .condition_channels = condition_channels,
            .head_scale = weights[cursor],
        };
    }

    fn buildLayerArray(allocator: std.mem.Allocator, ctx: *ExecContext, lc: *const nam_file.WaveNetLayerArray, chunk_hint: usize, weights: []const f32, cursor: *usize) !LayerArray {
        var rechannel = try Conv.init(allocator, ctx, lc.input_size, lc.channels, 1, 1, false, 1, chunk_hint, weights, cursor);
        errdefer rechannel.deinit();

        const layers = try allocator.alloc(Layer, lc.layerCount());
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*layer| deinitLayer(layer);
        for (layers, 0..) |*layer, l| {
            layer.* = try buildLayer(allocator, ctx, lc, l, chunk_hint, weights, cursor);
            built += 1;
        }

        const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
        const head_rechannel = try Conv.init(allocator, ctx, head_width, lc.head_out, lc.head_kernel, 1, lc.head_bias, 1, chunk_hint, weights, cursor);
        return .{ .rechannel = rechannel, .layers = layers, .head_rechannel = head_rechannel };
    }

    fn buildLayer(allocator: std.mem.Allocator, ctx: *ExecContext, lc: *const nam_file.WaveNetLayerArray, l: usize, chunk_hint: usize, weights: []const f32, cursor: *usize) !Layer {
        const bg = lc.gateWidth(l);
        var conv = try Conv.init(allocator, ctx, lc.channels, bg, lc.kernel_sizes[l], lc.dilations[l], true, lc.groups_input, chunk_hint, weights, cursor);
        errdefer conv.deinit();
        var input_mixin = try Conv.init(allocator, ctx, lc.condition_size, bg, 1, 1, false, lc.groups_input_mixin, chunk_hint, weights, cursor);
        errdefer input_mixin.deinit();

        var layer1x1: ?Conv = null;
        errdefer if (layer1x1) |*c| c.deinit();
        if (lc.layer1x1_active) layer1x1 = try Conv.init(allocator, ctx, lc.bottleneck, lc.channels, 1, 1, true, lc.layer1x1_groups, chunk_hint, weights, cursor);
        var head1x1: ?Conv = null;
        errdefer if (head1x1) |*c| c.deinit();
        if (lc.head1x1_active) head1x1 = try Conv.init(allocator, ctx, lc.bottleneck, lc.head1x1_out, 1, 1, true, lc.head1x1_groups, chunk_hint, weights, cursor);

        var conv_pre_film = try buildFilm(allocator, ctx, lc, lc.channels, lc.conv_pre_film, chunk_hint, weights, cursor);
        errdefer if (conv_pre_film) |*f| f.conv.deinit();
        var conv_post_film = try buildFilm(allocator, ctx, lc, bg, lc.conv_post_film, chunk_hint, weights, cursor);
        errdefer if (conv_post_film) |*f| f.conv.deinit();
        var input_mixin_pre_film = try buildFilm(allocator, ctx, lc, lc.condition_size, lc.input_mixin_pre_film, chunk_hint, weights, cursor);
        errdefer if (input_mixin_pre_film) |*f| f.conv.deinit();
        var input_mixin_post_film = try buildFilm(allocator, ctx, lc, bg, lc.input_mixin_post_film, chunk_hint, weights, cursor);
        errdefer if (input_mixin_post_film) |*f| f.conv.deinit();
        var activation_pre_film = try buildFilm(allocator, ctx, lc, bg, lc.activation_pre_film, chunk_hint, weights, cursor);
        errdefer if (activation_pre_film) |*f| f.conv.deinit();
        var activation_post_film = try buildFilm(allocator, ctx, lc, lc.bottleneck, lc.activation_post_film, chunk_hint, weights, cursor);
        errdefer if (activation_post_film) |*f| f.conv.deinit();
        var layer1x1_post_film = try buildFilm(allocator, ctx, lc, lc.channels, lc.layer1x1_post_film, chunk_hint, weights, cursor);
        errdefer if (layer1x1_post_film) |*f| f.conv.deinit();
        var head1x1_post_film = try buildFilm(allocator, ctx, lc, lc.head1x1_out, lc.head1x1_post_film, chunk_hint, weights, cursor);
        errdefer if (head1x1_post_film) |*f| f.conv.deinit();

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
        chunk_hint: usize,
        weights: []const f32,
        cursor: *usize,
    ) !?Film {
        if (!params.active) return null;
        const out_dim = input_dim * (if (params.shift) @as(usize, 2) else 1);
        const conv = try Conv.init(allocator, ctx, lc.condition_size, out_dim, 1, 1, true, params.groups, chunk_hint, weights, cursor);
        return .{ .conv = conv, .input_dim = input_dim, .shift = params.shift };
    }

    fn deinitFilm(film: *?Film) void {
        if (film.*) |*f| f.conv.deinit();
    }

    fn deinitLayer(layer: *Layer) void {
        layer.conv.deinit();
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
        for (self.post_head) |*block| block.conv.deinit();
        self.allocator.free(self.post_head);
        if (self.condition_child) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.input_slot.deinit();
        self.* = undefined;
    }

    pub fn outputChannels(self: *const WaveNet) usize {
        if (self.post_head.len > 0) return self.post_head[self.post_head.len - 1].conv.out_channels;
        return self.arrays[self.arrays.len - 1].head_rechannel.out_channels;
    }

    fn resetFilm(film: *?Film) void {
        if (film.*) |*f| f.conv.reset();
    }

    /// Zeroes every conv history (the caller drives prewarm, as with the
    /// reference engine).
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
                resetFilm(&layer.conv_pre_film);
                resetFilm(&layer.conv_post_film);
                resetFilm(&layer.input_mixin_pre_film);
                resetFilm(&layer.input_mixin_post_film);
                resetFilm(&layer.activation_pre_film);
                resetFilm(&layer.activation_post_film);
                resetFilm(&layer.layer1x1_post_film);
                resetFilm(&layer.head1x1_post_film);
            }
        }
        for (self.post_head) |*block| block.conv.reset();
    }

    /// Mono in -> mono out over one block. Every transient is owned by the
    /// exec scope opened here.
    pub fn process(self: *WaveNet, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        const input_t = try self.input_slot.view(ctx, input, frames);
        const out = try self.forwardBlock(ctx, &input_t);
        try out.copyTo(output[0..frames]);
    }

    /// The block forward inside the caller's exec scope (shared by `process`,
    /// by a parent's condition path, and by callers that bring the block as
    /// their own tensor).
    pub fn forwardBlock(self: *WaveNet, ctx: *ExecContext, input: *const TimeIn) anyerror!TimeOut {
        var condition_t: TimeIn = undefined;
        var condition: *const TimeIn = input;
        if (self.condition_child) |child| {
            const child_out = try child.forwardBlock(ctx, input);
            condition_t = try asIn(ctx, &child_out);
            condition = &condition_t;
        }

        var x: TimeIn = undefined;
        var head_prev: ?TimeOut = null;
        for (self.arrays, 0..) |*array, array_index| {
            const array_input: *const TimeIn = if (array_index == 0) input else &x;
            const rc = try array.rechannel.forward(ctx, array_input);
            x = try asIn(ctx, &rc);
            // Head accumulator: array 0 starts from the first layer's
            // contribution; later arrays are seeded by the previous head
            // output (mutated in place from here on).
            var acc: ?TimeOut = head_prev;
            for (array.layers) |*layer| try forwardLayer(ctx, layer, &x, condition, &acc);
            const head_in = try asIn(ctx, &acc.?);
            head_prev = try array.head_rechannel.forward(ctx, &head_in);
        }

        var current = try head_prev.?.scale(ctx, self.head_scale);
        for (self.post_head) |*block| {
            const activated = try activate(ctx, &block.activation, &current);
            const conv_in = try asIn(ctx, &activated);
            current = try block.conv.forward(ctx, &conv_in);
        }
        return current;
    }

    fn forwardLayer(ctx: *ExecContext, layer: *Layer, x: *TimeIn, condition: *const TimeIn, acc: *?TimeOut) !void {
        const b = layer.bottleneck;

        // z = conv(x) + bias + input_mixin(condition)
        var conv_input: TimeIn = x.*;
        if (layer.conv_pre_film) |*film| {
            const x_out = try asOut(ctx, x);
            const filmed = try film.forward(ctx, condition, &x_out);
            conv_input = try asIn(ctx, &filmed);
        }
        var z = try layer.conv.forward(ctx, &conv_input);
        if (layer.conv_post_film) |*film| z = try film.forward(ctx, condition, &z);

        var mixin_input: *const TimeIn = condition;
        var mixin_filmed: TimeIn = undefined;
        if (layer.input_mixin_pre_film) |*film| {
            const condition_out = try asOut(ctx, condition);
            const filmed = try film.forward(ctx, condition, &condition_out);
            mixin_filmed = try asIn(ctx, &filmed);
            mixin_input = &mixin_filmed;
        }
        var mix = try layer.input_mixin.forward(ctx, mixin_input);
        if (layer.input_mixin_post_film) |*film| mix = try film.forward(ctx, condition, &mix);
        try z.addScaledInPlace(ctx, &mix, 1.0);
        if (layer.activation_pre_film) |*film| z = try film.forward(ctx, condition, &z);

        // activation (gated: act(top) * act2(bottom); blended: pre + alpha*(act(pre) - pre))
        var activated: TimeOut = undefined;
        switch (layer.gating_mode) {
            .none => activated = try activate(ctx, &layer.activation, &z),
            .gated => {
                const top = try z.narrow(ctx, .out, 0, b);
                const bottom = try z.narrow(ctx, .out, b, b);
                const primary = try activate(ctx, &layer.activation, &top);
                activated = switch (layer.secondary_activation.kind) {
                    .sigmoid => try primary.glu(ctx, &bottom),
                    .silu => try primary.swiglu(ctx, &bottom),
                    else => blk: {
                        const gate = try activate(ctx, &layer.secondary_activation, &bottom);
                        break :blk try primary.mul(ctx, &gate);
                    },
                };
            },
            .blended => {
                const top = try z.narrow(ctx, .out, 0, b);
                const bottom = try z.narrow(ctx, .out, b, b);
                const primary = try activate(ctx, &layer.activation, &top);
                const alpha = try activate(ctx, &layer.secondary_activation, &bottom);
                const diff = try primary.sub(ctx, &top);
                var blended = try alpha.mul(ctx, &diff);
                try blended.addScaledInPlace(ctx, &top, 1.0);
                activated = blended;
            },
        }
        if (layer.activation_post_film) |*film| activated = try film.forward(ctx, condition, &activated);

        // residual (layer1x1 before head1x1, as in the reference)
        if (layer.layer1x1) |*conv| {
            const residual_in = try asIn(ctx, &activated);
            var residual = try conv.forward(ctx, &residual_in);
            if (layer.gating_mode == .blended) {
                if (layer.layer1x1_post_film) |*film| residual = try film.forward(ctx, condition, &residual);
            }
            try x.addScaledInPlace(ctx, &residual, 1.0);
        }

        // head contribution
        var contribution: TimeOut = activated;
        if (layer.head1x1) |*conv| {
            const head_in = try asIn(ctx, &activated);
            contribution = try conv.forward(ctx, &head_in);
            if (layer.head1x1_post_film) |*film| contribution = try film.forward(ctx, condition, &contribution);
        }
        if (acc.*) |*a| {
            try a.addScaledInPlace(ctx, &contribution, 1.0);
        } else {
            acc.* = contribution;
        }
    }
};

/// The cab IR as one long single-channel causal conv (`ir_cab.IrCab`'s
/// counterpart): `weight` is the reversed, gained IR (`IrCab.weight`).
pub const IrCab = struct {
    weight: DenseWeight,
    state: CausalState,
    taps: usize,
    input_slot: InputSlot = .{},

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, reversed_gained: []const f32, chunk_hint: usize) !IrCab {
        var weight = try DenseWeight.fromSlice(ctx, .{ reversed_gained.len, 1, 1 }, reversed_gained);
        errdefer weight.deinit();
        const state = try CausalState.init(allocator, 1, reversed_gained.len, 1, chunk_hint);
        return .{ .weight = weight, .state = state, .taps = reversed_gained.len };
    }

    pub fn deinit(self: *IrCab) void {
        self.weight.deinit();
        self.state.deinit();
        self.input_slot.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *IrCab) void {
        self.state.reset();
    }

    pub fn process(self: *IrCab, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        const x = try self.input_slot.view(ctx, input, frames);
        const y = try x.causalConv1dStreaming(ctx, .time, .in, .tap, .out, &self.weight, null, 1, &self.state);
        try y.copyTo(output[0..frames]);
    }
};

/// The LSTM runtime composed per sample from facade ops (`models.LstmEngine`'s
/// counterpart): the step's input is a tensor (`[x | h]` by `concat`), one
/// `dot` for the stacked gates, four narrowed activations, the cell and
/// hidden updates as elementwise ops, and the new `h` and `c` ARE the
/// state: ownership moves from the op outputs into the cell, nothing is
/// copied. Runs outside any exec scope so those outputs outlive the step.
pub const Lstm = struct {
    allocator: std.mem.Allocator,
    cells: []Cell,
    head_weight: Tensor(.{ .out, .h }),
    head_bias: []f32,

    const Cell = struct {
        w: Tensor(.{ .gate, .k }),
        b: []f32,
        /// Trained initial state, restored on `reset`.
        h0: []f32,
        c0: []f32,
        /// Live state: the previous step's outputs.
        h: Tensor(.{.h}),
        c: Tensor(.{.h}),
        input_size: usize,
        hidden: usize,
    };

    /// Copies the weights out of an initialized reference engine.
    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, engine: *const models.LstmEngine) !Lstm {
        const cells = try allocator.alloc(Cell, engine.cells.len);
        errdefer allocator.free(cells);
        var built: usize = 0;
        errdefer for (cells[0..built]) |*cell| deinitCell(allocator, cell);
        for (cells, engine.cells) |*cell, *src| {
            const h = src.hidden;
            var w = try Tensor(.{ .gate, .k }).fromSlice(ctx, .{ 4 * h, src.input_size + h }, src.w);
            errdefer w.deinit();
            var h_t = try Tensor(.{.h}).fromSlice(ctx, .{h}, src.h0);
            errdefer h_t.deinit();
            var c_t = try Tensor(.{.h}).fromSlice(ctx, .{h}, src.c0);
            errdefer c_t.deinit();
            cell.* = .{
                .w = w,
                .b = try allocator.dupe(f32, src.b),
                .h0 = try allocator.dupe(f32, src.h0),
                .c0 = try allocator.dupe(f32, src.c0),
                .h = h_t,
                .c = c_t,
                .input_size = src.input_size,
                .hidden = h,
            };
            built += 1;
        }
        const hidden = engine.cells[engine.cells.len - 1].hidden;
        var head_weight = try Tensor(.{ .out, .h }).fromSlice(ctx, .{ engine.head_bias.len, hidden }, engine.head_weight);
        errdefer head_weight.deinit();
        return .{
            .allocator = allocator,
            .cells = cells,
            .head_weight = head_weight,
            .head_bias = try allocator.dupe(f32, engine.head_bias),
        };
    }

    fn deinitCell(allocator: std.mem.Allocator, cell: *Cell) void {
        cell.w.deinit();
        allocator.free(cell.b);
        allocator.free(cell.h0);
        allocator.free(cell.c0);
        cell.h.deinit();
        cell.c.deinit();
    }

    pub fn deinit(self: *Lstm) void {
        for (self.cells) |*cell| deinitCell(self.allocator, cell);
        self.allocator.free(self.cells);
        self.head_weight.deinit();
        self.allocator.free(self.head_bias);
        self.* = undefined;
    }

    /// Back to the trained initial state.
    pub fn reset(self: *Lstm) !void {
        for (self.cells) |*cell| {
            try cell.h.copyFrom(cell.h0);
            try cell.c.copyFrom(cell.c0);
        }
    }

    pub fn processSample(self: *Lstm, ctx: *ExecContext, x: f32) !f32 {
        if (ctx.execScopeActive()) return error.ActiveExecScopeUnsupported;
        var input = try Tensor(.{.k}).fromSlice(ctx, .{1}, &[_]f32{x});
        defer input.deinit();
        for (self.cells) |*cell| {
            const h = cell.hidden;
            var h_k = try cell.h.withTags(ctx, .{.k});
            defer h_k.deinit();
            var xh = try input.concat(ctx, .k, &.{&h_k});
            defer xh.deinit();
            var gates = try cell.w.dot(ctx, &xh, .k);
            defer gates.deinit();
            try gates.addAxisVectorInPlace(ctx, cell.b, .gate);
            var i_pre = try gates.narrow(ctx, .gate, 0, h);
            defer i_pre.deinit();
            var f_pre = try gates.narrow(ctx, .gate, h, h);
            defer f_pre.deinit();
            var g_pre = try gates.narrow(ctx, .gate, 2 * h, h);
            defer g_pre.deinit();
            var o_pre = try gates.narrow(ctx, .gate, 3 * h, h);
            defer o_pre.deinit();
            var ig = try i_pre.sigmoid(ctx);
            defer ig.deinit();
            var fg = try f_pre.sigmoid(ctx);
            defer fg.deinit();
            var gg = try g_pre.tanh(ctx);
            defer gg.deinit();
            var og = try o_pre.sigmoid(ctx);
            defer og.deinit();
            // c = f*c + i*g; h = o*tanh(c). The outputs become the state.
            var fg_h = try fg.withTags(ctx, .{.h});
            defer fg_h.deinit();
            var c_new = try fg_h.mul(ctx, &cell.c);
            errdefer c_new.deinit();
            var ig_gg = try ig.mul(ctx, &gg);
            defer ig_gg.deinit();
            try c_new.addScaledInPlace(ctx, &ig_gg, 1.0);
            var tc = try c_new.tanh(ctx);
            defer tc.deinit();
            var og_h = try og.withTags(ctx, .{.h});
            defer og_h.deinit();
            const h_new = try og_h.mul(ctx, &tc);
            cell.c.deinit();
            cell.c = c_new;
            cell.h.deinit();
            cell.h = h_new;
            const next = try cell.h.withTags(ctx, .{.k});
            input.deinit();
            input = next;
        }
        const last = &self.cells[self.cells.len - 1];
        var out = try self.head_weight.dot(ctx, &last.h, .h);
        defer out.deinit();
        try out.addAxisVectorInPlace(ctx, self.head_bias, .out);
        return out.item();
    }

    pub fn process(self: *Lstm, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        for (0..frames) |t| output[t] = try self.processSample(ctx, input[t]);
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
