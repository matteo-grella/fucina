//! Streaming WaveNet engine — faithful port of NeuralAmpModelerCore's
//! wavenet runtime (NAM/wavenet/model.cpp, detail.h).
//!
//! Per block: condition = raw input; for each layer array (array 0 takes the
//! condition as layer input and zeroes its head accumulator; array i>0 takes
//! the previous array's residual outputs and head outputs):
//!   x = rechannel(input)                         [Conv1x1, no bias]
//!   per layer: z = dilated_conv(x) + input_mixin(condition)
//!              a = activation(z)                 [gated: act(top)*act2(bottom)]
//!              head_acc += head1x1(a) or a
//!              x = x + layer1x1(a)               [or x unchanged if inactive]
//!   head_out = head_rechannel(head_acc)          [causal conv, has memory when k>1]
//! Output = head_scale * head_out of the last array, optionally through the
//! post-stack head (activation BEFORE each conv, applied to the scaled
//! stream). All buffers are sized once at init; process() is allocation-free.

const std = @import("std");
const nam_file = @import("nam_file.zig");
const activations = @import("activations.zig");
const stream_conv = @import("stream_conv.zig");
const models = @import("models.zig");

const StreamConv = stream_conv.StreamConv;
const Activation = nam_file.Activation;

const FiLM = struct {
    conv: StreamConv,
    input_dim: usize,
    shift: bool,

    fn init(allocator: std.mem.Allocator, condition_dim: usize, input_dim: usize, params: nam_file.FiLMParams) !FiLM {
        const out_dim = input_dim * (if (params.shift) @as(usize, 2) else 1);
        return .{
            .conv = try StreamConv.initGrouped(allocator, condition_dim, out_dim, 1, 1, true, params.groups),
            .input_dim = input_dim,
            .shift = params.shift,
        };
    }

    fn deinit(self: *FiLM) void {
        self.conv.deinit();
        self.* = undefined;
    }

    fn loadNamWeights(self: *FiLM, weights: []const f32) usize {
        return self.conv.loadNamWeights(weights);
    }

    fn reset(self: *FiLM) void {
        self.conv.reset();
    }

    fn process(
        self: *FiLM,
        condition: []const f32,
        input: []const f32,
        output: []f32,
        scale_shift: []f32,
        frames: usize,
    ) void {
        const width = self.conv.out_channels;
        self.conv.process(condition, scale_shift, frames, false);
        self.conv.push(condition, frames);
        for (0..frames) |t| {
            const in_row = input[t * self.input_dim ..][0..self.input_dim];
            const out_row = output[t * self.input_dim ..][0..self.input_dim];
            const ss_row = scale_shift[t * width ..][0..width];
            for (0..self.input_dim) |i| {
                out_row[i] = in_row[i] * ss_row[i] + if (self.shift) ss_row[self.input_dim + i] else 0;
            }
        }
    }
};

const Layer = struct {
    conv: StreamConv, // C -> Bg, dilated, bias
    input_mixin: StreamConv, // Dc -> Bg, 1x1, no bias
    layer1x1: ?StreamConv, // B -> C, 1x1, bias
    head1x1: ?StreamConv, // B -> head1x1_out, 1x1, bias
    conv_pre_film: ?FiLM,
    conv_post_film: ?FiLM,
    input_mixin_pre_film: ?FiLM,
    input_mixin_post_film: ?FiLM,
    activation_pre_film: ?FiLM,
    activation_post_film: ?FiLM,
    layer1x1_post_film: ?FiLM,
    head1x1_post_film: ?FiLM,
    activation: Activation,
    secondary_activation: Activation,
    gating_mode: nam_file.GatingMode,
    bottleneck: usize,
};

const LayerArray = struct {
    rechannel: StreamConv, // input_size -> C, 1x1, no bias
    layers: []Layer,
    head_rechannel: StreamConv, // head_in -> head_out, k = head_kernel, bias?
    channels: usize,
    head_width: usize, // width of the head accumulator rows
};

const PostHeadBlock = struct {
    conv: StreamConv,
    activation: Activation,
};

const ConditionEngine = union(nam_file.Arch) {
    wavenet: *WaveNetEngine,
    lstm: models.LstmEngine,
    convnet: models.ConvNetEngine,
    linear: models.LinearEngine,

    fn init(allocator: std.mem.Allocator, dsp: *const nam_file.ConditionDsp) anyerror!ConditionEngine {
        return switch (dsp.config) {
            .wavenet => |*c| blk: {
                const child = try allocator.create(WaveNetEngine);
                errdefer allocator.destroy(child);
                child.* = try WaveNetEngine.init(allocator, c, dsp.weights);
                break :blk .{ .wavenet = child };
            },
            // The A2 reference models use a nested WaveNet condition DSP. Other
            // nested DSPs can be added when a real model needs them; the top-level
            // engines currently expose only mono output for these architectures.
            .lstm, .convnet, .linear => error.UnsupportedFeature,
        };
    }

    fn deinit(self: *ConditionEngine, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .wavenet => |child| {
                child.deinit();
                allocator.destroy(child);
            },
            .lstm => |*e| e.deinit(),
            .convnet => |*e| e.deinit(),
            .linear => |*e| e.deinit(),
        }
        self.* = undefined;
    }

    fn reset(self: *ConditionEngine, max_frames: usize) anyerror!void {
        switch (self.*) {
            .wavenet => |child| try child.reset(max_frames),
            .lstm => |*e| e.reset(),
            .convnet => |*e| try e.reset(max_frames),
            .linear => |*e| e.reset(),
        }
    }

    fn process(self: *ConditionEngine, input: []const f32, output: []f32, frames: usize) void {
        switch (self.*) {
            .wavenet => |child| child.process(input, output, frames),
            .lstm => |*e| e.process(input, output, frames),
            .convnet => |*e| e.process(input, output, frames),
            .linear => |*e| e.process(input, output, frames),
        }
    }

    fn prewarmSamples(self: *const ConditionEngine) usize {
        return switch (self.*) {
            .wavenet => |child| child.prewarmSamples(),
            .lstm => |*e| e.prewarmSamples(),
            .convnet => |*e| e.prewarmSamples(),
            .linear => 0,
        };
    }

    fn outputChannels(self: *const ConditionEngine) usize {
        return switch (self.*) {
            .wavenet => |child| child.outputChannels(),
            .lstm => |*e| e.head_bias.len,
            .convnet => |*e| e.head.out_channels,
            .linear => |*e| e.conv.out_channels,
        };
    }
};

pub const WaveNetEngine = struct {
    allocator: std.mem.Allocator,
    arrays: []LayerArray,
    post_head: []PostHeadBlock,
    condition_dsp: ?ConditionEngine,
    condition_channels: usize,
    head_scale: f32,
    prewarm_samples: usize,
    max_frames: usize,

    // Per-block scratch, sized for max_frames at reset():
    x: []f32, // residual stream rows [n, max channels]
    z: []f32, // pre-activation rows [n, max gate width]
    act: []f32, // post-activation rows [n, max bottleneck]
    head_acc: []f32, // head accumulator rows [n, max head width]
    head_out: []f32, // head_rechannel output rows [n, max head out]
    scratch: []f32, // 1x1 outputs [n, max channels]
    scratch2: []f32,
    film_affine: []f32,
    condition_out: []f32,

    pub fn init(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, weights: []const f32) anyerror!WaveNetEngine {
        const arrays = try allocator.alloc(LayerArray, config.layers.len);
        errdefer allocator.free(arrays);
        var arrays_built: usize = 0;
        errdefer for (arrays[0..arrays_built]) |*array| deinitLayerArray(allocator, array);

        var cursor: usize = 0;
        var prev_head_out: usize = 0;
        for (config.layers, arrays, 0..) |*lc, *array, i| {
            array.* = try buildLayerArray(allocator, lc, weights, &cursor);
            arrays_built += 1;
            // The head accumulator of array i>0 is seeded by the previous
            // array's head outputs, so the widths must agree.
            if (i > 0 and array.head_width != prev_head_out) return error.UnsupportedFeature;
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
                block.conv = try StreamConv.init(allocator, cin, cout, k, 1, true);
                post_built += 1;
                cursor += block.conv.loadNamWeights(weights[cursor..]);
                cin = cout;
            }
        }

        // head_scale: the final float of the stream overrides the JSON copy
        // (model.cpp:632).
        if (cursor + 1 != weights.len) return error.WeightCountMismatch;

        var condition_dsp: ?ConditionEngine = null;
        errdefer if (condition_dsp) |*engine| engine.deinit(allocator);
        var condition_channels: usize = 1;
        if (config.condition_dsp) |dsp| {
            condition_dsp = try ConditionEngine.init(allocator, dsp);
            if (condition_dsp) |*engine| condition_channels = engine.outputChannels();
        }
        for (config.layers) |*lc| {
            if (lc.condition_size != condition_channels) return error.UnsupportedChannels;
        }

        // Prewarm = 1 + sum of array receptive fields (+ post-head RF - 1)
        // (model.cpp:615-620).
        var prewarm: usize = if (condition_dsp) |*engine| engine.prewarmSamples() else 1;
        for (config.layers) |*lc| prewarm += lc.receptiveField() - 1;
        if (config.head) |*hc| {
            for (hc.kernel_sizes) |k| prewarm += k - 1;
        }

        return .{
            .allocator = allocator,
            .arrays = arrays,
            .post_head = post_head,
            .condition_dsp = condition_dsp,
            .condition_channels = condition_channels,
            .head_scale = weights[cursor],
            .prewarm_samples = prewarm,
            .max_frames = 0,
            .x = &.{},
            .z = &.{},
            .act = &.{},
            .head_acc = &.{},
            .head_out = &.{},
            .scratch = &.{},
            .scratch2 = &.{},
            .film_affine = &.{},
            .condition_out = &.{},
        };
    }

    fn buildLayerArray(allocator: std.mem.Allocator, lc: *const nam_file.WaveNetLayerArray, weights: []const f32, cursor: *usize) !LayerArray {
        var rechannel = try StreamConv.init(allocator, lc.input_size, lc.channels, 1, 1, false);
        errdefer rechannel.deinit();
        cursor.* += rechannel.loadNamWeights(weights[cursor.*..]);

        const layers = try allocator.alloc(Layer, lc.layerCount());
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*layer| deinitLayer(layer);
        for (layers, 0..) |*layer, l| {
            layer.* = try buildLayer(allocator, lc, l, weights, cursor);
            built += 1;
        }

        const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
        var head_rechannel = try StreamConv.init(allocator, head_width, lc.head_out, lc.head_kernel, 1, lc.head_bias);
        errdefer head_rechannel.deinit();
        cursor.* += head_rechannel.loadNamWeights(weights[cursor.*..]);

        return .{
            .rechannel = rechannel,
            .layers = layers,
            .head_rechannel = head_rechannel,
            .channels = lc.channels,
            .head_width = head_width,
        };
    }

    fn buildLayer(allocator: std.mem.Allocator, lc: *const nam_file.WaveNetLayerArray, l: usize, weights: []const f32, cursor: *usize) !Layer {
        const bg = lc.gateWidth(l);
        var conv = try StreamConv.initGrouped(allocator, lc.channels, bg, lc.kernel_sizes[l], lc.dilations[l], true, lc.groups_input);
        errdefer conv.deinit();
        cursor.* += conv.loadNamWeights(weights[cursor.*..]);

        var input_mixin = try StreamConv.initGrouped(allocator, lc.condition_size, bg, 1, 1, false, lc.groups_input_mixin);
        errdefer input_mixin.deinit();
        cursor.* += input_mixin.loadNamWeights(weights[cursor.*..]);

        var layer1x1: ?StreamConv = null;
        errdefer if (layer1x1) |*c| c.deinit();
        if (lc.layer1x1_active) {
            layer1x1 = try StreamConv.initGrouped(allocator, lc.bottleneck, lc.channels, 1, 1, true, lc.layer1x1_groups);
            cursor.* += layer1x1.?.loadNamWeights(weights[cursor.*..]);
        }

        var head1x1: ?StreamConv = null;
        if (lc.head1x1_active) {
            head1x1 = try StreamConv.initGrouped(allocator, lc.bottleneck, lc.head1x1_out, 1, 1, true, lc.head1x1_groups);
            cursor.* += head1x1.?.loadNamWeights(weights[cursor.*..]);
        }

        var conv_pre_film = try buildFilm(allocator, lc, lc.channels, lc.conv_pre_film, weights, cursor);
        errdefer if (conv_pre_film) |*f| f.deinit();
        var conv_post_film = try buildFilm(allocator, lc, bg, lc.conv_post_film, weights, cursor);
        errdefer if (conv_post_film) |*f| f.deinit();
        var input_mixin_pre_film = try buildFilm(allocator, lc, lc.condition_size, lc.input_mixin_pre_film, weights, cursor);
        errdefer if (input_mixin_pre_film) |*f| f.deinit();
        var input_mixin_post_film = try buildFilm(allocator, lc, bg, lc.input_mixin_post_film, weights, cursor);
        errdefer if (input_mixin_post_film) |*f| f.deinit();
        var activation_pre_film = try buildFilm(allocator, lc, bg, lc.activation_pre_film, weights, cursor);
        errdefer if (activation_pre_film) |*f| f.deinit();
        var activation_post_film = try buildFilm(allocator, lc, lc.bottleneck, lc.activation_post_film, weights, cursor);
        errdefer if (activation_post_film) |*f| f.deinit();
        var layer1x1_post_film = try buildFilm(allocator, lc, lc.channels, lc.layer1x1_post_film, weights, cursor);
        errdefer if (layer1x1_post_film) |*f| f.deinit();
        var head1x1_post_film = try buildFilm(allocator, lc, lc.head1x1_out, lc.head1x1_post_film, weights, cursor);
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
            .activation = lc.activations[l],
            .secondary_activation = lc.secondary_activations[l],
            .gating_mode = lc.gating_modes[l],
            .bottleneck = lc.bottleneck,
        };
    }

    fn buildFilm(
        allocator: std.mem.Allocator,
        lc: *const nam_file.WaveNetLayerArray,
        input_dim: usize,
        params: nam_file.FiLMParams,
        weights: []const f32,
        cursor: *usize,
    ) !?FiLM {
        if (!params.active) return null;
        var film = try FiLM.init(allocator, lc.condition_size, input_dim, params);
        errdefer film.deinit();
        cursor.* += film.loadNamWeights(weights[cursor.*..]);
        return film;
    }

    fn deinitLayer(layer: *Layer) void {
        layer.conv.deinit();
        layer.input_mixin.deinit();
        if (layer.layer1x1) |*c| c.deinit();
        if (layer.head1x1) |*c| c.deinit();
        if (layer.conv_pre_film) |*f| f.deinit();
        if (layer.conv_post_film) |*f| f.deinit();
        if (layer.input_mixin_pre_film) |*f| f.deinit();
        if (layer.input_mixin_post_film) |*f| f.deinit();
        if (layer.activation_pre_film) |*f| f.deinit();
        if (layer.activation_post_film) |*f| f.deinit();
        if (layer.layer1x1_post_film) |*f| f.deinit();
        if (layer.head1x1_post_film) |*f| f.deinit();
    }

    fn deinitLayerArray(allocator: std.mem.Allocator, array: *LayerArray) void {
        array.rechannel.deinit();
        array.head_rechannel.deinit();
        for (array.layers) |*layer| deinitLayer(layer);
        allocator.free(array.layers);
    }

    pub fn deinit(self: *WaveNetEngine) void {
        for (self.arrays) |*array| deinitLayerArray(self.allocator, array);
        self.allocator.free(self.arrays);
        for (self.post_head) |*block| block.conv.deinit();
        self.allocator.free(self.post_head);
        if (self.condition_dsp) |*engine| engine.deinit(self.allocator);
        self.freeScratch();
        self.* = undefined;
    }

    fn freeScratch(self: *WaveNetEngine) void {
        self.allocator.free(self.x);
        self.allocator.free(self.z);
        self.allocator.free(self.act);
        self.allocator.free(self.head_acc);
        self.allocator.free(self.head_out);
        self.allocator.free(self.scratch);
        self.allocator.free(self.scratch2);
        self.allocator.free(self.film_affine);
        self.allocator.free(self.condition_out);
        self.x = &.{};
        self.z = &.{};
        self.act = &.{};
        self.head_acc = &.{};
        self.head_out = &.{};
        self.scratch = &.{};
        self.scratch2 = &.{};
        self.film_affine = &.{};
        self.condition_out = &.{};
    }

    /// Sizes the per-block scratch and zeroes all streaming state. Does not
    /// prewarm — the engine wrapper drives that (block-rounded zeros).
    pub fn reset(self: *WaveNetEngine, max_frames: usize) anyerror!void {
        var max_channels: usize = 1;
        var max_gate: usize = 1;
        var max_bottleneck: usize = 1;
        var max_head_width: usize = 1;
        var max_head_out: usize = 1;
        var max_film_affine: usize = 1;
        for (self.arrays) |*array| {
            max_channels = @max(max_channels, array.channels);
            max_channels = @max(max_channels, array.rechannel.in_channels);
            max_channels = @max(max_channels, self.condition_channels);
            max_head_width = @max(max_head_width, array.head_width);
            max_head_out = @max(max_head_out, array.head_rechannel.out_channels);
            for (array.layers) |*layer| {
                max_gate = @max(max_gate, layer.conv.out_channels);
                max_bottleneck = @max(max_bottleneck, layer.bottleneck);
                max_filmAffine(&max_film_affine, layer);
            }
        }
        for (self.post_head) |*block| {
            max_head_out = @max(max_head_out, block.conv.out_channels);
            max_head_width = @max(max_head_width, block.conv.in_channels);
        }

        self.freeScratch();
        self.max_frames = max_frames;
        self.x = try self.allocator.alloc(f32, max_frames * max_channels);
        self.z = try self.allocator.alloc(f32, max_frames * max_gate);
        self.act = try self.allocator.alloc(f32, max_frames * max_bottleneck);
        self.head_acc = try self.allocator.alloc(f32, max_frames * @max(max_head_width, max_head_out));
        self.head_out = try self.allocator.alloc(f32, max_frames * @max(max_head_out, max_head_width));
        const scratch_width = @max(@max(max_channels, max_gate), max_head_width);
        self.scratch = try self.allocator.alloc(f32, max_frames * scratch_width);
        self.scratch2 = try self.allocator.alloc(f32, max_frames * scratch_width);
        self.film_affine = try self.allocator.alloc(f32, max_frames * max_film_affine);
        self.condition_out = try self.allocator.alloc(f32, max_frames * self.condition_channels);

        if (self.condition_dsp) |*engine| try engine.reset(max_frames);
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

    fn max_filmAffine(max_value: *usize, layer: *const Layer) void {
        inline for (.{ "conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film", "activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film" }) |field| {
            if (@field(layer, field)) |*film| max_value.* = @max(max_value.*, film.conv.out_channels);
        }
    }

    fn resetFilm(film: *?FiLM) void {
        if (film.*) |*f| f.reset();
    }

    /// Mono in -> mono out, `frames <= max_frames`. Allocation-free.
    pub fn process(self: *WaveNetEngine, input: []const f32, output: []f32, frames: usize) void {
        std.debug.assert(frames <= self.max_frames);
        const condition = blk: {
            if (self.condition_dsp) |*engine| {
                engine.process(input, self.condition_out, frames);
                break :blk self.condition_out[0 .. frames * self.condition_channels];
            }
            break :blk input[0..frames];
        };

        for (self.arrays, 0..) |*array, array_index| {
            const channels = array.channels;

            // Layer input: condition for array 0, previous residual stream
            // (already in self.x) afterwards. rechannel into scratch, then
            // adopt as the new x.
            if (array_index == 0) {
                array.rechannel.process(input, self.scratch, frames, false);
                array.rechannel.push(input, frames);
            } else {
                array.rechannel.process(self.x, self.scratch, frames, false);
                array.rechannel.push(self.x, frames);
            }
            @memcpy(self.x[0 .. frames * channels], self.scratch[0 .. frames * channels]);

            // Head accumulator: zero for array 0, previous head outputs after
            // (model.cpp:427-448).
            const head_width = array.head_width;
            if (array_index == 0) {
                @memset(self.head_acc[0 .. frames * head_width], 0);
            } else {
                @memcpy(self.head_acc[0 .. frames * head_width], self.head_out[0 .. frames * head_width]);
            }

            for (array.layers) |*layer| {
                const bg = layer.conv.out_channels;
                const b = layer.bottleneck;

                // z = dilated conv(x) + input_mixin(condition)
                const conv_input = blk: {
                    if (layer.conv_pre_film) |*film| {
                        film.process(condition, self.x[0 .. frames * channels], self.scratch, self.film_affine, frames);
                        break :blk self.scratch[0 .. frames * channels];
                    }
                    break :blk self.x[0 .. frames * channels];
                };
                layer.conv.process(conv_input, self.z, frames, false);
                layer.conv.push(conv_input, frames);
                if (layer.conv_post_film) |*film| {
                    film.process(condition, self.z[0 .. frames * bg], self.scratch, self.film_affine, frames);
                    @memcpy(self.z[0 .. frames * bg], self.scratch[0 .. frames * bg]);
                }

                const mixin_input = blk: {
                    if (layer.input_mixin_pre_film) |*film| {
                        film.process(condition, condition, self.scratch, self.film_affine, frames);
                        break :blk self.scratch[0 .. frames * self.condition_channels];
                    }
                    break :blk condition;
                };
                if (layer.input_mixin_post_film) |*film| {
                    layer.input_mixin.process(mixin_input, self.scratch2, frames, false);
                    layer.input_mixin.push(mixin_input, frames);
                    film.process(condition, self.scratch2[0 .. frames * bg], self.scratch, self.film_affine, frames);
                    for (self.z[0 .. frames * bg], self.scratch[0 .. frames * bg]) |*dst, v| dst.* += v;
                } else {
                    layer.input_mixin.process(mixin_input, self.scratch2, frames, false);
                    layer.input_mixin.push(mixin_input, frames);
                    for (self.z[0 .. frames * bg], self.scratch2[0 .. frames * bg]) |*dst, v| dst.* += v;
                }
                if (layer.activation_pre_film) |*film| {
                    film.process(condition, self.z[0 .. frames * bg], self.scratch, self.film_affine, frames);
                    @memcpy(self.z[0 .. frames * bg], self.scratch[0 .. frames * bg]);
                }

                // activation (gated: top half * act2(bottom half))
                var activated = switch (layer.gating_mode) {
                    .none => blk: {
                        activations.applyRows(&layer.activation, self.z[0 .. frames * b], b);
                        break :blk self.z[0 .. frames * b];
                    },
                    .gated => blk: {
                        activations.applyGated(&layer.activation, &layer.secondary_activation, self.z[0 .. frames * bg], self.act[0 .. frames * b], frames, b);
                        break :blk self.act[0 .. frames * b];
                    },
                    .blended => blk: {
                        activations.applyBlended(&layer.activation, &layer.secondary_activation, self.z[0 .. frames * bg], self.act[0 .. frames * b], frames, b);
                        break :blk self.act[0 .. frames * b];
                    },
                };
                if (layer.activation_post_film) |*film| {
                    film.process(condition, activated, self.scratch, self.film_affine, frames);
                    @memcpy(self.act[0 .. frames * b], self.scratch[0 .. frames * b]);
                    activated = self.act[0 .. frames * b];
                }

                // residual (reference computes layer1x1 before head1x1)
                if (layer.layer1x1) |*conv| {
                    conv.process(activated, self.scratch2, frames, false);
                    conv.push(activated, frames);
                    const residual = blk: {
                        // NeuralAmpModelerCore only applies this post-FiLM in
                        // the BLENDED branch (model.cpp:262-269). The object
                        // and weights can be present for other gating modes.
                        if (layer.gating_mode == .blended) {
                            if (layer.layer1x1_post_film) |*film| {
                                film.process(condition, self.scratch2[0 .. frames * channels], self.scratch, self.film_affine, frames);
                                break :blk self.scratch[0 .. frames * channels];
                            }
                        }
                        break :blk self.scratch2[0 .. frames * channels];
                    };
                    for (self.x[0 .. frames * channels], residual) |*dst, v| dst.* += v;
                }

                // head contribution
                if (layer.head1x1) |*conv| {
                    if (layer.head1x1_post_film) |*film| {
                        conv.process(activated, self.scratch2, frames, false);
                        conv.push(activated, frames);
                        film.process(condition, self.scratch2[0 .. frames * conv.out_channels], self.scratch, self.film_affine, frames);
                        for (self.head_acc[0 .. frames * conv.out_channels], self.scratch[0 .. frames * conv.out_channels]) |*dst, v| dst.* += v;
                    } else {
                        conv.process(activated, self.scratch, frames, false);
                        conv.push(activated, frames);
                        for (self.head_acc[0 .. frames * conv.out_channels], self.scratch[0 .. frames * conv.out_channels]) |*dst, v| dst.* += v;
                    }
                } else {
                    for (self.head_acc[0 .. frames * b], activated) |*dst, v| dst.* += v;
                }
            }

            array.head_rechannel.process(self.head_acc, self.head_out, frames, false);
            array.head_rechannel.push(self.head_acc, frames);
        }

        const final_out = self.arrays[self.arrays.len - 1].head_rechannel.out_channels;
        if (self.post_head.len == 0) {
            for (0..frames) |t| {
                const src = self.head_out[t * final_out ..][0..final_out];
                const dst = output[t * final_out ..][0..final_out];
                for (dst, src) |*d, v| d.* = self.head_scale * v;
            }
            return;
        }

        // Post-stack head: scale first, then activation-before-conv blocks
        // (model.cpp:776-805; detail::Head).
        for (self.head_acc[0 .. frames * final_out], self.head_out[0 .. frames * final_out]) |*dst, v| dst.* = self.head_scale * v;
        var current = self.head_acc;
        var other = self.head_out;
        var width = final_out;
        for (self.post_head) |*block| {
            activations.applyRows(&block.activation, current[0 .. frames * width], width);
            block.conv.process(current, other, frames, false);
            block.conv.push(current, frames);
            const tmp = current;
            current = other;
            other = tmp;
            width = block.conv.out_channels;
        }
        std.debug.assert(width == 1);
        for (output[0..frames], 0..) |*dst, t| dst.* = current[t];
    }

    pub fn prewarmSamples(self: *const WaveNetEngine) usize {
        return self.prewarm_samples;
    }

    pub fn outputChannels(self: *const WaveNetEngine) usize {
        if (self.post_head.len > 0) return self.post_head[self.post_head.len - 1].conv.out_channels;
        return self.arrays[self.arrays.len - 1].head_rechannel.out_channels;
    }
};

test {
    _ = @import("wavenet_tests.zig");
}
