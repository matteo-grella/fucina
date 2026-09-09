//! The NAM ConvNet and Linear architectures as fucina tensors (upstream
//! NAM/convnet.cpp, NAM/linear.cpp), streaming over the core's causal conv
//! with its context ring; the LSTM lives in `lstm.zig`.
//!
//! ConvNet: blocks of Conv1D(k=2, dilated) + folded BatchNorm + activation,
//! then a 1x1 head with bias and no activation. The batchnorm fold
//! (`y·scale[c] + loc[c]`, scale and loc in f64 as convnet.cpp:14-37 does)
//! goes into the block's conv: the weight rows scaled per out channel by a
//! tensor `mul` over the stream view, the offset as the conv's bias, so a
//! block is one streaming conv call plus the activation. Prewarm = 1 +
//! sum(dilations).
//!
//! Linear: `y[t] = bias + Σ_j w[j]·x[t−j]` with `weights[0]` multiplying the
//! newest sample (the C++ player semantics, pinned by upstream
//! test_linear.cpp:101-113; upstream's own Python exporter writes the
//! reverse orientation, a verified writer/reader discrepancy that every
//! player resolves the C++ way), i.e. the causal conv with the taps
//! reversed (tap `K−1` = newest).

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");

const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;
const Conv = wavenet.Conv;
const InputSlot = wavenet.InputSlot;

pub const Error = error{ WeightCountMismatch, UnsupportedChannels };

const ConvNetBlock = struct {
    conv: Conv,
};

pub const ConvNet = struct {
    allocator: std.mem.Allocator,
    blocks: []ConvNetBlock,
    head: Conv,
    activation: nam_file.Activation,
    activation_slopes: ?wavenet.Bias,
    prewarm_samples: usize,
    input_slot: InputSlot = .{},

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.ConvNetConfig, weights: []const f32, chunk_hint: usize) !ConvNet {
        const k = nam_file.ConvNetConfig.kernel_size;
        const options = wavenet.Options{ .trainable = false, .chunk_hint = chunk_hint };
        const blocks = try allocator.alloc(ConvNetBlock, config.dilations.len);
        errdefer allocator.free(blocks);
        var built: usize = 0;
        errdefer for (blocks[0..built]) |*block| block.conv.deinit();

        // The batchnorm fold in f64 (convnet.cpp:14-37): a scale and an
        // offset per out channel, computed into these once-allocated rows.
        const cout = config.channels;
        const scales = try allocator.alloc(f32, cout);
        defer allocator.free(scales);
        const offsets = try allocator.alloc(f32, cout);
        defer allocator.free(offsets);

        var cursor: usize = 0;
        var cin = config.in_channels;
        for (blocks, config.dilations) |*block, dilation| {
            const weight_len = cout * cin * k;
            if (config.batchnorm) {
                // Stream: the conv weights (no bias), then running_mean,
                // running_var, gamma, beta (cout each) and eps. The scale
                // goes into the weight rows (a tensor `mul` over the stream's
                // (out, in, tap) view, permuted into the conv's layout), the
                // offset is the conv's bias.
                if (cursor + weight_len + 4 * cout + 1 > weights.len) return Error.WeightCountMismatch;
                const mean = weights[cursor + weight_len ..][0..cout];
                const variance = weights[cursor + weight_len + cout ..][0..cout];
                const gamma = weights[cursor + weight_len + 2 * cout ..][0..cout];
                const beta = weights[cursor + weight_len + 3 * cout ..][0..cout];
                const eps = weights[cursor + weight_len + 4 * cout];
                for (0..cout) |c| {
                    const scale64 = @as(f64, gamma[c]) / @sqrt(@as(f64, eps) + @as(f64, variance[c]));
                    scales[c] = @floatCast(scale64);
                    offsets[c] = @floatCast(@as(f64, beta[c]) - scale64 * @as(f64, mean[c]));
                }
                var rows = try Tensor(.{ .out, .in_group, .tap }).fromBorrowedConstSlice(ctx, .{ cout, cin, k }, weights[cursor..][0..weight_len]);
                defer rows.deinit();
                var scale_t = try Tensor(.{.out}).fromBorrowedConstSlice(ctx, .{cout}, scales);
                defer scale_t.deinit();
                var scaled = try rows.mul(ctx, &scale_t);
                defer scaled.deinit();
                var permuted = try scaled.permuteTo(ctx, .{ .tap, .in_group, .out });
                defer permuted.deinit();
                var weight = try permuted.copy(ctx);
                errdefer weight.deinit();
                var bias = try wavenet.Bias.fromSlice(ctx, .{cout}, offsets);
                errdefer bias.deinit();
                block.conv = try Conv.fromTensors(allocator, weight, bias, cin, cout, 1, k, dilation, chunk_hint);
                cursor += weight_len + 4 * cout + 1;
            } else {
                block.conv = try Conv.init(allocator, ctx, cin, cout, k, dilation, true, 1, options, weights, &cursor);
            }
            built += 1;
            cin = cout;
        }

        var head = try Conv.init(allocator, ctx, config.channels, config.out_channels, 1, 1, true, 1, options, weights, &cursor);
        errdefer head.deinit();
        if (cursor != weights.len) return Error.WeightCountMismatch;

        const activation_slopes = try wavenet.preluSlopes(ctx, &config.activation, config.channels);
        var prewarm: usize = 1;
        for (config.dilations) |d| prewarm += d;
        return .{ .allocator = allocator, .blocks = blocks, .head = head, .activation = config.activation, .activation_slopes = activation_slopes, .prewarm_samples = prewarm };
    }

    pub fn deinit(self: *ConvNet) void {
        for (self.blocks) |*block| block.conv.deinit();
        self.allocator.free(self.blocks);
        self.head.deinit();
        if (self.activation_slopes) |*t| t.deinit();
        self.input_slot.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *ConvNet) void {
        for (self.blocks) |*block| block.conv.reset();
        self.head.reset();
    }

    pub fn prewarmSamples(self: *const ConvNet) usize {
        return self.prewarm_samples;
    }

    pub fn process(self: *ConvNet, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        var no_grad = fucina.noGrad();
        defer no_grad.close();
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        var current = try self.input_slot.view(ctx, input, frames);
        for (self.blocks) |*block| {
            const y = try block.conv.forward(ctx, &current);
            const activated = try wavenet.activate(ctx, &self.activation, &y, false, if (self.activation_slopes) |*t| t else null);
            current = try activated.withTags(ctx, .{ .time, .in });
        }
        const out = try self.head.forward(ctx, &current);
        try out.copyTo(output[0..frames]);
    }
};

pub const Linear = struct {
    conv: Conv,
    input_slot: InputSlot = .{},

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.LinearConfig, weights: []const f32, chunk_hint: usize) !Linear {
        const rf = config.receptive_field;
        const expected = rf + @as(usize, if (config.bias) 1 else 0);
        if (weights.len != expected) return Error.WeightCountMismatch;
        // NAM weights[j] = lag-j tap; our tap k multiplies lag K-1-k.
        const stream = try allocator.alloc(f32, expected);
        defer allocator.free(stream);
        for (0..rf) |j| stream[rf - 1 - j] = weights[j];
        if (config.bias) stream[rf] = weights[rf];
        var cursor: usize = 0;
        const conv = try Conv.init(allocator, ctx, 1, 1, rf, 1, config.bias, 1, .{ .trainable = false, .chunk_hint = chunk_hint }, stream, &cursor);
        return .{ .conv = conv };
    }

    pub fn deinit(self: *Linear) void {
        self.conv.deinit();
        self.input_slot.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *Linear) void {
        self.conv.reset();
    }

    pub fn process(self: *Linear, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        var no_grad = fucina.noGrad();
        defer no_grad.close();
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        const x = try self.input_slot.view(ctx, input, frames);
        const y = try self.conv.forward(ctx, &x);
        try y.copyTo(output[0..frames]);
    }
};

test {
    _ = @import("models_tests.zig");
}
