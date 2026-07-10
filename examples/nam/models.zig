//! Streaming LSTM, ConvNet, and Linear engines — faithful ports of
//! NeuralAmpModelerCore's lstm.cpp, convnet.cpp, linear.cpp runtimes.
//! Unlike upstream (whose ConvNet/LSTM allocate per process() call), all
//! engines here are allocation-free after init/reset — a deliberate,
//! numerics-preserving deviation.

const std = @import("std");
const nam_file = @import("nam_file.zig");
const activations = @import("activations.zig");
const stream_conv = @import("stream_conv.zig");

const StreamConv = stream_conv.StreamConv;

// ---------------------------------------------------------------------------
// LSTM (NAM/lstm.cpp): strictly sample-by-sample; trained initial h/c are
// part of the weight stream; gate order i,f,g,o over the combined
// [W_ih | W_hh] matrix; head = W*h_last + b.
// ---------------------------------------------------------------------------

const LstmCell = struct {
    /// (4H) x (input + H), row-major as PyTorch flattens.
    w: []f32,
    /// 4H (the exported sum of PyTorch's two bias vectors).
    b: []f32,
    /// [x (input) | h (H)] working vector.
    xh: []f32,
    c: []f32,
    /// Trained initial states, restored on reset.
    h0: []f32,
    c0: []f32,
    gates: []f32, // 4H scratch
    input_size: usize,
    hidden: usize,
};

pub const LstmEngine = struct {
    allocator: std.mem.Allocator,
    cells: []LstmCell,
    head_weight: []f32, // out x H row-major
    head_bias: []f32, // out
    out_scratch: []f32,
    /// 0.5 s of the model's expected rate (lstm.cpp:125-132); computed at
    /// engine init from the model sample rate.
    prewarm_samples: usize,

    pub fn init(allocator: std.mem.Allocator, config: *const nam_file.LstmConfig, weights: []const f32, expected_sample_rate: f64) !LstmEngine {
        const h = config.hidden_size;
        const cells = try allocator.alloc(LstmCell, config.num_layers);
        errdefer allocator.free(cells);
        var built: usize = 0;
        errdefer for (cells[0..built]) |*cell| deinitCell(allocator, cell);

        var cursor: usize = 0;
        for (cells, 0..) |*cell, l| {
            const in_l = if (l == 0) config.input_size else h;
            cell.* = .{
                .w = try allocator.alloc(f32, 4 * h * (in_l + h)),
                .b = undefined,
                .xh = undefined,
                .c = undefined,
                .h0 = undefined,
                .c0 = undefined,
                .gates = undefined,
                .input_size = in_l,
                .hidden = h,
            };
            // Partial-failure cleanup inside the cell.
            var ok = false;
            defer if (!ok) allocator.free(cell.w);
            cell.b = try allocator.alloc(f32, 4 * h);
            defer if (!ok) allocator.free(cell.b);
            cell.xh = try allocator.alloc(f32, in_l + h);
            defer if (!ok) allocator.free(cell.xh);
            cell.c = try allocator.alloc(f32, h);
            defer if (!ok) allocator.free(cell.c);
            cell.h0 = try allocator.alloc(f32, h);
            defer if (!ok) allocator.free(cell.h0);
            cell.c0 = try allocator.alloc(f32, h);
            defer if (!ok) allocator.free(cell.c0);
            cell.gates = try allocator.alloc(f32, 4 * h);
            ok = true;

            @memcpy(cell.w, weights[cursor..][0 .. 4 * h * (in_l + h)]);
            cursor += 4 * h * (in_l + h);
            @memcpy(cell.b, weights[cursor..][0 .. 4 * h]);
            cursor += 4 * h;
            @memcpy(cell.h0, weights[cursor..][0..h]);
            cursor += h;
            @memcpy(cell.c0, weights[cursor..][0..h]);
            cursor += h;
            built += 1;
        }

        const out = config.out_channels;
        const head_weight = try allocator.alloc(f32, out * h);
        errdefer allocator.free(head_weight);
        @memcpy(head_weight, weights[cursor..][0 .. out * h]);
        cursor += out * h;
        const head_bias = try allocator.alloc(f32, out);
        errdefer allocator.free(head_bias);
        @memcpy(head_bias, weights[cursor..][0..out]);
        cursor += out;
        if (cursor != weights.len) return error.WeightCountMismatch;

        const out_scratch = try allocator.alloc(f32, out);

        const prewarm_f = 0.5 * expected_sample_rate;
        const prewarm: usize = if (prewarm_f >= 1) @intFromFloat(prewarm_f) else 1;

        var self = LstmEngine{
            .allocator = allocator,
            .cells = cells,
            .head_weight = head_weight,
            .head_bias = head_bias,
            .out_scratch = out_scratch,
            .prewarm_samples = prewarm,
        };
        self.reset();
        return self;
    }

    fn deinitCell(allocator: std.mem.Allocator, cell: *LstmCell) void {
        allocator.free(cell.w);
        allocator.free(cell.b);
        allocator.free(cell.xh);
        allocator.free(cell.c);
        allocator.free(cell.h0);
        allocator.free(cell.c0);
        allocator.free(cell.gates);
    }

    pub fn deinit(self: *LstmEngine) void {
        for (self.cells) |*cell| deinitCell(self.allocator, cell);
        self.allocator.free(self.cells);
        self.allocator.free(self.head_weight);
        self.allocator.free(self.head_bias);
        self.allocator.free(self.out_scratch);
        self.* = undefined;
    }

    pub fn reset(self: *LstmEngine) void {
        for (self.cells) |*cell| {
            @memset(cell.xh[0..cell.input_size], 0);
            @memcpy(cell.xh[cell.input_size..], cell.h0);
            @memcpy(cell.c, cell.c0);
        }
    }

    fn processSample(self: *LstmEngine, x: f32) f32 {
        const single = [1]f32{x};
        var input_slice: []const f32 = &single;
        for (self.cells) |*cell| {
            const h = cell.hidden;
            @memcpy(cell.xh[0..cell.input_size], input_slice[0..cell.input_size]);
            // gates = W * xh + b
            const width = cell.input_size + h;
            for (0..4 * h) |row| {
                var acc: f32 = cell.b[row];
                const w_row = cell.w[row * width ..][0..width];
                for (w_row, cell.xh) |w, v| acc += w * v;
                cell.gates[row] = acc;
            }
            // i, f, g, o; c = sig(f)*c + sig(i)*tanh(g); h = sig(o)*tanh(c)
            const hp = cell.xh[cell.input_size..];
            for (0..h) |j| {
                const ig = activations.sigmoid(cell.gates[j]);
                const fg = activations.sigmoid(cell.gates[h + j]);
                const gg = std.math.tanh(cell.gates[2 * h + j]);
                const og = activations.sigmoid(cell.gates[3 * h + j]);
                cell.c[j] = fg * cell.c[j] + ig * gg;
                hp[j] = og * std.math.tanh(cell.c[j]);
            }
            input_slice = hp;
        }
        // head: out = W*h + b (out_channels = 1 in practice)
        const h_last = input_slice;
        const hidden = self.cells[self.cells.len - 1].hidden;
        var acc: f32 = self.head_bias[0];
        for (self.head_weight[0..hidden], h_last) |w, v| acc += w * v;
        return acc;
    }

    pub fn process(self: *LstmEngine, input: []const f32, output: []f32, frames: usize) void {
        for (0..frames) |t| output[t] = self.processSample(input[t]);
    }

    pub fn prewarmSamples(self: *const LstmEngine) usize {
        return self.prewarm_samples;
    }
};

// ---------------------------------------------------------------------------
// ConvNet (NAM/convnet.cpp): blocks of Conv1D(k=2, dilated) + folded
// BatchNorm + activation; head = plain linear, no activation. Prewarm =
// 1 + sum(dilations).
// ---------------------------------------------------------------------------

const ConvNetBlock = struct {
    conv: StreamConv,
    /// Folded batchnorm: y = y*scale[c] + loc[c]; empty when batchnorm off.
    scale: []f32,
    loc: []f32,
};

pub const ConvNetEngine = struct {
    allocator: std.mem.Allocator,
    blocks: []ConvNetBlock,
    head: StreamConv, // channels -> out, 1x1, bias
    activation: nam_file.Activation,
    prewarm_samples: usize,
    max_frames: usize,
    a: []f32,
    b: []f32,

    pub fn init(allocator: std.mem.Allocator, config: *const nam_file.ConvNetConfig, weights: []const f32) !ConvNetEngine {
        const blocks = try allocator.alloc(ConvNetBlock, config.dilations.len);
        errdefer allocator.free(blocks);
        var built: usize = 0;
        errdefer for (blocks[0..built]) |*block| deinitBlock(allocator, block);

        var cursor: usize = 0;
        var cin = config.in_channels;
        for (blocks, config.dilations) |*block, dilation| {
            const cout = config.channels;
            var conv = try StreamConv.init(allocator, cin, cout, nam_file.ConvNetConfig.kernel_size, dilation, !config.batchnorm);
            errdefer conv.deinit();
            cursor += conv.loadNamWeights(weights[cursor..]);

            var scale: []f32 = &.{};
            var loc: []f32 = &.{};
            if (config.batchnorm) {
                scale = try allocator.alloc(f32, cout);
                errdefer allocator.free(scale);
                loc = try allocator.alloc(f32, cout);
                // weight order: running_mean, running_var, gamma (w), beta
                // (b), eps (1 value); folded with DOUBLE sqrt
                // (convnet.cpp:14-37).
                const mean = weights[cursor..][0..cout];
                const variance = weights[cursor + cout ..][0..cout];
                const gamma = weights[cursor + 2 * cout ..][0..cout];
                const beta = weights[cursor + 3 * cout ..][0..cout];
                const eps = weights[cursor + 4 * cout];
                cursor += 4 * cout + 1;
                for (scale, loc, mean, variance, gamma, beta) |*s, *o, m, v, g, bt| {
                    const s64 = @as(f64, g) / @sqrt(@as(f64, eps) + @as(f64, v));
                    s.* = @floatCast(s64);
                    o.* = @floatCast(@as(f64, bt) - s64 * @as(f64, m));
                }
            }
            block.* = .{ .conv = conv, .scale = scale, .loc = loc };
            built += 1;
            cin = cout;
        }

        var head = try StreamConv.init(allocator, config.channels, config.out_channels, 1, 1, true);
        errdefer head.deinit();
        cursor += head.loadNamWeights(weights[cursor..]);
        if (cursor != weights.len) return error.WeightCountMismatch;

        var prewarm: usize = 1;
        for (config.dilations) |d| prewarm += d;

        return .{
            .allocator = allocator,
            .blocks = blocks,
            .head = head,
            .activation = config.activation,
            .prewarm_samples = prewarm,
            .max_frames = 0,
            .a = &.{},
            .b = &.{},
        };
    }

    fn deinitBlock(allocator: std.mem.Allocator, block: *ConvNetBlock) void {
        block.conv.deinit();
        allocator.free(block.scale);
        allocator.free(block.loc);
    }

    pub fn deinit(self: *ConvNetEngine) void {
        for (self.blocks) |*block| deinitBlock(self.allocator, block);
        self.allocator.free(self.blocks);
        self.head.deinit();
        self.allocator.free(self.a);
        self.allocator.free(self.b);
        self.* = undefined;
    }

    pub fn reset(self: *ConvNetEngine, max_frames: usize) !void {
        var max_ch: usize = 1;
        for (self.blocks) |*block| max_ch = @max(max_ch, block.conv.out_channels);
        max_ch = @max(max_ch, self.head.out_channels);
        self.allocator.free(self.a);
        self.allocator.free(self.b);
        self.a = &.{};
        self.b = &.{};
        self.a = try self.allocator.alloc(f32, max_frames * max_ch);
        self.b = try self.allocator.alloc(f32, max_frames * max_ch);
        self.max_frames = max_frames;
        for (self.blocks) |*block| block.conv.reset();
        self.head.reset();
    }

    pub fn process(self: *ConvNetEngine, input: []const f32, output: []f32, frames: usize) void {
        std.debug.assert(frames <= self.max_frames);
        var current: []const f32 = input;
        var dst = self.a;
        var other = self.b;
        for (self.blocks) |*block| {
            const cout = block.conv.out_channels;
            block.conv.process(current, dst, frames, false);
            block.conv.push(current, frames);
            if (block.scale.len > 0) {
                for (0..frames) |t| {
                    const row = dst[t * cout ..][0..cout];
                    for (row, block.scale, block.loc) |*v, s, o| v.* = v.* * s + o;
                }
            }
            activations.applyRows(&self.activation, dst[0 .. frames * cout], cout);
            current = dst;
            const tmp = dst;
            dst = other;
            other = tmp;
        }
        self.head.process(current, dst, frames, false);
        for (output[0..frames], 0..) |*v, t| v.* = dst[t * self.head.out_channels];
    }

    pub fn prewarmSamples(self: *const ConvNetEngine) usize {
        return self.prewarm_samples;
    }
};

// ---------------------------------------------------------------------------
// Linear (NAM/linear.cpp): y[t] = bias + sum_j w[j]*x[t-j] with weights[0]
// multiplying the NEWEST sample — the C++ player semantics, pinned by
// upstream test_linear.cpp:101-113 (upstream's own Python exporter writes
// the reverse orientation — a verified upstream writer/reader discrepancy;
// every player follows the C++ reading). Implemented over StreamConv by
// reverse-copying into tap order (tap K-1 = newest).
// ---------------------------------------------------------------------------

pub const LinearEngine = struct {
    conv: StreamConv,

    pub fn init(allocator: std.mem.Allocator, config: *const nam_file.LinearConfig, weights: []const f32) !LinearEngine {
        var conv = try StreamConv.init(allocator, 1, 1, config.receptive_field, 1, config.bias);
        errdefer conv.deinit();
        const rf = config.receptive_field;
        // NAM weights[j] = lag-j tap; our tap k multiplies lag K-1-k.
        for (0..rf) |j| conv.weight[rf - 1 - j] = weights[j];
        if (config.bias) conv.bias[0] = weights[rf];
        return .{ .conv = conv };
    }

    pub fn deinit(self: *LinearEngine) void {
        self.conv.deinit();
        self.* = undefined;
    }

    pub fn reset(self: *LinearEngine) void {
        self.conv.reset();
    }

    pub fn process(self: *LinearEngine, input: []const f32, output: []f32, frames: usize) void {
        self.conv.process(input, output, frames, false);
        self.conv.push(input, frames);
    }
};

test {
    _ = @import("models_tests.zig");
}
