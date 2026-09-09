//! The NAM LSTM model over `fucina.rnn`: the core's recurrent stack (one
//! sequence op per layer and block), a linear head on the last layer's
//! `h`, the NAM weight-stream layout, and the trainer's window contract
//! (burn-in without gradient, truncated backpropagation through time, MSE
//! over the segment's tail).
//!
//! NAM semantics (nam/models/recurrent.py, NAM/lstm.cpp) are the core's
//! PyTorch semantics: gate order i, f, g, o over `[x | h]`, one bias
//! vector (the trainer's `b_ih + b_hh`), learned initial states. The NAM
//! stream (spec §5.3) is, per layer, the stacked `[4H, in + H]` matrix
//! row-major, the bias `[4H]`, `h0`, `c0`; then the head `[out, H]` and
//! its bias. Import and export are views over that stream.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;
const rng = fucina.rng;
const rnn = fucina.rnn;

pub const Units = rnn.Units;
pub const HeadWeight = Tensor(.{ .unit, .out });
pub const HeadBias = Tensor(.{.out});
pub const Output = Tensor(.{ .time, .hout });

pub const Error = error{ WeightCountMismatch, UnsupportedChannels };

pub const Spec = struct {
    hidden_size: usize = 24,
    num_layers: usize = 1,
    input_size: usize = 1,
    /// Samples run without gradient before the training segment (the NAM
    /// trainer's `train_burn_in`).
    burn_in: usize = 4096,
    /// Gradient horizon: the state is detached every `truncate` steps of
    /// the training segment (NAM's `train_truncate`); 0 = whole segment.
    truncate: usize = 512,

    pub const standard = Spec{};

    pub fn name(_: Spec) []const u8 {
        return "lstm";
    }

    /// The trainer's window contract: `receptiveField() - 1` samples precede
    /// every target sample, here the burn-in.
    pub fn receptiveField(self: *const Spec) usize {
        return self.burn_in + 1;
    }

    pub fn engineConfig(self: *const Spec) nam_file.LstmConfig {
        return .{ .input_size = self.input_size, .hidden_size = self.hidden_size, .num_layers = self.num_layers, .in_channels = 1, .out_channels = 1 };
    }

    fn forwardOptions(self: *const Spec) rnn.Lstm.ForwardOptions {
        return .{ .burn_in = self.burn_in, .truncate = self.truncate };
    }
};

/// The training schedule an imported model trains with (`Spec`'s burn-in
/// and truncation, separate from the file's architecture).
pub const Training = struct {
    burn_in: usize = 4096,
    truncate: usize = 512,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    spec: Spec,
    lstm: rnn.Lstm,
    /// `[H, out]`: the NAM head `[out, H]` transposed.
    head_w: HeadWeight,
    head_b: HeadBias,

    /// PyTorch's LSTM/Linear initialization: uniform in ±1/sqrt(H) for the
    /// weights and biases; the initial states start at zero, as NAM's
    /// trainer parameters do.
    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, spec: Spec, seed: u64) !Model {
        const h = spec.hidden_size;
        if (h == 0 or spec.num_layers == 0 or spec.input_size == 0) return Error.UnsupportedChannels;
        var lstm = try rnn.Lstm.init(allocator, ctx, spec.input_size, h, spec.num_layers, seed);
        errdefer lstm.deinit();
        const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        const scratch = try allocator.alloc(f32, h);
        defer allocator.free(scratch);
        // The core's cells took the seed's first `num_layers` streams.
        rng.uniformFill(rng.at(seed, spec.num_layers), scratch, -bound, bound);
        var head_w = try HeadWeight.variableFromSlice(ctx, .{ h, 1 }, scratch);
        errdefer head_w.deinit();
        rng.uniformFill(rng.at(seed, spec.num_layers + 1), scratch[0..1], -bound, bound);
        var head_b = try HeadBias.variableFromSlice(ctx, .{1}, scratch[0..1]);
        errdefer head_b.deinit();
        return .{ .allocator = allocator, .spec = spec, .lstm = lstm, .head_w = head_w, .head_b = head_b };
    }

    /// From a NAM weight stream, as trainable variables or as constants:
    /// every layer's slices are borrowed views handed to the core, the
    /// head's `[out, H]` a permuted view materialized once.
    pub fn initFromNam(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.LstmConfig, weights: []const f32, requires_grad: bool, training: Training) !Model {
        const h = config.hidden_size;
        if (h == 0 or config.num_layers == 0 or config.input_size == 0) return Error.UnsupportedChannels;
        if (config.in_channels != 1 or config.out_channels != 1) return Error.UnsupportedChannels;
        const spec = Spec{ .hidden_size = h, .num_layers = config.num_layers, .input_size = config.input_size, .burn_in = training.burn_in, .truncate = training.truncate };

        var cursor: usize = 0;
        const cells = try buildCells(allocator, ctx, config, weights, &cursor, requires_grad);
        if (cursor + h + 1 != weights.len) {
            for (cells) |*cell| cell.deinit();
            allocator.free(cells);
            return Error.WeightCountMismatch;
        }
        // `fromCells` takes the cells over; until it has, they are ours.
        var lstm = rnn.Lstm.fromCells(allocator, cells) catch |err| {
            for (cells) |*cell| cell.deinit();
            allocator.free(cells);
            return err;
        };
        errdefer lstm.deinit();
        var stream_view = try Tensor(.{ .out, .unit }).fromBorrowedConstSlice(ctx, .{ 1, h }, weights[cursor..][0..h]);
        defer stream_view.deinit();
        var transposed = try stream_view.permuteTo(ctx, .{ .unit, .out });
        defer transposed.deinit();
        var head_w = try if (requires_grad) transposed.copyAsVariable(ctx) else transposed.copy(ctx);
        errdefer head_w.deinit();
        cursor += h;
        var bias_view = try HeadBias.fromBorrowedConstSlice(ctx, .{1}, weights[cursor..][0..1]);
        defer bias_view.deinit();
        var head_b = try if (requires_grad) bias_view.copyAsVariable(ctx) else bias_view.copy(ctx);
        errdefer head_b.deinit();
        return .{ .allocator = allocator, .spec = spec, .lstm = lstm, .head_w = head_w, .head_b = head_b };
    }

    /// The layers of the NAM stream from `cursor`, each a set of borrowed
    /// views handed to the core; owned by the caller until `fromCells`.
    fn buildCells(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.LstmConfig, weights: []const f32, cursor: *usize, requires_grad: bool) ![]rnn.LstmCell {
        const h = config.hidden_size;
        const cells = try allocator.alloc(rnn.LstmCell, config.num_layers);
        errdefer allocator.free(cells);
        var built: usize = 0;
        errdefer for (cells[0..built]) |*cell| cell.deinit();
        for (cells, 0..) |*cell, l| {
            const in_l = if (l == 0) config.input_size else h;
            const width = in_l + h;
            if (cursor.* + 4 * h * width + 4 * h + 2 * h > weights.len) return Error.WeightCountMismatch;
            var stacked = try rnn.StackedWeight.fromBorrowedConstSlice(ctx, .{ 4 * h, width }, weights[cursor.*..][0 .. 4 * h * width]);
            defer stacked.deinit();
            cursor.* += 4 * h * width;
            var bias = try Units.fromBorrowedConstSlice(ctx, .{4 * h}, weights[cursor.*..][0 .. 4 * h]);
            defer bias.deinit();
            cursor.* += 4 * h;
            var h0 = try Units.fromBorrowedConstSlice(ctx, .{h}, weights[cursor.*..][0..h]);
            defer h0.deinit();
            cursor.* += h;
            var c0 = try Units.fromBorrowedConstSlice(ctx, .{h}, weights[cursor.*..][0..h]);
            defer c0.deinit();
            cursor.* += h;
            cell.* = try rnn.LstmCell.fromStacked(ctx, &stacked, &bias, &h0, &c0, requires_grad);
            built += 1;
        }
        return cells;
    }

    pub fn deinit(self: *Model) void {
        self.lstm.deinit();
        self.head_w.deinit();
        self.head_b.deinit();
        self.* = undefined;
    }

    pub fn registerParams(self: *Model, opt: anytype) !void {
        try self.lstm.registerParams(opt);
        try opt.addParam(&self.head_w);
        try opt.addParam(&self.head_b);
    }

    pub fn requiresGrad(self: *const Model) bool {
        return self.lstm.cells[0].w.requiresGrad();
    }

    /// The head over one hidden vector: `h · W_head + b`, `[out]`.
    pub fn head(self: *const Model, ctx: *ExecContext, h: *const Units) !HeadBias {
        var out = try h.dot(ctx, &self.head_w, .unit);
        defer out.deinit();
        return out.add(ctx, &self.head_b);
    }

    /// The window forward: the core's recorded sequence (burn-in and
    /// truncation from `spec`) through the head. Runs inside the caller's
    /// exec scope, which owns every step's tensors; returns `[T, 1]`.
    pub fn forward(self: *const Model, ctx: *ExecContext, window: []const f32) !Output {
        var x = try rnn.Sequence.fromSlice(ctx, .{ window.len, 1 }, window);
        defer x.deinit();
        var hs = try self.lstm.forward(ctx, &x, self.spec.forwardOptions());
        defer hs.deinit();
        var pred = try hs.dot(ctx, &self.head_w, .unit);
        defer pred.deinit();
        var biased = try pred.add(ctx, &self.head_b);
        defer biased.deinit();
        return biased.withTags(ctx, .{ .time, .hout });
    }

    /// MSE over the last `target.len` predictions of `window` (the trainer's
    /// loss contract, torch `F.mse_loss` mean).
    pub fn segmentLoss(self: *const Model, ctx: *ExecContext, window: []const f32, target: []const f32) !Tensor(.{}) {
        var pred = try self.forward(ctx, window);
        defer pred.deinit();
        var target_tensor = try Output.fromSlice(ctx, .{ target.len, 1 }, target);
        defer target_tensor.deinit();
        var tail = try pred.narrow(ctx, .time, window.len - target.len, target.len);
        defer tail.deinit();
        var diff = try tail.sub(ctx, &target_tensor);
        defer diff.deinit();
        var sq = try diff.mul(ctx, &diff);
        defer sq.deinit();
        var total = try sq.sumAll(ctx);
        defer total.deinit();
        return total.scale(ctx, 1.0 / @as(f32, @floatFromInt(target.len)));
    }

    /// The NAM weight stream (the inverse of `initFromNam`): the core's
    /// stacked-layout views and the head's transpose, copied out
    /// stride-aware.
    pub fn extractWeights(self: *const Model, ctx: *ExecContext, allocator: std.mem.Allocator) ![]f32 {
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(allocator);
        for (self.lstm.cells) |*cell| {
            var stacked = try cell.stackedWeight(ctx);
            defer stacked.deinit();
            try appendView(allocator, &out, &stacked);
            try out.appendSlice(allocator, try cell.b.dataConst());
            try out.appendSlice(allocator, try cell.h0.dataConst());
            try out.appendSlice(allocator, try cell.c0.dataConst());
        }
        var head_order = try self.head_w.permuteTo(ctx, .{ .out, .unit });
        defer head_order.deinit();
        try appendView(allocator, &out, &head_order);
        try out.appendSlice(allocator, try self.head_b.dataConst());
        return out.toOwnedSlice(allocator);
    }

    fn appendView(allocator: std.mem.Allocator, out: *std.ArrayList(f32), view: anytype) !void {
        var count: usize = 1;
        for (view.shape()) |dim| count *= dim;
        const start = out.items.len;
        try out.resize(allocator, start + count);
        try view.copyTo(out.items[start..]);
    }
};

/// Streaming inference over a `Model`: the core's stream plus the head,
/// one block at a time inside a per-block exec scope, allocation-free once
/// the pool is warm.
pub const Stream = struct {
    model: *const Model,
    inner: rnn.Lstm.Stream,
    input_slot: wavenet.InputSlot = .{},

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, model: *const Model) !Stream {
        const inner = try rnn.Lstm.Stream.init(allocator, ctx, &model.lstm);
        return .{ .model = model, .inner = inner };
    }

    pub fn deinit(self: *Stream) void {
        self.inner.deinit();
        self.input_slot.deinit();
        self.* = undefined;
    }

    /// Back to the learned initial state.
    pub fn reset(self: *Stream, ctx: *ExecContext) !void {
        _ = ctx;
        try self.inner.reset();
    }

    pub fn process(self: *Stream, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        var offset: usize = 0;
        while (offset < frames) {
            const n = @min(frames - offset, max_block);
            try self.processBlock(ctx, input[offset..][0..n], output[offset..][0..n]);
            offset += n;
        }
    }

    /// Blocks longer than this are split: the per-block scope's transients
    /// (`[n, 2H]` and `[n, H]` rows) stay in the pool's working set.
    const max_block: usize = 4096;

    fn processBlock(self: *Stream, ctx: *ExecContext, input: []const f32, output: []f32) !void {
        var no_grad = fucina.noGrad();
        defer no_grad.close();
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        var x = try self.input_slot.view(ctx, input, input.len);
        const x_seq = try x.withTags(ctx, .{ .time, .k });
        const hs = try self.inner.step(ctx, &x_seq);
        const pred = try hs.dot(ctx, &self.model.head_w, .unit);
        const biased = try pred.add(ctx, &self.model.head_b);
        try biased.copyTo(output);
    }
};

test {
    _ = @import("lstm_tests.zig");
}
