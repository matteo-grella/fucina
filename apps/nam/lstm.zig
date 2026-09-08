//! The NAM LSTM model as fucina tensors: one `step` serves training and
//! streaming inference. Every transient is released by a `defer` right
//! after it is made; under an open exec scope (training, windowed
//! rendering) those releases are no-ops and the scope owns everything, so
//! the same code records the graph or runs allocation-free per sample.
//!
//! NAM semantics (nam/models/recurrent.py, NAM/lstm.cpp): PyTorch gate order
//! i, f, g, o over the concatenated `[x | h]` input; one bias vector (the
//! trainer's `b_ih + b_hh`); learned initial states `h0`, `c0`; a linear
//! head on the last layer's `h`; `c' = σ(f)·c + σ(i)·tanh(g)`,
//! `h' = σ(o)·tanh(c')`. Training follows the NAM trainer: a burn-in prefix
//! runs without gradient, then truncated backpropagation through time in
//! `truncate`-step segments; the loss covers the outputs after burn-in.
//!
//! Layout: the weight is stored `[k, unit]` (`[in + H + 1, 4H]`): the
//! transpose of the NAM stream's `[4H, in + H]` with the bias as one more
//! row against a constant-one input, so the gate pre-activation is one
//! vector-times-matrix product (the orientation the row kernels run
//! fastest) and the bias costs no separate pass. The NAM stream order is a
//! permuted view on import and export.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");

const Tensor = fucina.Tensor;
const ExecContext = fucina.ExecContext;
const rng = fucina.rng;

pub const Units = Tensor(.{.unit});
pub const Input = Tensor(.{.k});
pub const Weight = Tensor(.{ .k, .unit });
pub const HeadWeight = Tensor(.{ .unit, .out });
pub const HeadBias = Tensor(.{.out});
pub const Output = Tensor(.{ .time, .hout });

pub const Error = error{ WeightCountMismatch, ExecScopeRequired, ActiveExecScopeUnsupported, UnsupportedChannels };

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
};

/// The training schedule an imported model trains with (`Spec`'s burn-in
/// and truncation, separate from the file's architecture).
pub const Training = struct {
    burn_in: usize = 4096,
    truncate: usize = 512,
};

/// The state handed from one step to the next.
pub const State = struct {
    h: Units,
    c: Units,

    pub fn deinit(self: *State) void {
        self.h.deinit();
        self.c.deinit();
        self.* = undefined;
    }
};

pub const Cell = struct {
    /// `[in + H + 1, 4H]`: the NAM `[W_ih | W_hh]` matrix transposed, then
    /// the bias as the last row.
    w: Weight,
    h0: Units,
    c0: Units,
    input_size: usize,
    hidden: usize,

    fn deinit(self: *Cell) void {
        self.w.deinit();
        self.h0.deinit();
        self.c0.deinit();
        self.* = undefined;
    }

    /// One step: `[x | h | 1] · w`, the four gates, the cell and hidden
    /// updates. `x` is `[in]`, `h` and `c` are `[H]`, `one` the model's
    /// constant; the returned state is the caller's (scope-owned under an
    /// open scope).
    pub fn step(self: *const Cell, ctx: *ExecContext, x: *const Input, one: *const Input, h: *const Units, c: *const Units) !State {
        const hidden = self.hidden;
        var h_k = try h.withTags(ctx, .{.k});
        defer h_k.deinit();
        var xh = try x.concat(ctx, .k, &.{ &h_k, one });
        defer xh.deinit();
        var gates = try xh.dot(ctx, &self.w, .k);
        defer gates.deinit();
        var i_pre = try gates.narrow(ctx, .unit, 0, hidden);
        defer i_pre.deinit();
        var f_pre = try gates.narrow(ctx, .unit, hidden, hidden);
        defer f_pre.deinit();
        var g_pre = try gates.narrow(ctx, .unit, 2 * hidden, hidden);
        defer g_pre.deinit();
        var o_pre = try gates.narrow(ctx, .unit, 3 * hidden, hidden);
        defer o_pre.deinit();
        var g = try g_pre.tanh(ctx);
        defer g.deinit();
        // c' = σ(f)·c + σ(i)·tanh(g): two gated products (glu is x·σ(gate)).
        var forget = try c.glu(ctx, &f_pre);
        defer forget.deinit();
        var write = try g.glu(ctx, &i_pre);
        defer write.deinit();
        var c_new = try forget.add(ctx, &write);
        errdefer c_new.deinit();
        // h' = σ(o)·tanh(c').
        var tc = try c_new.tanh(ctx);
        defer tc.deinit();
        const h_new = try tc.glu(ctx, &o_pre);
        return .{ .h = h_new, .c = c_new };
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    spec: Spec,
    cells: []Cell,
    /// `[H, out]`: the NAM head `[out, H]` transposed.
    head_w: HeadWeight,
    head_b: HeadBias,
    /// The constant-one input the bias row multiplies.
    one: Input,

    /// PyTorch's LSTM/Linear initialization: uniform in ±1/sqrt(H) for the
    /// weights and biases; the initial states start at zero, as NAM's
    /// trainer parameters do.
    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, spec: Spec, seed: u64) !Model {
        const h = spec.hidden_size;
        if (h == 0 or spec.num_layers == 0 or spec.input_size == 0) return Error.UnsupportedChannels;
        const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        var scratch: std.ArrayList(f32) = .empty;
        defer scratch.deinit(allocator);
        var seed_counter: u64 = 0;

        const cells = try allocator.alloc(Cell, spec.num_layers);
        errdefer allocator.free(cells);
        var built: usize = 0;
        errdefer for (cells[0..built]) |*cell| cell.deinit();
        for (cells, 0..) |*cell, l| {
            const in_l = if (l == 0) spec.input_size else h;
            var w = try uniformVariable(Weight, ctx, allocator, &scratch, .{ in_l + h + 1, 4 * h }, bound, seed, &seed_counter);
            errdefer w.deinit();
            var h0 = try Units.variable(ctx, try ctx.zeros(.f32, &.{h}));
            errdefer h0.deinit();
            var c0 = try Units.variable(ctx, try ctx.zeros(.f32, &.{h}));
            errdefer c0.deinit();
            cell.* = .{ .w = w, .h0 = h0, .c0 = c0, .input_size = in_l, .hidden = h };
            built += 1;
        }
        var head_w = try uniformVariable(HeadWeight, ctx, allocator, &scratch, .{ h, 1 }, bound, seed, &seed_counter);
        errdefer head_w.deinit();
        var head_b = try uniformVariable(HeadBias, ctx, allocator, &scratch, .{1}, bound, seed, &seed_counter);
        errdefer head_b.deinit();
        var one = try Input.ones(ctx, .{1});
        errdefer one.deinit();
        return .{ .allocator = allocator, .spec = spec, .cells = cells, .head_w = head_w, .head_b = head_b, .one = one };
    }

    fn uniformVariable(comptime T: type, ctx: *ExecContext, allocator: std.mem.Allocator, scratch: *std.ArrayList(f32), shape: anytype, bound: f32, seed: u64, seed_counter: *u64) !T {
        var count: usize = 1;
        inline for (shape) |dim| count *= dim;
        try scratch.resize(allocator, count);
        rng.uniformFill(rng.at(seed, seed_counter.*), scratch.items, -bound, bound);
        seed_counter.* += 1;
        return T.variableFromSlice(ctx, shape, scratch.items);
    }

    /// From a NAM weight stream (spec §5.3: per layer `[4H, in + H]` row-major,
    /// bias `[4H]`, `h0`, `c0`; then the head `[out, H]` and its bias), as
    /// trainable variables or as constants. The layout change is a permuted
    /// view of the stream, copied out stride-aware.
    pub fn initFromNam(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.LstmConfig, weights: []const f32, requires_grad: bool, training: Training) !Model {
        const h = config.hidden_size;
        if (h == 0 or config.num_layers == 0 or config.input_size == 0) return Error.UnsupportedChannels;
        if (config.in_channels != 1 or config.out_channels != 1) return Error.UnsupportedChannels;
        const spec = Spec{ .hidden_size = h, .num_layers = config.num_layers, .input_size = config.input_size, .burn_in = training.burn_in, .truncate = training.truncate };
        var scratch: std.ArrayList(f32) = .empty;
        defer scratch.deinit(allocator);

        const cells = try allocator.alloc(Cell, config.num_layers);
        errdefer allocator.free(cells);
        var built: usize = 0;
        errdefer for (cells[0..built]) |*cell| cell.deinit();
        var cursor: usize = 0;
        for (cells, 0..) |*cell, l| {
            const in_l = if (l == 0) config.input_size else h;
            const width = in_l + h;
            if (cursor + 4 * h * width + 4 * h + 2 * h > weights.len) return Error.WeightCountMismatch;
            // `[W_ih | W_hh]` transposed, then the bias as the last row: the
            // stream's matrix as a permuted view concatenated with its bias
            // row, one materialization.
            var stream_view = try Tensor(.{ .unit, .k }).fromBorrowedConstSlice(ctx, .{ 4 * h, width }, weights[cursor..][0 .. 4 * h * width]);
            defer stream_view.deinit();
            var transposed = try stream_view.permuteTo(ctx, .{ .k, .unit });
            defer transposed.deinit();
            cursor += 4 * h * width;
            var bias_row = try Weight.fromBorrowedConstSlice(ctx, .{ 1, 4 * h }, weights[cursor..][0 .. 4 * h]);
            defer bias_row.deinit();
            cursor += 4 * h;
            var joined = try transposed.concat(ctx, .k, &.{&bias_row});
            defer joined.deinit();
            var w = try param(Weight, ctx, .{ width + 1, 4 * h }, try joined.dataConst(), requires_grad);
            errdefer w.deinit();
            var h0 = try param(Units, ctx, .{h}, weights[cursor..][0..h], requires_grad);
            errdefer h0.deinit();
            cursor += h;
            var c0 = try param(Units, ctx, .{h}, weights[cursor..][0..h], requires_grad);
            errdefer c0.deinit();
            cursor += h;
            cell.* = .{ .w = w, .h0 = h0, .c0 = c0, .input_size = in_l, .hidden = h };
            built += 1;
        }
        if (cursor + h + 1 != weights.len) return Error.WeightCountMismatch;
        try scratch.resize(allocator, h);
        {
            var stream_view = try Tensor(.{ .out, .unit }).fromBorrowedConstSlice(ctx, .{ 1, h }, weights[cursor..][0..h]);
            defer stream_view.deinit();
            var ours = try stream_view.permuteTo(ctx, .{ .unit, .out });
            defer ours.deinit();
            try ours.copyTo(scratch.items);
        }
        cursor += h;
        var head_w = try param(HeadWeight, ctx, .{ h, 1 }, scratch.items, requires_grad);
        errdefer head_w.deinit();
        var head_b = try param(HeadBias, ctx, .{1}, weights[cursor..][0..1], requires_grad);
        errdefer head_b.deinit();
        var one = try Input.ones(ctx, .{1});
        errdefer one.deinit();
        return .{ .allocator = allocator, .spec = spec, .cells = cells, .head_w = head_w, .head_b = head_b, .one = one };
    }

    fn param(comptime T: type, ctx: *ExecContext, shape: anytype, values: []const f32, requires_grad: bool) !T {
        return if (requires_grad) T.variableFromSlice(ctx, shape, values) else T.fromSlice(ctx, shape, values);
    }

    pub fn deinit(self: *Model) void {
        for (self.cells) |*cell| cell.deinit();
        self.allocator.free(self.cells);
        self.head_w.deinit();
        self.head_b.deinit();
        self.one.deinit();
        self.* = undefined;
    }

    pub fn registerParams(self: *Model, opt: anytype) !void {
        for (self.cells) |*cell| {
            try opt.addParam(&cell.w);
            try opt.addParam(&cell.h0);
            try opt.addParam(&cell.c0);
        }
        try opt.addParam(&self.head_w);
        try opt.addParam(&self.head_b);
    }

    pub fn requiresGrad(self: *const Model) bool {
        return self.cells[0].w.requiresGrad();
    }

    /// The head over one hidden vector: `h · W_head + b`, `[out]`.
    pub fn head(self: *const Model, ctx: *ExecContext, h: *const Units) !HeadBias {
        var out = try h.dot(ctx, &self.head_w, .unit);
        defer out.deinit();
        return out.add(ctx, &self.head_b);
    }

    /// The window forward: every step from the learned initial state, the
    /// first `spec.burn_in` steps without gradient, the rest recorded in
    /// `spec.truncate`-step segments (states detached between segments).
    /// Runs inside the caller's exec scope, which owns every step's
    /// tensors; returns `[T, 1]` predictions.
    pub fn forward(self: *const Model, ctx: *ExecContext, window: []const f32) !Output {
        if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
        const frames = window.len;
        var x_all = try Tensor(.{ .time, .k }).fromSlice(ctx, .{ frames, 1 }, window);
        defer x_all.deinit();

        const states = try ctx.allocator().alloc(State, self.cells.len);
        defer ctx.allocator().free(states);
        for (states, self.cells) |*state, *cell| state.* = .{ .h = try cell.h0.withTags(ctx, .{.unit}), .c = try cell.c0.withTags(ctx, .{.unit}) };
        const outputs = try ctx.allocator().alloc(Units, frames);
        defer ctx.allocator().free(outputs);
        const output_ptrs = try ctx.allocator().alloc(*const Units, frames);
        defer ctx.allocator().free(output_ptrs);

        const burn_in = @min(self.spec.burn_in, frames);
        {
            // Burn-in: the same steps, nothing recorded.
            var no_grad = fucina.noGrad();
            defer no_grad.close();
            for (0..burn_in) |t| outputs[t] = try self.stepAll(ctx, &x_all, t, states);
        }
        for (burn_in..frames) |t| {
            const into_segment = t - burn_in;
            if (self.spec.truncate != 0 and into_segment != 0 and into_segment % self.spec.truncate == 0) {
                for (states) |*state| {
                    state.h = try state.h.detach(ctx);
                    state.c = try state.c.detach(ctx);
                }
            }
            outputs[t] = try self.stepAll(ctx, &x_all, t, states);
        }
        for (outputs, output_ptrs) |*o, *p| p.* = o;
        var hs = try outputs[0].stack(ctx, .time, 0, output_ptrs[1..]);
        defer hs.deinit();
        var pred = try hs.dot(ctx, &self.head_w, .unit);
        defer pred.deinit();
        var biased = try pred.add(ctx, &self.head_b);
        defer biased.deinit();
        return biased.withTags(ctx, .{ .time, .hout });
    }

    /// Step `t` of the window through every layer, advancing `states`;
    /// returns the last layer's `h`.
    fn stepAll(self: *const Model, ctx: *ExecContext, x_all: *const Tensor(.{ .time, .k }), t: usize, states: []State) !Units {
        var x_t = try x_all.select(ctx, .time, @intCast(t));
        defer x_t.deinit();
        var input: *const Input = &x_t;
        var carried: ?Input = null;
        defer if (carried) |*v| v.deinit();
        for (self.cells, states) |*cell, *state| {
            state.* = try cell.step(ctx, input, &self.one, &state.h, &state.c);
            if (carried) |*v| v.deinit();
            carried = try state.h.withTags(ctx, .{.k});
            input = &carried.?;
        }
        return states[states.len - 1].h.withTags(ctx, .{.unit});
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

    /// The NAM weight stream (the inverse of `initFromNam`): the stored
    /// `[k, unit]` weight is the NAM `[4H, in + H]` matrix's transpose, a
    /// permuted view copied out stride-aware.
    pub fn extractWeights(self: *const Model, ctx: *ExecContext, allocator: std.mem.Allocator) ![]f32 {
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(allocator);
        for (self.cells) |*cell| {
            const width = cell.input_size + cell.hidden;
            var matrix = try cell.w.narrow(ctx, .k, 0, width);
            defer matrix.deinit();
            var nam_order = try matrix.permuteTo(ctx, .{ .unit, .k });
            defer nam_order.deinit();
            try appendView(allocator, &out, &nam_order);
            var bias_row = try cell.w.narrow(ctx, .k, width, 1);
            defer bias_row.deinit();
            try appendView(allocator, &out, &bias_row);
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

/// Streaming inference over a `Model`: the live state is the previous
/// step's outputs, replaced by ownership move each sample (no scope, so
/// every `step` transient is released at once and the outputs survive).
pub const Stream = struct {
    model: *const Model,
    allocator: std.mem.Allocator,
    states: []State,
    x_t: Input,

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, model: *const Model) !Stream {
        if (ctx.execScopeActive()) return Error.ActiveExecScopeUnsupported;
        const states = try allocator.alloc(State, model.cells.len);
        errdefer allocator.free(states);
        var built: usize = 0;
        errdefer for (states[0..built]) |*state| state.deinit();
        for (states, model.cells) |*state, *cell| {
            var h = try cell.h0.materialize(ctx);
            errdefer h.deinit();
            const c = try cell.c0.materialize(ctx);
            state.* = .{ .h = h, .c = c };
            built += 1;
        }
        var x_t = try Input.zeros(ctx, .{1});
        errdefer x_t.deinit();
        return .{ .model = model, .allocator = allocator, .states = states, .x_t = x_t };
    }

    pub fn deinit(self: *Stream) void {
        for (self.states) |*state| state.deinit();
        self.allocator.free(self.states);
        self.x_t.deinit();
        self.* = undefined;
    }

    /// Back to the learned initial state.
    pub fn reset(self: *Stream, ctx: *ExecContext) !void {
        for (self.states, self.model.cells) |*state, *cell| {
            var h = try cell.h0.materialize(ctx);
            errdefer h.deinit();
            const c = try cell.c0.materialize(ctx);
            state.deinit();
            state.* = .{ .h = h, .c = c };
        }
    }

    pub fn processSample(self: *Stream, ctx: *ExecContext, x: f32) !f32 {
        if (ctx.execScopeActive()) return Error.ActiveExecScopeUnsupported;
        try self.x_t.copyFrom(&[_]f32{x});
        var input: *const Input = &self.x_t;
        var carried: ?Input = null;
        defer if (carried) |*v| v.deinit();
        for (self.model.cells, self.states) |*cell, *state| {
            var next = try cell.step(ctx, input, &self.model.one, &state.h, &state.c);
            errdefer next.deinit();
            state.deinit();
            state.* = next;
            if (carried) |*v| v.deinit();
            carried = try state.h.withTags(ctx, .{.k});
            input = &carried.?;
        }
        var y = try self.model.head(ctx, &self.states[self.states.len - 1].h);
        defer y.deinit();
        return y.item();
    }

    pub fn process(self: *Stream, ctx: *ExecContext, input: []const f32, output: []f32, frames: usize) !void {
        for (0..frames) |t| output[t] = try self.processSample(ctx, input[t]);
    }
};

test {
    _ = @import("lstm_tests.zig");
}
