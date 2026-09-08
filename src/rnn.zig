//! Recurrent layers over the autograd facade: the LSTM as one `step` that
//! serves training and streaming inference alike.
//!
//! Every transient a step makes is released by a `defer` right after it is
//! made; under an open exec scope (training, windowed rendering) those
//! releases are no-ops and the scope owns everything, outside a scope
//! (streaming) the returned state is the caller's and the step runs
//! allocation-free once the runtime's pool is warm. The same code records
//! the graph or runs per sample.
//!
//! Semantics are PyTorch's `nn.LSTM`: gates i, f, g, o over the
//! concatenated `[x | h]` input, `c' = σ(f)·c + σ(i)·tanh(g)`,
//! `h' = σ(o)·tanh(c')`, with the initial state `h0`, `c0` a learnable
//! parameter per layer. Layout: a cell stores one `[in + H + 1, 4H]`
//! matrix, PyTorch's stacked `[W_ih | W_hh]` transposed with the bias
//! (`b_ih + b_hh`) as the last row against a constant-one input, so the
//! gate pre-activation is one vector-times-matrix `dot` (the orientation
//! the row kernels run fastest) with no separate bias pass. The stacked
//! gate-major form is a permuted view on import and export
//! (`LstmCell.fromStacked`, `stackedWeight`, `bias`).
//!
//! Sequence training follows the usual recurrent recipe: `burn_in` steps
//! without gradient from the learned initial state, then truncated
//! backpropagation through time in `truncate`-step segments (the state
//! detached between segments; values never change, only the gradient
//! horizon).
//!
//! Tags are fixed: `.k` is a step's input axis (the contraction axis of the
//! gate product), `.unit` the hidden/gate axis, `.time` the sequence axis.
//! Callers retag views to and from their own names.

const std = @import("std");
const ag = @import("ag.zig");
const exec_mod = @import("exec.zig");
const rng = @import("rng.zig");

const Tensor = ag.Tensor;
const ExecContext = exec_mod.ExecContext;

pub const Units = Tensor(.{.unit});
pub const Input = Tensor(.{.k});
pub const Weight = Tensor(.{ .k, .unit });
/// PyTorch's stacked gate-major matrix `[4H, in + H]`.
pub const StackedWeight = Tensor(.{ .unit, .k });
pub const Sequence = Tensor(.{ .time, .k });
pub const HiddenSequence = Tensor(.{ .time, .unit });

pub const Error = error{ ExecScopeRequired, ActiveExecScopeUnsupported, InvalidShape };

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

pub const LstmCell = struct {
    /// `[in + H + 1, 4H]`: the stacked `[W_ih | W_hh]` transposed, then the
    /// bias as the last row.
    w: Weight,
    h0: Units,
    c0: Units,
    input_size: usize,
    hidden: usize,

    /// PyTorch's initialization: uniform in ±1/sqrt(H) for the weights and
    /// the bias, zeros for the initial state; all trainable.
    pub fn init(ctx: *ExecContext, input_size: usize, hidden: usize, seed: u64, seed_counter: *u64) !LstmCell {
        if (input_size == 0 or hidden == 0) return Error.InvalidShape;
        const allocator = ctx.allocator();
        const width = input_size + hidden;
        const scratch = try allocator.alloc(f32, (width + 1) * 4 * hidden);
        defer allocator.free(scratch);
        const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
        rng.uniformFill(rng.at(seed, seed_counter.*), scratch, -bound, bound);
        seed_counter.* += 1;
        var w = try Weight.variableFromSlice(ctx, .{ width + 1, 4 * hidden }, scratch);
        errdefer w.deinit();
        var h0 = try Units.variable(ctx, try ctx.zeros(.f32, &.{hidden}));
        errdefer h0.deinit();
        var c0 = try Units.variable(ctx, try ctx.zeros(.f32, &.{hidden}));
        errdefer c0.deinit();
        return .{ .w = w, .h0 = h0, .c0 = c0, .input_size = input_size, .hidden = hidden };
    }

    /// From PyTorch's layout: the stacked gate-major matrix `[4H, in + H]`
    /// (rows i, f, g, o; `[W_ih | W_hh]`), the bias `[4H]` (`b_ih + b_hh`),
    /// and the initial state; any views. The layout change is a permuted
    /// view concatenated with the bias row, copied out once; the cell owns
    /// its four tensors whatever the scope, as variables when `trainable`.
    pub fn fromStacked(ctx: *ExecContext, stacked: *const StackedWeight, bias_in: *const Units, h0: *const Units, c0: *const Units, trainable: bool) !LstmCell {
        const shape = stacked.shape();
        const hidden = shape[0] / 4;
        if (hidden == 0 or shape[0] != 4 * hidden or shape[1] <= hidden) return Error.InvalidShape;
        if (bias_in.shape()[0] != 4 * hidden or h0.shape()[0] != hidden or c0.shape()[0] != hidden) return Error.InvalidShape;
        var transposed = try stacked.permuteTo(ctx, .{ .k, .unit });
        defer transposed.deinit();
        var bias_row = try bias_in.insertAxis(ctx, .k, 0);
        defer bias_row.deinit();
        var joined = try transposed.concat(ctx, .k, &.{&bias_row});
        defer joined.deinit();
        var w = try own(Weight, ctx, &joined, trainable);
        errdefer w.deinit();
        var h0_own = try own(Units, ctx, h0, trainable);
        errdefer h0_own.deinit();
        const c0_own = try own(Units, ctx, c0, trainable);
        return .{ .w = w, .h0 = h0_own, .c0 = c0_own, .input_size = shape[1] - hidden, .hidden = hidden };
    }

    /// A caller-owned copy of `src` (stride-aware), a variable when
    /// `trainable`: explicit constructors are the caller's under any scope.
    fn own(comptime T: type, ctx: *ExecContext, src: *const T, trainable: bool) !T {
        const shape = src.shape();
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        const allocator = ctx.allocator();
        const scratch = try allocator.alloc(f32, count);
        defer allocator.free(scratch);
        try src.copyTo(scratch);
        return if (trainable) T.variableFromSlice(ctx, shape, scratch) else T.fromSlice(ctx, shape, scratch);
    }

    pub fn deinit(self: *LstmCell) void {
        self.w.deinit();
        self.h0.deinit();
        self.c0.deinit();
        self.* = undefined;
    }

    pub fn registerParams(self: *LstmCell, opt: anytype) !void {
        try opt.addParam(&self.w);
        try opt.addParam(&self.h0);
        try opt.addParam(&self.c0);
    }

    /// The stacked gate-major matrix `[4H, in + H]` as a view of the stored
    /// weight (copy it out with `copyTo`).
    pub fn stackedWeight(self: *const LstmCell, ctx: *ExecContext) !StackedWeight {
        var matrix = try self.w.narrow(ctx, .k, 0, self.input_size + self.hidden);
        defer matrix.deinit();
        return matrix.permuteTo(ctx, .{ .unit, .k });
    }

    /// The bias `[4H]` as a view of the stored weight's last row.
    pub fn bias(self: *const LstmCell, ctx: *ExecContext) !Units {
        var row = try self.w.narrow(ctx, .k, self.input_size + self.hidden, 1);
        defer row.deinit();
        return row.select(ctx, .k, 0);
    }

    /// One step: `[x | h | 1] · w`, the four gates, the cell and hidden
    /// updates. `x` is `[in]`, `h` and `c` are `[H]`, `one` the stack's
    /// constant; the returned state is the caller's (scope-owned under an
    /// open scope).
    pub fn step(self: *const LstmCell, ctx: *ExecContext, x: *const Input, one: *const Input, h: *const Units, c: *const Units) !State {
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

/// A stack of LSTM layers, each feeding the next.
pub const Lstm = struct {
    allocator: std.mem.Allocator,
    cells: []LstmCell,
    /// The constant-one input the bias row multiplies.
    one: Input,

    pub const ForwardOptions = struct {
        /// Steps run without gradient before the recorded segment.
        burn_in: usize = 0,
        /// Gradient horizon inside the recorded segment; 0 = the whole segment.
        truncate: usize = 0,
    };

    pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, input_size: usize, hidden: usize, num_layers: usize, seed: u64) !Lstm {
        if (num_layers == 0) return Error.InvalidShape;
        const cells = try allocator.alloc(LstmCell, num_layers);
        errdefer allocator.free(cells);
        var built: usize = 0;
        errdefer for (cells[0..built]) |*cell| cell.deinit();
        var seed_counter: u64 = 0;
        for (cells, 0..) |*cell, l| {
            cell.* = try LstmCell.init(ctx, if (l == 0) input_size else hidden, hidden, seed, &seed_counter);
            built += 1;
        }
        return fromCells(allocator, ctx, cells);
    }

    /// Takes ownership of `cells` (allocated with `allocator`).
    pub fn fromCells(allocator: std.mem.Allocator, ctx: *ExecContext, cells: []LstmCell) !Lstm {
        if (cells.len == 0) return Error.InvalidShape;
        for (cells[1..], cells[0 .. cells.len - 1]) |*cell, *previous| {
            if (cell.input_size != previous.hidden) return Error.InvalidShape;
        }
        var one = try Input.ones(ctx, .{1});
        errdefer one.deinit();
        return .{ .allocator = allocator, .cells = cells, .one = one };
    }

    pub fn deinit(self: *Lstm) void {
        for (self.cells) |*cell| cell.deinit();
        self.allocator.free(self.cells);
        self.one.deinit();
        self.* = undefined;
    }

    pub fn registerParams(self: *Lstm, opt: anytype) !void {
        for (self.cells) |*cell| try cell.registerParams(opt);
    }

    pub fn inputSize(self: *const Lstm) usize {
        return self.cells[0].input_size;
    }

    pub fn hiddenSize(self: *const Lstm) usize {
        return self.cells[self.cells.len - 1].hidden;
    }

    /// The last layer's hidden sequence `[T, H]` for `x` `[T, in]`, every
    /// step from the learned initial state: the first `burn_in` steps
    /// without gradient, the rest recorded in `truncate`-step segments.
    /// Runs inside the caller's exec scope, which owns every step's
    /// tensors.
    pub fn forward(self: *const Lstm, ctx: *ExecContext, x: *const Sequence, options: ForwardOptions) !HiddenSequence {
        if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
        const frames = x.shape()[0];
        if (frames == 0 or x.shape()[1] != self.inputSize()) return Error.InvalidShape;
        const allocator = ctx.allocator();
        const states = try allocator.alloc(State, self.cells.len);
        defer allocator.free(states);
        for (states, self.cells) |*state, *cell| state.* = .{ .h = try cell.h0.withTags(ctx, .{.unit}), .c = try cell.c0.withTags(ctx, .{.unit}) };
        const outputs = try allocator.alloc(Units, frames);
        defer allocator.free(outputs);
        const output_ptrs = try allocator.alloc(*const Units, frames);
        defer allocator.free(output_ptrs);

        const burn_in = @min(options.burn_in, frames);
        {
            var no_grad = ag.noGrad();
            defer no_grad.close();
            for (0..burn_in) |t| outputs[t] = try self.stepAll(ctx, x, t, states);
        }
        for (burn_in..frames) |t| {
            const into_segment = t - burn_in;
            if (options.truncate != 0 and into_segment != 0 and into_segment % options.truncate == 0) {
                for (states) |*state| {
                    state.h = try state.h.detach(ctx);
                    state.c = try state.c.detach(ctx);
                }
            }
            outputs[t] = try self.stepAll(ctx, x, t, states);
        }
        for (outputs, output_ptrs) |*o, *p| p.* = o;
        return outputs[0].stack(ctx, .time, 0, output_ptrs[1..]);
    }

    /// Step `t` of the sequence through every layer, advancing `states`;
    /// returns the last layer's `h`.
    fn stepAll(self: *const Lstm, ctx: *ExecContext, x: *const Sequence, t: usize, states: []State) !Units {
        var x_t = try x.select(ctx, .time, @intCast(t));
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

    /// Streaming inference: the live state is the previous step's outputs,
    /// replaced by ownership move each step (no scope, so every `step`
    /// transient is released at once and the outputs survive).
    pub const Stream = struct {
        lstm: *const Lstm,
        allocator: std.mem.Allocator,
        states: []State,

        pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, lstm: *const Lstm) !Stream {
            if (ctx.execScopeActive()) return Error.ActiveExecScopeUnsupported;
            const states = try allocator.alloc(State, lstm.cells.len);
            errdefer allocator.free(states);
            var built: usize = 0;
            errdefer for (states[0..built]) |*state| state.deinit();
            for (states, lstm.cells) |*state, *cell| {
                var h = try cell.h0.materialize(ctx);
                errdefer h.deinit();
                const c = try cell.c0.materialize(ctx);
                state.* = .{ .h = h, .c = c };
                built += 1;
            }
            return .{ .lstm = lstm, .allocator = allocator, .states = states };
        }

        pub fn deinit(self: *Stream) void {
            for (self.states) |*state| state.deinit();
            self.allocator.free(self.states);
            self.* = undefined;
        }

        /// Back to the learned initial state.
        pub fn reset(self: *Stream, ctx: *ExecContext) !void {
            for (self.states, self.lstm.cells) |*state, *cell| {
                var h = try cell.h0.materialize(ctx);
                errdefer h.deinit();
                const c = try cell.c0.materialize(ctx);
                state.deinit();
                state.* = .{ .h = h, .c = c };
            }
        }

        /// One step of `x` `[in]` through the stack; returns the last
        /// layer's new `h`, borrowed until the next step or `deinit`.
        pub fn step(self: *Stream, ctx: *ExecContext, x: *const Input) !*const Units {
            if (ctx.execScopeActive()) return Error.ActiveExecScopeUnsupported;
            var input: *const Input = x;
            var carried: ?Input = null;
            defer if (carried) |*v| v.deinit();
            for (self.lstm.cells, self.states) |*cell, *state| {
                var next = try cell.step(ctx, input, &self.lstm.one, &state.h, &state.c);
                errdefer next.deinit();
                state.deinit();
                state.* = next;
                if (carried) |*v| v.deinit();
                carried = try state.h.withTags(ctx, .{.k});
                input = &carried.?;
            }
            return &self.states[self.states.len - 1].h;
        }
    };
};

test {
    _ = @import("rnn_tests.zig");
}
