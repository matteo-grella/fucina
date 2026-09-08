//! Recurrent layers over the autograd facade: the LSTM as one sequence op
//! (`Tensor.lstm`) that serves training and streaming inference alike.
//!
//! Semantics are PyTorch's `nn.LSTM`: gates i, f, g, o over the
//! concatenated `[x | h]` input, `c' = σ(f)·c + σ(i)·tanh(g)`,
//! `h' = σ(o)·tanh(c')`, with the initial state `h0`, `c0` a learnable
//! parameter per layer. A cell stores `w` `[in + H, 4H]` (PyTorch's stacked
//! `[W_ih | W_hh]` transposed) and `b` `[4H]` (`b_ih + b_hh`); the stacked
//! gate-major form is a permuted view on import and export
//! (`LstmCell.fromStacked`, `stackedWeight`).
//!
//! One op for the block: the recurrence over a sequence runs inside the
//! kernel (serial by nature, weights streamed once per step, the lane
//! nonlinearities) and returns `[2T, H]`, the hidden rows then the cell
//! rows, so a block costs one dispatch per layer whatever the hidden size
//! and the hidden sequence is a contiguous view; the backward is one BPTT
//! pass over the saved gates.
//! Under an open exec scope `Lstm.forward` records it; `Lstm.Stream` feeds
//! blocks without gradients and carries the last row as the next block's
//! initial state.
//!
//! Sequence training follows the usual recurrent recipe: `burn_in` steps
//! without gradient from the learned initial state, then truncated
//! backpropagation through time in `truncate`-step segments (the state
//! detached between segments; values never change, only the gradient
//! horizon).
//!
//! Tags are fixed: `.k` is the input axis (the concatenated `[x | h]` axis
//! of the weight), `.unit` the hidden/gate axis, `.time` the sequence axis.
//! Callers retag views to and from their own names.

const std = @import("std");
const ag = @import("ag.zig");
const exec_mod = @import("exec.zig");
const rng = @import("rng.zig");

const Tensor = ag.Tensor;
const ExecContext = exec_mod.ExecContext;

pub const Units = Tensor(.{.unit});
pub const Weight = Tensor(.{ .k, .unit });
/// PyTorch's stacked gate-major matrix `[4H, in + H]`.
pub const StackedWeight = Tensor(.{ .unit, .k });
pub const Sequence = Tensor(.{ .time, .k });
pub const HiddenSequence = Tensor(.{ .time, .unit });

pub const Error = error{ ExecScopeRequired, ActiveExecScopeUnsupported, InvalidShape };

/// The state handed from one block or segment to the next.
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
    /// `[in + H, 4H]`: the stacked `[W_ih | W_hh]` transposed.
    w: Weight,
    /// `[4H]`: `b_ih + b_hh`.
    b: Units,
    h0: Units,
    c0: Units,
    input_size: usize,
    hidden: usize,

    /// PyTorch's initialization: uniform in ±1/sqrt(H) for the weight and
    /// the bias, zeros for the initial state; all trainable.
    pub fn init(ctx: *ExecContext, input_size: usize, hidden: usize, seed: u64, seed_counter: *u64) !LstmCell {
        if (input_size == 0 or hidden == 0) return Error.InvalidShape;
        const allocator = ctx.allocator();
        const width = input_size + hidden;
        const scratch = try allocator.alloc(f32, width * 4 * hidden);
        defer allocator.free(scratch);
        const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
        rng.uniformFill(rng.at(seed, seed_counter.*), scratch, -bound, bound);
        seed_counter.* += 1;
        var w = try Weight.variableFromSlice(ctx, .{ width, 4 * hidden }, scratch);
        errdefer w.deinit();
        rng.uniformFill(rng.at(seed, seed_counter.*), scratch[0 .. 4 * hidden], -bound, bound);
        seed_counter.* += 1;
        var b = try Units.variableFromSlice(ctx, .{4 * hidden}, scratch[0 .. 4 * hidden]);
        errdefer b.deinit();
        var h0 = try Units.variable(ctx, try ctx.zeros(.f32, &.{hidden}));
        errdefer h0.deinit();
        var c0 = try Units.variable(ctx, try ctx.zeros(.f32, &.{hidden}));
        errdefer c0.deinit();
        return .{ .w = w, .b = b, .h0 = h0, .c0 = c0, .input_size = input_size, .hidden = hidden };
    }

    /// From PyTorch's layout: the stacked gate-major matrix `[4H, in + H]`
    /// (rows i, f, g, o; `[W_ih | W_hh]`), the bias `[4H]` (`b_ih + b_hh`),
    /// and the initial state; any views. The matrix is a permuted view
    /// copied out once; the cell owns its four tensors whatever the scope,
    /// as variables when `trainable`.
    pub fn fromStacked(ctx: *ExecContext, stacked: *const StackedWeight, bias: *const Units, h0: *const Units, c0: *const Units, trainable: bool) !LstmCell {
        const shape = stacked.shape();
        const hidden = shape[0] / 4;
        if (hidden == 0 or shape[0] != 4 * hidden or shape[1] <= hidden) return Error.InvalidShape;
        if (bias.shape()[0] != 4 * hidden or h0.shape()[0] != hidden or c0.shape()[0] != hidden) return Error.InvalidShape;
        var transposed = try stacked.permuteTo(ctx, .{ .k, .unit });
        defer transposed.deinit();
        var w = try own(Weight, ctx, &transposed, trainable);
        errdefer w.deinit();
        var b = try own(Units, ctx, bias, trainable);
        errdefer b.deinit();
        var h0_own = try own(Units, ctx, h0, trainable);
        errdefer h0_own.deinit();
        const c0_own = try own(Units, ctx, c0, trainable);
        return .{ .w = w, .b = b, .h0 = h0_own, .c0 = c0_own, .input_size = shape[1] - hidden, .hidden = hidden };
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
        self.b.deinit();
        self.h0.deinit();
        self.c0.deinit();
        self.* = undefined;
    }

    pub fn registerParams(self: *LstmCell, opt: anytype) !void {
        try opt.addParam(&self.w);
        try opt.addParam(&self.b);
        try opt.addParam(&self.h0);
        try opt.addParam(&self.c0);
    }

    /// The stacked gate-major matrix `[4H, in + H]` as a view of the stored
    /// weight (copy it out with `copyTo`).
    pub fn stackedWeight(self: *const LstmCell, ctx: *ExecContext) !StackedWeight {
        return self.w.permuteTo(ctx, .{ .unit, .k });
    }

    /// The sequence `x` `[T, in]` from the state `h`, `c`: the op's
    /// `[2T, H]` (`h` rows then `c` rows), recorded when anything requires
    /// grad.
    pub fn forward(self: *const LstmCell, ctx: *ExecContext, x: *const Sequence, h: *const Units, c: *const Units) !HiddenSequence {
        return x.lstm(ctx, .time, .k, .unit, &self.w, &self.b, h, c);
    }
};

/// A stack of LSTM layers, each feeding the next.
pub const Lstm = struct {
    allocator: std.mem.Allocator,
    cells: []LstmCell,

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
        return fromCells(allocator, cells);
    }

    /// Takes ownership of `cells` (allocated with `allocator`).
    pub fn fromCells(allocator: std.mem.Allocator, cells: []LstmCell) !Lstm {
        if (cells.len == 0) return Error.InvalidShape;
        for (cells[1..], cells[0 .. cells.len - 1]) |*cell, *previous| {
            if (cell.input_size != previous.hidden) return Error.InvalidShape;
        }
        return .{ .allocator = allocator, .cells = cells };
    }

    pub fn deinit(self: *Lstm) void {
        for (self.cells) |*cell| cell.deinit();
        self.allocator.free(self.cells);
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
    /// Runs inside the caller's exec scope, which owns every segment's
    /// tensors.
    pub fn forward(self: *const Lstm, ctx: *ExecContext, x: *const Sequence, options: ForwardOptions) !HiddenSequence {
        if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
        const frames = x.shape()[0];
        if (frames == 0 or x.shape()[1] != self.inputSize()) return Error.InvalidShape;
        const allocator = ctx.allocator();
        const states = try allocator.alloc(State, self.cells.len);
        defer allocator.free(states);
        for (states, self.cells) |*state, *cell| state.* = .{ .h = try cell.h0.withTags(ctx, .{.unit}), .c = try cell.c0.withTags(ctx, .{.unit}) };

        const burn_in = @min(options.burn_in, frames);
        const recorded = frames - burn_in;
        const segment = if (options.truncate == 0 or options.truncate > recorded) @max(recorded, 1) else options.truncate;
        const segments = if (recorded == 0) 0 else (recorded + segment - 1) / segment;
        const parts_len = segments + @as(usize, if (burn_in > 0) 1 else 0);
        const parts = try allocator.alloc(HiddenSequence, parts_len);
        defer allocator.free(parts);
        const part_ptrs = try allocator.alloc(*const HiddenSequence, parts_len);
        defer allocator.free(part_ptrs);
        var index: usize = 0;

        if (burn_in > 0) {
            // The burn-in rows are values: no gradient reaches the initial
            // state through them.
            var no_grad = ag.noGrad();
            defer no_grad.close();
            var prefix = try x.narrow(ctx, .time, 0, burn_in);
            defer prefix.deinit();
            parts[index] = try self.stackSegment(ctx, &prefix, states);
            index += 1;
        }
        for (0..segments) |s| {
            const start = burn_in + s * segment;
            const len = @min(segment, frames - start);
            var chunk = try x.narrow(ctx, .time, start, len);
            defer chunk.deinit();
            if (s > 0) {
                for (states) |*state| {
                    state.h = try state.h.detach(ctx);
                    state.c = try state.c.detach(ctx);
                }
            }
            parts[index] = try self.stackSegment(ctx, &chunk, states);
            index += 1;
        }
        for (parts, part_ptrs) |*o, *p| p.* = o;
        if (parts_len == 1) return parts[0];
        return parts[0].concat(ctx, .time, part_ptrs[1..]);
    }

    /// One segment through every layer, advancing `states` to the segment's
    /// last row; returns the last layer's `[len, H]` hidden rows.
    fn stackSegment(self: *const Lstm, ctx: *ExecContext, x: *const Sequence, states: []State) !HiddenSequence {
        var input: *const Sequence = x;
        var carried: ?Sequence = null;
        defer if (carried) |*v| v.deinit();
        var hidden: ?HiddenSequence = null;
        defer if (hidden) |*v| v.deinit();
        for (self.cells, states) |*cell, *state| {
            var out = try cell.forward(ctx, input, &state.h, &state.c);
            defer out.deinit();
            const frames = out.shape()[0] / 2;
            state.h = try out.select(ctx, .time, @intCast(frames - 1));
            state.c = try out.select(ctx, .time, @intCast(2 * frames - 1));
            const hs = try out.narrow(ctx, .time, 0, frames);
            if (hidden) |*v| v.deinit();
            hidden = hs;
            if (carried) |*v| v.deinit();
            carried = try hs.withTags(ctx, .{ .time, .k });
            input = &carried.?;
        }
        const result = hidden.?;
        hidden = null;
        return result;
    }

    /// Streaming inference: blocks through the stack without gradients,
    /// the last row of every block carried as the next block's initial
    /// state in persistent tensors (`copyFrom`), so a block leaves nothing
    /// behind whatever scope the caller runs it in.
    pub const Stream = struct {
        lstm: *const Lstm,
        allocator: std.mem.Allocator,
        states: []State,

        pub fn init(allocator: std.mem.Allocator, ctx: *ExecContext, lstm: *const Lstm) !Stream {
            const states = try allocator.alloc(State, lstm.cells.len);
            errdefer allocator.free(states);
            var built: usize = 0;
            errdefer for (states[0..built]) |*state| state.deinit();
            for (states, lstm.cells) |*state, *cell| {
                var h = try Units.zeros(ctx, .{cell.hidden});
                errdefer h.deinit();
                const c = try Units.zeros(ctx, .{cell.hidden});
                state.* = .{ .h = h, .c = c };
                built += 1;
            }
            var stream = Stream{ .lstm = lstm, .allocator = allocator, .states = states };
            try stream.reset();
            return stream;
        }

        pub fn deinit(self: *Stream) void {
            for (self.states) |*state| state.deinit();
            self.allocator.free(self.states);
            self.* = undefined;
        }

        /// Back to the learned initial state.
        pub fn reset(self: *Stream) !void {
            for (self.states, self.lstm.cells) |*state, *cell| {
                try state.h.copyFrom(try cell.h0.dataConst());
                try state.c.copyFrom(try cell.c0.dataConst());
            }
        }

        /// One block `x` `[T, in]` through the stack; the last layer's
        /// `[T, H]` hidden rows, owned as any op result of the caller's
        /// scope is.
        pub fn step(self: *Stream, ctx: *ExecContext, x: *const Sequence) !HiddenSequence {
            var no_grad = ag.noGrad();
            defer no_grad.close();
            var input: *const Sequence = x;
            var carried: ?Sequence = null;
            defer if (carried) |*v| v.deinit();
            var hidden: ?HiddenSequence = null;
            defer if (hidden) |*v| v.deinit();
            for (self.lstm.cells, self.states) |*cell, *state| {
                var out = try cell.forward(ctx, input, &state.h, &state.c);
                defer out.deinit();
                const frames = out.shape()[0] / 2;
                const rows = try out.dataConst();
                try state.h.copyFrom(rows[(frames - 1) * cell.hidden ..][0..cell.hidden]);
                try state.c.copyFrom(rows[(2 * frames - 1) * cell.hidden ..][0..cell.hidden]);
                const hs = try out.narrow(ctx, .time, 0, frames);
                if (hidden) |*v| v.deinit();
                hidden = hs;
                if (carried) |*v| v.deinit();
                carried = try hs.withTags(ctx, .{ .time, .k });
                input = &carried.?;
            }
            const result = hidden.?;
            hidden = null;
            return result;
        }
    };
};

test {
    _ = @import("rnn_tests.zig");
}
