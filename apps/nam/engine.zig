//! Unified NAM DSP engine: load a parsed `.nam` model, Reset, process mono
//! blocks. Mirrors the upstream DSP contract (NAM/dsp.h): Reset(sampleRate,
//! maxBufferSize) prewarms by default with zero samples **rounded up to
//! whole buffers** (ceil(prewarm/maxBuf)*maxBuf, dsp.cpp:47-81) — golden
//! parity vs upstream tools/render depends on reproducing exactly that.
//! Deviation from upstream (which asserts only in debug builds): frames >
//! max_frames is a checked error, not UB.
//!
//! Every architecture runs as fucina tensors (`wavenet.zig`, `lstm.zig`,
//! `models.zig`) on an `ExecContext` the engine owns: one buffer pool per
//! engine, warm after the first blocks, so `process` is allocation-free at
//! steady state and safe on the audio thread.

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
const lstm = @import("lstm.zig");
const models = @import("models.zig");

const ExecContext = fucina.ExecContext;

pub const Error = error{
    UnsupportedChannels,
    FramesExceedMaxBuffer,
    NotReset,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    ctx: *ExecContext,
    impl: Impl,
    /// -1 when the model doesn't declare one.
    expected_sample_rate: f64,
    max_frames: usize,
    /// Zero-filled input scratch for prewarm.
    warm_in: []f32,
    warm_out: []f32,

    const LstmImpl = struct {
        model: *lstm.Model,
        stream: lstm.Stream,
        /// 0.5 s of the model's expected rate (lstm.cpp:125-132).
        prewarm_samples: usize,
    };

    pub const Impl = union(nam_file.Arch) {
        wavenet: wavenet.WaveNet,
        lstm: LstmImpl,
        convnet: models.ConvNet,
        linear: models.Linear,
    };

    pub fn init(allocator: std.mem.Allocator, model: *const nam_file.NamModel) !Engine {
        // v1 is mono-in mono-out (every real-world NAM profile).
        switch (model.config) {
            .wavenet => |*c| {
                // The engine feeds a MONO input as the conv "condition": array 0
                // rechannels it (width = layers[0].input_size) and every layer's
                // input_mixin reads it (width = condition_size), so both must be 1
                // or those convs index past the mono buffer. Higher arrays'
                // input_size == prev.channels is already enforced at parse.
                if (c.layers.len == 0) return Error.UnsupportedChannels;
                if (c.in_channels != 1 or c.layers[0].input_size != 1) return Error.UnsupportedChannels;
                for (c.layers) |*arr| {
                    if (c.condition_dsp == null and arr.condition_size != 1) return Error.UnsupportedChannels;
                    if (arr.channels == 0 or arr.bottleneck == 0 or arr.head_out == 0) return Error.UnsupportedChannels;
                }
                const out = if (c.head) |*h| h.out_channels else c.layers[c.layers.len - 1].head_out;
                if (out != 1) return Error.UnsupportedChannels;
            },
            .lstm => |*c| {
                if (c.in_channels != 1 or c.out_channels != 1 or c.input_size != 1) return Error.UnsupportedChannels;
            },
            .convnet => |*c| {
                if (c.in_channels != 1 or c.out_channels != 1) return Error.UnsupportedChannels;
            },
            .linear => |*c| {
                if (c.in_channels != 1 or c.out_channels != 1) return Error.UnsupportedChannels;
            },
        }

        const ctx = try allocator.create(ExecContext);
        errdefer allocator.destroy(ctx);
        ctx.init(allocator);
        errdefer ctx.deinit();

        const chunk_hint = 64;
        const impl: Impl = switch (model.config) {
            .wavenet => |*c| .{ .wavenet = try wavenet.WaveNet.init(allocator, ctx, c, model.weights, .{ .chunk_hint = chunk_hint }) },
            .lstm => |*c| blk: {
                const lstm_model = try allocator.create(lstm.Model);
                errdefer allocator.destroy(lstm_model);
                lstm_model.* = try lstm.Model.initFromNam(allocator, ctx, c, model.weights, false, .{});
                errdefer lstm_model.deinit();
                const stream = try lstm.Stream.init(allocator, ctx, lstm_model);
                const prewarm_f = 0.5 * model.sample_rate;
                const prewarm: usize = if (prewarm_f >= 1) @intFromFloat(prewarm_f) else 1;
                break :blk .{ .lstm = .{ .model = lstm_model, .stream = stream, .prewarm_samples = prewarm } };
            },
            .convnet => |*c| .{ .convnet = try models.ConvNet.init(allocator, ctx, c, model.weights, chunk_hint) },
            .linear => |*c| .{ .linear = try models.Linear.init(allocator, ctx, c, model.weights, chunk_hint) },
        };

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .impl = impl,
            .expected_sample_rate = model.sample_rate,
            .max_frames = 0,
            .warm_in = &.{},
            .warm_out = &.{},
        };
    }

    pub fn deinit(self: *Engine) void {
        switch (self.impl) {
            .wavenet => |*e| e.deinit(),
            .lstm => |*e| {
                e.stream.deinit();
                e.model.deinit();
                self.allocator.destroy(e.model);
            },
            .convnet => |*e| e.deinit(),
            .linear => |*e| e.deinit(),
        }
        self.ctx.deinit();
        self.allocator.destroy(self.ctx);
        self.allocator.free(self.warm_in);
        self.allocator.free(self.warm_out);
        self.* = undefined;
    }

    pub fn prewarmSamples(self: *const Engine) usize {
        return switch (self.impl) {
            .wavenet => |*e| e.prewarmSamples(),
            .lstm => |*e| e.prewarm_samples,
            .convnet => |*e| e.prewarmSamples(),
            // Linear is FIR with zero-initialized history: prewarming with
            // zeros is a no-op, so none is needed.
            .linear => 0,
        };
    }

    /// Sizes buffers for blocks of up to `max_frames`, zeroes all streaming
    /// state, and (by default) prewarms with block-rounded zeros.
    pub fn reset(self: *Engine, max_frames: usize, prewarm: bool) !void {
        std.debug.assert(max_frames > 0);
        self.max_frames = max_frames;
        self.allocator.free(self.warm_in);
        self.allocator.free(self.warm_out);
        self.warm_in = &.{};
        self.warm_out = &.{};
        self.warm_in = try self.allocator.alloc(f32, max_frames);
        self.warm_out = try self.allocator.alloc(f32, max_frames);
        @memset(self.warm_in, 0);

        switch (self.impl) {
            .wavenet => |*e| e.reset(),
            .lstm => |*e| try e.stream.reset(self.ctx),
            .convnet => |*e| e.reset(),
            .linear => |*e| e.reset(),
        }

        if (prewarm) {
            const samples = self.prewarmSamples();
            const blocks = (samples + max_frames - 1) / max_frames;
            for (0..blocks) |_| try self.processUnchecked(self.warm_in, self.warm_out, max_frames);
        }
    }

    /// Mono in -> mono out; allocation-free once the pool is warm.
    pub fn process(self: *Engine, input: []const f32, output: []f32, frames: usize) !void {
        if (self.max_frames == 0) return Error.NotReset;
        if (frames > self.max_frames) return Error.FramesExceedMaxBuffer;
        try self.processUnchecked(input, output, frames);
    }

    fn processUnchecked(self: *Engine, input: []const f32, output: []f32, frames: usize) !void {
        switch (self.impl) {
            .wavenet => |*e| try e.process(self.ctx, input, output, frames),
            .lstm => |*e| try e.stream.process(self.ctx, input, output, frames),
            .convnet => |*e| try e.process(self.ctx, input, output, frames),
            .linear => |*e| try e.process(self.ctx, input, output, frames),
        }
    }
};

test {
    _ = @import("engine_tests.zig");
}
