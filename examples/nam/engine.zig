//! Unified NAM DSP engine: load a parsed `.nam` model, Reset, process mono
//! blocks. Mirrors the upstream DSP contract (NAM/dsp.h): Reset(sampleRate,
//! maxBufferSize) prewarms by default with zero samples **rounded up to
//! whole buffers** (ceil(prewarm/maxBuf)*maxBuf, dsp.cpp:47-81) — golden
//! parity vs upstream tools/render depends on reproducing exactly that.
//! Deviation from upstream (which asserts only in debug builds): frames >
//! max_frames is a checked error, not UB.

const std = @import("std");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
const models = @import("models.zig");

pub const Error = error{
    UnsupportedChannels,
    FramesExceedMaxBuffer,
    NotReset,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    impl: Impl,
    /// -1 when the model doesn't declare one.
    expected_sample_rate: f64,
    max_frames: usize,
    /// Zero-filled input scratch for prewarm.
    warm_in: []f32,
    warm_out: []f32,

    pub const Impl = union(nam_file.Arch) {
        wavenet: wavenet.WaveNetEngine,
        lstm: models.LstmEngine,
        convnet: models.ConvNetEngine,
        linear: models.LinearEngine,
    };

    pub fn init(allocator: std.mem.Allocator, model: *const nam_file.NamModel) !Engine {
        // v1 is mono-in mono-out (every real-world NAM profile).
        switch (model.config) {
            .wavenet => |*c| {
                // The engine feeds a MONO input as the conv "condition": array 0
                // rechannels it (width = layers[0].input_size) and every layer's
                // input_mixin reads it (width = condition_size), so both must be 1
                // or those convs index past the mono buffer (panic in Debug, OOB
                // read in ReleaseFast). Higher arrays' input_size == prev.channels
                // is already enforced at parse. Also reject empty/zero dims.
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

        const impl: Impl = switch (model.config) {
            .wavenet => |*c| .{ .wavenet = try wavenet.WaveNetEngine.init(allocator, c, model.weights) },
            .lstm => |*c| .{ .lstm = try models.LstmEngine.init(allocator, c, model.weights, model.sample_rate) },
            .convnet => |*c| .{ .convnet = try models.ConvNetEngine.init(allocator, c, model.weights) },
            .linear => |*c| .{ .linear = try models.LinearEngine.init(allocator, c, model.weights) },
        };

        return .{
            .allocator = allocator,
            .impl = impl,
            .expected_sample_rate = model.sample_rate,
            .max_frames = 0,
            .warm_in = &.{},
            .warm_out = &.{},
        };
    }

    pub fn deinit(self: *Engine) void {
        switch (self.impl) {
            inline else => |*engine| engine.deinit(),
        }
        self.allocator.free(self.warm_in);
        self.allocator.free(self.warm_out);
        self.* = undefined;
    }

    pub fn prewarmSamples(self: *const Engine) usize {
        return switch (self.impl) {
            .wavenet => |*e| e.prewarmSamples(),
            .lstm => |*e| e.prewarmSamples(),
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
            .wavenet => |*e| try e.reset(max_frames),
            .lstm => |*e| e.reset(),
            .convnet => |*e| try e.reset(max_frames),
            .linear => |*e| e.reset(),
        }

        if (prewarm) {
            const samples = self.prewarmSamples();
            const blocks = (samples + max_frames - 1) / max_frames;
            for (0..blocks) |_| {
                self.processUnchecked(self.warm_in, self.warm_out, max_frames);
            }
        }
    }

    /// Mono in -> mono out; allocation-free.
    pub fn process(self: *Engine, input: []const f32, output: []f32, frames: usize) Error!void {
        if (self.max_frames == 0) return Error.NotReset;
        if (frames > self.max_frames) return Error.FramesExceedMaxBuffer;
        self.processUnchecked(input, output, frames);
    }

    fn processUnchecked(self: *Engine, input: []const f32, output: []f32, frames: usize) void {
        switch (self.impl) {
            .wavenet => |*e| e.process(input, output, frames),
            .lstm => |*e| e.process(input, output, frames),
            .convnet => |*e| e.process(input, output, frames),
            .linear => |*e| e.process(input, output, frames),
        }
    }
};

test {
    _ = @import("engine_tests.zig");
}
