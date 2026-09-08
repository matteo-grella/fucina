//! `fucina-nam facade-bench`: the measurement side of the assess/nam-facade
//! assessment. Runs the hand-rolled streaming engine and its facade
//! counterpart (`facade_engine.zig`) on the same model and stream, reports
//! parity, ns/sample, steady-state allocations, then microbenchmarks each
//! distinct conv shape of the model three ways (StreamConv kernel, the core
//! `groupedCausalConv1dInto` kernel called raw, the facade op) plus the
//! tanh row pass, the cab IR, and the per-sample LSTM composition.

const std = @import("std");
const fucina = @import("fucina");
const facade = @import("facade_engine.zig");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
const stream_conv = @import("stream_conv.zig");
const activations = @import("activations.zig");
const ir_cab = @import("ir_cab.zig");
const models = @import("models.zig");
const gguf_compat = @import("gguf_compat.zig");
const lstm = @import("lstm.zig");

const ExecContext = fucina.ExecContext;
const kernels = fucina.internal.backend_mod.kernels;
const RawTensor = fucina.internal.tensor_mod.Tensor;
const ParallelConfig = fucina.internal.backend_mod.ParallelConfig;

const usage =
    \\usage: zig build nam -- facade-bench <model.nam | standard> [--blocksize N] [--seconds S]
    \\
;

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn elapsedNs(io: std.Io, start: i96) f64 {
    return @floatFromInt(nowNs(io) - start);
}

pub fn run(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, args: []const []const u8) !void {
    if (args.len < 1) {
        try stdout.writeAll(usage);
        return;
    }
    var blocksize: usize = 64;
    var seconds: f64 = 3.0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--blocksize")) {
            i += 1;
            if (i >= args.len) return error.MissingBlocksize;
            blocksize = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--seconds")) {
            i += 1;
            if (i >= args.len) return error.MissingSeconds;
            seconds = try std.fmt.parseFloat(f64, args[i]);
        } else return error.UnknownArgument;
    }
    if (blocksize == 0 or !std.math.isFinite(seconds) or seconds <= 0 or seconds > 600) return error.InvalidArgument;

    var model: ?nam_file.NamModel = null;
    defer if (model) |*m| m.deinit();
    var synthetic_weights: ?[]f32 = null;
    defer if (synthetic_weights) |w| allocator.free(w);
    var config: *const nam_file.WaveNetConfig = undefined;
    var weights: []const f32 = undefined;
    var rate: f64 = 48000;
    if (std.mem.eql(u8, args[0], "standard")) {
        config = &facade.standard_config;
        synthetic_weights = try facade.syntheticWeights(allocator, config, 11, 0.25);
        weights = synthetic_weights.?;
    } else {
        model = try gguf_compat.loadAny(io, allocator, args[0]);
        const m = &model.?;
        if (m.config != .wavenet) return error.UnsupportedFeature;
        config = &m.config.wavenet;
        weights = m.weights;
        if (m.sample_rate > 0) rate = m.sample_rate;
    }

    try stdout.print("model:        {s} ({d} weights, {d} arrays)\n", .{ args[0], weights.len, config.layers.len });
    for (config.layers, 0..) |*lc, ai| {
        var films: usize = 0;
        inline for (.{ "conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film", "activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film" }) |field| {
            if (@field(lc, field).active) films += 1;
        }
        try stdout.print("  array {d}: {d}->{d} ch, bottleneck {d}, {d} layers k{d}, gating {s}, act {s}, groups in/mixin/1x1 {d}/{d}/{d}, head1x1 {s}, head {d}x{d}{s}, {d} FiLM slots\n", .{
            ai,                 lc.input_size,                          lc.channels,                      lc.bottleneck,   lc.layerCount(),
            lc.kernel_sizes[0], @tagName(lc.gating_modes[0]),           @tagName(lc.activations[0].kind), lc.groups_input, lc.groups_input_mixin,
            lc.layer1x1_groups, if (lc.head1x1_active) "on" else "off", lc.head_kernel,                   lc.head_out,     if (lc.head_bias) " +bias" else "",
            films,
        });
    }
    if (config.head) |*hc| try stdout.print("  post-head: {d} blocks, {d} channels, act {s}\n", .{ hc.kernel_sizes.len, hc.channels, @tagName(hc.activation.kind) });
    if (config.condition_dsp != null) try stdout.writeAll("  condition DSP: nested\n");

    var counting = facade.CountingAllocator{ .backing = allocator };
    var ctx: ExecContext = undefined;
    ctx.init(counting.allocator());
    defer ctx.deinit();

    var reference = try wavenet.WaveNetEngine.init(allocator, config, weights);
    defer reference.deinit();
    try reference.reset(blocksize);
    var candidate = try facade.WaveNet.init(allocator, &ctx, config, weights, blocksize);
    defer candidate.deinit();
    candidate.reset();

    const total_blocks: usize = @intFromFloat(@max(1.0, seconds * rate / @as(f64, @floatFromInt(blocksize))));
    const input = try allocator.alloc(f32, blocksize);
    defer allocator.free(input);
    facade.fillSignal(input, 42);
    const out_ref = try allocator.alloc(f32, blocksize);
    defer allocator.free(out_ref);
    const out_fac = try allocator.alloc(f32, blocksize);
    defer allocator.free(out_fac);

    // Parity over the whole run (same stream, both from reset).
    var max_abs: f32 = 0;
    var max_ref: f32 = 0;
    var exact: usize = 0;
    for (0..total_blocks) |_| {
        reference.process(input, out_ref, blocksize);
        try candidate.process(&ctx, input, out_fac, blocksize);
        for (out_ref, out_fac) |r, g| {
            max_abs = @max(max_abs, @abs(r - g));
            max_ref = @max(max_ref, @abs(r));
            if (r == g) exact += 1;
        }
    }
    try stdout.print("parity:       max|d|={e:.2} over {d} samples (max|ref|={d:.3}, exact {d})\n", .{ max_abs, total_blocks * blocksize, max_ref, exact });

    // Timing, A / B / A so DVFS drift shows up as a disagreement between the two A runs.
    const budget_ns = @as(f64, @floatFromInt(blocksize)) / rate * 1e9;
    try stdout.print("blocksize:    {d} frames @ {d} Hz (budget {d:.0} us/block), {d} blocks per run\n", .{ blocksize, rate, budget_ns / 1e3, total_blocks });
    const ref_a = try timeReference(io, &reference, input, out_ref, blocksize, total_blocks);
    const allocs_before = counting.allocs.load(.monotonic);
    const fac = try timeFacade(io, &ctx, &candidate, input, out_fac, blocksize, total_blocks);
    const allocs_during = counting.allocs.load(.monotonic) - allocs_before;
    const allocs_before_borrowed = counting.allocs.load(.monotonic);
    const fac_borrowed = try timeFacadeBorrowed(io, &ctx, &candidate, input, out_fac, blocksize, total_blocks);
    const allocs_during_borrowed = counting.allocs.load(.monotonic) - allocs_before_borrowed;
    const ref_b = try timeReference(io, &reference, input, out_ref, blocksize, total_blocks);
    const per_sample = @as(f64, @floatFromInt(blocksize));
    try stdout.print("hand-rolled:  {d:.1} us/block  {d:.1} ns/sample  ({d:.1}x headroom)   [second run {d:.1} us/block]\n", .{ ref_a / 1e3, ref_a / per_sample, budget_ns / ref_a, ref_b / 1e3 });
    try stdout.print("facade:       {d:.1} us/block  {d:.1} ns/sample  ({d:.1}x headroom)   input via persistent slot + copyFrom; allocations during the timed run: {d}\n", .{ fac / 1e3, fac / per_sample, budget_ns / fac, allocs_during });
    try stdout.print("facade:       {d:.1} us/block  {d:.1} ns/sample  ({d:.1}x headroom)   input via fromBorrowedConstSlice per block; allocations during the timed run: {d}\n", .{ fac_borrowed / 1e3, fac_borrowed / per_sample, budget_ns / fac_borrowed, allocs_during_borrowed });
    try stdout.print("facade/hand:  {d:.2}x time (slot), {d:.2}x (borrowed)\n", .{ fac / @min(ref_a, ref_b), fac_borrowed / @min(ref_a, ref_b) });

    try convMicrobench(io, allocator, stdout, &ctx, config, blocksize);
    try tanhMicrobench(io, allocator, stdout, &ctx, blocksize);
    try cabMicrobench(io, allocator, stdout, &ctx, blocksize);
    try lstmMicrobench(io, allocator, stdout, &ctx);
}

fn timeReference(io: std.Io, engine: *wavenet.WaveNetEngine, input: []const f32, output: []f32, blocksize: usize, blocks: usize) !f64 {
    for (0..@min(blocks, 64)) |_| engine.process(input, output, blocksize);
    const start = nowNs(io);
    for (0..blocks) |_| engine.process(input, output, blocksize);
    return elapsedNs(io, start) / @as(f64, @floatFromInt(blocks));
}

fn timeFacade(io: std.Io, ctx: *ExecContext, engine: *facade.WaveNet, input: []const f32, output: []f32, blocksize: usize, blocks: usize) !f64 {
    for (0..@min(blocks, 64)) |_| try engine.process(ctx, input, output, blocksize);
    const start = nowNs(io);
    for (0..blocks) |_| try engine.process(ctx, input, output, blocksize);
    return elapsedNs(io, start) / @as(f64, @floatFromInt(blocks));
}

/// The zero-copy alternative for the block input: a borrowed tensor per
/// block (one storage header allocation each) straight into `forwardBlock`.
fn processBorrowed(ctx: *ExecContext, engine: *facade.WaveNet, input: []const f32, output: []f32, frames: usize) !void {
    const mark = ctx.openExecScope();
    defer ctx.closeExecScope(mark);
    var x = try facade.TimeIn.fromBorrowedConstSlice(ctx, .{ frames, 1 }, input[0..frames]);
    defer x.deinit();
    const out = try engine.forwardBlock(ctx, &x);
    try out.copyTo(output[0..frames]);
}

fn timeFacadeBorrowed(io: std.Io, ctx: *ExecContext, engine: *facade.WaveNet, input: []const f32, output: []f32, blocksize: usize, blocks: usize) !f64 {
    for (0..@min(blocks, 64)) |_| try processBorrowed(ctx, engine, input, output, blocksize);
    const start = nowNs(io);
    for (0..blocks) |_| try processBorrowed(ctx, engine, input, output, blocksize);
    return elapsedNs(io, start) / @as(f64, @floatFromInt(blocks));
}

// ---------------------------------------------------------------------------
// Conv shapes of the model, three ways, plus the main-branch generic kernel
// as the fourth column.
// ---------------------------------------------------------------------------

/// The general causal conv forward as main (e356ac7) runs it for every shape
/// without a fixed-shape route: per output row, zero, then per tap and
/// in-channel an axpy over the out-channel row (unfused multiply-add, the
/// accumulator round-tripping through memory). Kept here verbatim so the
/// before/after column is measured, not remembered.
fn legacyGenericForward(out: []f32, input: []const f32, weight: []const f32, state: ?[]const f32, in_channels: usize, out_channels: usize, taps: usize, dilation: usize, seq: usize) void {
    const V = @Vector(vector_len, f32);
    const pad = dilation * (taps - 1);
    for (0..seq) |t| {
        const out_row = out[t * out_channels ..][0..out_channels];
        @memset(out_row, 0);
        for (0..taps) |k| {
            const shifted = t + k * dilation;
            const x_row = if (shifted >= pad) input[(shifted - pad) * in_channels ..][0..in_channels] else if (state) |s| s[shifted * in_channels ..][0..in_channels] else continue;
            for (0..in_channels) |i| {
                const row = weight[(k * in_channels + i) * out_channels ..][0..out_channels];
                const sv: V = @splat(x_row[i]);
                var o: usize = 0;
                while (o + vector_len <= out_channels) : (o += vector_len) {
                    const cur: V = out_row[o..][0..vector_len].*;
                    const rv: V = row[o..][0..vector_len].*;
                    out_row[o..][0..vector_len].* = cur + sv * rv;
                }
                while (o < out_channels) : (o += 1) out_row[o] += x_row[i] * row[o];
            }
        }
    }
}

const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;

/// Shapes of the other in-tree consumer of this kernel, the qwen3tts codec
/// decoder (pre_conv, the DAC entry conv, a residual conv per stage width,
/// the 1x1 residual conv, the post conv).
const codec_shapes = [_]ConvShape{
    .{ .in = 512, .out = 1024, .taps = 3, .dilation = 1, .bias = true, .groups = 1 },
    .{ .in = 1024, .out = 1536, .taps = 7, .dilation = 1, .bias = true, .groups = 1 },
    .{ .in = 768, .out = 768, .taps = 7, .dilation = 3, .bias = true, .groups = 1 },
    .{ .in = 192, .out = 192, .taps = 7, .dilation = 9, .bias = true, .groups = 1 },
    .{ .in = 96, .out = 96, .taps = 1, .dilation = 1, .bias = true, .groups = 1 },
    .{ .in = 96, .out = 1, .taps = 7, .dilation = 1, .bias = true, .groups = 1 },
};

const ConvShape = struct { in: usize, out: usize, taps: usize, dilation: usize, bias: bool, groups: usize };

fn addShape(list: *std.ArrayList(ConvShape), allocator: std.mem.Allocator, shape: ConvShape) !void {
    for (list.items) |s| {
        if (std.meta.eql(s, shape)) return;
    }
    try list.append(allocator, shape);
}

fn collectShapes(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig) !std.ArrayList(ConvShape) {
    var list: std.ArrayList(ConvShape) = .empty;
    errdefer list.deinit(allocator);
    for (config.layers) |*lc| {
        try addShape(&list, allocator, .{ .in = lc.input_size, .out = lc.channels, .taps = 1, .dilation = 1, .bias = false, .groups = 1 });
        for (0..lc.layerCount()) |l| {
            const bg = lc.gateWidth(l);
            try addShape(&list, allocator, .{ .in = lc.channels, .out = bg, .taps = lc.kernel_sizes[l], .dilation = lc.dilations[l], .bias = true, .groups = lc.groups_input });
            try addShape(&list, allocator, .{ .in = lc.condition_size, .out = bg, .taps = 1, .dilation = 1, .bias = false, .groups = lc.groups_input_mixin });
            if (lc.layer1x1_active) try addShape(&list, allocator, .{ .in = lc.bottleneck, .out = lc.channels, .taps = 1, .dilation = 1, .bias = true, .groups = lc.layer1x1_groups });
            if (lc.head1x1_active) try addShape(&list, allocator, .{ .in = lc.bottleneck, .out = lc.head1x1_out, .taps = 1, .dilation = 1, .bias = true, .groups = lc.head1x1_groups });
        }
        const head_width = if (lc.head1x1_active) lc.head1x1_out else lc.bottleneck;
        try addShape(&list, allocator, .{ .in = head_width, .out = lc.head_out, .taps = lc.head_kernel, .dilation = 1, .bias = lc.head_bias, .groups = 1 });
    }
    return list;
}

fn convMicrobench(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, ctx: *ExecContext, config: *const nam_file.WaveNetConfig, frames: usize) !void {
    var shapes = try collectShapes(allocator, config);
    defer shapes.deinit(allocator);
    for (codec_shapes) |shape| try addShape(&shapes, allocator, shape);
    try stdout.print("\nconv kernels at {d} frames (ns per call; MAC = frames*in/groups*out*taps; legacy = main's generic kernel, '-' where main took a fixed-shape route):\n", .{frames});
    try stdout.writeAll("  shape                      StreamConv   core kernel   facade op      legacy   core/SC   facade/SC   core/legacy\n");
    for (shapes.items) |shape| {
        const in_per_group = shape.in / shape.groups;
        const stream_len = shape.taps * in_per_group * shape.out + (if (shape.bias) shape.out else 0);
        const stream = try allocator.alloc(f32, stream_len);
        defer allocator.free(stream);
        var prng = std.Random.DefaultPrng.init(stream_len);
        for (stream) |*v| v.* = (prng.random().float(f32) * 2 - 1) * 0.3;

        var sc = try stream_conv.StreamConv.initGrouped(allocator, shape.in, shape.out, shape.taps, shape.dilation, shape.bias, shape.groups);
        defer sc.deinit();
        _ = sc.loadNamWeights(stream);
        var cursor: usize = 0;
        var fc = try facade.Conv.init(allocator, ctx, shape.in, shape.out, shape.taps, shape.dilation, shape.bias, shape.groups, frames, stream, &cursor);
        defer fc.deinit();

        const input = try allocator.alloc(f32, frames * shape.in);
        defer allocator.free(input);
        facade.fillSignal(input, 3);
        const output = try allocator.alloc(f32, frames * shape.out);
        defer allocator.free(output);

        const macs_per_call = frames * in_per_group * shape.out * shape.taps;
        const reps: usize = @max(5, @min(200, 20_000_000 / @max(1, macs_per_call)));

        // (a) StreamConv: process + push, as the engine calls it.
        for (0..reps / 10) |_| {
            sc.process(input, output, frames, false);
            sc.push(input, frames);
        }
        var start = nowNs(io);
        for (0..reps) |_| {
            sc.process(input, output, frames, false);
            sc.push(input, frames);
        }
        const sc_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

        // (b) The core kernel called raw (serial) with the bias epilogue and
        //     the same state carry: what the facade op costs minus dispatch.
        const shape_in = [_]usize{ frames, shape.in };
        const shape_out = [_]usize{ frames, shape.out };
        const shape_w = [_]usize{ shape.taps, in_per_group, shape.out };
        var raw_in = try RawTensor.fromBorrowedSlice(allocator, &shape_in, input);
        defer raw_in.deinit();
        var raw_out = try RawTensor.fromBorrowedSlice(allocator, &shape_out, output);
        defer raw_out.deinit();
        const wdata = try allocator.dupe(f32, try fc.weight.dataConst());
        defer allocator.free(wdata);
        var raw_w = try RawTensor.fromBorrowedSlice(allocator, &shape_w, wdata);
        defer raw_w.deinit();
        const pc = ParallelConfig{};
        for (0..reps / 10) |_| {
            kernels.groupedCausalConv1dInto(pc, &raw_out, &raw_in, &raw_w, fc.state.slice(), fc.biasOrNull(), frames, shape.in, shape.out, shape.taps, shape.dilation, shape.groups);
            fc.state.advance(input);
        }
        start = nowNs(io);
        for (0..reps) |_| {
            kernels.groupedCausalConv1dInto(pc, &raw_out, &raw_in, &raw_w, fc.state.slice(), fc.biasOrNull(), frames, shape.in, shape.out, shape.taps, shape.dilation, shape.groups);
            fc.state.advance(input);
        }
        const core_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

        // (c) The facade op inside a scope, with the block as a persistent-slot view.
        var slot = try facade.TimeIn.zeros(ctx, .{ frames, shape.in });
        defer slot.deinit();
        @memcpy(try slot.data(), input);
        for (0..reps / 10) |_| {
            const mark = ctx.openExecScope();
            defer ctx.closeExecScope(mark);
            const y = try fc.forward(ctx, &slot);
            try y.copyTo(output);
        }
        start = nowNs(io);
        for (0..reps) |_| {
            const mark = ctx.openExecScope();
            defer ctx.closeExecScope(mark);
            const y = try fc.forward(ctx, &slot);
            try y.copyTo(output);
        }
        const fac_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

        // (d) main's generic kernel (+ the bias row pass it needed), only
        //     where main actually took it.
        const legacy_fixed = shape.groups == 1 and (shape.taps == 1 and (isPair(shape, 8, 8) or isPair(shape, 4, 4) or isPair(shape, 3, 3) or isPair(shape, 2, 2)) or
            shape.taps > 1 and (isPair(shape, 8, 16) or isPair(shape, 8, 8) or isPair(shape, 8, 1) or isPair(shape, 4, 8) or isPair(shape, 4, 4) or isPair(shape, 3, 3) or isPair(shape, 3, 1) or isPair(shape, 2, 4) or isPair(shape, 2, 2) or isPair(shape, 2, 1)));
        var legacy_ns: f64 = 0;
        if (shape.groups == 1 and !legacy_fixed) {
            for (0..reps / 10) |_| {
                legacyGenericForward(output, input, wdata, fc.state.slice(), shape.in, shape.out, shape.taps, shape.dilation, frames);
                if (shape.bias) kernels.addRowVectorSlice(null, output, fc.bias, frames, shape.out);
            }
            start = nowNs(io);
            for (0..reps) |_| {
                legacyGenericForward(output, input, wdata, fc.state.slice(), shape.in, shape.out, shape.taps, shape.dilation, frames);
                if (shape.bias) kernels.addRowVectorSlice(null, output, fc.bias, frames, shape.out);
            }
            legacy_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));
        }

        var label_buf: [40]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "{d}->{d} k{d} d{d} g{d}{s}", .{ shape.in, shape.out, shape.taps, shape.dilation, shape.groups, if (shape.bias) " +b" else "" });
        if (legacy_ns > 0) {
            try stdout.print("  {s:<26} {d:>9.0}   {d:>9.0}   {d:>9.0}   {d:>9.0}   {d:>6.2}x   {d:>6.2}x   {d:>8.2}x\n", .{ label, sc_ns, core_ns, fac_ns, legacy_ns, core_ns / sc_ns, fac_ns / sc_ns, core_ns / legacy_ns });
        } else {
            try stdout.print("  {s:<26} {d:>9.0}   {d:>9.0}   {d:>9.0}   {s:>9}   {d:>6.2}x   {d:>6.2}x   {s:>9}\n", .{ label, sc_ns, core_ns, fac_ns, "-", core_ns / sc_ns, fac_ns / sc_ns, "-" });
        }
    }
}

fn isPair(shape: ConvShape, in: usize, out: usize) bool {
    return shape.in == in and shape.out == out;
}

fn tanhMicrobench(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, ctx: *ExecContext, frames: usize) !void {
    const channels = 16;
    const n = frames * channels;
    const data = try allocator.alloc(f32, n);
    defer allocator.free(data);
    const reps: usize = 20_000_000 / n + 100;
    const act = nam_file.Activation{ .kind = .tanh };

    facade.fillSignal(data, 5);
    for (0..reps / 10) |_| activations.applyRows(&act, data, channels);
    var start = nowNs(io);
    for (0..reps) |_| activations.applyRows(&act, data, channels);
    const nam_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

    var slot = try fucina.Tensor(.{ .time, .out }).zeros(ctx, .{ frames, channels });
    defer slot.deinit();
    facade.fillSignal(try slot.data(), 5);
    for (0..reps / 10) |_| {
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        _ = try slot.tanh(ctx);
    }
    start = nowNs(io);
    for (0..reps) |_| {
        const mark = ctx.openExecScope();
        defer ctx.closeExecScope(mark);
        _ = try slot.tanh(ctx);
    }
    const fac_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

    // In place through the fused bias+activation pass (zero bias).
    const zero_bias = [_]f32{0} ** channels;
    for (0..reps / 10) |_| try slot.addAxisVectorUnaryInPlace(ctx, .tanh, &zero_bias, .out);
    start = nowNs(io);
    for (0..reps) |_| try slot.addAxisVectorUnaryInPlace(ctx, .tanh, &zero_bias, .out);
    const fused_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

    try stdout.print("\ntanh over [{d}, {d}] (ns per pass): NAM tanhLanes in place {d:.0}   facade tanh (new tensor) {d:.0}   facade bias+tanh in place {d:.0}   ({d:.2} / {d:.2} ns per element)\n", .{ frames, channels, nam_ns, fac_ns, fused_ns, nam_ns / @as(f64, @floatFromInt(n)), fac_ns / @as(f64, @floatFromInt(n)) });
}

fn cabMicrobench(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, ctx: *ExecContext, frames: usize) !void {
    const taps = 2048;
    const ir = try allocator.alloc(f32, taps);
    defer allocator.free(ir);
    var prng = std.Random.DefaultPrng.init(9);
    for (ir, 0..) |*v, i| v.* = (prng.random().float(f32) * 2 - 1) * @exp(-@as(f32, @floatFromInt(i)) / 512.0);
    var reference = try ir_cab.IrCab.init(allocator, ir, 48000, 48000, frames);
    defer reference.deinit();
    var candidate = try facade.IrCab.init(allocator, ctx, reference.weight, frames);
    defer candidate.deinit();

    const input = try allocator.alloc(f32, frames);
    defer allocator.free(input);
    facade.fillSignal(input, 8);
    const output = try allocator.alloc(f32, frames);
    defer allocator.free(output);
    const reps: usize = 200_000_000 / (frames * taps) + 20;

    for (0..reps / 10) |_| reference.process(input, output, frames);
    var start = nowNs(io);
    for (0..reps) |_| reference.process(input, output, frames);
    const ref_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

    for (0..reps / 10) |_| try candidate.process(ctx, input, output, frames);
    start = nowNs(io);
    for (0..reps) |_| try candidate.process(ctx, input, output, frames);
    const fac_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(reps));

    const per_sample = @as(f64, @floatFromInt(frames));
    try stdout.print("\ncab IR {d} taps at {d} frames: IrCab {d:.0} ns/sample   facade causalConv1d(1->1) {d:.0} ns/sample   ({d:.2}x)\n", .{ taps, frames, ref_ns / per_sample, fac_ns / per_sample, fac_ns / ref_ns });
}

fn lstmMicrobench(io: std.Io, allocator: std.mem.Allocator, stdout: *std.Io.Writer, ctx: *ExecContext) !void {
    const config = nam_file.LstmConfig{ .input_size = 1, .hidden_size = 24, .num_layers = 1, .in_channels = 1, .out_channels = 1 };
    const file_config = nam_file.Config{ .lstm = config };
    const count = nam_file.expectedWeightCount(&file_config);
    const weights = try allocator.alloc(f32, count);
    defer allocator.free(weights);
    var prng = std.Random.DefaultPrng.init(17);
    for (weights) |*v| v.* = (prng.random().float(f32) * 2 - 1) * 0.3;
    var reference = try models.LstmEngine.init(allocator, &config, weights, 48000);
    defer reference.deinit();
    var model = try lstm.Model.initFromNam(allocator, ctx, &config, weights, false, .{});
    defer model.deinit();
    var candidate = try lstm.Stream.init(allocator, ctx, &model);
    defer candidate.deinit();

    const total = 24000;
    const input = try allocator.alloc(f32, total);
    defer allocator.free(input);
    facade.fillSignal(input, 6);
    const out_ref = try allocator.alloc(f32, total);
    defer allocator.free(out_ref);
    const out_fac = try allocator.alloc(f32, total);
    defer allocator.free(out_fac);

    reference.process(input, out_ref, 2000);
    var start = nowNs(io);
    reference.process(input, out_ref, total);
    const ref_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(total));
    try candidate.process(ctx, input, out_fac, 2000);
    start = nowNs(io);
    try candidate.process(ctx, input, out_fac, total);
    const fac_ns = elapsedNs(io, start) / @as(f64, @floatFromInt(total));
    var max_abs: f32 = 0;
    reference.reset();
    try candidate.reset(ctx);
    reference.process(input, out_ref, total);
    try candidate.process(ctx, input, out_fac, total);
    for (out_ref, out_fac) |r, g| max_abs = @max(max_abs, @abs(r - g));
    try stdout.print("\nlstm hidden 24 x 1 layer: LstmEngine {d:.0} ns/sample   tensor lstm.Stream {d:.0} ns/sample   ({d:.2}x, max|d|={e:.2})\n", .{ ref_ns, fac_ns, fac_ns / ref_ns, max_abs });
    try lstmOpCosts(io, stdout, ctx, config.hidden_size);
}

/// The cost of each facade op at the LSTM step's shapes (ns per call):
/// where a per-sample composition spends its time.
fn lstmOpCosts(io: std.Io, stdout: *std.Io.Writer, ctx: *ExecContext, hidden: usize) !void {
    const H = hidden;
    const K = 1 + H + 1;
    const reps: usize = 20000;
    var w_data: [4 * 32 * 34]f32 = undefined;
    for (&w_data, 0..) |*v, i| v.* = 0.01 * @as(f32, @floatFromInt(i % 17));
    var w = try fucina.Tensor(.{ .unit, .k }).fromSlice(ctx, .{ 4 * H, K }, w_data[0 .. 4 * H * K]);
    defer w.deinit();
    var h = try fucina.Tensor(.{.unit}).zeros(ctx, .{H});
    defer h.deinit();
    var c = try fucina.Tensor(.{.unit}).zeros(ctx, .{H});
    defer c.deinit();
    var x_t = try fucina.Tensor(.{.k}).zeros(ctx, .{1});
    defer x_t.deinit();
    var one = try fucina.Tensor(.{.k}).ones(ctx, .{1});
    defer one.deinit();
    var gates = try fucina.Tensor(.{.unit}).zeros(ctx, .{4 * H});
    defer gates.deinit();
    var bias: [128]f32 = undefined;
    @memset(&bias, 0.1);
    const sample = [_]f32{0.3};

    var start = nowNs(io);
    for (0..reps) |_| try x_t.copyFrom(&sample);
    const t_copyfrom = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try h.withTags(ctx, .{.k});
        v.deinit();
    }
    const t_view = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try gates.narrow(ctx, .unit, H, H);
        v.deinit();
    }
    const t_narrow = elapsedNs(io, start) / reps;

    var h_k = try h.withTags(ctx, .{.k});
    defer h_k.deinit();
    start = nowNs(io);
    for (0..reps) |_| {
        var v = try x_t.concat(ctx, .k, &.{ &h_k, &one });
        v.deinit();
    }
    const t_concat = elapsedNs(io, start) / reps;

    var xh = try x_t.concat(ctx, .k, &.{ &h_k, &one });
    defer xh.deinit();
    start = nowNs(io);
    for (0..reps) |_| {
        var v = try w.dot(ctx, &xh, .k);
        v.deinit();
    }
    const t_dot = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| try gates.addAxisVectorInPlace(ctx, bias[0 .. 4 * H], .unit);
    const t_bias = elapsedNs(io, start) / reps;

    // matvec alternatives: matmul against a [k, 1] view, and mul + sum.
    var xh_col = try xh.split(ctx, .k, .{ .k, .one }, .{ K, 1 });
    defer xh_col.deinit();
    start = nowNs(io);
    for (0..reps) |_| {
        var v = try w.matmul(ctx, &xh_col, .plain, .{ .unit, .one });
        v.deinit();
    }
    const t_matmul = elapsedNs(io, start) / reps;
    start = nowNs(io);
    for (0..reps) |_| {
        var prod = try w.mul(ctx, &xh);
        defer prod.deinit();
        var v = try prod.sum(ctx, .k, .{});
        v.deinit();
    }
    const t_mulsum = elapsedNs(io, start) / reps;
    // Transposed orientation: weights stored [k, unit], vector times matrix.
    var w_kn = try w.permuteTo(ctx, .{ .k, .unit });
    defer w_kn.deinit();
    var w_kn_c = try w_kn.materialize(ctx);
    defer w_kn_c.deinit();
    start = nowNs(io);
    for (0..reps) |_| {
        var v = try xh.dot(ctx, &w_kn_c, .k);
        v.deinit();
    }
    const t_dot_t = elapsedNs(io, start) / reps;
    var xh_row = try xh.split(ctx, .k, .{ .one, .k }, .{ 1, K });
    defer xh_row.deinit();
    start = nowNs(io);
    for (0..reps) |_| {
        var v = try xh_row.matmul(ctx, &w_kn_c, .plain, .{ .one, .unit });
        v.deinit();
    }
    const t_matmul_t = elapsedNs(io, start) / reps;
    // The raw kernel for the transposed orientation (no dispatch).
    var raw_out = try fucina.Tensor(.{.unit}).zeros(ctx, .{4 * H});
    defer raw_out.deinit();
    const raw_a = xh.asRawTensor();
    const raw_b = w_kn_c.asRawTensor();
    var raw_o = try fucina.internal.tensor_mod.Tensor.fromBorrowedSlice(ctx.allocator(), &.{ 1, 4 * H }, try raw_out.data());
    defer raw_o.deinit();
    start = nowNs(io);
    for (0..reps) |_| kernels.gemm(.{}, .{ .kind = .plain }, &raw_o, raw_a, raw_b, 1, 4 * H, K);
    const t_raw_t = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try c.sigmoid(ctx);
        v.deinit();
    }
    const t_sigmoid = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try c.tanh(ctx);
        v.deinit();
    }
    const t_tanh = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try c.glu(ctx, &h);
        v.deinit();
    }
    const t_glu = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try c.mul(ctx, &h);
        v.deinit();
    }
    const t_mul = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| try c.addScaledInPlace(ctx, &h, 1.0);
    const t_axpy = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| {
        var v = try fucina.Tensor(.{.k}).fromSlice(ctx, .{1}, &sample);
        v.deinit();
    }
    const t_fromslice = elapsedNs(io, start) / reps;

    start = nowNs(io);
    for (0..reps) |_| _ = try x_t.item();
    const t_item = elapsedNs(io, start) / reps;

    try stdout.print("  matvec orientations (ns): [4H x K]·[K] dot {d:.0}  matmul {d:.0}  mul+sum {d:.0}  |  [K]·[K x 4H] dot {d:.0}  matmul {d:.0}  raw gemm kernel {d:.0}\n", .{ t_dot, t_matmul, t_mulsum, t_dot_t, t_matmul_t, t_raw_t });
    try stdout.print("  op costs at H={d} (ns): copyFrom[1] {d:.0}  withTags view {d:.0}  narrow view {d:.0}  concat[1+H+1] {d:.0}  dot[4H x {d}] {d:.0}  matmul[4H x {d}]x[{d} x 1] {d:.0}  mul+sum {d:.0}  bias add[4H] {d:.0}  sigmoid[H] {d:.0}  tanh[H] {d:.0}  glu[H] {d:.0}  mul[H] {d:.0}  addScaledInPlace[H] {d:.0}  fromSlice[1] {d:.0}  item {d:.0}\n", .{ H, t_copyfrom, t_view, t_narrow, t_concat, K, t_dot, K, K, t_matmul, t_mulsum, t_bias, t_sigmoid, t_tanh, t_glu, t_mul, t_axpy, t_fromslice, t_item });
}
