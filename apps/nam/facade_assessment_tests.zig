//! Feasibility gates for the assess/nam-facade assessment: every NAM engine
//! kernel that has a facade counterpart is run both ways on the same
//! streams, with the numeric distance printed (the assessment records it)
//! and the parity gate asserted. Also the steady-state allocation count of
//! the facade path, and the tanh contracts side by side.

const std = @import("std");
const fucina = @import("fucina");
const facade = @import("facade_engine.zig");
const nam_file = @import("nam_file.zig");
const wavenet = @import("wavenet.zig");
const stream_conv = @import("stream_conv.zig");
const activations = @import("activations.zig");
const ir_cab = @import("ir_cab.zig");
const models = @import("models.zig");
const lstm = @import("lstm.zig");

const ExecContext = fucina.ExecContext;
const Activation = nam_file.Activation;

const Distance = struct {
    max_abs: f32 = 0,
    max_ref: f32 = 0,
    exact: usize = 0,
    total: usize = 0,

    fn add(self: *Distance, ref: f32, got: f32) void {
        self.total += 1;
        if (ref == got) self.exact += 1;
        self.max_abs = @max(self.max_abs, @abs(ref - got));
        self.max_ref = @max(self.max_ref, @abs(ref));
    }

    fn compare(ref: []const f32, got: []const f32) Distance {
        var d = Distance{};
        for (ref, got) |r, g| d.add(r, g);
        return d;
    }

    fn report(self: Distance, label: []const u8) void {
        std.debug.print("  [facade] {s}: max|d|={e:.2} (max|ref|={d:.3}) exact {d}/{d}\n", .{ label, self.max_abs, self.max_ref, self.exact, self.total });
    }
};

fn fillUniform(buf: []f32, seed: u64, amp: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (buf) |*v| v.* = (random.float(f32) * 2 - 1) * amp;
}

// ---------------------------------------------------------------------------
// StreamConv vs groupedCausalConv1d(state) on the NAM conv shapes.
// ---------------------------------------------------------------------------

const ConvShape = struct { in: usize, out: usize, taps: usize, dilation: usize, bias: bool, groups: usize };

const conv_shapes = [_]ConvShape{
    .{ .in = 1, .out = 16, .taps = 1, .dilation = 1, .bias = false, .groups = 1 }, // rechannel
    .{ .in = 16, .out = 16, .taps = 3, .dilation = 1, .bias = true, .groups = 1 }, // standard array-0 dilated conv
    .{ .in = 16, .out = 16, .taps = 3, .dilation = 512, .bias = true, .groups = 1 },
    .{ .in = 16, .out = 32, .taps = 3, .dilation = 4, .bias = true, .groups = 1 }, // gated width
    .{ .in = 16, .out = 8, .taps = 1, .dilation = 1, .bias = true, .groups = 1 }, // head1x1 / rechannel
    .{ .in = 8, .out = 8, .taps = 3, .dilation = 16, .bias = true, .groups = 1 }, // array-1 conv (core fixed-shape path)
    .{ .in = 8, .out = 1, .taps = 1, .dilation = 1, .bias = true, .groups = 1 }, // head rechannel
    .{ .in = 8, .out = 16, .taps = 1, .dilation = 1, .bias = true, .groups = 1 }, // 0.7.0 head 1x16
    .{ .in = 4, .out = 8, .taps = 2, .dilation = 1, .bias = true, .groups = 2 }, // grouped
    .{ .in = 1, .out = 1, .taps = 64, .dilation = 1, .bias = true, .groups = 1 }, // LinearEngine / short FIR
};

test "facade: grouped causal conv matches StreamConv across NAM shapes and chunkings" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const total = 300;
    const chunks = [_]usize{ 64, 7, 33, 64, 64, 1, 64, 3 };
    for (conv_shapes, 0..) |shape, si| {
        const in_per_group = shape.in / shape.groups;
        const stream_len = shape.taps * in_per_group * shape.out + (if (shape.bias) shape.out else 0);
        const stream = try allocator.alloc(f32, stream_len);
        defer allocator.free(stream);
        fillUniform(stream, 100 + si, 0.5);

        var reference = try stream_conv.StreamConv.initGrouped(allocator, shape.in, shape.out, shape.taps, shape.dilation, shape.bias, shape.groups);
        defer reference.deinit();
        try std.testing.expectEqual(stream_len, reference.loadNamWeights(stream));
        var cursor: usize = 0;
        var candidate = try facade.Conv.init(allocator, &ctx, shape.in, shape.out, shape.taps, shape.dilation, shape.bias, shape.groups, 64, stream, &cursor);
        defer candidate.deinit();
        try std.testing.expectEqual(stream_len, cursor);

        const input = try allocator.alloc(f32, total * shape.in);
        defer allocator.free(input);
        fillUniform(input, 200 + si, 1.0);
        const expected = try allocator.alloc(f32, total * shape.out);
        defer allocator.free(expected);
        const got = try allocator.alloc(f32, total * shape.out);
        defer allocator.free(got);

        // Reference: one chunking. Candidate: a different, ragged chunking,
        // so the state hand-off is exercised at every boundary.
        var offset: usize = 0;
        while (offset < total) {
            const n = @min(@as(usize, 64), total - offset);
            reference.process(input[offset * shape.in ..], expected[offset * shape.out ..], n, false);
            reference.push(input[offset * shape.in ..], n);
            offset += n;
        }
        offset = 0;
        var ci: usize = 0;
        while (offset < total) : (ci += 1) {
            const n = @min(chunks[ci % chunks.len], total - offset);
            const mark = ctx.openExecScope();
            defer ctx.closeExecScope(mark);
            var x = try facade.TimeIn.fromBorrowedConstSlice(&ctx, .{ n, shape.in }, input[offset * shape.in ..][0 .. n * shape.in]);
            defer x.deinit();
            const y = try candidate.forward(&ctx, &x);
            try y.copyTo(got[offset * shape.out ..][0 .. n * shape.out]);
            offset += n;
        }

        const d = Distance.compare(expected, got);
        var label_buf: [96]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "conv {d}->{d} k{d} d{d} g{d}{s}", .{ shape.in, shape.out, shape.taps, shape.dilation, shape.groups, if (shape.bias) " +bias" else "" });
        d.report(label);
        try std.testing.expect(d.max_abs <= 4e-6 * @max(1.0, d.max_ref));
    }
}

// ---------------------------------------------------------------------------
// Whole WaveNet engines.
// ---------------------------------------------------------------------------

fn runReference(allocator: std.mem.Allocator, config: *const nam_file.WaveNetConfig, weights: []const f32, input: []const f32, output: []f32, block: usize) !void {
    var engine = try wavenet.WaveNetEngine.init(allocator, config, weights);
    defer engine.deinit();
    try engine.reset(block);
    var offset: usize = 0;
    while (offset < input.len) {
        const n = @min(block, input.len - offset);
        engine.process(input[offset..], output[offset..], n);
        offset += n;
    }
}

fn runFacade(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.WaveNetConfig, weights: []const f32, input: []const f32, output: []f32, block: usize) !void {
    var engine = try facade.WaveNet.init(allocator, ctx, config, weights, block);
    defer engine.deinit();
    engine.reset();
    var offset: usize = 0;
    while (offset < input.len) {
        const n = @min(block, input.len - offset);
        try engine.process(ctx, input[offset..], output[offset..], n);
        offset += n;
    }
}

fn expectWaveNetParity(allocator: std.mem.Allocator, ctx: *ExecContext, config: *const nam_file.WaveNetConfig, weights: []const f32, total: usize, label: []const u8) !void {
    const input = try allocator.alloc(f32, total);
    defer allocator.free(input);
    facade.fillSignal(input, 7);
    const expected = try allocator.alloc(f32, total);
    defer allocator.free(expected);
    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    const got_chunked = try allocator.alloc(f32, total);
    defer allocator.free(got_chunked);

    try runReference(allocator, config, weights, input, expected, 64);
    try runFacade(allocator, ctx, config, weights, input, got, 64);
    try runFacade(allocator, ctx, config, weights, input, got_chunked, 37);

    const d = Distance.compare(expected, got);
    d.report(label);
    // The upstream render golden gate is 5e-6 absolute on outputs of order 1.
    try std.testing.expect(d.max_abs <= 5e-6 * @max(1.0, d.max_ref));
    // Chunk independence of the facade path is exact, like the reference's.
    try std.testing.expectEqualSlices(f32, got, got_chunked);
    var energy: f64 = 0;
    for (expected) |v| energy += @as(f64, v) * v;
    try std.testing.expect(energy > 1e-9);
}

test "facade: wavenet matches the streaming engine on the upstream tiny model" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try nam_file.loadFromSlice(allocator, @embedFile("testdata/wavenet.nam"));
    defer model.deinit();
    try expectWaveNetParity(allocator, &ctx, &model.config.wavenet, model.weights, 999, "wavenet tiny (testdata/wavenet.nam)");
}

test "facade: wavenet matches the streaming engine on the standard 16/8 WaveNet" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const weights = try facade.syntheticWeights(allocator, &facade.standard_config, 11, 0.25);
    defer allocator.free(weights);
    try expectWaveNetParity(allocator, &ctx, &facade.standard_config, weights, 4096, "wavenet standard 16/8 x 10+10 (synthetic weights)");
}

// A2-style layer array: gated or blended, grouped convs, every FiLM slot on.
fn a2Layers(comptime gating: nam_file.GatingMode) [1]nam_file.WaveNetLayerArray {
    const dilations = [_]usize{ 1, 2 };
    const kernels = [_]usize{ 3, 2 };
    const acts = [_]Activation{ .{ .kind = .tanh }, .{ .kind = .fasttanh } };
    const secondary = [_]Activation{ .{ .kind = .sigmoid }, .{ .kind = .silu } };
    const gates = [_]nam_file.GatingMode{ gating, gating };
    return .{.{
        .input_size = 1,
        .condition_size = 1,
        .channels = 4,
        .bottleneck = 4,
        .head_out = 1,
        .head_kernel = 2,
        .head_bias = true,
        .dilations = &dilations,
        .kernel_sizes = &kernels,
        .activations = &acts,
        .gating_modes = &gates,
        .secondary_activations = &secondary,
        .layer1x1_active = true,
        .layer1x1_groups = 2,
        .head1x1_active = true,
        .head1x1_out = 2,
        .head1x1_groups = 1,
        .groups_input = 2,
        .groups_input_mixin = 1,
        .conv_pre_film = .{ .active = true, .shift = false },
        .conv_post_film = .{ .active = true, .shift = true },
        .input_mixin_pre_film = .{ .active = true, .shift = true },
        .input_mixin_post_film = .{ .active = true, .shift = false },
        .activation_pre_film = .{ .active = true, .shift = true },
        .activation_post_film = .{ .active = true, .shift = true },
        .layer1x1_post_film = .{ .active = true, .shift = true },
        .head1x1_post_film = .{ .active = true, .shift = true },
    }};
}

const a2_gated_layers = a2Layers(.gated);
const a2_blended_layers = a2Layers(.blended);
const a2_post_head_kernels = [_]usize{ 2, 1 };

fn a2Config(layers: []const nam_file.WaveNetLayerArray) nam_file.WaveNetConfig {
    return .{
        .layers = layers,
        .head = .{ .channels = 3, .out_channels = 1, .kernel_sizes = &a2_post_head_kernels, .activation = .{ .kind = .relu } },
        .head_scale = 0.5,
        .in_channels = 1,
        .condition_dsp = null,
    };
}

test "facade: wavenet matches the streaming engine on A2 gated and blended FiLM stacks" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    inline for (.{ .{ &a2_gated_layers, "wavenet A2 gated+FiLM+grouped+post-head" }, .{ &a2_blended_layers, "wavenet A2 blended+FiLM+grouped+post-head" } }) |case| {
        const config = a2Config(case[0]);
        const weights = try facade.syntheticWeights(allocator, &config, 23, 0.4);
        defer allocator.free(weights);
        try expectWaveNetParity(allocator, &ctx, &config, weights, 777, case[1]);
    }
}

// ---------------------------------------------------------------------------
// Steady-state allocation count of the facade block path.
// ---------------------------------------------------------------------------

test "facade: the wavenet block path stops allocating once the buffer pool is warm" {
    var counting = facade.CountingAllocator{ .backing = std.testing.allocator };
    const allocator = counting.allocator();
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    const weights = try facade.syntheticWeights(std.testing.allocator, &facade.standard_config, 11, 0.25);
    defer std.testing.allocator.free(weights);
    var engine = try facade.WaveNet.init(std.testing.allocator, &ctx, &facade.standard_config, weights, 64);
    defer engine.deinit();

    const block = 64;
    var input: [block]f32 = undefined;
    facade.fillSignal(&input, 3);
    var output: [block]f32 = undefined;
    for (0..32) |_| try engine.process(&ctx, &input, &output, block);
    const warm_allocs = counting.allocs.load(.monotonic);
    const warm_bytes = counting.bytes.load(.monotonic);
    for (0..32) |_| try engine.process(&ctx, &input, &output, block);
    const steady_allocs = counting.allocs.load(.monotonic) - warm_allocs;
    std.debug.print("  [facade] standard wavenet 64-frame block: {d} allocations / {d} bytes over the first 32 blocks, {d} allocations over the next 32\n", .{ warm_allocs, warm_bytes, steady_allocs });
    try std.testing.expectEqual(@as(usize, 0), steady_allocs);

    // The cab IR and the per-sample LSTM composition, same discipline.
    var ir: [512]f32 = undefined;
    fillUniform(&ir, 4, 0.5);
    var reference_cab = try ir_cab.IrCab.init(std.testing.allocator, &ir, 48000, 48000, block);
    defer reference_cab.deinit();
    var cab = try facade.IrCab.init(std.testing.allocator, &ctx, reference_cab.weight, block);
    defer cab.deinit();
    for (0..16) |_| try cab.process(&ctx, &input, &output, block);
    const cab_warm = counting.allocs.load(.monotonic);
    for (0..16) |_| try cab.process(&ctx, &input, &output, block);
    const cab_steady = counting.allocs.load(.monotonic) - cab_warm;

    const lstm_config = nam_file.LstmConfig{ .input_size = 1, .hidden_size = 8, .num_layers = 1, .in_channels = 1, .out_channels = 1 };
    const lstm_file_config = nam_file.Config{ .lstm = lstm_config };
    const lstm_weights = try std.testing.allocator.alloc(f32, nam_file.expectedWeightCount(&lstm_file_config));
    defer std.testing.allocator.free(lstm_weights);
    fillUniform(lstm_weights, 5, 0.3);
    var lstm_model = try lstm.Model.initFromNam(std.testing.allocator, &ctx, &lstm_config, lstm_weights, false, .{});
    defer lstm_model.deinit();
    var lstm_stream = try lstm.Stream.init(std.testing.allocator, &ctx, &lstm_model);
    defer lstm_stream.deinit();
    try lstm_stream.process(&ctx, &input, &output, block);
    const lstm_warm = counting.allocs.load(.monotonic);
    try lstm_stream.process(&ctx, &input, &output, block);
    const lstm_steady = counting.allocs.load(.monotonic) - lstm_warm;
    std.debug.print("  [facade] steady-state allocations: cab IR {d} per 16 blocks, lstm {d} per 64 samples\n", .{ cab_steady, lstm_steady });
    try std.testing.expectEqual(@as(usize, 0), cab_steady);
    try std.testing.expectEqual(@as(usize, 0), lstm_steady);
}

// ---------------------------------------------------------------------------
// The two tanh contracts side by side, and fast_tanh parity.
// ---------------------------------------------------------------------------

test "facade: core tanh and NAM tanhF32 against correctly rounded tanh" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const n = 20001;
    const xs = try allocator.alloc(f32, n);
    defer allocator.free(xs);
    for (xs, 0..) |*x, i| x.* = -10.0 + 20.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
    // Dense small-argument sweep where the two contracts branch differently.
    var small: [4001]f32 = undefined;
    for (&small, 0..) |*x, i| x.* = -0.5 + @as(f32, @floatFromInt(i)) * (1.0 / 4000.0);

    inline for (.{ .{ xs, "[-10, 10]" }, .{ &small, "[-0.5, 0.5]" } }) |sweep| {
        const values: []const f32 = sweep[0];
        var nam_err: f64 = 0;
        var core_err: f64 = 0;
        var agree: usize = 0;
        const core_vals = try allocator.alloc(f32, values.len);
        defer allocator.free(core_vals);
        {
            var t = try fucina.Tensor(.{.x}).fromSlice(&ctx, .{values.len}, values);
            defer t.deinit();
            var y = try t.tanh(&ctx);
            defer y.deinit();
            try y.copyTo(core_vals);
        }
        for (values, core_vals) |x, core| {
            const exact = std.math.tanh(@as(f64, x));
            const nam = activations.tanhF32(x);
            nam_err = @max(nam_err, @abs(@as(f64, nam) - exact));
            core_err = @max(core_err, @abs(@as(f64, core) - exact));
            if (nam == core) agree += 1;
        }
        std.debug.print("  [facade] tanh {s}: max|err| NAM tanhF32={e:.2} core tanh={e:.2}; bit-equal {d}/{d}\n", .{ sweep[1], nam_err, core_err, agree, values.len });
        try std.testing.expect(nam_err <= 3e-7);
        try std.testing.expect(core_err <= 1e-6);
    }

    // fast_tanh: the same rational approximation on both sides.
    var fast_agree: usize = 0;
    var fast_max: f32 = 0;
    {
        var t = try fucina.Tensor(.{.x}).fromSlice(&ctx, .{xs.len}, xs);
        defer t.deinit();
        var y = try t.fastTanh(&ctx);
        defer y.deinit();
        for (xs, try y.dataConst()) |x, core| {
            const nam = activations.fastTanh(x);
            if (nam == core) fast_agree += 1;
            fast_max = @max(fast_max, @abs(nam - core));
        }
    }
    std.debug.print("  [facade] fast_tanh [-10, 10]: bit-equal {d}/{d}, max|d|={e:.2}\n", .{ fast_agree, xs.len, fast_max });
    try std.testing.expect(fast_max <= 1e-6);
}

// ---------------------------------------------------------------------------
// IR cab as one long causal conv.
// ---------------------------------------------------------------------------

test "facade: cab IR through causalConv1d matches IrCab" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    for ([_]usize{ 3, 129, 4096 }) |taps| {
        const ir = try allocator.alloc(f32, taps);
        defer allocator.free(ir);
        fillUniform(ir, taps, 1.0);
        for (ir, 0..) |*v, i| v.* *= @exp(-@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(taps)) * 4.0);

        var reference = try ir_cab.IrCab.init(allocator, ir, 48000, 48000, 64);
        defer reference.deinit();
        var candidate = try facade.IrCab.init(allocator, &ctx, reference.weight, 64);
        defer candidate.deinit();

        const total = 1000;
        const input = try allocator.alloc(f32, total);
        defer allocator.free(input);
        facade.fillSignal(input, 5);
        const expected = try allocator.alloc(f32, total);
        defer allocator.free(expected);
        const got = try allocator.alloc(f32, total);
        defer allocator.free(got);
        var offset: usize = 0;
        while (offset < total) {
            const n = @min(@as(usize, 64), total - offset);
            reference.process(input[offset..], expected[offset..], n);
            try candidate.process(&ctx, input[offset..], got[offset..], n);
            offset += n;
        }
        const d = Distance.compare(expected, got);
        var label_buf: [48]u8 = undefined;
        d.report(try std.fmt.bufPrint(&label_buf, "cab IR {d} taps", .{taps}));
        try std.testing.expect(d.max_abs <= 4e-6 * @max(1.0, d.max_ref));
    }
}

// ---------------------------------------------------------------------------
// LSTM composed per sample from facade ops.
// ---------------------------------------------------------------------------

test "facade: per-sample LSTM composition matches LstmEngine" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const config = nam_file.LstmConfig{ .input_size = 1, .hidden_size = 8, .num_layers = 2, .in_channels = 1, .out_channels = 1 };
    const file_config = nam_file.Config{ .lstm = config };
    const count = nam_file.expectedWeightCount(&file_config);
    const weights = try allocator.alloc(f32, count);
    defer allocator.free(weights);
    fillUniform(weights, 31, 0.4);

    var reference = try models.LstmEngine.init(allocator, &config, weights, 48000);
    defer reference.deinit();
    var model = try lstm.Model.initFromNam(allocator, &ctx, &config, weights, false, .{});
    defer model.deinit();
    var candidate = try lstm.Stream.init(allocator, &ctx, &model);
    defer candidate.deinit();

    const total = 600;
    const input = try allocator.alloc(f32, total);
    defer allocator.free(input);
    facade.fillSignal(input, 9);
    const expected = try allocator.alloc(f32, total);
    defer allocator.free(expected);
    const got = try allocator.alloc(f32, total);
    defer allocator.free(got);
    reference.process(input, expected, total);
    try candidate.process(&ctx, input, got, total);
    const d = Distance.compare(expected, got);
    d.report("lstm hidden 8 x 2 layers (per-sample facade ops)");
    try std.testing.expect(d.max_abs <= 2e-5 * @max(1.0, d.max_ref));
}
