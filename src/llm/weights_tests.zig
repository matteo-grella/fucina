//! Behavioral tests for the LLM weight loader (`weights.zig`): resident-bf16
//! linear/embedding/fuse parity vs the f32-widened reference, GGUF-load
//! routing to the resident bf16 arm, and quantized-weight storage ownership
//! (a cloneView'd weight outlives the source it shares blocks with).
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("weights.zig");

const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;

const LinearWeight = weights.LinearWeight;
const WeightF32 = weights.WeightF32;
const fuseLinear = weights.fuseLinear;

test "loadMoeRhs rejects expert tensor dims that disagree with config" {
    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var info = gguf.TensorInfo{
        .name = "blk.0.ffn_gate_exps.weight",
        .dims = .{ 64, 32, 4, 0 },
        .n_dims = 3,
        .ggml_type = .q4_k,
        .offset = 0,
        .data = &.{},
    };

    try std.testing.expectError(error.InvalidWeightShape, weights.loadMoeRhs(&ctx, &info, 64, 64, 4, false));
    try std.testing.expectError(error.InvalidWeightShape, weights.loadMoeRhs(&ctx, &info, 32, 32, 4, false));
    try std.testing.expectError(error.InvalidWeightShape, weights.loadMoeRhs(&ctx, &info, 64, 32, 8, false));
}

test "LinearWeight.load q8_0: cloneView shares block storage and survives source deinit" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = 32; // one q8_0 block per row
    const out_dim = 4;
    const seq_len = 2;

    // d = 1.0 (f16 bits) with small-int qs: the dequantized weight IS qs, and
    // the LHS below quantizes exactly (per-block amax 127 -> scale 1), so the
    // integer reference is reached bitwise in f32.
    var blocks: [out_dim]fucina.BlockQ8_0 = undefined;
    for (&blocks, 0..) |*blk, o| {
        blk.d = 0x3C00;
        for (&blk.qs, 0..) |*q, j| q.* = @intCast(@as(i64, @intCast((o * 7 + j * 3) % 11)) - 5);
    }

    const info = gguf.TensorInfo{
        .name = "w",
        .dims = .{ in_dim, out_dim, 0, 0 },
        .n_dims = 2,
        .ggml_type = .q8_0,
        .offset = 0,
        .data = std.mem.sliceAsBytes(blocks[0..]),
    };

    // On gpu builds with a device, this load routes through the resident arm
    // (device-owned blocks); either way the clone must keep the shared block
    // storage alive after the source weight is gone.
    var source = try LinearWeight.load(&ctx, &info, out_dim, in_dim);
    var source_owned = true;
    defer if (source_owned) source.deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).q8_0, std.meta.activeTag(source));

    var clone = try source.cloneView(&ctx);
    defer clone.deinit();
    source.deinit();
    source_owned = false;

    var x_vals: [seq_len * in_dim]f32 = undefined;
    for (&x_vals, 0..) |*v, i| {
        // Per-block amax exactly 127 -> exact LHS q8_0 quantization (scale 1).
        v.* = if (i % in_dim == 0) 127 else @floatFromInt(@as(i64, @intCast((i * 5) % 9)) - 4);
    }
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
    defer x.deinit();

    var y = try clone.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();
    const yd = try y.dataConst();
    for (0..seq_len) |s| {
        for (0..out_dim) |o| {
            var expected: i64 = 0;
            for (0..in_dim) |j| {
                expected += @as(i64, @intFromFloat(x_vals[s * in_dim + j])) * blocks[o].qs[j];
            }
            try std.testing.expectEqual(@as(f32, @floatFromInt(expected)), yd[s * out_dim + o]);
        }
    }
}

/// Integer-valued q8_0 blocks (d = 1.0): the dequantized weight IS qs, so the
/// linear output is reached exactly in f32 (see the q8_0 test above).
fn testFillQ8_0Blocks(blocks: []fucina.BlockQ8_0, salt: usize) void {
    for (blocks, 0..) |*blk, o| {
        blk.d = 0x3C00;
        for (&blk.qs, 0..) |*q, j| q.* = @intCast(@as(i64, @intCast((o * 7 + j * 3 + salt) % 11)) - 5);
    }
}

test "loadForFusion q8_0: parts skip residency but the fused weight matches the separate loads" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = 32; // one q8_0 block per row
    const out_a = 4; // out dims stay x4-pack friendly (n % 4 == 0)
    const out_b = 8;

    var blocks_a: [out_a]fucina.BlockQ8_0 = undefined;
    testFillQ8_0Blocks(&blocks_a, 0);
    var blocks_b: [out_b]fucina.BlockQ8_0 = undefined;
    testFillQ8_0Blocks(&blocks_b, 4);

    const info_a = gguf.TensorInfo{
        .name = "a",
        .dims = .{ in_dim, out_a, 0, 0 },
        .n_dims = 2,
        .ggml_type = .q8_0,
        .offset = 0,
        .data = std.mem.sliceAsBytes(blocks_a[0..]),
    };
    const info_b = gguf.TensorInfo{
        .name = "b",
        .dims = .{ in_dim, out_b, 0, 0 },
        .n_dims = 2,
        .ggml_type = .q8_0,
        .offset = 0,
        .data = std.mem.sliceAsBytes(blocks_b[0..]),
    };

    // Fuse-destined parts (no device residency) + default-loaded references.
    var a = try LinearWeight.loadForFusion(&ctx, &info_a, out_a, in_dim);
    var b = try LinearWeight.loadForFusion(&ctx, &info_b, out_b, in_dim);
    var a_ref = try LinearWeight.load(&ctx, &info_a, out_a, in_dim);
    defer a_ref.deinit();
    var b_ref = try LinearWeight.load(&ctx, &info_b, out_b, in_dim);
    defer b_ref.deinit();

    var x_vals: [2 * in_dim]f32 = undefined;
    for (&x_vals, 0..) |*v, i| {
        // Per-block amax exactly 127 -> exact LHS q8_0 quantization (scale 1).
        v.* = if (i % in_dim == 0) 127 else @floatFromInt(@as(i64, @intCast((i * 5) % 9)) - 4);
    }
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 2, in_dim }, &x_vals);
    defer x.deinit();

    var y_a = try a_ref.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_a.deinit();
    var y_b = try b_ref.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_b.deinit();

    var fuse_parts = [_]*LinearWeight{ &a, &b };
    const maybe_fused = try fuseLinear(&ctx, &fuse_parts);
    if (maybe_fused == null) { // fuse must succeed; clean up if it ever doesn't
        a.deinit();
        b.deinit();
    }
    try std.testing.expect(maybe_fused != null);
    var fused = maybe_fused.?; // parts were consumed by the successful fuse
    defer fused.deinit();

    var y = try fused.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();
    const yd = try y.dataConst();
    const ad = try y_a.dataConst();
    const bd = try y_b.dataConst();
    const out_total = out_a + out_b;
    for (0..2) |s| {
        try std.testing.expectEqualSlices(f32, ad[s * out_a ..][0..out_a], yd[s * out_total ..][0..out_a]);
        try std.testing.expectEqualSlices(f32, bd[s * out_b ..][0..out_b], yd[s * out_total + out_a ..][0..out_b]);
    }
}

test "linearSeqQ5_K: compact decode route (m < 4) matches the packed path bitwise; m = 4 stays packed" {
    // Route-level A/B of the Q5_K decode gate: with the toggle forced ON the
    // m<4 inputs go through the compact GGUF-native blocks (weight.value ->
    // quantized-RHS dot), with it forced OFF through the byte-expanded packed
    // layout (dotPacked) — and the outputs must be BITWISE equal (the kernel
    // proof lives in q5_k_tests.zig "compact-vs-packed cross-layout"). m=4 is
    // the gate boundary: both toggles take the packed path there, pinning the
    // boundary row against a numerics-diverging change to either family.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = 512; // 2 Q5_K blocks per row
    const out_dim = 16; // packed Q5_Kx8 layout requires n % 8 == 0

    // Arbitrary-but-deterministic valid Q5_K encodings (any byte pattern is a
    // valid encoding; getScaleMinK4 decodes the scale bytes the same way on
    // both routes). dm holds raw f16 bits: ~0.1 / ~0.05.
    var blocks: [out_dim * (in_dim / 256)]fucina.BlockQ5_K = undefined;
    for (&blocks, 0..) |*b, bi| {
        b.dm = .{ 0x2E66, 0x2A66 };
        for (&b.scales, 0..) |*s, i| s.* = @intCast((i * 7 + bi * 3) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
        for (&b.qs, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
    }

    const info = gguf.TensorInfo{
        .name = "w",
        .dims = .{ in_dim, out_dim, 0, 0 },
        .n_dims = 2,
        .ggml_type = .q5_k,
        .offset = 0,
        .data = std.mem.sliceAsBytes(blocks[0..]),
    };
    var w = try LinearWeight.load(&ctx, &info, out_dim, in_dim);
    defer w.deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).q5_k, std.meta.activeTag(w));

    // Restore the env/arch default once done (the setter pre-seeds the
    // read-once gate cache; null resets it to unread).
    defer weights.setQ5kDecodeCompact(null);

    inline for ([_]usize{ 1, 2, 3, 4 }) |seq_len| {
        var x_vals: [seq_len * in_dim]f32 = undefined;
        for (&x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i64, @intCast((i * 17 + seq_len) % 251)) - 125);
        var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
        defer x.deinit();

        weights.setQ5kDecodeCompact(true);
        var y_compact = try w.linearSeq(&ctx, &x, .embed, .ffn);
        defer y_compact.deinit();

        weights.setQ5kDecodeCompact(false);
        var y_packed = try w.linearSeq(&ctx, &x, .embed, .ffn);
        defer y_packed.deinit();

        try std.testing.expectEqualSlices(f32, try y_packed.dataConst(), try y_compact.dataConst());
    }
}

test "linearSeqQ6_K: compact decode route (m < 4) matches the packed path bitwise; m = 4 stays packed" {
    // Q6_K ride-along of the Q5_K decode gate (same structure): toggle ON
    // routes m<4 through the compact GGUF-native blocks (weight.value ->
    // quantized-RHS dot), OFF through the byte-expanded packed Q6_Kx4 layout
    // (dotPacked) — outputs must be BITWISE equal (kernel proof in
    // q6_k_tests.zig "compact-vs-packed cross-layout"). m=4 pins the gate
    // boundary.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = 512; // 2 Q6_K blocks per row
    const out_dim = 16; // packed Q6_Kx4 layout requires n % 4 == 0

    // Arbitrary-but-deterministic valid Q6_K encodings; d holds raw f16 bits
    // (~0.1).
    var blocks: [out_dim * (in_dim / 256)]fucina.BlockQ6_K = undefined;
    for (&blocks, 0..) |*b, bi| {
        b.d = 0x2E66;
        for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + bi * 3) % 64)) - 32);
        for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 31 + bi * 5) % 256);
        for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 13 + bi * 11) % 256);
    }

    const info = gguf.TensorInfo{
        .name = "w",
        .dims = .{ in_dim, out_dim, 0, 0 },
        .n_dims = 2,
        .ggml_type = .q6_k,
        .offset = 0,
        .data = std.mem.sliceAsBytes(blocks[0..]),
    };
    var w = try LinearWeight.load(&ctx, &info, out_dim, in_dim);
    defer w.deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).q6_k, std.meta.activeTag(w));

    // Restore the env/arch default once done (the setter pre-seeds the
    // read-once gate cache; null resets it to unread).
    defer weights.setQ6kDecodeCompact(null);

    inline for ([_]usize{ 1, 2, 3, 4 }) |seq_len| {
        var x_vals: [seq_len * in_dim]f32 = undefined;
        for (&x_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i64, @intCast((i * 17 + seq_len) % 251)) - 125);
        var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
        defer x.deinit();

        weights.setQ6kDecodeCompact(true);
        var y_compact = try w.linearSeq(&ctx, &x, .embed, .ffn);
        defer y_compact.deinit();

        weights.setQ6kDecodeCompact(false);
        var y_packed = try w.linearSeq(&ctx, &x, .embed, .ffn);
        defer y_packed.deinit();

        try std.testing.expectEqualSlices(f32, try y_packed.dataConst(), try y_compact.dataConst());
    }
}

/// f32 -> bf16 bit truncation; exact (== round-to-nearest) for the
/// small-integer test values used here.
fn testBf16Bits(value: f32) u16 {
    return @truncate(@as(u32, @bitCast(value)) >> 16);
}

fn testIntegerFill(values: []f32, modulus: i64, bias: i64) void {
    for (values, 0..) |*v, i| {
        v.* = @floatFromInt(@as(i64, @intCast(i % @as(usize, @intCast(modulus)))) - bias);
    }
}

/// Build the same logical weight twice: the resident-bf16 arm and the
/// f32-widened reference arm (what `load` produced before bf16 residency).
fn testBf16AndF32Pair(ctx: *ExecContext, values: []const f32, out_dim: usize, in_dim: usize) ![2]LinearWeight {
    var w32 = try WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values);
    defer w32.deinit();
    var w_bf16 = try w32.to(ctx, .bf16);
    errdefer w_bf16.deinit();
    const w_ref = try WeightF32.fromSlice(ctx, .{ out_dim, in_dim }, values);
    return .{ .{ .bf16 = w_bf16 }, .{ .f32 = w_ref } };
}

test "linearSeq: resident bf16 weight matches the f32-widened reference bitwise" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const out_dim = 4;
    const in_dim = 64;
    const seq_len = 3;

    var w_vals: [out_dim * in_dim]f32 = undefined;
    testIntegerFill(&w_vals, 5, 2); // {-2..2}: exact in bf16
    var pair = try testBf16AndF32Pair(&ctx, &w_vals, out_dim, in_dim);
    defer pair[0].deinit();
    defer pair[1].deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).bf16, std.meta.activeTag(pair[0]));

    var x_vals: [seq_len * in_dim]f32 = undefined;
    testIntegerFill(&x_vals, 7, 3); // {-3..3}
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
    defer x.deinit();

    var y_bf16 = try pair[0].linearSeq(&ctx, &x, .embed, .ffn);
    defer y_bf16.deinit();
    var y_ref = try pair[1].linearSeq(&ctx, &x, .embed, .ffn);
    defer y_ref.deinit();
    try std.testing.expectEqualSlices(f32, try y_ref.dataConst(), try y_bf16.dataConst());

    // Decode shape (seq=1): the m=1 GEMV path of the same kernel.
    var x1 = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 1, in_dim }, x_vals[0..in_dim]);
    defer x1.deinit();
    var y1_bf16 = try pair[0].linearSeq(&ctx, &x1, .embed, .ffn);
    defer y1_bf16.deinit();
    var y1_ref = try pair[1].linearSeq(&ctx, &x1, .embed, .ffn);
    defer y1_ref.deinit();
    try std.testing.expectEqualSlices(f32, try y1_ref.dataConst(), try y1_bf16.dataConst());
}

test "fuseLinear: bf16 parts fuse on the out axis and match the separate projections" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const in_dim = 32;
    const out_a = 4;
    const out_b = 6;
    const seq_len = 2;

    var a_vals: [out_a * in_dim]f32 = undefined;
    testIntegerFill(&a_vals, 5, 2);
    var b_vals: [out_b * in_dim]f32 = undefined;
    testIntegerFill(&b_vals, 9, 4);

    var a32 = try WeightF32.fromSlice(&ctx, .{ out_a, in_dim }, &a_vals);
    defer a32.deinit();
    var b32 = try WeightF32.fromSlice(&ctx, .{ out_b, in_dim }, &b_vals);
    defer b32.deinit();
    var a = LinearWeight{ .bf16 = try a32.to(&ctx, .bf16) };
    var b = LinearWeight{ .bf16 = try b32.to(&ctx, .bf16) };

    var x_vals: [seq_len * in_dim]f32 = undefined;
    testIntegerFill(&x_vals, 7, 3);
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
    defer x.deinit();

    var y_a = try a.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_a.deinit();
    var y_b = try b.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_b.deinit();

    var fuse_parts = [_]*LinearWeight{ &a, &b };
    const maybe_fused = try fuseLinear(&ctx, &fuse_parts);
    if (maybe_fused == null) { // fuse must succeed; clean up if it ever doesn't
        a.deinit();
        b.deinit();
    }
    try std.testing.expect(maybe_fused != null);
    var fused = maybe_fused.?; // parts were consumed by the successful fuse
    defer fused.deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).bf16, std.meta.activeTag(fused));

    var y = try fused.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();
    const yd = try y.dataConst();
    const ad = try y_a.dataConst();
    const bd = try y_b.dataConst();
    const out_total = out_a + out_b;
    for (0..seq_len) |s| {
        try std.testing.expectEqualSlices(f32, ad[s * out_a ..][0..out_a], yd[s * out_total ..][0..out_a]);
        try std.testing.expectEqualSlices(f32, bd[s * out_b ..][0..out_b], yd[s * out_total + out_a ..][0..out_b]);
    }
}

fn testGgufWriteBytes(buf: []u8, offset: *usize, bytes: []const u8) !void {
    const end = try std.math.add(usize, offset.*, bytes.len);
    if (end > buf.len) return error.NoSpaceLeft;
    @memcpy(buf[offset.*..end], bytes);
    offset.* = end;
}

fn testGgufWriteInt(buf: []u8, offset: *usize, comptime Int: type, value: Int) !void {
    const end = try std.math.add(usize, offset.*, @sizeOf(Int));
    if (end > buf.len) return error.NoSpaceLeft;
    std.mem.writeInt(Int, buf[offset.*..end][0..@sizeOf(Int)], value, .little);
    offset.* = end;
}

test "GGUF load: bf16 tensors stay resident bf16 and linearSeq matches the widened reference" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const out_dim = 3;
    const in_dim = 8;
    var w_vals: [out_dim * in_dim]f32 = undefined;
    testIntegerFill(&w_vals, 5, 2);

    // Minimal GGUF (same fixture pattern as src/gguf.zig's parser tests):
    // one bf16 tensor "w", GGUF dims [in, out] -> logical [out, in].
    var raw: [256]u8 = undefined;
    @memset(&raw, 0);
    var offset: usize = 0;
    try testGgufWriteBytes(&raw, &offset, "GGUF");
    try testGgufWriteInt(&raw, &offset, u32, 3); // version
    try testGgufWriteInt(&raw, &offset, u64, 1); // tensor_count
    try testGgufWriteInt(&raw, &offset, u64, 0); // metadata_count
    try testGgufWriteInt(&raw, &offset, u64, 1); // name length
    try testGgufWriteBytes(&raw, &offset, "w");
    try testGgufWriteInt(&raw, &offset, u32, 2); // n_dims
    try testGgufWriteInt(&raw, &offset, u64, in_dim);
    try testGgufWriteInt(&raw, &offset, u64, out_dim);
    try testGgufWriteInt(&raw, &offset, u32, @intFromEnum(gguf.GgmlType.bf16));
    try testGgufWriteInt(&raw, &offset, u64, 0); // payload offset

    offset = std.mem.alignForward(usize, offset, 32);
    for (w_vals) |v| try testGgufWriteInt(&raw, &offset, u16, testBf16Bits(v));

    const owned = try allocator.dupe(u8, raw[0..offset]);
    var file = try gguf.File.parseOwned(allocator, owned);
    defer file.deinit();

    var weight = try LinearWeight.load(&ctx, try file.get("w"), out_dim, in_dim);
    defer weight.deinit();
    // The load must route to the RESIDENT bf16 arm, not widen to f32.
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).bf16, std.meta.activeTag(weight));

    var reference = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ out_dim, in_dim }, &w_vals) };
    defer reference.deinit();

    var x_vals: [2 * in_dim]f32 = undefined;
    testIntegerFill(&x_vals, 7, 3);
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 2, in_dim }, &x_vals);
    defer x.deinit();

    var y = try weight.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();
    var y_ref = try reference.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_ref.deinit();
    try std.testing.expectEqualSlices(f32, try y_ref.dataConst(), try y.dataConst());

    // Embedding-table read through the same loaded weight.
    const ids = [_]usize{ 2, 0, 1 };
    var rows = try weight.getRowsAs(&ctx, &ids, .embed);
    defer rows.deinit();
    var rows_ref = try reference.getRowsAs(&ctx, &ids, .embed);
    defer rows_ref.deinit();
    try std.testing.expectEqualSlices(f32, try rows_ref.dataConst(), try rows.dataConst());
}

test "toPtqtp: dense f32 decorates to dual planes; linearSeq matches the dequantized reference" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const out_dim = 8;
    const in_dim = 256; // one TQ2_0 block per weight row
    const seq_len = 3;

    var prng = std.Random.DefaultPrng.init(31);
    var w_vals: [out_dim * in_dim]f32 = undefined;
    for (&w_vals) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
    var weight = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ out_dim, in_dim }, &w_vals) };
    defer weight.deinit();

    const stats = try weight.toPtqtp(&ctx, .{});
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).ptqtp, std.meta.activeTag(weight));
    try std.testing.expect(weight.ptqtp.p2 != null);
    try std.testing.expectEqual(@as(usize, out_dim), weight.outDim());
    try std.testing.expectEqual(@as(usize, in_dim), weight.inDim());
    try std.testing.expect(stats.rel_frob_err > 0 and stats.rel_frob_err < 0.5);
    // Decoration is terminal: no double decoration.
    try std.testing.expect(!weight.ptqtpEligible());
    try std.testing.expectError(weights.Error.UnsupportedWeightType, weight.toPtqtp(&ctx, .{}));

    // getRowsAs sums the dequantized planes — that reconstruction is the
    // linear's ground truth for what inference should compute.
    var ids: [out_dim]usize = undefined;
    for (&ids, 0..) |*id, i| id.* = i;
    var w_hat_rows = try weight.getRowsAs(&ctx, &ids, .embed);
    defer w_hat_rows.deinit();
    const w_hat = try w_hat_rows.dataConst();

    var x_vals: [seq_len * in_dim]f32 = undefined;
    for (&x_vals) |*v| v.* = prng.random().floatNorm(f32);
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
    defer x.deinit();
    var y = try weight.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();
    const y_data = try y.dataConst();

    // linearSeq runs the deployed int8 path (Q8_K activations), so allow an
    // activation-quantization margin against the exact dequantized product.
    for (0..seq_len) |s| {
        for (0..out_dim) |o| {
            var want: f64 = 0;
            for (0..in_dim) |j| want += @as(f64, x_vals[s * in_dim + j]) * @as(f64, w_hat[o * in_dim + j]);
            try std.testing.expectApproxEqAbs(want, @as(f64, y_data[s * out_dim + o]), 0.05);
        }
    }
}

test "toPtqtp: rejects contract dims off the 256-block grid" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var w_vals: [4 * 128]f32 = undefined;
    @memset(&w_vals, 0.25);
    var weight = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ 4, 128 }, &w_vals) };
    defer weight.deinit();

    try std.testing.expect(!weight.ptqtpEligible());
    try std.testing.expectError(weights.Error.UnsupportedWeightType, weight.toPtqtp(&ctx, .{}));
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).f32, std.meta.activeTag(weight));
}

test "toPtqtp planes=3: triple-plane arm, linearSeq matches the three-plane reconstruction" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const out_dim = 4;
    const in_dim = 256;
    var prng = std.Random.DefaultPrng.init(303);
    var w_vals: [out_dim * in_dim]f32 = undefined;
    for (&w_vals) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
    var weight = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ out_dim, in_dim }, &w_vals) };
    defer weight.deinit();

    const stats = try weight.toPtqtp(&ctx, .{ .planes = 3 });
    try std.testing.expect(weight.ptqtp.p2 != null);
    try std.testing.expect(weight.ptqtp.p3 != null);
    try std.testing.expect(stats.rel_frob_err < 0.15); // ~3x under the dual bound

    var ids: [out_dim]usize = undefined;
    for (&ids, 0..) |*id, i| id.* = i;
    var w_hat_rows = try weight.getRowsAs(&ctx, &ids, .embed);
    defer w_hat_rows.deinit();
    const w_hat = try w_hat_rows.dataConst();

    var x_vals: [in_dim]f32 = undefined;
    for (&x_vals) |*v| v.* = prng.random().floatNorm(f32);
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 1, in_dim }, &x_vals);
    defer x.deinit();
    var y = try weight.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();
    const y_data = try y.dataConst();
    for (0..out_dim) |o| {
        var want: f64 = 0;
        for (0..in_dim) |j| want += @as(f64, x_vals[j]) * @as(f64, w_hat[o * in_dim + j]);
        try std.testing.expectApproxEqAbs(want, @as(f64, y_data[o]), 0.05);
    }
}

test "tie-fitted ptqtp serves the folded one-pass semantics" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const out_dim = 8;
    const in_dim = 512;
    const seq_len = 2;
    var prng = std.Random.DefaultPrng.init(2727);
    const w_vals = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(w_vals);
    for (w_vals) |*v| v.* = prng.random().floatNorm(f32) * 0.05;

    var weight = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ out_dim, in_dim }, w_vals) };
    defer weight.deinit();
    _ = try weight.toPtqtp(&ctx, .{ .planes = 2, .tie_scales = true });

    // The wiring pin: the tie survives decoration and the folded pack built.
    try std.testing.expect(weight.ptqtp.tied);
    try std.testing.expect(weight.ptqtp.pfold != null);

    var x_vals: [seq_len * in_dim]f32 = undefined;
    for (&x_vals) |*v| v.* = prng.random().floatNorm(f32);
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
    defer x.deinit();

    var y = try weight.linearSeq(&ctx, &x, .embed, .ffn);
    defer y.deinit();

    // Reference: the folded kernel run directly over the same pack with the
    // same activation quantization — the fused dispatch must be bitwise
    // equal to it under any column partition.
    const bpr = in_dim / 256;
    const qlhs = try allocator.alloc(fucina.internal.backend_mod.quantized_matmul.BlockQ8_K, seq_len * bpr);
    defer allocator.free(qlhs);
    for (0..seq_len) |r| {
        try fucina.internal.backend_mod.quantized_matmul.quantizeRowQ8_KInto(qlhs[r * bpr ..][0..bpr], x_vals[r * in_dim ..][0..in_dim]);
    }
    const want = try allocator.alloc(f32, seq_len * out_dim);
    defer allocator.free(want);
    fucina.internal.backend_mod.quantized_matmul.matmulTQ2_0FoldedX4RhsRange(want, qlhs, weight.ptqtp.pfold.?, bpr, out_dim, 0, seq_len);
    try std.testing.expectEqualSlices(f32, want, try y.dataConst());
}

test "fused ptqtp linear is bitwise identical to the per-plane facade chain" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const out_dim = 96; // above the column-chunk gate when threaded
    const in_dim = 512;
    const seq_len = 3;
    var prng = std.Random.DefaultPrng.init(1717);
    const w_vals = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(w_vals);
    for (w_vals) |*v| v.* = prng.random().floatNorm(f32) * 0.05;

    for ([_]u8{ 1, 2, 3 }) |planes| {
        var weight = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ out_dim, in_dim }, w_vals) };
        defer weight.deinit();
        _ = try weight.toPtqtp(&ctx, .{ .planes = planes });

        var x_vals: [seq_len * in_dim]f32 = undefined;
        for (&x_vals) |*v| v.* = prng.random().floatNorm(f32);
        var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq_len, in_dim }, &x_vals);
        defer x.deinit();

        // The deployed path (fused single-dispatch).
        var y = try weight.linearSeq(&ctx, &x, .embed, .ffn);
        defer y.deinit();

        // Reference: the equivalent per-plane facade dot chain.
        var p1 = try weight.ptqtp.p1.withTags(&ctx, .{ .ffn, .embed });
        defer p1.deinit();
        var acc = try x.dot(&ctx, &p1, .embed);
        defer acc.deinit();
        inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
            if (@field(weight.ptqtp, plane_field)) |*plane| {
                var tagged = try plane.withTags(&ctx, .{ .ffn, .embed });
                defer tagged.deinit();
                var yp = try x.dot(&ctx, &tagged, .embed);
                defer yp.deinit();
                const sum = try acc.add(&ctx, &yp);
                acc.deinit();
                acc = sum;
            }
        }
        try std.testing.expectEqualSlices(f32, try acc.dataConst(), try y.dataConst());
    }
}
