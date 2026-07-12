//! Behavioral tests for PTQTP GGUF persistence (`ptqtp_gguf.zig`): plane
//! tensors replace decorated bases byte-exactly (undecorated tensors and
//! metadata pass through verbatim), loader pair-detection reconstructs the
//! decorated weight bitwise (plane bytes and the served linear output),
//! fused weights row-slice to per-source planes that re-fuse losslessly,
//! and save→load→save is byte-stable.
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("weights.zig");
const ptqtp_gguf = @import("ptqtp_gguf.zig");

const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;
const WeightF32 = weights.WeightF32;

const in_dim = fucina.ptqtp.block_len; // one TQ2_0 block per row

fn testWeightValues(comptime len: usize, salt: usize) [len]f32 {
    var values: [len]f32 = undefined;
    for (&values, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt((i * 7 + salt * 13) % 23)) - 11.0) / 4.0;
    }
    return values;
}

/// A parsed three-tensor source file: `a.weight` [4, 256], `b.weight`
/// [2, 256], `c.weight` [3, 256], all f32, plus marker metadata. Caller
/// deinits.
fn buildSourceFile(allocator: std.mem.Allocator) !gguf.File {
    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    try w.addMetaString("general.architecture", "qwen3");
    try w.addMetaString("general.name", "ptqtp-persistence-test");

    const a_vals = testWeightValues(4 * in_dim, 1);
    try w.addTensor("a.weight", .f32, &.{ in_dim, 4 }, std.mem.sliceAsBytes(&a_vals));
    const b_vals = testWeightValues(2 * in_dim, 2);
    try w.addTensor("b.weight", .f32, &.{ in_dim, 2 }, std.mem.sliceAsBytes(&b_vals));
    const c_vals = testWeightValues(3 * in_dim, 4);
    try w.addTensor("c.weight", .f32, &.{ in_dim, 3 }, std.mem.sliceAsBytes(&c_vals));

    var buf: [32768]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    return gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
}

fn decorateFromFile(ctx: *ExecContext, file: *const gguf.File, name: []const u8, rows: usize, planes: u8) !LinearWeight {
    var weight = try LinearWeight.load(ctx, try file.get(name), rows, in_dim);
    errdefer weight.deinit();
    _ = try weight.toPtqtp(ctx, .{ .planes = planes });
    return weight;
}

fn planeBytes(weight: *const LinearWeight, plane: usize) ![]const u8 {
    const arm = &weight.ptqtp;
    const tensor = switch (plane) {
        0 => &arm.p1,
        1 => &(arm.p2.?),
        2 => &(arm.p3.?),
        else => unreachable,
    };
    return std.mem.sliceAsBytes(try tensor.dataConst());
}

fn saveToBuffer(allocator: std.mem.Allocator, src: *const gguf.File, entries: []const ptqtp_gguf.SaveEntry, buf: []u8) ![]const u8 {
    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    _ = try ptqtp_gguf.build(allocator, src, entries, .{}, &w);
    var sink = std.Io.Writer.fixed(buf);
    try w.finish(&sink);
    return sink.buffered();
}

test "save: planes replace the decorated base; the rest passes through byte-verbatim" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer decorated.deinit();

    var buf: [16384]u8 = undefined;
    const saved = try saveToBuffer(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated },
    }, &buf);

    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, saved));
    defer out.deinit();

    try std.testing.expectEqual(@as(?i64, ptqtp_gguf.format_version), out.getInt(ptqtp_gguf.version_key));
    try std.testing.expectEqualStrings("ptqtp-persistence-test", out.getString("general.name").?);

    try std.testing.expectEqual(@as(?*const gguf.TensorInfo, null), out.maybeGet("a.weight"));
    const p0 = try out.get("a.weight.ptqtp0");
    try std.testing.expectEqual(gguf.GgmlType.tq2_0, p0.ggml_type);
    try std.testing.expectEqualSlices(usize, &.{ 4, in_dim }, &(try p0.logicalMatrixShape()));
    try std.testing.expectEqualSlices(u8, try planeBytes(&decorated, 0), p0.data);
    const p1 = try out.get("a.weight.ptqtp1");
    try std.testing.expectEqualSlices(u8, try planeBytes(&decorated, 1), p1.data);
    try std.testing.expectEqual(@as(?*const gguf.TensorInfo, null), out.maybeGet("a.weight.ptqtp2"));

    const b_src = try src.get("b.weight");
    const b_out = try out.get("b.weight");
    try std.testing.expectEqual(b_src.ggml_type, b_out.ggml_type);
    try std.testing.expectEqualSlices(u8, b_src.data, b_out.data);
}

test "load: pair-detection rebuilds the decorated weight bitwise, undecorated names fall through" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer decorated.deinit();

    var buf: [16384]u8 = undefined;
    const saved = try saveToBuffer(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated },
    }, &buf);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, saved));
    defer out.deinit();

    // The source file carries no version key: detection is a no-op there.
    try std.testing.expectEqual(@as(?LinearWeight, null), try ptqtp_gguf.maybeLoadPlanes(&ctx, &src, "a.weight", 4, in_dim));
    // Undecorated names inside a decorated file fall through to the base.
    try std.testing.expectEqual(@as(?LinearWeight, null), try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "b.weight", 2, in_dim));

    var loaded = (try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "a.weight", 4, in_dim)).?;
    defer loaded.deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).ptqtp, std.meta.activeTag(loaded));
    try std.testing.expectEqual(@as(usize, 2), loaded.ptqtp.planeCount());
    try std.testing.expectEqualSlices(u8, try planeBytes(&decorated, 0), try planeBytes(&loaded, 0));
    try std.testing.expectEqualSlices(u8, try planeBytes(&decorated, 1), try planeBytes(&loaded, 1));

    // Served output parity: identical planes through the identical path.
    const x_vals = testWeightValues(2 * in_dim, 3);
    var x = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 2, in_dim }, &x_vals);
    defer x.deinit();
    var y_decorated = try decorated.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_decorated.deinit();
    var y_loaded = try loaded.linearSeq(&ctx, &x, .embed, .ffn);
    defer y_loaded.deinit();
    try std.testing.expectEqualSlices(f32, try y_decorated.dataConst(), try y_loaded.dataConst());
}

test "fused weight round-trip: row-sliced planes equal per-part decoration and re-fuse losslessly (3 parts, qkv shape)" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();

    // A fused [9, 256] matrix whose rows are a.weight (4) ++ b.weight (2)
    // ++ c.weight (3), decorated as one weight — the in-memory shape a
    // fused q/k/v projection has after decoratePtqtp.
    const a_vals = testWeightValues(4 * in_dim, 1);
    const b_vals = testWeightValues(2 * in_dim, 2);
    const c_vals = testWeightValues(3 * in_dim, 4);
    var fused_vals: [9 * in_dim]f32 = undefined;
    @memcpy(fused_vals[0 .. 4 * in_dim], &a_vals);
    @memcpy(fused_vals[4 * in_dim .. 6 * in_dim], &b_vals);
    @memcpy(fused_vals[6 * in_dim ..], &c_vals);
    var fused = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ 9, in_dim }, &fused_vals) };
    defer fused.deinit();
    _ = try fused.toPtqtp(&ctx, .{ .planes = 2 });

    var buf: [32768]u8 = undefined;
    const saved = try saveToBuffer(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &fused, .row0 = 0, .rows = 4 },
        .{ .name = "b.weight", .weight = &fused, .row0 = 4, .rows = 2 },
        .{ .name = "c.weight", .weight = &fused, .row0 = 6, .rows = 3 },
    }, &buf);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, saved));
    defer out.deinit();

    // Group independence: the fused rows' planes are byte-identical to
    // decorating each part alone — the property the on-disk per-source
    // naming rests on. Checked for the first and last slice.
    var a_alone = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer a_alone.deinit();
    const a_p0 = try out.get("a.weight.ptqtp0");
    try std.testing.expectEqualSlices(u8, try planeBytes(&a_alone, 0), a_p0.data);
    const a_p1 = try out.get("a.weight.ptqtp1");
    try std.testing.expectEqualSlices(u8, try planeBytes(&a_alone, 1), a_p1.data);
    var c_alone = try decorateFromFile(&ctx, &src, "c.weight", 3, 2);
    defer c_alone.deinit();
    const c_p0 = try out.get("c.weight.ptqtp0");
    try std.testing.expectEqualSlices(u8, try planeBytes(&c_alone, 0), c_p0.data);

    // Loading all three parts and re-fusing reproduces the fused planes
    // bitwise — the 3-part arm AttentionProjection.load takes for qkv.
    var a_loaded = (try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "a.weight", 4, in_dim)).?;
    errdefer a_loaded.deinit();
    var b_loaded = (try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "b.weight", 2, in_dim)).?;
    errdefer b_loaded.deinit();
    var c_loaded = (try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "c.weight", 3, in_dim)).?;
    errdefer c_loaded.deinit();
    var fuse_parts = [_]*LinearWeight{ &a_loaded, &b_loaded, &c_loaded };
    var refused = (try weights.fuseLinear(&ctx, &fuse_parts)).?;
    defer refused.deinit();
    try std.testing.expectEqual(std.meta.Tag(LinearWeight).ptqtp, std.meta.activeTag(refused));
    try std.testing.expectEqualSlices(u8, try planeBytes(&fused, 0), try planeBytes(&refused, 0));
    try std.testing.expectEqualSlices(u8, try planeBytes(&fused, 1), try planeBytes(&refused, 1));
}

test "resave of a loaded decorated file is byte-identical" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer decorated.deinit();

    var buf_first: [16384]u8 = undefined;
    const first = try saveToBuffer(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated },
    }, &buf_first);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, first));
    defer out.deinit();

    var loaded = (try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "a.weight", 4, in_dim)).?;
    defer loaded.deinit();

    var buf_second: [16384]u8 = undefined;
    const second = try saveToBuffer(allocator, &out, &.{
        .{ .name = "a.weight", .weight = &loaded },
    }, &buf_second);
    try std.testing.expectEqualSlices(u8, first, second);
}

test "appended entries and SaveReport accounting; undecorated saves get no version stamp" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var a_decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer a_decorated.deinit();
    var b_plain = try LinearWeight.load(&ctx, try src.get("b.weight"), 2, in_dim);
    defer b_plain.deinit();

    // A decorated weight with no base tensor in the source — the decorated
    // head of a tied-embedding model.
    const head_vals = testWeightValues(4 * in_dim, 9);
    var head = LinearWeight{ .f32 = try WeightF32.fromSlice(&ctx, .{ 4, in_dim }, &head_vals) };
    defer head.deinit();
    _ = try head.toPtqtp(&ctx, .{ .planes = 2 });

    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    const report = try ptqtp_gguf.build(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &a_decorated },
        .{ .name = "head.weight", .weight = &head },
        // Non-ptqtp entry: ignored, its base passes through verbatim.
        .{ .name = "b.weight", .weight = &b_plain },
    }, .{}, &w);
    try std.testing.expectEqual(@as(usize, 2), report.decorated);
    try std.testing.expectEqual(@as(usize, 4), report.planes);
    try std.testing.expectEqual(@as(usize, 2), report.passthrough); // b, c
    try std.testing.expectEqual(@as(usize, 1), report.appended);

    var buf: [32768]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer out.deinit();

    // Appended planes land after every source-order tensor and load back.
    try std.testing.expectEqualStrings("head.weight.ptqtp0", out.tensors[out.tensors.len - 2].name);
    try std.testing.expectEqualStrings("head.weight.ptqtp1", out.tensors[out.tensors.len - 1].name);
    var head_loaded = (try ptqtp_gguf.maybeLoadPlanes(&ctx, &out, "head.weight", 4, in_dim)).?;
    defer head_loaded.deinit();
    try std.testing.expectEqualSlices(u8, try planeBytes(&head, 0), try planeBytes(&head_loaded, 0));

    // Nothing decorated -> pure re-emit, no PTQTP format claim.
    const plain = try saveToBuffer(allocator, &src, &.{
        .{ .name = "b.weight", .weight = &b_plain },
    }, &buf);
    var plain_out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, plain));
    defer plain_out.deinit();
    try std.testing.expectEqual(@as(?i64, null), plain_out.getInt(ptqtp_gguf.version_key));
    try std.testing.expectEqual(@as(?LinearWeight, null), try ptqtp_gguf.maybeLoadPlanes(&ctx, &plain_out, "b.weight", 2, in_dim));
}

test "save entry validation: duplicates, row windows, base-shape mismatch" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer decorated.deinit();

    var w1 = gguf.Writer.init(allocator);
    defer w1.deinit();
    try std.testing.expectError(ptqtp_gguf.Error.DuplicateSaveEntry, ptqtp_gguf.build(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated },
        .{ .name = "a.weight", .weight = &decorated },
    }, .{}, &w1));

    var w2 = gguf.Writer.init(allocator);
    defer w2.deinit();
    try std.testing.expectError(ptqtp_gguf.Error.InvalidRowRange, ptqtp_gguf.build(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated, .row0 = 2, .rows = 3 },
    }, .{}, &w2));

    // Row window valid for the weight but disagreeing with the base tensor
    // it would replace — the row-range bookkeeping guard.
    var w3 = gguf.Writer.init(allocator);
    defer w3.deinit();
    try std.testing.expectError(ptqtp_gguf.Error.PlaneShapeMismatch, ptqtp_gguf.build(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated, .row0 = 0, .rows = 2 },
    }, .{}, &w3));
}

test "load validation: broken plane sets and wrong plane dtypes are refused" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer decorated.deinit();
    var buf: [32768]u8 = undefined;
    const saved = try saveToBuffer(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &decorated },
    }, &buf);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, saved));
    defer out.deinit();

    // ptqtp2 without ptqtp1: planes must fill in order.
    var w1 = gguf.Writer.init(allocator);
    defer w1.deinit();
    try w1.copyAllMetadata(&out, &.{});
    const p0 = try out.get("a.weight.ptqtp0");
    const p1 = try out.get("a.weight.ptqtp1");
    try w1.addTensor("a.weight.ptqtp0", .tq2_0, p0.dims[0..p0.n_dims], p0.data);
    try w1.addTensor("a.weight.ptqtp2", .tq2_0, p1.dims[0..p1.n_dims], p1.data);
    var sink1 = std.Io.Writer.fixed(&buf);
    try w1.finish(&sink1);
    var broken = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink1.buffered()));
    defer broken.deinit();
    try std.testing.expectError(ptqtp_gguf.Error.InvalidPlaneSet, ptqtp_gguf.maybeLoadPlanes(&ctx, &broken, "a.weight", 4, in_dim));

    // A `.ptqtp0` tensor that is not TQ2_0.
    var w2 = gguf.Writer.init(allocator);
    defer w2.deinit();
    try w2.copyAllMetadata(&out, &.{});
    const f32_vals = testWeightValues(4 * in_dim, 5);
    try w2.addTensor("a.weight.ptqtp0", .f32, &.{ in_dim, 4 }, std.mem.sliceAsBytes(&f32_vals));
    var sink2 = std.Io.Writer.fixed(&buf);
    try w2.finish(&sink2);
    var wrong_type = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink2.buffered()));
    defer wrong_type.deinit();
    try std.testing.expectError(ptqtp_gguf.Error.PlaneTypeMismatch, ptqtp_gguf.maybeLoadPlanes(&ctx, &wrong_type, "a.weight", 4, in_dim));
}

test "SaveOptions.header_bytes reproduces the default metadata copy byte-for-byte" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();
    var decorated = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer decorated.deinit();
    const entries = [_]ptqtp_gguf.SaveEntry{.{ .name = "a.weight", .weight = &decorated }};

    var buf_default: [32768]u8 = undefined;
    const default_bytes = try saveToBuffer(allocator, &src, &entries, &buf_default);

    // The takeMapping seam: metadata is read from the explicit region
    // instead of `src.bytes`.
    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    _ = try ptqtp_gguf.build(allocator, &src, &entries, .{ .header_bytes = src.bytes }, &w);
    var buf_explicit: [32768]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf_explicit);
    try w.finish(&sink);
    try std.testing.expectEqualSlices(u8, default_bytes, sink.buffered());
}

test "fuseLinear: mixed ptqtp plane counts stay separate; version guard refuses newer files" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var src = try buildSourceFile(allocator);
    defer src.deinit();

    var k2 = try decorateFromFile(&ctx, &src, "a.weight", 4, 2);
    defer k2.deinit();
    var k1 = try decorateFromFile(&ctx, &src, "b.weight", 2, 1);
    defer k1.deinit();
    var fuse_parts = [_]*LinearWeight{ &k2, &k1 };
    try std.testing.expectEqual(@as(?LinearWeight, null), try weights.fuseLinear(&ctx, &fuse_parts));

    // A decorated file claiming a newer format version must refuse to load.
    var buf: [16384]u8 = undefined;
    const saved = try saveToBuffer(allocator, &src, &.{
        .{ .name = "a.weight", .weight = &k2 },
    }, &buf);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, saved));
    defer out.deinit();

    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    try w.copyAllMetadata(&out, &.{});
    try w.addMetaInt(ptqtp_gguf.version_key, u32, ptqtp_gguf.format_version + 1);
    for (out.tensors) |*info| {
        try w.addTensor(info.name, info.ggml_type, info.dims[0..info.n_dims], info.data);
    }
    var buf_newer: [16384]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf_newer);
    try w.finish(&sink);
    var newer = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer newer.deinit();
    try std.testing.expectError(ptqtp_gguf.Error.UnsupportedPtqtpVersion, ptqtp_gguf.maybeLoadPlanes(&ctx, &newer, "a.weight", 4, in_dim));
}

test "quantizeMoeStack: expert row-blocks equal direct per-expert quantizeMatrix; streamed write loads back bitwise through maybeLoadMoeRhs" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const n_expert: usize = 3;
    const out_dim: usize = 2;
    const options = fucina.ptqtp.Options{ .planes = 3 };

    // Expert-major contiguous f32 stack — the raw bytes of a GGUF
    // `[in, out, n_expert]` `*_exps.weight` tensor.
    const stack_vals = testWeightValues(n_expert * out_dim * in_dim, 6);

    var quant = try ptqtp_gguf.quantizeMoeStack(&ctx, .f32, std.mem.sliceAsBytes(&stack_vals), in_dim, out_dim, n_expert, options, false);
    defer quant.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), quant.plane_count);

    // Group independence: each expert's row-block in each plane is
    // byte-identical to quantizing that expert's [out x in] matrix alone —
    // the slicing contract loadMoeRhsPtqtp and the ExpertStore rely on.
    const blocks_per_expert = out_dim * (in_dim / fucina.ptqtp.block_len);
    var rel_sum: f64 = 0;
    var rel_max: f64 = 0;
    for (0..n_expert) |e| {
        var pair = try fucina.ptqtp.quantizeMatrix(&ctx, stack_vals[e * out_dim * in_dim ..][0 .. out_dim * in_dim], out_dim, in_dim, options);
        defer pair.deinit(allocator);
        const dst0 = e * blocks_per_expert;
        const expert_planes = [3][]const fucina.BlockTQ2_0{ pair.plane1, pair.plane2, pair.plane3 };
        for (expert_planes, 0..) |expert_plane, p| {
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expert_plane), std.mem.sliceAsBytes(quant.planes[p][dst0..][0..blocks_per_expert]));
        }
        rel_sum += pair.stats.rel_frob_err;
        rel_max = @max(rel_max, pair.stats.rel_frob_err);
    }
    try std.testing.expectApproxEqAbs(rel_sum / @as(f64, @floatFromInt(n_expert)), quant.stats.mean_rel_err, 1e-12);
    try std.testing.expectEqual(rel_max, quant.stats.max_rel_err);

    // Persist the plane stacks exactly as export-gguf --ptqtp does
    // (declareTensor with the base 3D dims + beginStream) and load back
    // through the MoE pair-detection: the round-trip is bitwise.
    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    try w.addMetaString("general.architecture", "qwen3");
    try w.addMetaInt(ptqtp_gguf.version_key, u32, ptqtp_gguf.format_version);
    var name_buf: [64]u8 = undefined;
    for (0..quant.plane_count) |p| {
        try w.declareTensor(try ptqtp_gguf.planeName(&name_buf, "e.weight", p), .tq2_0, &.{ in_dim, out_dim, n_expert });
    }
    var buf: [32768]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    var streamer = try w.beginStream(&sink);
    for (quant.planes[0..quant.plane_count]) |plane| try streamer.writeTensorData(std.mem.sliceAsBytes(plane));
    try streamer.finish();
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer out.deinit();

    var loaded = (try ptqtp_gguf.maybeLoadMoeRhs(&ctx, &out, "e.weight", in_dim, out_dim, n_expert, false)).?;
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, 3), loaded.ptqtp.plane_count);
    for (0..quant.plane_count) |p| {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(quant.planes[p]), std.mem.sliceAsBytes(loaded.ptqtp.planes[p]));
    }
}

test "quantizeMoeStack: quantized sources slice per expert; bad geometry is refused" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const n_expert: usize = 2;
    const out_dim: usize = 2;
    const options = fucina.ptqtp.Options{ .planes = 2 };
    const expert_len = out_dim * in_dim;

    const stack_vals = testWeightValues(n_expert * expert_len, 8);
    const encoded = try allocator.alloc(u8, try gguf.tensorByteLen(.q8_0, &.{ in_dim, out_dim, n_expert }));
    defer allocator.free(encoded);
    try gguf.encodeF32(.q8_0, &stack_vals, encoded);

    var quant = try ptqtp_gguf.quantizeMoeStack(&ctx, .q8_0, encoded, in_dim, out_dim, n_expert, options, false);
    defer quant.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), quant.plane_count);
    try std.testing.expectEqual(@as(usize, 0), quant.planes[2].len);

    // Per-expert slicing of a block-quantized source: decode the expert's
    // byte window directly and quantize it alone — must match bitwise.
    const expert_bytes = try gguf.tensorByteLen(.q8_0, &.{ in_dim, out_dim });
    const blocks_per_expert = out_dim * (in_dim / fucina.ptqtp.block_len);
    const decoded = try allocator.alloc(f32, expert_len);
    defer allocator.free(decoded);
    for (0..n_expert) |e| {
        try gguf.decodeF32(.q8_0, encoded[e * expert_bytes ..][0..expert_bytes], decoded);
        var pair = try fucina.ptqtp.quantizeMatrix(&ctx, decoded, out_dim, in_dim, options);
        defer pair.deinit(allocator);
        const dst0 = e * blocks_per_expert;
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(pair.plane1), std.mem.sliceAsBytes(quant.planes[0][dst0..][0..blocks_per_expert]));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(pair.plane2), std.mem.sliceAsBytes(quant.planes[1][dst0..][0..blocks_per_expert]));
    }

    // Geometry refusals: contract dim not a 256 multiple; byte length that
    // disagrees with [in, out, n_expert].
    try std.testing.expectError(fucina.ptqtp.Error.InvalidShape, ptqtp_gguf.quantizeMoeStack(&ctx, .f32, std.mem.sliceAsBytes(&stack_vals), in_dim / 2, out_dim * 2, n_expert, options, false));
    try std.testing.expectError(fucina.ptqtp.Error.InvalidShape, ptqtp_gguf.quantizeMoeStack(&ctx, .f32, std.mem.sliceAsBytes(&stack_vals), in_dim, out_dim, n_expert + 1, options, false));
}

test "MoE expert stacks: plane pair-detection loads the ptqtp arm; streamed ProjSpec gathers sibling planes" {
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const n_expert: usize = 2;
    const out_dim: usize = 2;
    const rows = n_expert * out_dim;

    // Patterned, byte-valid TQ2_0 plane stacks (the loader checks dtype and
    // geometry; crumb values only need to be in 0..2).
    var p0_blocks: [rows]fucina.BlockTQ2_0 = undefined;
    var p1_blocks: [rows]fucina.BlockTQ2_0 = undefined;
    for (&p0_blocks, 0..) |*b, i| {
        b.d = @bitCast(@as(f16, @floatCast(0.5 + 0.01 * @as(f32, @floatFromInt(i)))));
        for (&b.qs, 0..) |*q, j| q.* = @intCast((i * 3 + j) % 3);
    }
    for (&p1_blocks, 0..) |*b, i| {
        b.d = @bitCast(@as(f16, @floatCast(0.25 + 0.01 * @as(f32, @floatFromInt(i)))));
        for (&b.qs, 0..) |*q, j| q.* = @intCast((i * 7 + j * 2) % 3);
    }

    var w = gguf.Writer.init(allocator);
    defer w.deinit();
    try w.addMetaString("general.architecture", "qwen3");
    try w.addMetaInt(ptqtp_gguf.version_key, u32, ptqtp_gguf.format_version);
    try w.addTensor("e.weight.ptqtp0", .tq2_0, &.{ in_dim, out_dim, n_expert }, std.mem.sliceAsBytes(&p0_blocks));
    try w.addTensor("e.weight.ptqtp1", .tq2_0, &.{ in_dim, out_dim, n_expert }, std.mem.sliceAsBytes(&p1_blocks));
    // A broken set: plane 1 without plane 0.
    try w.addTensor("g.weight.ptqtp1", .tq2_0, &.{ in_dim, out_dim, n_expert }, std.mem.sliceAsBytes(&p1_blocks));
    const undecorated = testWeightValues(rows * in_dim, 9);
    try w.addTensor("f.weight", .f32, &.{ in_dim, out_dim, n_expert }, std.mem.sliceAsBytes(&undecorated));
    var buf: [32768]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    var out = try gguf.File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer out.deinit();

    // Undecorated names fall through; broken sets are refused.
    try std.testing.expectEqual(@as(?fucina.MoeRhs, null), try ptqtp_gguf.maybeLoadMoeRhs(&ctx, &out, "f.weight", in_dim, out_dim, n_expert, false));
    try std.testing.expectError(ptqtp_gguf.Error.InvalidPlaneSet, ptqtp_gguf.maybeLoadMoeRhs(&ctx, &out, "g.weight", in_dim, out_dim, n_expert, false));

    var loaded = (try ptqtp_gguf.maybeLoadMoeRhs(&ctx, &out, "e.weight", in_dim, out_dim, n_expert, false)).?;
    defer loaded.deinit();
    try std.testing.expectEqual(std.meta.Tag(fucina.MoeRhs).ptqtp, std.meta.activeTag(loaded));
    try std.testing.expectEqual(@as(usize, 2), loaded.ptqtp.plane_count);
    try std.testing.expectEqual(in_dim, loaded.ptqtp.k);
    try std.testing.expectEqual(rows, loaded.ptqtp.n);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&p0_blocks), std.mem.sliceAsBytes(loaded.ptqtp.planes[0]));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&p1_blocks), std.mem.sliceAsBytes(loaded.ptqtp.planes[1]));

    // Geometry that disagrees with the plane tensors is refused.
    try std.testing.expectError(error.InvalidWeightShape, ptqtp_gguf.maybeLoadMoeRhs(&ctx, &out, "e.weight", in_dim, out_dim + 1, n_expert, false));

    // Streamed: the ProjSpec points the ExpertStore at both sibling planes.
    const spec = (try ptqtp_gguf.maybeStreamedMoeProjSpec(&out, "e.weight", in_dim, out_dim, n_expert)).?;
    try std.testing.expectEqual(fucina.expert_store.StreamedQuant.tq2_0, spec.quant);
    try std.testing.expectEqual(@as(u8, 2), spec.plane_count);
    const info0 = try out.get("e.weight.ptqtp0");
    const info1 = try out.get("e.weight.ptqtp1");
    try std.testing.expectEqual(out.partDataOffset(info0.part) + info0.offset, spec.file_offset);
    try std.testing.expectEqual(out.partDataOffset(info1.part) + info1.offset, spec.plane_offsets[0]);
    try std.testing.expectEqual(info0.data.len, spec.byte_len);
    try std.testing.expectEqual(@as(?fucina.expert_store.ProjSpec, null), try ptqtp_gguf.maybeStreamedMoeProjSpec(&out, "f.weight", in_dim, out_dim, n_expert));
}
