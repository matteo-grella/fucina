//! Behavioral tests for the GGUF parser/writer module (`gguf.zig`): tensor
//! payload indexing, metadata capture, the writer round-trip + alignment/skip/
//! duplicate rules, the encode/decode f32 quantize seam (K-quant transcode,
//! gated encoders, non-finite rejection), byte-length checks, and a real-model
//! re-emit (skipped without models/).
const std = @import("std");

const gguf = @import("gguf.zig");
const dtype_mod = @import("dtype.zig");
const quant = @import("backend/quant.zig");

const GgmlType = gguf.GgmlType;
const File = gguf.File;
const Writer = gguf.Writer;
const MetaType = gguf.MetaType;
const Error = gguf.Error;
const tensorByteLen = gguf.tensorByteLen;
const encodeF32 = gguf.encodeF32;
const decodeF32 = gguf.decodeF32;

test "GGUF parser indexes tensor payloads" {
    const allocator = std.testing.allocator;

    var raw: [256]u8 = undefined;
    @memset(&raw, 0);
    var offset: usize = 0;

    try writeBytes(&raw, &offset, "GGUF");
    try writeInt(&raw, &offset, u32, 3);
    try writeInt(&raw, &offset, u64, 1);
    try writeInt(&raw, &offset, u64, 0);
    try writeString(&raw, &offset, "w");
    try writeInt(&raw, &offset, u32, 2);
    try writeInt(&raw, &offset, u64, 2);
    try writeInt(&raw, &offset, u64, 3);
    try writeInt(&raw, &offset, u32, @intFromEnum(GgmlType.f32));
    try writeInt(&raw, &offset, u64, 0);

    offset = std.mem.alignForward(usize, offset, 32);
    const payload_offset = offset;
    for (0..6) |i| {
        const bits: u32 = @bitCast(@as(f32, @floatFromInt(i + 1)));
        try writeInt(&raw, &offset, u32, bits);
    }

    const owned = try allocator.dupe(u8, raw[0..offset]);
    var file = try File.parseOwned(allocator, owned);
    defer file.deinit();

    try std.testing.expectEqual(@as(usize, 32), file.alignment);
    try std.testing.expectEqual(payload_offset, file.data_offset);

    const info = try file.get("w");
    try std.testing.expectEqual(@as(usize, 2), try info.dim(0));
    try std.testing.expectEqual(@as(usize, 3), try info.dim(1));
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, &(try info.logicalMatrixShape()));
    try std.testing.expectEqual(@as(usize, 24), info.data.len);
}

test "GGUF parser captures metadata values" {
    const allocator = std.testing.allocator;

    var raw: [256]u8 = undefined;
    @memset(&raw, 0);
    var offset: usize = 0;

    try writeBytes(&raw, &offset, "GGUF");
    try writeInt(&raw, &offset, u32, 3);
    try writeInt(&raw, &offset, u64, 0); // tensor_count
    try writeInt(&raw, &offset, u64, 4); // metadata_count

    try writeString(&raw, &offset, "general.name");
    try writeInt(&raw, &offset, u32, 8); // string
    try writeString(&raw, &offset, "test");

    try writeString(&raw, &offset, "tokenizer.ggml.eos_token_id");
    try writeInt(&raw, &offset, u32, 4); // u32
    try writeInt(&raw, &offset, u32, 42);

    try writeString(&raw, &offset, "tokenizer.ggml.add_bos_token");
    try writeInt(&raw, &offset, u32, 7); // bool
    try writeInt(&raw, &offset, u8, 0);

    try writeString(&raw, &offset, "tokenizer.ggml.tokens");
    try writeInt(&raw, &offset, u32, 9); // array
    try writeInt(&raw, &offset, u32, 8); // item type string
    try writeInt(&raw, &offset, u64, 2); // len
    try writeString(&raw, &offset, "a");
    try writeString(&raw, &offset, "bb");

    const owned = try allocator.dupe(u8, raw[0..offset]);
    var file = try File.parseOwned(allocator, owned);
    defer file.deinit();

    try std.testing.expectEqualStrings("test", file.getString("general.name").?);
    try std.testing.expectEqual(@as(i64, 42), file.getInt("tokenizer.ggml.eos_token_id").?);
    try std.testing.expectEqual(false, file.getBool("tokenizer.ggml.add_bos_token").?);
    try std.testing.expectEqual(@as(?[]const u8, null), file.getString("missing"));

    const arr = file.getArray("tokenizer.ggml.tokens").?;
    try std.testing.expectEqual(@as(usize, 2), arr.len);
    const strings = try arr.stringSlices(allocator);
    defer allocator.free(strings);
    try std.testing.expectEqualStrings("a", strings[0]);
    try std.testing.expectEqualStrings("bb", strings[1]);
}

test "GGUF parser rejects impossible header counts before allocation" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(Error.InvalidTensorInfo, parseHeaderCounts(allocator, 64, 0));
    try std.testing.expectError(Error.InvalidTensorInfo, parseHeaderCounts(allocator, 0, 64));
    try std.testing.expectError(Error.InvalidTensorInfo, parseHeaderCounts(allocator, std.math.maxInt(u64), 0));
    try std.testing.expectError(Error.InvalidTensorInfo, parseHeaderCounts(allocator, 0, std.math.maxInt(u64)));
}

test "GGUF File.load malformed parse frees owned bytes once" {
    const allocator = std.testing.allocator;
    const malformed = [_]u8{ 'N', 'O', 'P', 'E' };

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "gguf_bad_{d}.gguf", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [64]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(&malformed);
        try writer.interface.flush();
    }

    try std.testing.expectError(Error.InvalidMagic, File.load(allocator, std.testing.io, path));
}

test "GGUF writer round-trips metadata and tensors through the parser" {
    const allocator = std.testing.allocator;

    var w = Writer.init(allocator);
    defer w.deinit();

    try w.addMetaString("general.name", "fucina-writer-test");
    try w.addMetaInt("test.u8", u8, 7);
    try w.addMetaInt("test.i8", i8, -7);
    try w.addMetaInt("test.u16", u16, 300);
    try w.addMetaInt("test.i16", i16, -300);
    try w.addMetaInt("test.u32", u32, 70_000);
    try w.addMetaInt("test.i32", i32, -70_000);
    try w.addMetaInt("test.u64", u64, 1 << 40);
    try w.addMetaInt("test.i64", i64, -(1 << 40));
    try w.addMetaFloat("test.f32", f32, 1.5);
    try w.addMetaFloat("test.f64", f64, -2.25);
    try w.addMetaBool("test.bool", true);
    try w.addMetaArray("test.arr_i32", i32, &.{ 1, -2, 3 });
    try w.addMetaArray("test.arr_f32", f32, &.{ 0.5, -1.5 });
    try w.addMetaStringArray("test.arr_str", &.{ "a", "bb", "" });

    // f32 [3, 2] in ne order = logical [2, 3] rows-by-cols.
    const f32_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var f32_bytes: [24]u8 = undefined;
    try encodeF32(.f32, &f32_values, &f32_bytes);
    try w.addTensor("w_f32", .f32, &.{ 3, 2 }, &f32_bytes);

    const f16_values = [_]f32{ 0.25, -0.5, 8, 16 };
    var f16_bytes: [8]u8 = undefined;
    try encodeF32(.f16, &f16_values, &f16_bytes);
    try w.addTensor("w_f16", .f16, &.{4}, &f16_bytes);

    var q8_values: [64]f32 = undefined;
    for (&q8_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) - 31.5;
    var q8_bytes: [68]u8 align(2) = undefined;
    try encodeF32(.q8_0, &q8_values, &q8_bytes);
    try w.addTensor("w_q8", .q8_0, &.{64}, &q8_bytes);

    var buf: [8192]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    const written = sink.buffered();

    var file = try File.parseOwned(allocator, try allocator.dupe(u8, written));
    defer file.deinit();

    try std.testing.expectEqualStrings("fucina-writer-test", file.getString("general.name").?);
    try std.testing.expectEqual(@as(i64, 7), file.getInt("test.u8").?);
    try std.testing.expectEqual(@as(i64, -7), file.getInt("test.i8").?);
    try std.testing.expectEqual(@as(i64, 300), file.getInt("test.u16").?);
    try std.testing.expectEqual(@as(i64, -300), file.getInt("test.i16").?);
    try std.testing.expectEqual(@as(i64, 70_000), file.getInt("test.u32").?);
    try std.testing.expectEqual(@as(i64, -70_000), file.getInt("test.i32").?);
    try std.testing.expectEqual(@as(i64, 1 << 40), file.getInt("test.u64").?);
    try std.testing.expectEqual(@as(i64, -(1 << 40)), file.getInt("test.i64").?);
    try std.testing.expectEqual(@as(f64, 1.5), file.getFloat("test.f32").?);
    try std.testing.expectEqual(@as(f64, -2.25), file.getFloat("test.f64").?);
    try std.testing.expectEqual(true, file.getBool("test.bool").?);

    const arr_i32 = file.getArray("test.arr_i32").?;
    try std.testing.expectEqual(@as(u32, @intFromEnum(MetaType.int32)), arr_i32.item_type);
    try std.testing.expectEqual(@as(usize, 3), arr_i32.len);
    try std.testing.expectEqual(@as(i32, -2), std.mem.readInt(i32, arr_i32.data[4..8], .little));

    const arr_f32 = file.getArray("test.arr_f32").?;
    try std.testing.expectEqual(@as(u32, @intFromEnum(MetaType.float32)), arr_f32.item_type);
    try std.testing.expectEqual(@as(usize, 2), arr_f32.len);
    try std.testing.expectEqual(@as(f32, -1.5), @as(f32, @bitCast(std.mem.readInt(u32, arr_f32.data[4..8], .little))));

    const arr_str = file.getArray("test.arr_str").?;
    const strings = try arr_str.stringSlices(allocator);
    defer allocator.free(strings);
    try std.testing.expectEqual(@as(usize, 3), strings.len);
    try std.testing.expectEqualStrings("a", strings[0]);
    try std.testing.expectEqualStrings("bb", strings[1]);
    try std.testing.expectEqualStrings("", strings[2]);

    try std.testing.expectEqual(@as(usize, 32), file.alignment);
    try std.testing.expectEqual(@as(usize, 0), file.data_offset % 32);

    const w_f32 = try file.get("w_f32");
    try std.testing.expectEqual(GgmlType.f32, w_f32.ggml_type);
    try std.testing.expectEqual(@as(usize, 2), w_f32.n_dims);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, &(try w_f32.logicalMatrixShape()));
    try std.testing.expectEqualSlices(u8, &f32_bytes, w_f32.data);

    const w_f16 = try file.get("w_f16");
    try std.testing.expectEqual(GgmlType.f16, w_f16.ggml_type);
    try std.testing.expectEqualSlices(u8, &f16_bytes, w_f16.data);
    try std.testing.expectEqual(@as(f16, 0.25), @as(f16, @bitCast(std.mem.readInt(u16, w_f16.data[0..2], .little))));

    const w_q8 = try file.get("w_q8");
    try std.testing.expectEqual(GgmlType.q8_0, w_q8.ggml_type);
    try std.testing.expectEqualSlices(u8, &q8_bytes, w_q8.data);
    try std.testing.expectEqual(@as(usize, 0), w_q8.offset % 32);
    var decoded: [64]f32 = undefined;
    var block_copy: [2]dtype_mod.BlockQ8_0 = undefined;
    @memcpy(std.mem.sliceAsBytes(&block_copy), w_q8.data);
    try quant.dequantizeRowQ8_0Into(&decoded, &block_copy);
    for (q8_values, decoded) |expected, got| {
        try std.testing.expectApproxEqAbs(expected, got, 0.13); // half a q8_0 step at amax 31.5
    }

    // Re-emitting the parsed file must reproduce it byte-identically.
    var w2 = Writer.init(allocator);
    defer w2.deinit();
    try w2.copyAllMetadata(&file, &.{});
    for (file.tensors) |*info| {
        try w2.addTensor(info.name, info.ggml_type, info.dims[0..info.n_dims], info.data);
    }
    var buf2: [8192]u8 = undefined;
    var sink2 = std.Io.Writer.fixed(&buf2);
    try w2.finish(&sink2);
    try std.testing.expectEqualSlices(u8, written, sink2.buffered());
}

test "GGUF writer streams declared tensors byte-identically to finish" {
    const allocator = std.testing.allocator;

    const f32_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var f32_bytes: [24]u8 = undefined;
    try encodeF32(.f32, &f32_values, &f32_bytes);
    var q8_values: [64]f32 = undefined;
    for (&q8_values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) - 31.5;
    var q8_bytes: [68]u8 align(2) = undefined;
    try encodeF32(.q8_0, &q8_values, &q8_bytes);

    // Reference: the borrow-everything finish path.
    var w = Writer.init(allocator);
    defer w.deinit();
    try w.addMetaString("general.name", "stream-test");
    try w.addTensor("a", .f32, &.{ 3, 2 }, &f32_bytes);
    try w.addTensor("b", .q8_0, &.{64}, &q8_bytes);
    var buf: [4096]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);
    const reference = sink.buffered();

    // Streaming: declare (no bytes), write header, then feed data in order.
    var ws = Writer.init(allocator);
    defer ws.deinit();
    try ws.addMetaString("general.name", "stream-test");
    try ws.declareTensor("a", .f32, &.{ 3, 2 });
    try ws.declareTensor("b", .q8_0, &.{64});

    // finish refuses data-less declarations — they belong to the stream path.
    var reject: [4096]u8 = undefined;
    var reject_sink = std.Io.Writer.fixed(&reject);
    try std.testing.expectError(Error.TensorDataMissing, ws.finish(&reject_sink));

    var buf2: [4096]u8 = undefined;
    var sink2 = std.Io.Writer.fixed(&buf2);
    var streamer = try ws.beginStream(&sink2);
    try std.testing.expectEqualStrings("a", streamer.nextTensorName().?);
    try std.testing.expectError(Error.TensorDataMissing, streamer.finish()); // incomplete
    try std.testing.expectError(Error.InvalidTensorInfo, streamer.writeTensorData(f32_bytes[0..8])); // wrong length
    try streamer.writeTensorData(&f32_bytes);
    try std.testing.expectEqualStrings("b", streamer.nextTensorName().?);
    try streamer.writeTensorData(&q8_bytes);
    try std.testing.expectEqual(@as(?[]const u8, null), streamer.nextTensorName());
    try std.testing.expectError(Error.TensorDataMissing, streamer.writeTensorData(&q8_bytes)); // past end
    try streamer.finish();

    try std.testing.expectEqualSlices(u8, reference, sink2.buffered());

    // The streamed bytes parse back to the same tensors.
    var file = try File.parseOwned(allocator, try allocator.dupe(u8, sink2.buffered()));
    defer file.deinit();
    try std.testing.expectEqualSlices(u8, &f32_bytes, (try file.get("a")).data);
    try std.testing.expectEqualSlices(u8, &q8_bytes, (try file.get("b")).data);
}

test "GGUF writer honors general.alignment, key replacement, and skip lists" {
    const allocator = std.testing.allocator;

    var w = Writer.init(allocator);
    defer w.deinit();
    try w.addMetaInt("general.alignment", u32, 64);
    try w.addMetaString("general.name", "first");
    try w.addMetaString("general.name", "second"); // replaced in place
    try w.addMetaString("test.dropme", "x");

    const values = [_]f32{ 1, 2, 3 };
    var bytes: [12]u8 = undefined;
    try encodeF32(.f32, &values, &bytes);
    try w.addTensor("t", .f32, &.{3}, &bytes);

    var buf: [4096]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);

    var file = try File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();
    try std.testing.expectEqual(@as(usize, 64), file.alignment);
    try std.testing.expectEqual(@as(usize, 0), file.data_offset % 64);
    try std.testing.expectEqual(@as(u32, 3), file.metadata.count());
    try std.testing.expectEqualStrings("second", file.getString("general.name").?);
    try std.testing.expectEqualSlices(u8, &bytes, (try file.get("t")).data);

    // copyAllMetadata skip list drops the requested key; alignment is carried.
    var w2 = Writer.init(allocator);
    defer w2.deinit();
    try w2.copyAllMetadata(&file, &.{"test.dropme"});
    try std.testing.expectEqual(@as(usize, 64), w2.alignment);
    try std.testing.expectEqual(@as(usize, 2), w2.kvs.items.len);

    // addMetaCopy: present key copied with its wire type, absent key errors.
    var w3 = Writer.init(allocator);
    defer w3.deinit();
    try w3.addMetaCopy(&file, "general.alignment");
    try std.testing.expectEqual(@as(usize, 64), w3.alignment);
    try std.testing.expectEqual(@as(u32, @intFromEnum(MetaType.uint32)), w3.kvs.items[0].value_type);
    try std.testing.expectError(Error.KeyNotFound, w3.addMetaCopy(&file, "missing.key"));

    // Non-power-of-two alignment is rejected.
    var w4 = Writer.init(allocator);
    defer w4.deinit();
    try std.testing.expectError(Error.InvalidAlignment, w4.addMetaInt("general.alignment", u32, 48));

    // llama.cpp asserts that general.alignment is encoded specifically as u32.
    var w5 = Writer.init(allocator);
    defer w5.deinit();
    try std.testing.expectError(Error.InvalidAlignment, w5.addMetaInt("general.alignment", u8, 64));
    try std.testing.expectError(Error.InvalidAlignment, w5.addMetaInt("general.alignment", u16, 64));
    try std.testing.expectError(Error.InvalidAlignment, w5.addMetaInt("general.alignment", u64, 64));
}

test "GGUF writer rejects duplicate and malformed tensors" {
    const allocator = std.testing.allocator;

    var w = Writer.init(allocator);
    defer w.deinit();
    const values = [_]f32{ 1, 2 };
    var bytes: [8]u8 = undefined;
    try encodeF32(.f32, &values, &bytes);
    try w.addTensor("t", .f32, &.{2}, &bytes);
    try std.testing.expectError(Error.DuplicateTensorName, w.addTensor("t", .f32, &.{2}, &bytes));
    try std.testing.expectError(Error.InvalidTensorInfo, w.addTensor("u", .f32, &.{3}, &bytes));
    try std.testing.expectError(Error.InvalidTensorInfo, w.addTensor("", .f32, &.{2}, &bytes));
    try std.testing.expectError(Error.InvalidTensorInfo, w.addTensor("v", .f32, &.{ 1, 1, 1, 1, 2 }, &bytes));
}

test "encodeF32 K-quant transcode round-trips through the parity-tested dequant" {
    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13) * 3.5;
    inline for (.{ .q4_k, .q5_k, .q6_k }) |dt| {
        const Block = dtype_mod.Storage(dt);
        var blocks: [1]Block = undefined;
        try encodeF32(@field(GgmlType, @tagName(dt)), &src, std.mem.asBytes(&blocks));
        var back: [256]f32 = undefined;
        try quant.dequantizeRowForDType(dt, &back, &blocks);
        var err_sq: f64 = 0;
        for (src, back) |a, b| err_sq += (a - b) * (a - b);
        const rmse = @sqrt(err_sq / 256.0);
        // K-quants on smooth data: RMSE well under one quant step of the range.
        try std.testing.expect(rmse < 0.2);
        try std.testing.expect(rmse > 0); // lossy, not pass-through
    }
}

test "encodeF32 gates unimplemented encoders" {
    const src = [_]f32{0} ** 256;
    var dst2: [84]u8 = undefined; // q2_k block byte size for one 256-block
    try std.testing.expectError(Error.EncoderUnavailable, encodeF32(.q2_k, &src, &dst2));
}

test "decodeF32 mirrors encodeF32 (scalars exact, blocks via the parity-tested dequant)" {
    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13) * 3.5;

    // Scalar formats round-trip exactly (f32) or to the cast precision.
    var f32_wire: [256 * 4]u8 = undefined;
    try encodeF32(.f32, &src, &f32_wire);
    var back: [256]f32 = undefined;
    try decodeF32(.f32, &f32_wire, &back);
    try std.testing.expectEqualSlices(f32, &src, &back);

    var bf16_wire: [256 * 2]u8 = undefined;
    try encodeF32(.bf16, &src, &bf16_wire);
    try decodeF32(.bf16, &bf16_wire, &back);
    for (src, back) |a, b| try std.testing.expect(@abs(a - b) <= 0.02 * @max(@abs(a), 1.0));

    // Block formats: decodeF32 must equal the parity-tested row dequant —
    // the requantize seam (`export-gguf --experts-dtype`) rides on this.
    inline for (.{ .q8_0, .q4_k, .q5_k, .q6_k, .tq2_0 }) |dt| {
        const Block = dtype_mod.Storage(dt);
        var blocks: [256 / dtype_mod.blockSize(dt)]Block = undefined;
        try encodeF32(@field(GgmlType, @tagName(dt)), &src, std.mem.asBytes(&blocks));
        var want: [256]f32 = undefined;
        try quant.dequantizeRowForDType(dt, &want, &blocks);
        try decodeF32(@field(GgmlType, @tagName(dt)), std.mem.asBytes(&blocks), &back);
        try std.testing.expectEqualSlices(f32, &want, &back);
    }

    var iq_dst: [2]f32 = undefined;
    var iq_src: [8]u8 = undefined;
    @memset(&iq_src, 0);
    try std.testing.expectError(Error.DecoderUnavailable, decodeF32(.i32, &iq_src, &iq_dst));
}

test "GGUF tq2_0 transcode writes, reads back, and matches the direct encoder" {
    const allocator = std.testing.allocator;

    // 2 rows x 512 columns (ne order: dims[0] = innermost) — two 256-element
    // TQ2_0 blocks per row.
    const rows = 2;
    const cols = 512;
    const Block = dtype_mod.Storage(.tq2_0);
    const block_count = rows * cols / 256;

    var values: [rows * cols]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 1.5;

    var wire: [block_count]Block = undefined;
    try encodeF32(.tq2_0, &values, std.mem.asBytes(&wire));

    var w = Writer.init(allocator);
    defer w.deinit();
    try w.addMetaString("general.name", "fucina-tq2_0-test");
    try w.addTensor("w_tq2", .tq2_0, &.{ cols, rows }, std.mem.asBytes(&wire));

    var buf: [4096]u8 = undefined;
    var sink = std.Io.Writer.fixed(&buf);
    try w.finish(&sink);

    var file = try File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer file.deinit();

    const info = try file.get("w_tq2");
    // Loading yields the ternary dtype.
    try std.testing.expectEqual(GgmlType.tq2_0, info.ggml_type);
    try std.testing.expectEqual(dtype_mod.DType.tq2_0, gguf.dtypeForGgmlType(info.ggml_type).?);
    // Payload length matches tensorByteLen; the wire bytes survive verbatim.
    try std.testing.expectEqual(try tensorByteLen(.tq2_0, &.{ cols, rows }), info.data.len);
    try std.testing.expectEqual(block_count * @sizeOf(Block), info.data.len);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&wire), info.data);

    // encodeF32 must equal quantizeRowForDType(.tq2_0) directly, and the
    // parsed bytes must dequantize to the same values as that direct
    // encode → dequantizeRowForDType round-trip.
    var direct: [block_count]Block = undefined;
    try quant.quantizeRowForDType(.tq2_0, &direct, &values);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&direct), std.mem.asBytes(&wire));

    var want: [rows * cols]f32 = undefined;
    try quant.dequantizeRowForDType(.tq2_0, &want, &direct);
    var block_copy: [block_count]Block = undefined;
    @memcpy(std.mem.sliceAsBytes(&block_copy), info.data);
    var got: [rows * cols]f32 = undefined;
    try quant.dequantizeRowForDType(.tq2_0, &got, &block_copy);
    try std.testing.expectEqualSlices(f32, &want, &got);

    // decodeF32 (the requantize seam) agrees with the block dequant.
    var via_decode: [rows * cols]f32 = undefined;
    try decodeF32(.tq2_0, std.mem.asBytes(&block_copy), &via_decode);
    try std.testing.expectEqualSlices(f32, &want, &via_decode);
}

test "encodeF32 rejects non-finite input at the block-quantize seam" {
    var src: [256]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13);

    var q4k_dst: [@sizeOf(dtype_mod.BlockQ4_K)]u8 align(@alignOf(dtype_mod.BlockQ4_K)) = undefined;
    var q50_dst: [8 * @sizeOf(dtype_mod.BlockQ5_0)]u8 align(@alignOf(dtype_mod.BlockQ5_0)) = undefined;
    var tq2_dst: [@sizeOf(dtype_mod.BlockTQ2_0)]u8 align(@alignOf(dtype_mod.BlockTQ2_0)) = undefined;

    // Clean inputs encode fine.
    try encodeF32(.q4_k, &src, &q4k_dst);
    try encodeF32(.q5_0, &src, &q50_dst);
    try encodeF32(.tq2_0, &src, &tq2_dst);

    // A single NaN or inf anywhere in the row errors.
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |poison| {
        src[173] = poison;
        try std.testing.expectError(Error.NonFiniteValue, encodeF32(.q4_k, &src, &q4k_dst));
        try std.testing.expectError(Error.NonFiniteValue, encodeF32(.q5_0, &src, &q50_dst));
        try std.testing.expectError(Error.NonFiniteValue, encodeF32(.tq2_0, &src, &tq2_dst));
    }
    // Restored clean input is unaffected.
    src[173] = 0.25;
    try encodeF32(.q4_k, &src, &q4k_dst);
    try encodeF32(.q5_0, &src, &q50_dst);
    try encodeF32(.tq2_0, &src, &tq2_dst);
}

test "tensorByteLen rejects blocks straddling the innermost dim" {
    // total 64 % 32 == 0 but ne[0] = 16 is not a whole number of q8_0 blocks.
    try std.testing.expectError(Error.InvalidTensorInfo, tensorByteLen(.q8_0, &.{ 16, 4 }));
    // Valid layouts (1-D and 2-D) still size correctly.
    try std.testing.expectEqual(@as(usize, 2 * 34), tensorByteLen(.q8_0, &.{64}));
    try std.testing.expectEqual(@as(usize, 8 * 34), tensorByteLen(.q8_0, &.{ 64, 4 }));
    try std.testing.expectError(Error.InvalidTensorInfo, tensorByteLen(.q4_k, &.{ 128, 2 }));
    try std.testing.expectEqual(@as(usize, 2 * @sizeOf(dtype_mod.BlockQ4_K)), tensorByteLen(.q4_k, &.{ 256, 2 }));
}

test "GGUF writer re-emits real-model metadata and tensors (skips without models/)" {
    const allocator = std.testing.allocator;

    var file = File.loadMmap(allocator, std.testing.io, "models/Qwen3-0.6B-Q4_K_S.gguf") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();

    var w = Writer.init(allocator);
    defer w.deinit();
    try w.copyAllMetadata(&file, &.{});

    // A tiny tensor subset — full-file re-emits are the export tool's job.
    var picked: usize = 0;
    for (file.tensors) |*info| {
        if (info.data.len > (1 << 20)) continue;
        try w.addTensor(info.name, info.ggml_type, info.dims[0..info.n_dims], info.data);
        picked += 1;
        if (picked == 3) break;
    }
    try std.testing.expect(picked > 0);

    const capacity = file.data_offset + (4 << 20);
    const buf = try allocator.alloc(u8, capacity);
    defer allocator.free(buf);
    var sink = std.Io.Writer.fixed(buf);
    try w.finish(&sink);

    var rt = try File.parseOwned(allocator, try allocator.dupe(u8, sink.buffered()));
    defer rt.deinit();

    try std.testing.expectEqual(file.alignment, rt.alignment);
    try std.testing.expectEqual(file.metadata.count(), rt.metadata.count());
    var it = file.metadata.iterator();
    while (it.next()) |entry| {
        const got = rt.meta(entry.key_ptr.*) orelse return error.TestUnexpectedResult;
        switch (entry.value_ptr.*) {
            .int => |v| try std.testing.expectEqual(v, got.int),
            .float => |v| try std.testing.expectEqual(v, got.float),
            .boolean => |v| try std.testing.expectEqual(v, got.boolean),
            .string => |v| try std.testing.expectEqualStrings(v, got.string),
            .array => |v| {
                try std.testing.expectEqual(v.item_type, got.array.item_type);
                try std.testing.expectEqual(v.len, got.array.len);
                try std.testing.expectEqualSlices(u8, v.data, got.array.data);
            },
        }
    }

    try std.testing.expectEqual(picked, rt.tensors.len);
    for (rt.tensors) |*info| {
        const src = try file.get(info.name);
        try std.testing.expectEqual(src.ggml_type, info.ggml_type);
        try std.testing.expectEqual(src.n_dims, info.n_dims);
        try std.testing.expectEqualSlices(usize, src.dims[0..src.n_dims], info.dims[0..info.n_dims]);
        try std.testing.expectEqualSlices(u8, src.data, info.data);
    }
}

fn writeBytes(buf: []u8, offset: *usize, bytes: []const u8) !void {
    const end = try std.math.add(usize, offset.*, bytes.len);
    if (end > buf.len) return error.NoSpaceLeft;
    @memcpy(buf[offset.*..end], bytes);
    offset.* = end;
}

fn writeString(buf: []u8, offset: *usize, value: []const u8) !void {
    try writeInt(buf, offset, u64, value.len);
    try writeBytes(buf, offset, value);
}

fn writeInt(buf: []u8, offset: *usize, comptime Int: type, value: Int) !void {
    const end = try std.math.add(usize, offset.*, @sizeOf(Int));
    if (end > buf.len) return error.NoSpaceLeft;
    std.mem.writeInt(Int, buf[offset.*..end][0..@sizeOf(Int)], value, .little);
    offset.* = end;
}

fn parseHeaderCounts(allocator: std.mem.Allocator, tensor_count: u64, metadata_count: u64) !File {
    var raw: [24]u8 = undefined;
    @memset(&raw, 0);
    var offset: usize = 0;
    try writeBytes(&raw, &offset, "GGUF");
    try writeInt(&raw, &offset, u32, 3);
    try writeInt(&raw, &offset, u64, tensor_count);
    try writeInt(&raw, &offset, u64, metadata_count);
    const owned = try allocator.dupe(u8, raw[0..offset]);
    return File.parseOwned(allocator, owned);
}

const AlignCase = union(enum) { str, neg_i64, huge_u64, u32v: u32 };

fn parseAlignmentGguf(allocator: std.mem.Allocator, c: AlignCase) !File {
    var raw: [256]u8 = undefined;
    @memset(&raw, 0);
    var offset: usize = 0;
    try writeBytes(&raw, &offset, "GGUF");
    try writeInt(&raw, &offset, u32, 3);
    try writeInt(&raw, &offset, u64, 0); // tensor_count
    try writeInt(&raw, &offset, u64, 1); // metadata_count
    try writeString(&raw, &offset, "general.alignment");
    switch (c) {
        .str => {
            try writeInt(&raw, &offset, u32, 8); // string type (non-int)
            try writeString(&raw, &offset, "oops");
        },
        .neg_i64 => {
            try writeInt(&raw, &offset, u32, 11); // int64
            try writeInt(&raw, &offset, i64, -5);
        },
        .huge_u64 => {
            try writeInt(&raw, &offset, u32, 10); // uint64 >= 2^63 (would overflow i64)
            try writeInt(&raw, &offset, u64, @as(u64, 1) << 63);
        },
        .u32v => |v| {
            try writeInt(&raw, &offset, u32, 4); // uint32
            try writeInt(&raw, &offset, u32, v);
        },
    }
    // parseOwned takes ownership of `owned` (frees it on error, or via File.deinit).
    const owned = try allocator.dupe(u8, raw[0..offset]);
    return File.parseOwned(allocator, owned);
}

test "GGUF reader rejects malformed general.alignment without UB" {
    const a = std.testing.allocator;
    // non-int, negative, >= 2^63 (uint64 overflow), zero, and non-power-of-two
    // each return InvalidAlignment — none reach the @intCast / alignForward UB.
    try std.testing.expectError(Error.InvalidAlignment, parseAlignmentGguf(a, .str));
    try std.testing.expectError(Error.InvalidAlignment, parseAlignmentGguf(a, .neg_i64));
    try std.testing.expectError(Error.InvalidAlignment, parseAlignmentGguf(a, .huge_u64));
    try std.testing.expectError(Error.InvalidAlignment, parseAlignmentGguf(a, .{ .u32v = 0 }));
    try std.testing.expectError(Error.InvalidAlignment, parseAlignmentGguf(a, .{ .u32v = 3 }));
    // a valid power-of-two alignment parses and is applied.
    var file = try parseAlignmentGguf(a, .{ .u32v = 64 });
    defer file.deinit();
    try std.testing.expectEqual(@as(usize, 64), file.alignment);
}

test "loadMmapAuto merges llama.cpp split GGUFs with part-tagged tensors" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // Two tiny parts: metadata + tensor "a" in part 1, tensor "b" in part 2.
    var stamp_buf: [160]u8 = undefined;
    const stamp = std.Io.Clock.real.now(io).nanoseconds;
    const path1 = try std.fmt.bufPrint(stamp_buf[0..80], "gguf_split_{d}-00001-of-00002.gguf", .{stamp});
    const path2 = try std.fmt.bufPrint(stamp_buf[80..], "gguf_split_{d}-00002-of-00002.gguf", .{stamp});
    defer std.Io.Dir.cwd().deleteFile(io, path1) catch {};
    defer std.Io.Dir.cwd().deleteFile(io, path2) catch {};

    const a_vals = [_]f32{ 1, 2, 3, 4 };
    const b_vals = [_]f32{ 5, 6, 7, 8, 9, 10 };
    inline for (.{ .{ path1, "a", a_vals[0..] }, .{ path2, "b", b_vals[0..] } }) |case| {
        var w = Writer.init(allocator);
        defer w.deinit();
        try w.addMetaString("general.architecture", "split-test");
        try w.addTensor(case[1], .f32, &.{case[2].len}, std.mem.sliceAsBytes(case[2]));
        var buf: [4096]u8 = undefined;
        var sink = std.Io.Writer.fixed(&buf);
        try w.finish(&sink);
        var file = try std.Io.Dir.cwd().createFile(io, case[0], .{});
        defer file.close(io);
        try file.writePositionalAll(io, sink.buffered(), 0);
    }

    // Non-first parts and non-split names are not split entry points.
    try std.testing.expectEqual(@as(?[][]u8, null), try File.splitPartPaths(allocator, path2));
    try std.testing.expectEqual(@as(?[][]u8, null), try File.splitPartPaths(allocator, "plain.gguf"));

    var merged = try File.loadMmapAuto(allocator, io, path1);
    defer merged.deinit();
    try std.testing.expect(merged.isSplit());
    try std.testing.expectEqualStrings("split-test", merged.getString("general.architecture").?);

    const a = try merged.get("a");
    try std.testing.expectEqual(@as(u16, 0), a.part);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a_vals[0..]), a.data);
    const b = try merged.get("b");
    try std.testing.expectEqual(@as(u16, 1), b.part);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b_vals[0..]), b.data);

    // The absolute on-disk position round-trips: partDataOffset(part) +
    // offset must point at the tensor bytes within its own file.
    var part2 = try std.Io.Dir.cwd().openFile(io, path2, .{});
    defer part2.close(io);
    var back: [6 * 4]u8 = undefined;
    _ = try part2.readPositionalAll(io, &back, merged.partDataOffset(1) + b.offset);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b_vals[0..]), &back);

    // Multiple mappings cannot be handed over as one region.
    try std.testing.expectEqual(@as(?File.MappedRegion, null), merged.takeMapping());
}
