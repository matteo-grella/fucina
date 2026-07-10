const std = @import("std");

const safetensors = @import("safetensors.zig");

const allocator = std.testing.allocator;

test "safetensors serialization matches upstream f32 golden bytes" {
    const shape = [_]usize{ 1, 2, 3 };
    const data = [_]u8{
        0, 0, 0,   0,  0, 0, 128, 63,
        0, 0, 0,   64, 0, 0, 64,  64,
        0, 0, 128, 64, 0, 0, 160, 64,
    };
    const tensors = [_]safetensors.Tensor{.{
        .name = "attn.0",
        .dtype = .F32,
        .shape = &shape,
        .data = &data,
    }};

    const out = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(out);

    const expected = [_]u8{
        64,  0,   0,   0,   0,  0,  0,   0,   123, 34, 97,  116, 116, 110, 46,  48,  34,  58,  123, 34,  100,
        116, 121, 112, 101, 34, 58, 34,  70,  51,  50, 34,  44,  34,  115, 104, 97,  112, 101, 34,  58,  91,
        49,  44,  50,  44,  51, 93, 44,  34,  100, 97, 116, 97,  95,  111, 102, 102, 115, 101, 116, 115, 34,
        58,  91,  48,  44,  50, 52, 93,  125, 125, 0,  0,   0,   0,   0,   0,   128, 63,  0,   0,   0,   64,
        0,   0,   64,  64,  0,  0,  128, 64,  0,   0,  160, 64,
    };
    try std.testing.expectEqualSlices(u8, &expected, out);

    var parsed = try safetensors.File.parse(allocator, out);
    defer parsed.deinit();
    const tensor = try parsed.tensor("attn.0");
    try std.testing.expectEqual(safetensors.DType.F32, tensor.dtype);
    try std.testing.expectEqualSlices(usize, &shape, tensor.shape);
    try std.testing.expectEqualSlices(u8, &data, tensor.data);
}

test "safetensors serialization matches upstream fp4 golden bytes" {
    const shape = [_]usize{ 1, 2 };
    const data = [_]u8{0};
    const tensors = [_]safetensors.Tensor{.{
        .name = "attn.0",
        .dtype = .F4,
        .shape = &shape,
        .data = &data,
    }};

    const out = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(out);

    const expected = [_]u8{
        64,  0,   0,   0,   0,   0,   0,  0,   123, 34, 97,  116, 116, 110, 46,  48,  34,  58, 123, 34, 100,
        116, 121, 112, 101, 34,  58,  34, 70,  52,  34, 44,  34,  115, 104, 97,  112, 101, 34, 58,  91, 49,
        44,  50,  93,  44,  34,  100, 97, 116, 97,  95, 111, 102, 102, 115, 101, 116, 115, 34, 58,  91, 48,
        44,  49,  93,  125, 125, 32,  32, 32,  32,  0,
    };
    try std.testing.expectEqualSlices(u8, &expected, out);

    var parsed = try safetensors.File.parse(allocator, out);
    defer parsed.deinit();
    const tensor = try parsed.tensor("attn.0");
    try std.testing.expectEqual(safetensors.DType.F4, tensor.dtype);
    try std.testing.expectEqualSlices(usize, &shape, tensor.shape);
    try std.testing.expectEqualSlices(u8, &data, tensor.data);
}

test "safetensors fp4 validates bit alignment and byte length" {
    const misaligned_shape = [_]usize{ 1, 3 };
    const invalid_shape = [_]usize{ 1, 2 };
    const data = [_]u8{ 0, 1 };

    try std.testing.expectError(safetensors.Error.MisalignedSlice, safetensors.serializeAlloc(allocator, &[_]safetensors.Tensor{.{
        .name = "attn.0",
        .dtype = .F4,
        .shape = &misaligned_shape,
        .data = &data,
    }}, null));
    try std.testing.expectError(safetensors.Error.TensorInvalidInfo, safetensors.serializeAlloc(allocator, &[_]safetensors.Tensor{.{
        .name = "attn.0",
        .dtype = .F4,
        .shape = &invalid_shape,
        .data = &data,
    }}, null));
}

test "safetensors empty files and metadata match upstream golden bytes" {
    const out = try safetensors.serializeAlloc(allocator, &.{}, null);
    defer allocator.free(out);
    const expected_empty = [_]u8{ 8, 0, 0, 0, 0, 0, 0, 0, 123, 125, 32, 32, 32, 32, 32, 32 };
    try std.testing.expectEqualSlices(u8, &expected_empty, out);
    var parsed_empty = try safetensors.File.parse(allocator, out);
    defer parsed_empty.deinit();
    try std.testing.expectEqual(@as(usize, 0), parsed_empty.tensors.len);

    const metadata = [_]safetensors.MetadataEntry{.{ .key = "framework", .value = "pt" }};
    const with_meta = try safetensors.serializeAlloc(allocator, &.{}, &metadata);
    defer allocator.free(with_meta);
    const expected_meta = [_]u8{
        40,  0,  0,  0,   0,  0,   0,   0,  123, 34,  95,  95,  109, 101, 116, 97, 100, 97,  116, 97, 95,
        95,  34, 58, 123, 34, 102, 114, 97, 109, 101, 119, 111, 114, 107, 34,  58, 34,  112, 116, 34, 125,
        125, 32, 32, 32,  32, 32,
    };
    try std.testing.expectEqualSlices(u8, &expected_meta, with_meta);
    var parsed_meta = try safetensors.File.parse(allocator, with_meta);
    defer parsed_meta.deinit();
    try std.testing.expectEqualStrings("pt", parsed_meta.metadata.get("framework").?);
}

test "safetensors rejects duplicate names, invalid metadata, and reserved metadata tensor name" {
    const shape = [_]usize{1};
    const data = [_]u8{0};
    const duplicate_tensors = [_]safetensors.Tensor{
        .{ .name = "w", .dtype = .U8, .shape = &shape, .data = &data },
        .{ .name = "w", .dtype = .U8, .shape = &shape, .data = &data },
    };
    try std.testing.expectError(safetensors.Error.DuplicateTensorName, safetensors.serializeAlloc(allocator, &duplicate_tensors, null));

    const reserved = [_]safetensors.Tensor{.{ .name = "__metadata__", .dtype = .U8, .shape = &shape, .data = &data }};
    try std.testing.expectError(safetensors.Error.InvalidTensorName, safetensors.serializeAlloc(allocator, &reserved, null));

    const duplicate_metadata = [_]safetensors.MetadataEntry{
        .{ .key = "framework", .value = "pt" },
        .{ .key = "framework", .value = "jax" },
    };
    try std.testing.expectError(safetensors.Error.InvalidMetadata, safetensors.serializeAlloc(allocator, &.{}, &duplicate_metadata));

    const invalid_metadata = "\x20\x00\x00\x00\x00\x00\x00\x00{\"__metadata__\":{\"framework\":1}}";
    try std.testing.expectError(safetensors.Error.InvalidMetadata, safetensors.File.parse(allocator, invalid_metadata));

    const duplicate_json_key = "\x3c\x00\x00\x00\x00\x00\x00\x00{\"w\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"w\":{}}\x00";
    try std.testing.expectError(safetensors.Error.InvalidHeaderDeserialization, safetensors.File.parse(allocator, duplicate_json_key));
}

test "safetensors rejects serialization header over upstream size limit" {
    const big = try std.heap.page_allocator.alloc(u8, safetensors.max_header_size);
    defer std.heap.page_allocator.free(big);
    @memset(big, 'a');
    const metadata = [_]safetensors.MetadataEntry{.{ .key = "very_large_metadata", .value = big }};
    try std.testing.expectError(safetensors.Error.HeaderTooLarge, safetensors.serializeAlloc(std.heap.page_allocator, &.{}, &metadata));
}

test "safetensors forced 8-byte header alignment matches upstream golden bytes" {
    const shape = [_]usize{ 1, 1, 2, 3 };
    const data = [_]u8{
        0, 0, 0,   0,  0, 0, 128, 63,
        0, 0, 0,   64, 0, 0, 64,  64,
        0, 0, 128, 64, 0, 0, 160, 64,
    };
    const tensors = [_]safetensors.Tensor{.{
        .name = "attn0",
        .dtype = .F32,
        .shape = &shape,
        .data = &data,
    }};

    const out = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(out);
    const expected = [_]u8{
        72,  0,   0,   0,  0,  0,  0,  0,  123, 34,  97, 116, 116, 110, 48,  34,  58,  123, 34,  100, 116,
        121, 112, 101, 34, 58, 34, 70, 51, 50,  34,  44, 34,  115, 104, 97,  112, 101, 34,  58,  91,  49,
        44,  49,  44,  50, 44, 51, 93, 44, 34,  100, 97, 116, 97,  95,  111, 102, 102, 115, 101, 116, 115,
        34,  58,  91,  48, 44, 50, 52, 93, 125, 125, 32, 32,  32,  32,  32,  32,  32,  0,   0,   0,   0,
        0,   0,   128, 63, 0,  0,  0,  64, 0,   0,   64, 64,  0,   0,   128, 64,  0,   0,   160, 64,
    };
    try std.testing.expectEqualSlices(u8, &expected, out);
}

test "safetensors slices row-major tensor bytes like upstream" {
    const shape = [_]usize{ 1, 2, 3 };
    const data = [_]u8{
        0, 0, 0,   0,
        0, 0, 128, 63,
        0, 0, 0,   64,
        0, 0, 64,  64,
        0, 0, 128, 64,
        0, 0, 160, 64,
    };
    const tensors = [_]safetensors.Tensor{.{
        .name = "attn.0",
        .dtype = .F32,
        .shape = &shape,
        .data = &data,
    }};
    const out = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(out);
    var parsed = try safetensors.File.parse(allocator, out);
    defer parsed.deinit();
    const tensor = try parsed.tensor("attn.0");

    const first_row = try tensor.sliceBytesAlloc(allocator, &.{
        .{},
        .{ .end = 1 },
    });
    defer allocator.free(first_row);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64 }, first_row);

    const first_column = try tensor.sliceBytesAlloc(allocator, &.{
        .{},
        .{},
        .{ .end = 1 },
    });
    defer allocator.free(first_column);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0, 0, 0, 64, 64 }, first_column);
}

test "safetensors parses empty shapes and ordinary tensors" {
    {
        const serialized = "8\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}\x00\x00\x00\x00";
        var loaded = try safetensors.File.parse(allocator, serialized);
        defer loaded.deinit();
        const tensor = try loaded.tensor("test");
        try std.testing.expectEqual(safetensors.DType.I32, tensor.dtype);
        try std.testing.expectEqual(@as(usize, 0), tensor.shape.len);
        try std.testing.expectEqualSlices(u8, "\x00\x00\x00\x00", tensor.data);
    }
    {
        const serialized = "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        var loaded = try safetensors.File.parse(allocator, serialized);
        defer loaded.deinit();
        const shape = [_]usize{ 2, 2 };
        const tensor = try loaded.tensor("test");
        try std.testing.expectEqual(safetensors.DType.I32, tensor.dtype);
        try std.testing.expectEqualSlices(usize, &shape, tensor.shape);
        try std.testing.expectEqualSlices(u8, "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", tensor.data);
    }
}

test "safetensors rejects overlapping offsets json attack" {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    for (0..10) |i| {
        if (i != 0) try header.append(allocator, ',');
        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "\"weight_{d}\":", .{i});
        try header.appendSlice(allocator, name);
        try header.appendSlice(allocator, "{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}");
    }
    try header.append(allocator, '}');

    const bytes = try allocator.alloc(u8, 8 + header.items.len + 16);
    defer allocator.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.items.len, .little);
    @memcpy(bytes[8..][0..header.items.len], header.items);
    @memset(bytes[8 + header.items.len ..], 0);

    try std.testing.expectError(safetensors.Error.InvalidOffset, safetensors.File.parse(allocator, bytes));
}

test "safetensors rejects incomplete or polyglot buffers" {
    const extra = "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file";
    try std.testing.expectError(safetensors.Error.MetadataIncompleteBuffer, safetensors.File.parse(allocator, extra));

    const missing = "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    try std.testing.expectError(safetensors.Error.MetadataIncompleteBuffer, safetensors.File.parse(allocator, missing));
}

test "safetensors rejects malformed headers" {
    const too_large = "<\x00\x00\x00\x00\xff\xff\xff{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    try std.testing.expectError(safetensors.Error.HeaderTooLarge, safetensors.File.parse(allocator, too_large));
    try std.testing.expectError(safetensors.Error.HeaderTooSmall, safetensors.File.parse(allocator, ""));
    try std.testing.expectError(safetensors.Error.InvalidHeaderLength, safetensors.File.parse(allocator, "<\x00\x00\x00\x00\x00\x00\x00"));
    try std.testing.expectError(safetensors.Error.InvalidHeader, safetensors.File.parse(allocator, "\x01\x00\x00\x00\x00\x00\x00\x00\xff"));
    try std.testing.expectError(safetensors.Error.InvalidHeaderDeserialization, safetensors.File.parse(allocator, "\x01\x00\x00\x00\x00\x00\x00\x00{"));
}

test "safetensors accepts leading and trailing JSON whitespace padding" {
    {
        var loaded = try safetensors.File.parse(allocator, "\x06\x00\x00\x00\x00\x00\x00\x00{}\x0d\x20\x09\x0a");
        defer loaded.deinit();
        try std.testing.expectEqual(@as(usize, 0), loaded.tensors.len);
    }
    {
        var loaded = try safetensors.File.parse(allocator, "\x06\x00\x00\x00\x00\x00\x00\x00\x09\x0a{}\x0d\x20");
        defer loaded.deinit();
        try std.testing.expectEqual(@as(usize, 0), loaded.tensors.len);
    }
}

test "safetensors supports zero-sized tensors and rejects invalid tensor info" {
    {
        const serialized = "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,0],\"data_offsets\":[0, 0]}}";
        var loaded = try safetensors.File.parse(allocator, serialized);
        defer loaded.deinit();
        const shape = [_]usize{ 2, 0 };
        const tensor = try loaded.tensor("test");
        try std.testing.expectEqual(safetensors.DType.I32, tensor.dtype);
        try std.testing.expectEqualSlices(usize, &shape, tensor.shape);
        try std.testing.expectEqual(@as(usize, 0), tensor.data.len);
    }
    {
        const serialized = "<\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,2],\"data_offsets\":[0, 4]}}";
        try std.testing.expectError(safetensors.Error.TensorInvalidInfo, safetensors.File.parse(allocator, serialized));
    }
}

test "safetensors rejects validation overflow" {
    const overflow_shape = "O\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,18446744073709551614],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    try std.testing.expectError(safetensors.Error.ValidationOverflow, safetensors.File.parse(allocator, overflow_shape));

    const overflow_bits = "N\x00\x00\x00\x00\x00\x00\x00{\"test\":{\"dtype\":\"I32\",\"shape\":[2,9223372036854775807],\"data_offsets\":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    try std.testing.expectError(safetensors.Error.ValidationOverflow, safetensors.File.parse(allocator, overflow_bits));
}

test "safetensors readPrefix consumes one safetensors prefix and leaves trailing frames" {
    const shape = [_]usize{3};
    const data = [_]u8{ 1, 2, 3 };
    const tensors = [_]safetensors.Tensor{.{
        .name = "w",
        .dtype = .U8,
        .shape = &shape,
        .data = &data,
    }};
    const prefix = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(prefix);

    const frame = "TAIL\x05\x00\x00\x00\x00\x00\x00\x00";
    const combined = try allocator.alloc(u8, prefix.len + frame.len);
    defer allocator.free(combined);
    @memcpy(combined[0..prefix.len], prefix);
    @memcpy(combined[prefix.len..], frame);

    var reader = std.Io.Reader.fixed(combined);
    var loaded = try safetensors.readPrefix(allocator, &reader);
    defer loaded.deinit();
    const tensor = try loaded.tensor("w");
    try std.testing.expectEqualSlices(u8, &data, tensor.data);
    const magic = try reader.takeArray(4);
    try std.testing.expectEqualSlices(u8, "TAIL", magic[0..]);
}

test "safetensors File.load malformed parse frees owned bytes once" {
    const malformed = [_]u8{ 1, 0, 0, 0, 0, 0, 0, 0, '{' };

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "safetensors_bad_{d}.safetensors", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    {
        var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{});
        defer file.close(std.testing.io);
        var write_buffer: [64]u8 = undefined;
        var writer = file.writer(std.testing.io, &write_buffer);
        try writer.interface.writeAll(&malformed);
        try writer.interface.flush();
    }

    try std.testing.expectError(safetensors.Error.InvalidHeaderDeserialization, safetensors.File.load(allocator, std.testing.io, path));
}

test "safetensors readPrefix malformed parse frees owned bytes once under OOM" {
    const shape = [_]usize{3};
    const data = [_]u8{ 1, 2, 3 };
    const tensors = [_]safetensors.Tensor{.{
        .name = "w",
        .dtype = .U8,
        .shape = &shape,
        .data = &data,
    }};
    const prefix = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(prefix);

    for (0..64) |fail_index| {
        var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = fail_index });
        var reader = std.Io.Reader.fixed(prefix);
        var loaded = safetensors.readPrefix(failing.allocator(), &reader) catch |err| {
            try std.testing.expect(err == error.OutOfMemory or err == safetensors.Error.InvalidHeaderDeserialization);
            continue;
        };
        loaded.deinit();
    }
}

test "safetensors readPrefix rejects wrapped total length" {
    const huge = std.math.maxInt(usize) - 4;
    const header = try std.fmt.allocPrint(
        allocator,
        "{{\"w\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}",
        .{ huge, huge },
    );
    defer allocator.free(header);

    const bytes = try allocator.alloc(u8, 8 + header.len);
    defer allocator.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..], header);

    var reader = std.Io.Reader.fixed(bytes);
    try std.testing.expectError(safetensors.Error.ValidationOverflow, safetensors.readPrefix(allocator, &reader));
}

test "safetensors mmap load can be saved back to the same path atomically" {
    const shape = [_]usize{3};
    const data = [_]u8{
        0, 0, 128, 63,
        0, 0, 0,   64,
        0, 0, 64,  64,
    };
    const tensors = [_]safetensors.Tensor{.{
        .name = "w",
        .dtype = .F32,
        .shape = &shape,
        .data = &data,
    }};

    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "safetensors_test_{d}.safetensors", .{std.Io.Clock.real.now(std.testing.io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    try safetensors.saveFileAtomic(allocator, std.testing.io, path, &tensors, null);
    {
        var mapped = try safetensors.File.loadMmap(allocator, std.testing.io, path);
        defer mapped.deinit();
        const w = try mapped.tensor("w");
        const mmap_tensors = [_]safetensors.Tensor{.{
            .name = w.name,
            .dtype = w.dtype,
            .shape = w.shape,
            .data = w.data,
        }};
        try safetensors.saveFileAtomic(allocator, std.testing.io, path, &mmap_tensors, null);
    }

    var reloaded = try safetensors.File.load(allocator, std.testing.io, path);
    defer reloaded.deinit();
    const w = try reloaded.tensor("w");
    try std.testing.expectEqualSlices(u8, &data, w.data);
}

test "safetensors many-tensor model-like roundtrip" {
    const shape2 = [_]usize{ 2, 3 };
    const shape1 = [_]usize{3};
    const wte = [_]u8{0} ** (2 * 3 * 4);
    const wpe = [_]u8{1} ** (2 * 3 * 4);
    const ln = [_]u8{2} ** (3 * 4);
    const bias = [_]u8{3} ** (3 * 4);
    const tensors = [_]safetensors.Tensor{
        .{ .name = "wte", .dtype = .F32, .shape = &shape2, .data = &wte },
        .{ .name = "wpe", .dtype = .F32, .shape = &shape2, .data = &wpe },
        .{ .name = "h.0.ln_1.weight", .dtype = .F32, .shape = &shape1, .data = &ln },
        .{ .name = "h.0.ln_1.bias", .dtype = .F32, .shape = &shape1, .data = &bias },
    };
    const out = try safetensors.serializeAlloc(allocator, &tensors, null);
    defer allocator.free(out);
    var loaded = try safetensors.File.parse(allocator, out);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, tensors.len), loaded.tensors.len);
    for (tensors) |tensor| {
        const got = try loaded.tensor(tensor.name);
        try std.testing.expectEqual(tensor.dtype, got.dtype);
        try std.testing.expectEqualSlices(usize, tensor.shape, got.shape);
        try std.testing.expectEqualSlices(u8, tensor.data, got.data);
    }
}

test "safetensors roundtrips every dtype tag" {
    const dtypes = [_]safetensors.DType{
        .BOOL,
        .F4,
        .F6_E2M3,
        .F6_E3M2,
        .U8,
        .I8,
        .F8_E5M2,
        .F8_E4M3,
        .F8_E8M0,
        .F8_E4M3FNUZ,
        .F8_E5M2FNUZ,
        .I16,
        .U16,
        .F16,
        .BF16,
        .I32,
        .U32,
        .F32,
        .C64,
        .F64,
        .I64,
        .U64,
    };
    for (dtypes, 0..) |dtype, i| {
        const shape = if (dtype.bitsize() == 4)
            &[_]usize{2}
        else if (dtype.bitsize() == 6)
            &[_]usize{4}
        else
            &[_]usize{2};
        const byte_len = try dtypeByteLen(dtype, shape);
        const data = try allocator.alloc(u8, byte_len);
        defer allocator.free(data);
        for (data, 0..) |*b, j| b.* = @truncate(i + j);

        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "dtype_{d}", .{i});
        const tensors = [_]safetensors.Tensor{.{ .name = name, .dtype = dtype, .shape = shape, .data = data }};
        const out = try safetensors.serializeAlloc(allocator, &tensors, null);
        defer allocator.free(out);
        var loaded = try safetensors.File.parse(allocator, out);
        defer loaded.deinit();
        const tensor = try loaded.tensor(name);
        try std.testing.expectEqual(dtype, tensor.dtype);
        try std.testing.expectEqualSlices(usize, shape, tensor.shape);
        try std.testing.expectEqualSlices(u8, data, tensor.data);
    }
}

fn dtypeByteLen(dtype: safetensors.DType, shape: []const usize) !usize {
    var elems: usize = 1;
    for (shape) |dim| elems = try std.math.mul(usize, elems, dim);
    const bits = try std.math.mul(usize, elems, dtype.bitsize());
    if (bits % 8 != 0) return error.MisalignedTestShape;
    return bits / 8;
}
