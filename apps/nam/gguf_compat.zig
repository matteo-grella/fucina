//! GGUF <-> .nam interchange: the GGUF carries one flat f32 tensor
//! `nam.weights` mirroring the .nam weights contract, plus the ENTIRE
//! original .nam JSON byte-verbatim in the string KV `nam.file_json` (GGUF
//! KV metadata cannot represent NAM's nested `config` objects/arrays) — so
//! GGUF -> .nam export is byte-identical by construction, the strongest
//! possible round-trip guarantee. Convenience KVs (general.architecture =
//! "nam", nam.architecture/version/sample_rate) make the file
//! self-describing to GGUF tooling. Quantization is refused by design
//! (13.8k-param models: no block-divisible dims, no bandwidth win, real
//! ESR risk).

const std = @import("std");
const fucina = @import("fucina");
const nam_file = @import("nam_file.zig");

pub const file_json_key = "nam.file_json";

/// .nam -> GGUF. `model` must have been loaded with its raw bytes retained
/// (the default loader behavior).
pub fn exportGguf(io: std.Io, allocator: std.mem.Allocator, model: *const nam_file.NamModel, path: []const u8) !void {
    var writer = fucina.gguf.Writer.init(allocator);
    defer writer.deinit();

    try writer.addMetaString("general.architecture", "nam");
    const arch_name = switch (model.architecture) {
        .wavenet => "WaveNet",
        .lstm => "LSTM",
        .convnet => "ConvNet",
        .linear => "Linear",
    };
    try writer.addMetaString("nam.architecture", arch_name);
    var version_buf: [32]u8 = undefined;
    const version = try std.fmt.bufPrint(&version_buf, "{d}.{d}.{d}", .{ model.version.major, model.version.minor, model.version.patch });
    try writer.addMetaString("nam.version", version);
    try writer.addMetaFloat("nam.sample_rate", f64, model.sample_rate);
    try writer.addMetaString(file_json_key, model.raw_bytes);

    try writer.addTensor("nam.weights", .f32, &.{model.weights.len}, std.mem.sliceAsBytes(model.weights));

    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var file_writer = file.writer(io, &buffer);
    try writer.finish(&file_writer.interface);
    try file_writer.interface.flush();
}

/// GGUF -> NamModel: parses the embedded byte-verbatim .nam document. The
/// returned model is exactly what loading the original .nam yields
/// (including raw bytes for a further byte-identical re-export).
pub fn loadFromGguf(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !nam_file.NamModel {
    var file = try fucina.gguf.File.load(allocator, io, path);
    defer file.deinit();
    const json = file.getString(file_json_key) orelse return error.NotANamGguf;
    return nam_file.loadFromSlice(allocator, json);
}

/// Loads a profile from either format by extension.
pub fn loadAny(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !nam_file.NamModel {
    if (std.ascii.endsWithIgnoreCase(path, ".gguf")) {
        return loadFromGguf(io, allocator, path);
    }
    return nam_file.loadFile(io, allocator, path);
}

/// GGUF -> .nam file on disk; byte-identical to the original .nam.
pub fn exportNamFromGguf(io: std.Io, allocator: std.mem.Allocator, gguf_path: []const u8, nam_path: []const u8) !void {
    var model = try loadFromGguf(io, allocator, gguf_path);
    defer model.deinit();
    var file = try std.Io.Dir.cwd().createFile(io, nam_path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(model.raw_bytes);
    try writer.interface.flush();
}
