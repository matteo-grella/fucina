//! GGUF host-side readers: `blk.N` tensor-name formatting (`layerName`),
//! owned host f32 vectors and matrices (`hostVector`, `hostVectorInfo`,
//! `hostMatrix`), the 1-D tensor loader (`loadVector`), and the widening
//! fill (`fillF32`) the dense loaders share.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const ag_mod = @import("../ag.zig");
const gguf = @import("../gguf.zig");

const common = @import("common.zig");

const Tensor = ag_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const Error = common.Error;
const Tag = @TypeOf(.tag);
const Allocator = std.mem.Allocator;

pub fn layerName(buf: []u8, layer_i: usize, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ layer_i, suffix });
}

/// 1-D tensor as an owned host f32 slice (any decodable source dtype).
pub fn hostVector(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, expected: usize) ![]f32 {
    return hostVectorInfo(allocator, try file.get(tensor_name), expected);
}

pub fn hostVectorInfo(allocator: Allocator, info: *const gguf.TensorInfo, expected: usize) ![]f32 {
    if (info.n_dims != 1 or info.dims[0] != expected) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, expected);
    errdefer allocator.free(out);
    try fillF32(out, info);
    return out;
}

/// 2-D tensor (GGUF dims `{cols, rows}`) as an owned host row-major
/// `[rows][cols]` f32 slice.
pub fn hostMatrix(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, cols: usize, rows: usize) ![]f32 {
    const info = try file.get(tensor_name);
    if (info.n_dims != 2 or info.dims[0] != cols or info.dims[1] != rows) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);
    try fillF32(out, info);
    return out;
}

pub fn loadVector(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_len: usize, comptime tag: Tag) !Tensor(.{tag}) {
    if (info.n_dims != 1 or info.dims[0] != expected_len) return Error.InvalidWeightShape;

    var v = try Tensor(.{tag}).empty(ctx, .{expected_len});
    errdefer v.deinit();
    try fillF32(try v.data(), info);
    return v;
}

pub fn fillF32(out: []f32, info: *const gguf.TensorInfo) !void {
    switch (info.ggml_type) {
        .f32 => {
            if (info.data.len != out.len * @sizeOf(f32)) return Error.InvalidWeightShape;
            @memcpy(std.mem.sliceAsBytes(out), info.data);
        },
        .f16 => {
            if (info.data.len != out.len * @sizeOf(u16)) return Error.InvalidWeightShape;
            for (out, 0..) |*dst, i| {
                const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
                const half: f16 = @bitCast(bits);
                dst.* = @floatCast(half);
            }
        },
        .bf16 => {
            if (info.data.len != out.len * @sizeOf(u16)) return Error.InvalidWeightShape;
            for (out, 0..) |*dst, i| {
                const bits = std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little);
                dst.* = dtype_mod.bf16ToF32(bits);
            }
        },
        .f64 => {
            if (info.data.len != out.len * @sizeOf(f64)) return Error.InvalidWeightShape;
            for (out, 0..) |*dst, i| {
                const bits = std.mem.readInt(u64, info.data[i * 8 ..][0..8], .little);
                const value: f64 = @bitCast(bits);
                dst.* = @floatCast(value);
            }
        },
        else => return Error.UnsupportedWeightType,
    }
}
