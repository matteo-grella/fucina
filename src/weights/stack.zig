//! Contiguous byte stacks of same-shaped quantized linear weights
//! (`QuantByteStack`, `makeQuantByteStack`): the building block for eager
//! dispatch batching, with the same device-residency policy the generic
//! loaders use.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");

const common = @import("common.zig");
const gpu = @import("gpu.zig");

const offload = backend_mod.offload;
const DType = dtype_mod.DType;
const Error = common.Error;
const Allocator = std.mem.Allocator;

/// One raw quantized linear weight to be copied into a contiguous stack.
/// `data` is the GGUF block payload for a `[out, in]` matrix.
pub const QuantByteStackPart = struct {
    data: []const u8,
    in: usize,
    out: usize,
};

pub const QuantByteStackOptions = struct {
    /// Prefer device-owned storage when the active provider implements the
    /// dtype's quantized kernel.
    prefer_device: bool = true,
    /// Return null instead of heap-allocating when device-owned storage is not
    /// available. Use this for GPU-only command-batched paths.
    require_device: bool = false,
};

/// Internal byte stack for same-shaped quantized linear weights. The stack is
/// CPU-readable either way; `device_owned=true` additionally means the bytes
/// live in provider-owned storage and are safe for cached wraps until deinit.
pub const QuantByteStack = struct {
    dtype: DType,
    count: usize,
    in: usize,
    out: usize,
    bytes_per_weight: usize,
    data: []u8,
    device_owned: bool,

    pub fn deinit(self: *QuantByteStack, allocator: Allocator) void {
        if (self.device_owned) {
            if (comptime offload.enabled) offload.freeResidentBytes(self.data);
        } else {
            allocator.free(self.data);
        }
        self.* = undefined;
    }

    pub fn bytesPerRow(self: *const QuantByteStack) usize {
        return self.bytes_per_weight / self.out;
    }

    pub fn totalOutRows(self: *const QuantByteStack) usize {
        return self.count * self.out;
    }
};

/// Build a contiguous stack of same-shaped quantized linear weights with the
/// same residency policy used by the generic loaders. This is a private
/// building block for eager dispatch batching: callers still return ordinary
/// tensors and still fall back when the backend declines the GPU path.
pub fn makeQuantByteStack(
    comptime dtype: DType,
    allocator: Allocator,
    parts: []const QuantByteStackPart,
    options: QuantByteStackOptions,
) !?QuantByteStack {
    if (parts.len == 0) return null;
    const first = parts[0];
    if (first.in == 0 or first.out == 0 or first.data.len == 0) return Error.InvalidWeightShape;
    if (first.data.len % first.out != 0) return Error.InvalidWeightShape;
    const bytes_per_weight = first.data.len;
    for (parts[1..]) |part| {
        if (part.in != first.in or part.out != first.out or part.data.len != bytes_per_weight) {
            return Error.InvalidWeightShape;
        }
    }

    const total_len = try std.math.mul(usize, bytes_per_weight, parts.len);
    _ = try std.math.mul(usize, first.out, parts.len);
    var device_owned = false;
    const data: []u8 = blk: {
        if (options.prefer_device and comptime offload.enabled and gpu.dtypeHasDenseQuantGpuKernel(dtype)) {
            if (offload.allocResidentBytes(total_len)) |dev| {
                device_owned = true;
                break :blk dev;
            }
        }
        if (options.require_device) return null;
        break :blk try allocator.alloc(u8, total_len);
    };
    errdefer if (!device_owned) allocator.free(data);

    for (parts, 0..) |part, i| {
        @memcpy(data[i * bytes_per_weight ..][0..bytes_per_weight], part.data);
    }

    return .{
        .dtype = dtype,
        .count = parts.len,
        .in = first.in,
        .out = first.out,
        .bytes_per_weight = bytes_per_weight,
        .data = data,
        .device_owned = device_owned,
    };
}
