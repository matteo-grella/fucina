//! GPU residency for weight storage: the session `ResidentByteRegistry`,
//! wrapping freshly copied device bytes as owning tensors
//! (`gpuResidentQuantTensor`, `gpuResidentDenseTensor`), and in-place
//! promotion of already-loaded weights (`makeGpuResidentDenseWeight`,
//! `makeGpuResidentQuantWeight`). Comptime-elided on non-gpu builds.

const std = @import("std");

const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");

const common = @import("common.zig");

const offload = backend_mod.offload;
const DType = dtype_mod.DType;
const ExecContext = exec_mod.ExecContext;
const Error = common.Error;
const QuantWeight = common.QuantWeight;
const BlockStorage = common.BlockStorage;
const Allocator = std.mem.Allocator;

/// Session/model-owned registry for immutable byte payloads that should be
/// copied once into device-owned storage when a capable GPU provider is built.
/// The returned bytes are still CPU-readable and remain ordinary RHS storage;
/// this only changes the backing allocation used by GPU matmul accelerators.
pub const ResidentByteRegistry = struct {
    allocator: Allocator,
    map: std.AutoHashMapUnmanaged(usize, []const u8) = .empty,

    pub fn init(allocator: Allocator) ResidentByteRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ResidentByteRegistry) void {
        if (comptime offload.enabled) {
            var it = self.map.iterator();
            while (it.next()) |e| {
                offload.freeResidentBytes(e.value_ptr.*);
            }
        }
        self.map.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn bytes(self: *ResidentByteRegistry, src: []const u8) []const u8 {
        if (comptime !offload.enabled) return src;
        const key = @intFromPtr(src.ptr);
        if (self.map.get(key)) |dev| return dev;
        self.map.ensureUnusedCapacity(self.allocator, 1) catch return src;
        const dev = offload.allocResidentBytes(src.len) orelse return src;
        @memcpy(dev, src);
        self.map.putAssumeCapacityNoClobber(key, dev);
        return dev;
    }
};

pub fn dtypeHasDenseQuantGpuKernel(comptime dtype: DType) bool {
    return comptime offload.supportsQuantDType(dtype);
}

/// Wrap freshly copied device-resident blocks (`internal.gpu.allocResidentBytes`)
/// in a tensor whose storage OWNS the device bytes: the final buffer release
/// (refs==0, counting cloneView'd weights that share it) frees them via
/// `freeResidentBytes`, which also evicts the shim's cached wrap for that base
/// address. Takes ownership of `dev` — freed here on error.
pub fn gpuResidentQuantTensor(comptime dtype: DType, ctx: *ExecContext, shape: [2]usize, dev: []u8) !QuantWeight(dtype) {
    const Raw = tensor_mod.TensorOf(dtype);
    const DevBuffer = std.meta.Child(@FieldType(Raw, "buffer"));
    const hook = struct {
        fn releaseDeviceBytes(_: *anyopaque, buffer: *DevBuffer) void {
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            offload.freeResidentBytes(bytes);
        }
    };
    var dev_owned: ?[]u8 = dev;
    errdefer if (dev_owned) |bytes| offload.freeResidentBytes(bytes);
    const dev_blocks = try blockSliceMut(BlockStorage(dtype), dev);
    const buffer = try DevBuffer.fromBorrowedSliceWithRelease(ctx.allocator, dev_blocks, hook.releaseDeviceBytes);
    dev_owned = null; // from here the buffer's release hook frees the device bytes
    var raw = Raw.fromOwnedBuffer(buffer, &shape) catch |err| {
        buffer.release();
        return err;
    };
    errdefer raw.deinit();
    return QuantWeight(dtype).fromTensor(ctx, raw);
}

/// Dense scalar analog of `gpuResidentQuantTensor`: wrap device-resident
/// bytes as a dense [out, in] weight tensor whose storage OWNS them (same
/// release-hook contract). Managed residency keeps the bytes CPU-readable
/// AND CPU-writable at the same pointer, so in-place trainers (fucina.es)
/// can mutate resident weights and GPU dispatches read the live values —
/// the dense f32/f16 GEMM paths never adopt-copy, so there is no stale
/// snapshot to fence.
pub fn gpuResidentDenseTensor(comptime dtype: DType, comptime Facade: type, ctx: *ExecContext, shape: [2]usize, dev: []u8) !Facade {
    const Raw = tensor_mod.TensorOf(dtype);
    const DevBuffer = std.meta.Child(@FieldType(Raw, "buffer"));
    const hook = struct {
        fn releaseDeviceBytes(_: *anyopaque, buffer: *DevBuffer) void {
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            offload.freeResidentBytes(bytes);
        }
    };
    var dev_owned: ?[]u8 = dev;
    errdefer if (dev_owned) |bytes| offload.freeResidentBytes(bytes);
    const Elem = std.meta.Child(@FieldType(DevBuffer, "data"));
    if (dev.len % @sizeOf(Elem) != 0) return Error.InvalidWeightShape;
    const elems: []Elem = @alignCast(std.mem.bytesAsSlice(Elem, dev));
    const buffer = try DevBuffer.fromBorrowedSliceWithRelease(ctx.allocator, elems, hook.releaseDeviceBytes);
    dev_owned = null; // from here the buffer's release hook frees the device bytes
    var raw = Raw.fromOwnedBuffer(buffer, &shape) catch |err| {
        buffer.release();
        return err;
    };
    errdefer raw.deinit();
    return Facade.fromTensor(ctx, raw);
}

/// Dense analog of `makeGpuResidentQuantWeight`: move a dense weight's
/// storage into GPU-resident bytes (fused concat results are heap tensors —
/// their fusion parts were loaded with `loadForFusion`, skipping per-part
/// residency on purpose). No-op (false) when the GPU is off or the budget
/// is exhausted.
pub fn makeGpuResidentDenseWeight(comptime dtype: DType, comptime Facade: type, ctx: *ExecContext, value: *Facade) !bool {
    if (comptime !offload.enabled) return false;
    const elems = try value.dataConst();
    const bytes = std.mem.sliceAsBytes(elems);
    const dev = offload.allocResidentBytes(bytes.len) orelse return false;
    @memcpy(dev, bytes);
    const raw_shape = value.asRawTensor().shape.slice();
    const shape = [2]usize{ raw_shape[0], raw_shape[1] };
    var resident = gpuResidentDenseTensor(dtype, Facade, ctx, shape, dev) catch |err| {
        return err;
    };
    errdefer resident.deinit();
    value.deinit();
    value.* = resident;
    return true;
}

pub fn makeGpuResidentQuantWeight(comptime dtype: DType, ctx: *ExecContext, value: *QuantWeight(dtype)) !bool {
    if (comptime !offload.supportsQuantDType(dtype)) return false;
    const blocks = try value.dataConst();
    const bytes = std.mem.sliceAsBytes(blocks);
    const dev = offload.allocResidentBytes(bytes.len) orelse return false;
    @memcpy(dev, bytes);
    var resident = try gpuResidentQuantTensor(dtype, ctx, value.shape(), dev);
    errdefer resident.deinit();
    value.deinit();
    value.* = resident;
    return true;
}

fn blockSliceMut(comptime Elem: type, bytes: []u8) ![]Elem {
    if (bytes.len % @sizeOf(Elem) != 0) return Error.InvalidWeightShape;
    if (@intFromPtr(bytes.ptr) % @alignOf(Elem) != 0) return Error.InvalidWeightShape;
    const aligned: []align(@alignOf(Elem)) u8 = @alignCast(bytes);
    return std.mem.bytesAsSlice(Elem, aligned);
}
