//! CUDA driver API + cuBLAS bindings for the `-Dgpu=cuda` provider.
//!
//! Everything here is a hand-declared FUNCTION-POINTER prototype resolved at
//! runtime through `std.DynLib` — deliberately NOT link-time `extern fn`s
//! (metal.zig's style): those would make libcuda an undefined symbol at link
//! and break the zero-CUDA-SDK build/cross-compile story. No `@cImport`
//! anywhere, per repo convention; no CUDA toolkit is needed to build, and a
//! machine with only the NVIDIA driver (no toolkit) still loads the driver
//! API half.
//!
//! Missing libraries degrade per-capability: no `libcuda` disables the
//! provider entirely; no `libcublas` disables only the f32/f16 GEMM arms.
const std = @import("std");

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUdeviceptr = u64;
pub const CUcontext = ?*anyopaque;
pub const CUstream = ?*anyopaque;

pub const CUmodule = ?*anyopaque;
pub const CUfunction = ?*anyopaque;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: c_int = 89;
pub const CU_MEM_ATTACH_GLOBAL: c_uint = 1;
pub const CU_MEM_ADVISE_SET_READ_MOSTLY: c_int = 1;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

fn Sym(comptime F: type) type {
    return *const F;
}

/// Struct-field name → exported symbol name. The CUDA driver and cuBLAS keep
/// ABI-versioned suffixes on some entry points; the field names stay clean.
fn symName(comptime field: []const u8) [:0]const u8 {
    @setEvalBranchQuota(100_000);
    const v2 = [_][]const u8{
        "cuDeviceTotalMem", "cuMemAlloc",    "cuMemFree",       "cuMemcpyHtoD",
        "cuMemcpyDtoH",     "cuStreamDestroy", "cublasCreate",  "cublasDestroy",
        "cublasSetStream",  "cublasSgemm",   "cuMemGetInfo",
    };
    inline for (v2) |name| {
        if (comptime std.mem.eql(u8, field, name)) return field ++ "_v2";
    }
    return field ++ "";
}

/// Resolve every function-pointer field of `T` from its already-open `lib`.
fn loadAll(comptime T: type, self: *T) error{MissingSymbol}!void {
    @setEvalBranchQuota(100_000);
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (comptime std.mem.eql(u8, f.name, "lib")) continue;
        @field(self, f.name) = self.lib.lookup(f.type, comptime symName(f.name)) orelse {
            std.log.warn("fucina-cuda: {s} missing symbol {s}", .{ @typeName(T), comptime symName(f.name) });
            return error.MissingSymbol;
        };
    }
}

/// CUDA driver API (`libcuda.so.1`, installed by the NVIDIA driver itself).
/// The symbol set below is ABI-stable since CUDA 11.0,
/// the supported floor.
pub const Driver = struct {
    lib: std.DynLib,

    cuInit: Sym(fn (c_uint) callconv(.c) CUresult),
    cuDriverGetVersion: Sym(fn (*c_int) callconv(.c) CUresult),
    cuDeviceGetCount: Sym(fn (*c_int) callconv(.c) CUresult),
    cuDeviceGet: Sym(fn (*CUdevice, c_int) callconv(.c) CUresult),
    cuDeviceGetName: Sym(fn ([*]u8, c_int, CUdevice) callconv(.c) CUresult),
    cuDeviceTotalMem: Sym(fn (*usize, CUdevice) callconv(.c) CUresult),
    cuDeviceGetAttribute: Sym(fn (*c_int, c_int, CUdevice) callconv(.c) CUresult),
    cuDevicePrimaryCtxRetain: Sym(fn (*CUcontext, CUdevice) callconv(.c) CUresult),
    cuCtxSetCurrent: Sym(fn (CUcontext) callconv(.c) CUresult),
    cuMemAlloc: Sym(fn (*CUdeviceptr, usize) callconv(.c) CUresult),
    cuMemFree: Sym(fn (CUdeviceptr) callconv(.c) CUresult),
    cuMemcpyHtoD: Sym(fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult),
    cuMemcpyDtoH: Sym(fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult),
    cuStreamCreate: Sym(fn (*CUstream, c_uint) callconv(.c) CUresult),
    cuStreamSynchronize: Sym(fn (CUstream) callconv(.c) CUresult),
    cuGetErrorString: Sym(fn (CUresult, *?[*:0]const u8) callconv(.c) CUresult),
    cuMemAllocManaged: Sym(fn (*CUdeviceptr, usize, c_uint) callconv(.c) CUresult),
    cuMemAdvise: Sym(fn (CUdeviceptr, usize, c_int, CUdevice) callconv(.c) CUresult),
    cuMemPrefetchAsync: Sym(fn (CUdeviceptr, usize, CUdevice, CUstream) callconv(.c) CUresult),
    cuMemGetInfo: Sym(fn (*usize, *usize) callconv(.c) CUresult),
    cuMemHostAlloc: Sym(fn (*?*anyopaque, usize, c_uint) callconv(.c) CUresult),
    cuMemFreeHost: Sym(fn (?*anyopaque) callconv(.c) CUresult),
    cuModuleLoadData: Sym(fn (*CUmodule, *const anyopaque) callconv(.c) CUresult),
    cuModuleGetFunction: Sym(fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult),
    cuFuncSetAttribute: Sym(fn (CUfunction, c_int, c_int) callconv(.c) CUresult),
    cuLaunchKernel: Sym(fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, ?[*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) CUresult),

    pub fn load() error{ LibraryNotFound, MissingSymbol }!Driver {
        var self: Driver = undefined;
        self.lib = std.DynLib.open("libcuda.so.1") catch return error.LibraryNotFound;
        errdefer self.lib.close();
        try loadAll(Driver, &self);
        return self;
    }

    pub fn errName(self: *const Driver, res: CUresult) []const u8 {
        var msg: ?[*:0]const u8 = null;
        _ = self.cuGetErrorString(res, &msg);
        return if (msg) |m| std.mem.span(m) else "unknown";
    }
};

// ---- cuBLAS (v2 ABI) --------------------------------------------------------

pub const CublasHandle = ?*anyopaque;
pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;
pub const CUBLAS_DEFAULT_MATH: c_int = 0;
pub const CUBLAS_TF32_TENSOR_OP_MATH: c_int = 3;
pub const CUDA_R_16F: c_int = 2;
pub const CUBLAS_COMPUTE_32F: c_int = 68;
pub const CUBLAS_GEMM_DEFAULT: c_int = -1;

/// cuBLAS ships with the toolkit/driver metapackages, not the bare driver.
/// Soname ladder instead of one hardcoded major: every symbol used here is
/// ABI-stable across cuBLAS 11/12/13.
pub const cublas_sonames = [_][]const u8{ "libcublas.so.13", "libcublas.so.12", "libcublas.so.11" };

// ---- NVRTC (optional dev-loop fallback: recompile kernels.cu at runtime) ----

pub const NvrtcProgram = ?*anyopaque;
pub const nvrtc_sonames = [_][]const u8{ "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11" };

pub const Nvrtc = struct {
    lib: std.DynLib,

    nvrtcCreateProgram: Sym(fn (*NvrtcProgram, [*:0]const u8, ?[*:0]const u8, c_int, ?[*]const [*:0]const u8, ?[*]const [*:0]const u8) callconv(.c) c_int),
    nvrtcCompileProgram: Sym(fn (NvrtcProgram, c_int, ?[*]const [*:0]const u8) callconv(.c) c_int),
    nvrtcGetPTXSize: Sym(fn (NvrtcProgram, *usize) callconv(.c) c_int),
    nvrtcGetPTX: Sym(fn (NvrtcProgram, [*]u8) callconv(.c) c_int),
    nvrtcGetProgramLogSize: Sym(fn (NvrtcProgram, *usize) callconv(.c) c_int),
    nvrtcGetProgramLog: Sym(fn (NvrtcProgram, [*]u8) callconv(.c) c_int),
    nvrtcDestroyProgram: Sym(fn (*NvrtcProgram) callconv(.c) c_int),

    pub fn load() error{LibraryNotFound}!Nvrtc {
        for (nvrtc_sonames) |soname| {
            var self: Nvrtc = undefined;
            self.lib = std.DynLib.open(soname) catch continue;
            loadAll(Nvrtc, &self) catch {
                self.lib.close();
                continue;
            };
            return self;
        }
        return error.LibraryNotFound;
    }
};

pub const Cublas = struct {
    lib: std.DynLib,

    cublasCreate: Sym(fn (*CublasHandle) callconv(.c) c_int),
    cublasDestroy: Sym(fn (CublasHandle) callconv(.c) c_int),
    cublasSetStream: Sym(fn (CublasHandle, CUstream) callconv(.c) c_int),
    cublasSetMathMode: Sym(fn (CublasHandle, c_int) callconv(.c) c_int),
    cublasSgemm: Sym(fn (CublasHandle, c_int, c_int, c_int, c_int, c_int, *const f32, CUdeviceptr, c_int, CUdeviceptr, c_int, *const f32, CUdeviceptr, c_int) callconv(.c) c_int),
    cublasSgemmStridedBatched: Sym(fn (CublasHandle, c_int, c_int, c_int, c_int, c_int, *const f32, CUdeviceptr, c_int, c_longlong, CUdeviceptr, c_int, c_longlong, *const f32, CUdeviceptr, c_int, c_longlong, c_int) callconv(.c) c_int),
    cublasGemmEx: Sym(fn (CublasHandle, c_int, c_int, c_int, c_int, c_int, *const anyopaque, CUdeviceptr, c_int, c_int, CUdeviceptr, c_int, c_int, *const anyopaque, CUdeviceptr, c_int, c_int, c_int, c_int) callconv(.c) c_int),

    /// Returns the loaded binding plus which soname resolved (for tracing).
    /// A soname that opens but misses a symbol (broken/partial package) falls
    /// through to the next rung instead of aborting the ladder.
    pub fn load() error{LibraryNotFound}!struct { api: Cublas, soname: []const u8 } {
        for (cublas_sonames) |soname| {
            var self: Cublas = undefined;
            self.lib = std.DynLib.open(soname) catch continue;
            loadAll(Cublas, &self) catch {
                self.lib.close();
                continue;
            };
            return .{ .api = self, .soname = soname };
        }
        return error.LibraryNotFound;
    }
};
