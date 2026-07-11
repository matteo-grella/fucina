//! Compile the vendored CUDA kernels to PTX through NVRTC.
//!
//! NVRTC is also Fucina's `FUCINA_GPU_KERNELS=src` compiler. Keeping artifact
//! generation on the same frontend prevents the shipped PTX and the measured
//! development path from silently receiving different optimization decisions.

const std = @import("std");
const cuda_api = @import("cuda_api");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 4) return error.InvalidArguments;

    const source = try std.Io.Dir.cwd().readFileAlloc(io, args[1], allocator, .limited(16 * 1024 * 1024));
    const source_z = try allocator.dupeZ(u8, source);
    var arch_buf: [128]u8 = undefined;
    const arch_opt = try std.fmt.bufPrintZ(&arch_buf, "--gpu-architecture={s}", .{args[3]});

    var nvrtc = try cuda_api.Nvrtc.load();
    defer nvrtc.lib.close();
    var program: cuda_api.NvrtcProgram = null;
    if (nvrtc.nvrtcCreateProgram(&program, source_z.ptr, "fucina_kernels.cu", 0, null, null) != 0) {
        return error.NvrtcCreateFailed;
    }
    defer _ = nvrtc.nvrtcDestroyProgram(&program);

    const options = [_][*:0]const u8{
        arch_opt.ptr,
        "--std=c++17",
        "-I/usr/include",
        "-I/usr/local/cuda/include",
    };
    const compile_result = nvrtc.nvrtcCompileProgram(program, options.len, &options);
    var log_size: usize = 0;
    _ = nvrtc.nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
        const log = try allocator.alloc(u8, log_size);
        _ = nvrtc.nvrtcGetProgramLog(program, log.ptr);
        var stderr_buf: [4096]u8 = undefined;
        var stderr_writer = std.Io.File.stderr().writer(io, &stderr_buf);
        try stderr_writer.interface.writeAll(log[0 .. log_size - 1]);
        try stderr_writer.interface.flush();
    }
    if (compile_result != 0) return error.NvrtcCompileFailed;

    var ptx_size: usize = 0;
    if (nvrtc.nvrtcGetPTXSize(program, &ptx_size) != 0 or ptx_size <= 1) {
        return error.NvrtcGetPtxFailed;
    }
    const ptx = try allocator.alloc(u8, ptx_size);
    if (nvrtc.nvrtcGetPTX(program, ptx.ptr) != 0) return error.NvrtcGetPtxFailed;

    var output = try std.Io.Dir.cwd().createFile(io, args[2], .{});
    defer output.close(io);
    var output_buf: [64 * 1024]u8 = undefined;
    var output_writer = output.writer(io, &output_buf);
    try output_writer.interface.writeAll(ptx[0 .. ptx_size - 1]);
    try output_writer.interface.flush();
}
