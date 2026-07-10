//! Comptime GPU provider selector (`-Dgpu`): resolves `gpu_impl` to the
//! active provider module. A leaf on purpose — native.zig imports this
//! instead of a concrete provider (it cannot import backend.zig without
//! creating an import cycle; `zig build arch-check` enforces zero SCCs).
//!
//! Dead switch arms are parsed but never semantically analyzed, so the
//! unselected provider costs nothing and needs none of its target's
//! libraries: cuda.zig is fully inert on macOS builds, metal.zig on Linux.
//! (`zig build cuda-check` is the compile-only cross-target leg that keeps
//! the CUDA arm honest on GPU-less dev machines.)
const build_options = @import("build_options");

pub const impl = switch (build_options.gpu_kind) {
    // metal.zig doubles as the `.none` resolution: its `enabled` flag is
    // false then and every call site comptime-elides past it (the pre-cuda
    // status quo).
    .none, .metal => @import("metal.zig"),
    .cuda => @import("cuda.zig"),
};

comptime {
    // One source of truth: build.zig derives both spellings from -Dgpu.
    if (build_options.use_gpu != (build_options.gpu_kind != .none))
        @compileError("build_options.use_gpu and build_options.gpu_kind are out of sync");
}

test {
    _ = impl; // forward the active provider's tests to the backend test root
}
