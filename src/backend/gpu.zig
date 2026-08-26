//! Comptime GPU provider selector (`-Dgpu`): resolves `gpu_impl` to the
//! active provider module. A leaf on purpose — native.zig imports this
//! instead of a concrete provider (it cannot import backend.zig without
//! creating an import cycle; `zig build arch-check` enforces zero SCCs).
//!
//! Dead switch arms are parsed but never semantically analyzed, so the
//! unselected providers cost nothing and need none of their targets'
//! libraries: cuda.zig is fully inert on macOS builds, metal.zig on Linux,
//! and the default `.none` build analyzes neither (it resolves to the null
//! provider `gpu_none.zig`). The compile-only legs keep the unselected
//! arms honest: `zig build -Dgpu=metal` on macOS, `zig build cuda-check`
//! for the CUDA arm on GPU-less dev machines.
const build_options = @import("build_options");

pub const provider = @import("gpu_provider.zig");

pub const impl = switch (build_options.gpu_kind) {
    // The null provider: `enabled = false`, every capability false, every
    // call site comptime-elides past it.
    .none => @import("gpu_none.zig"),
    .metal => @import("metal.zig"),
    .cuda => @import("cuda.zig"),
};

comptime {
    // One source of truth: build.zig derives both spellings from -Dgpu.
    if (build_options.use_gpu != (build_options.gpu_kind != .none))
        @compileError("build_options.use_gpu and build_options.gpu_kind are out of sync");

    // The selected provider implements the whole interface, checked on
    // signatures only (no function addresses taken). The unselected real
    // providers ride their compile legs: `zig build metal-check` for the
    // Metal arm, `zig build cuda-check` for the CUDA arm.
    provider.assertConforms(impl);
}

test {
    _ = impl; // forward the active provider's tests to the backend test root
    _ = @import("gpu_policy.zig"); // shared gate policy: provider-independent
    // The shared suite calls the provider directly, so it belongs to GPU
    // builds only: on `-Dgpu=none` the null provider has no device and no
    // kernels to drive. Same guard backend.zig puts on this whole import.
    if (comptime build_options.use_gpu) _ = @import("gpu_conformance.zig");
}
