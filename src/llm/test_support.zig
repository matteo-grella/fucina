//! Shared gates for model- and fixture-dependent tests: ONE definition of
//! the native-only guard and of open-or-skip file loading, replacing the
//! per-file hand-rolled variants that made silent skips unauditable.
//! Setting `FUCINA_TEST_REQUIRE_MODELS=1` turns a missing model or fixture
//! into a FAILURE instead of a skip, so a rig run cannot silently lose
//! coverage (the historical trap: model-gated tests skipping in a checkout
//! without the models/ symlink).
const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");

/// Skip unless the native backend is active. Real-model goldens verify
/// model WIRING, which is backend-independent; re-running them on the
/// scalar reference kernels re-buys kernel coverage the scalar leg already
/// owns, at model-forward cost.
pub fn requireNative() error{SkipZigTest}!void {
    if (comptime fucina.internal.backend_mod.active_kind != .native) return error.SkipZigTest;
}

fn requireModels() bool {
    // Same env-read caveat as the tuning band: libc-free builds have no
    // `std.c.getenv`; there the switch reads as unset and skips stay skips.
    if (!builtin.link_libc) return false;
    return std.c.getenv("FUCINA_TEST_REQUIRE_MODELS") != null;
}

fn missing(path: []const u8) error{ SkipZigTest, MissingTestInput } {
    if (requireModels()) {
        std.debug.print("FUCINA_TEST_REQUIRE_MODELS=1 and test input is missing: {s}\n", .{path});
        return error.MissingTestInput;
    }
    return error.SkipZigTest;
}

/// Mmap-open a GGUF, skipping the test when the file is absent (failing
/// under `FUCINA_TEST_REQUIRE_MODELS=1`).
pub fn openGgufOrSkip(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !fucina.gguf.File {
    return fucina.gguf.File.loadMmap(allocator, io, path) catch |err| switch (err) {
        error.FileNotFound => missing(path),
        else => err,
    };
}

/// Read a fixture file fully (caller frees), skipping the test when it is
/// absent (failing under `FUCINA_TEST_REQUIRE_MODELS=1`).
pub fn readFileOrSkip(allocator: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(limit)) catch |err| switch (err) {
        error.FileNotFound => missing(path),
        else => err,
    };
}

/// Exact-bit assertions hold on the recording machine class only (aarch64
/// native, Accelerate BLAS); other ISAs and BLAS providers reassociate at
/// the ulp level. Recorded-golden tests assert argmax chains everywhere
/// and gate their FNV hash checks on this.
pub const strict_bits = builtin.cpu.arch == .aarch64;

pub fn argmaxRow(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |x, i| {
        if (x > row[best]) best = i;
    }
    return best;
}

/// FNV-1a 64 over the raw f32 bytes: the exact-bit fingerprint the strict
/// recorded-golden gates compare.
pub fn fnvHash(values: []const f32) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (std.mem.sliceAsBytes(values)) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    }
    return h;
}
