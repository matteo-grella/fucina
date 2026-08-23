//! The GPU provider interface: ONE statement of the surface every `-Dgpu`
//! provider implements, plus the wire types they share with their kernel
//! side. `backend/gpu.zig` asserts the selected provider against it, so a
//! signature that drifts in one provider is a compile error on that
//! provider's own leg (`zig build` with `-Dgpu=metal`, `zig build
//! cuda-check` for the CUDA arm) instead of a link error or a silent
//! capability hole.
//!
//! Two things are deliberately NOT unified:
//!
//!   * `KernelFormatTag` is per-provider. Its integer values are the ABI
//!     the provider's own shaders/kernels switch on, and the sets differ:
//!     Metal has `tq2_0`/`tq2_0_folded` where CUDA has `q5_k`, so tag 3
//!     means different formats on the two sides. It is a wire tag, not a
//!     format identity (`DType` is): callers map a dtype to its tag with
//!     the provider's `kernelTag(dt)` (null when no kernel exists) and
//!     gate on the capability flags, never by assuming a tag exists;
//!     `assertConforms` checks flag and tag agree.
//!   * Provider-private test hooks (CUDA's `setDecodeForTest`,
//!     `setTransientFloorForTest`) are extras, not interface members.
//!
//! Layer stack: docs/ARCHITECTURE.md.

const std = @import("std");
const tensor = @import("../tensor.zig");
const thread = @import("../thread.zig");

const Tensor = tensor.Tensor;
const TensorF16 = tensor.TensorOf(.f16);
const TensorBf16 = tensor.TensorOf(.bf16);

/// GEMM operand orientation, as the provider kernels take it.
pub const Orient = enum(c_int) { nn = 0, tn = 1, nt = 2 };

/// One grouped-MoE tile: which expert, which rows of the batch, how many.
/// `extern` because the tile table crosses to the kernel side verbatim.
pub const QMMTile = extern struct {
    expert: i32,
    base_row: i32,
    m: i32,
    tile_m: i32,
};

/// Pinned staging buffers for the grouped-MoE path (host-visible in/out).
pub const QMoeStage = struct {
    in: [*]f32,
    out: [*]f32,
};

/// Element type for the ES device kernels (perturb/update/anchor).
pub const EsDType = enum(usize) { f16 = 0, f32 = 1 };

/// Capability flags every provider declares. `enabled` says the provider
/// is the selected one; the rest gate per-arm support and MUST agree with
/// the provider's own `KernelFormatTag` tag set (checked below).
const capability_flags = [_][]const u8{
    "enabled",
    "has_quant_gemm",
    "has_q5_k_quant",
    "has_tq2_0_quant",
    "has_tq2_0_folded_quant",
    "has_attention_fwd",
};

/// Capability flag -> the `KernelFormatTag` tag it promises. A flag that is true
/// without its tag (or a tag present with the flag false) is a compile
/// error: that pairing is exactly what dispatch code keys on.
const format_capabilities = [_]struct { flag: []const u8, tag: []const u8 }{
    .{ .flag = "has_q5_k_quant", .tag = "q5_k" },
    .{ .flag = "has_tq2_0_quant", .tag = "tq2_0" },
    .{ .flag = "has_tq2_0_folded_quant", .tag = "tq2_0_folded" },
};

/// The wire types a provider must re-export from THIS module (identical
/// type identity, not a structural copy — they cross to the kernel side).
const shared_types = [_]struct { name: []const u8, T: type }{
    .{ .name = "Orient", .T = Orient },
    .{ .name = "QMMTile", .T = QMMTile },
    .{ .name = "QMoeStage", .T = QMoeStage },
    .{ .name = "EsDType", .T = EsDType },
};

/// Process-global serialization for the two staging-panel protocols. Both
/// providers own one grow-only panel pair per protocol and dispatch
/// blocking, so callers (and the conformance suite) hold these across a
/// stage/write/dispatch/read sequence.
const shared_vars = [_]struct { name: []const u8, T: type }{
    .{ .name = "f16_lock", .T = thread.Mutex },
    .{ .name = "qmoe_lock", .T = thread.Mutex },
};

const Signature = struct {
    name: []const u8,
    params: []const type,
    ret: type,
};

/// Every function on the interface, with its exact parameter and return
/// types. `P.KernelFormatTag` resolves per-provider (see the header).
fn signatures(comptime P: type) []const Signature {
    const KernelFormatTag = P.KernelFormatTag;
    return &[_]Signature{
        // Dispatch tracing (`FUCINA_GPU_TRACE`); the reset/dump pair is a
        // no-op when tracing is off, so callers invoke unconditionally.
        .{ .name = "traceEnabled", .params = &.{}, .ret = bool },
        .{ .name = "traceReset", .params = &.{}, .ret = void },
        .{ .name = "traceDump", .params = &.{}, .ret = void },
        .{ .name = "deviceName", .params = &.{}, .ret = ?[]const u8 },

        // Work gates: should this shape go to the GPU at all.
        .{ .name = "shouldUseGpu", .params = &.{ usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuBatched", .params = &.{ usize, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuF16", .params = &.{ usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuF16ForRhs", .params = &.{ *const TensorF16, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuBf16ForRhs", .params = &.{ *const TensorBf16, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuGemv", .params = &.{ *const Tensor, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuForRhs", .params = &.{ *const Tensor, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuBatchedForRhs", .params = &.{ *const Tensor, usize, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuAttentionFwd", .params = &.{ usize, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuQMoe", .params = &.{u64}, .ret = bool },
        .{ .name = "qmoeFillAcceptable", .params = &.{ usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuDenseQuant", .params = &.{ KernelFormatTag, u64 }, .ret = bool },
        .{ .name = "shouldUseGpuDenseQuantPacked", .params = &.{ KernelFormatTag, u64 }, .ret = bool },
        .{ .name = "shouldUseGpuQuantDecode", .params = &.{ KernelFormatTag, usize, usize, usize }, .ret = bool },
        .{ .name = "shouldUseGpuAttn", .params = &.{ usize, usize, usize, usize }, .ret = bool },

        // Seams the shared conformance suite drives (see gpu_conformance.zig):
        // a device probe that is exactly "the provider has a live context",
        // and save/restore access to the grouped-MoE occupancy gate.
        .{ .name = "deviceAvailableForTest", .params = &.{}, .ret = bool },
        .{ .name = "setMinWorkQMoeForTest", .params = &.{u64}, .ret = void },
        .{ .name = "qmoeMinFillForTest", .params = &.{}, .ret = u64 },
        .{ .name = "setQmoeMinFillForTest", .params = &.{u64}, .ret = void },

        // Attention forward.
        .{
            .name = "attentionFwdF32",
            .params = &.{ []const f32, []const f32, []const f32, []f32, ?[]f32, usize, usize, usize, usize, usize, usize, bool, usize, f32 },
            .ret = bool,
        },
        .{
            .name = "attentionFwdF16Kv",
            .params = &.{ []const f32, []const f16, []const f16, []f32, ?[]f32, usize, usize, usize, usize, usize, usize, bool, usize, f32 },
            .ret = bool,
        },
        .{
            .name = "attnPrefillF16",
            .params = &.{ []const f32, []const f16, []const f16, []f32, []const i32, usize, usize, usize, usize, usize, usize, f32, usize, bool },
            .ret = bool,
        },

        // Dense GEMM: blocking host-slice forms and eager async forms.
        .{ .name = "gemmF16Nt", .params = &.{ []const f16, []const f16, usize, usize, usize, bool }, .ret = ?[]const f16 },
        .{ .name = "gemmF32", .params = &.{ Orient, []const f32, []const f32, []f32, usize, usize, usize }, .ret = bool },
        .{
            .name = "gemmBatchedF32",
            .params = &.{ Orient, []const f32, []const f32, []f32, usize, usize, usize, usize, usize, usize, usize },
            .ret = bool,
        },
        .{
            .name = "gemmBatchedF32Async",
            .params = &.{ Orient, *const Tensor, *const Tensor, *Tensor, usize, usize, usize, usize, usize, usize, usize },
            .ret = bool,
        },
        .{ .name = "gemmF32Async", .params = &.{ Orient, *const Tensor, *const Tensor, *Tensor, usize, usize, usize }, .ret = bool },
        .{ .name = "gemmF16NtAsync", .params = &.{ *const TensorF16, *const TensorF16, *Tensor, usize, usize, usize }, .ret = bool },
        .{ .name = "gemmBf16NtAsync", .params = &.{ *const Tensor, *const TensorBf16, *Tensor, usize, usize, usize }, .ret = bool },

        // Quantized GEMM: dense (blocking + async) and grouped MoE.
        .{
            .name = "gemmQuantNtAsync",
            .params = &.{ KernelFormatTag, []const u8, bool, usize, usize, *const Tensor, *Tensor, usize, usize, usize, usize },
            .ret = bool,
        },
        .{
            .name = "gemmQuantNt",
            .params = &.{ KernelFormatTag, []const u8, bool, usize, []const f32, []f32, usize, usize, usize },
            .ret = bool,
        },
        .{
            .name = "gemmQuantNtSharedABatch",
            .params = &.{ KernelFormatTag, []const u8, bool, usize, usize, []const f32, []f32, usize, usize, usize, usize },
            .ret = bool,
        },
        .{
            .name = "gemmQGroupedNt",
            .params = &.{ KernelFormatTag, []const u8, bool, usize, usize, usize, usize, []const QMMTile },
            .ret = bool,
        },
        .{ .name = "qmoeStage", .params = &.{ usize, usize }, .ret = ?QMoeStage },

        // Device-owned bytes for loader residency.
        .{ .name = "allocResidentBytes", .params = &.{usize}, .ret = ?[]u8 },
        .{ .name = "freeResidentBytes", .params = &.{[]const u8}, .ret = void },

        // Evolution-strategies device kernels over flat parameter bytes.
        .{ .name = "esPerturb", .params = &.{ EsDType, []u8, u64, f32, usize }, .ret = bool },
        .{ .name = "esUpdate", .params = &.{ EsDType, []u8, []const u64, []const f32, f32, usize }, .ret = bool },
        .{ .name = "esAnchor", .params = &.{ EsDType, []u8, []const u8, f32, bool, usize }, .ret = bool },
    };
}

/// Compile-error unless `P` implements the whole interface. Signature-only:
/// this never takes a function's address, so asserting conformance does not
/// pull a provider's bodies (or its shim) into a build that elided them.
pub fn assertConforms(comptime P: type) void {
    comptime {
        const who = @typeName(P);

        for (capability_flags) |name| {
            if (!@hasDecl(P, name)) @compileError(who ++ " is missing GPU capability flag `" ++ name ++ "`");
            if (@TypeOf(@field(P, name)) != bool)
                @compileError(who ++ "." ++ name ++ " must be `bool`, found " ++ @typeName(@TypeOf(@field(P, name))));
        }

        for (shared_vars) |entry| {
            if (!@hasDecl(P, entry.name)) @compileError(who ++ " is missing shared lock `" ++ entry.name ++ "`");
            if (@TypeOf(@field(P, entry.name)) != entry.T)
                @compileError(who ++ "." ++ entry.name ++ " must be a `" ++ @typeName(entry.T) ++ "`, found " ++
                    @typeName(@TypeOf(@field(P, entry.name))));
        }

        for (shared_types) |entry| {
            if (!@hasDecl(P, entry.name)) @compileError(who ++ " is missing shared wire type `" ++ entry.name ++ "`");
            if (@field(P, entry.name) != entry.T)
                @compileError(who ++ "." ++ entry.name ++ " must re-export backend/gpu_provider.zig's `" ++
                    entry.name ++ "`, found a distinct type " ++ @typeName(@field(P, entry.name)));
        }

        if (!@hasDecl(P, "KernelFormatTag")) @compileError(who ++ " is missing `KernelFormatTag`");
        if (!@hasDecl(P, "kernelTag")) @compileError(who ++ " is missing `kernelTag` (the DType -> KernelFormatTag map)");
        const KernelFormatTag = P.KernelFormatTag;
        const qinfo = @typeInfo(KernelFormatTag);
        if (qinfo != .@"enum" or qinfo.@"enum".tag_type != c_int)
            @compileError(who ++ ".KernelFormatTag must be an `enum(c_int)` (its values are the kernel-side ABI)");
        if (!@hasDecl(KernelFormatTag, "kMultiple"))
            @compileError(who ++ ".KernelFormatTag is missing `kMultiple` (the per-format K block granularity)");
        for (format_capabilities) |entry| {
            const flag = @field(P, entry.flag);
            const has_tag = @hasField(KernelFormatTag, entry.tag);
            if (flag and !has_tag)
                @compileError(who ++ "." ++ entry.flag ++ " is true but its KernelFormatTag tag `" ++ entry.tag ++ "` does not exist");
            if (!flag and has_tag)
                @compileError(who ++ ".KernelFormatTag declares `" ++ entry.tag ++ "` but " ++ entry.flag ++ " is false");
        }

        for (signatures(P)) |sig| {
            if (!@hasDecl(P, sig.name)) @compileError(who ++ " is missing GPU provider function `" ++ sig.name ++ "`");
            const info = @typeInfo(@TypeOf(@field(P, sig.name)));
            if (info != .@"fn") @compileError(who ++ "." ++ sig.name ++ " must be a function");
            const f = info.@"fn";
            if (f.params.len != sig.params.len)
                @compileError(std.fmt.comptimePrint("{s}.{s} takes {d} parameters, the GPU provider interface declares {d}", .{
                    who, sig.name, f.params.len, sig.params.len,
                }));
            for (f.params, sig.params, 0..) |got, want, i| {
                if (got.type != want)
                    @compileError(std.fmt.comptimePrint("{s}.{s} parameter {d} is {s}, the GPU provider interface declares {s}", .{
                        who, sig.name, i, @typeName(got.type orelse void), @typeName(want),
                    }));
            }
            if (f.return_type != sig.ret)
                @compileError(std.fmt.comptimePrint("{s}.{s} returns {s}, the GPU provider interface declares {s}", .{
                    who, sig.name, @typeName(f.return_type orelse void), @typeName(sig.ret),
                }));
        }
    }
}
