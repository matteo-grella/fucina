//! The GPU provider interface: ONE statement of the surface every `-Dgpu`
//! provider implements, plus the wire and request types they share. The
//! interface itself is DEFINED by the null provider's public declarations
//! (`gpu_none.zig` — one stub per entry, so the list cannot drift from its
//! reference implementation); `backend/gpu.zig` asserts the selected
//! provider against it with `assertConforms`, so a signature that drifts
//! in one provider is a compile error on that provider's own leg
//! (`zig build` with `-Dgpu=metal`, `zig build cuda-check` for the CUDA
//! arm) instead of a link error or a silent capability hole.
//!
//! Multi-parameter entries take the request structs below (`GemmRequest`,
//! `QuantGemmRequest`, `AttentionRequest`). They are host-side
//! descriptions: the provider unpacks them into its kernel ABI, so none
//! are `extern`. `QMMTile` alone crosses to the kernel side verbatim.
//!
//! The format vocabulary is ONE enum: `QuantFormat` below names every
//! quantized RHS layout an accelerator may take, and every interface entry
//! speaks it. A provider's kernel-side integer for a format is its private
//! ABI (Metal has ternary kernels where CUDA has Q5_K, so the integers
//! differ), exported only as `abiValue(fmt)` — null when the provider has
//! no kernel, which IS the capability answer (`offload.supportsQuant`
//! derives from it). Provider-private test hooks (CUDA's
//! `setDecodeForTest`, `setTransientFloorForTest`) are extras, not
//! interface members.
//!
//! Layer stack: docs/ARCHITECTURE.md.

const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const ops = @import("ops.zig");

/// GEMM operand orientation, as the provider kernels take it.
pub const Orient = ops.MatmulKind;

/// The quantized RHS layouts an accelerator may take — the one format
/// vocabulary above the providers. `tq2_0_folded` is the ternary
/// folded-plane layout (`weights/ptqtp.zig`); it has no `DType` of its own.
pub const QuantFormat = enum {
    q8_0,
    q4_k,
    q5_k,
    q6_k,
    tq2_0,
    tq2_0_folded,

    pub fn fromDType(comptime dt: dtype_mod.DType) ?QuantFormat {
        return switch (dt) {
            .q8_0 => .q8_0,
            .q4_k => .q4_k,
            .q5_k => .q5_k,
            .q6_k => .q6_k,
            .tq2_0 => .tq2_0,
            else => null,
        };
    }

    /// K (the reduced dim) must be a whole number of blocks: the format's
    /// block length, straight from `dtype.blockSize` (the folded pair
    /// layout keeps TQ2_0's).
    pub fn kMultiple(self: QuantFormat) usize {
        return switch (self) {
            inline .q8_0, .q4_k, .q5_k, .q6_k, .tq2_0 => |f| comptime dtype_mod.blockSize(@field(dtype_mod.DType, @tagName(f))),
            .tq2_0_folded => comptime dtype_mod.blockSize(.tq2_0),
        };
    }
};

/// One dense GEMM description: `c[m,n] = op(a)·op(b)` under `orient`,
/// optionally strided-batched — a `batch` above 1 is ONE dispatch with
/// grid depth = batch, strides in elements. The operand slices stay
/// parameters (their element types name the entry).
pub const GemmRequest = struct {
    orient: Orient,
    m: usize,
    n: usize,
    k: usize,
    batch: usize = 1,
    stride_a: usize = 0,
    stride_b: usize = 0,
    stride_c: usize = 0,
};

/// One quantized-RHS NT GEMM description: `input[m,k] · dequant(rhs[n,k])ᵀ`.
/// `rhs` is the raw block bytes with row stride `nb01` and per-matrix
/// (expert/batch) stride `nb02`. `rhs_cacheable` is the RHS lifetime
/// statement: true only for bytes that stay mapped for the process
/// lifetime (resident weights) — a cached wrap of a freed-and-reused page
/// reads stale data. `batch` is the shared-input batch count of the
/// batched forms (1 elsewhere); the grouped entry takes its rows from the
/// tile table and receives `m` = 0.
pub const QuantGemmRequest = struct {
    format: QuantFormat,
    rhs: []const u8,
    rhs_cacheable: bool,
    nb01: usize,
    nb02: usize = 0,
    batch: usize = 1,
    m: usize,
    n: usize,
    k: usize,
};

/// One grouped-attention call description (the Q/K/V/out slices stay
/// parameters: the f32 and f16-KV entries differ in element type). The
/// `attentionFwd*` entries use `heads_per_kv` (the uniform GQA map,
/// kv_head = head / heads_per_kv) and derive the query offset as
/// `kv_seq - q_seq`; the prefill entry takes an explicit per-head map
/// parameter plus `source_offset` and ignores `heads_per_kv`.
pub const AttentionRequest = struct {
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    kv_heads: usize,
    d: usize,
    window: usize,
    causal: bool,
    scale: f32,
    heads_per_kv: usize = 1,
    source_offset: usize = 0,
};

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
pub const FlatDType = enum(usize) { f16 = 0, f32 = 1 };

/// Compile-error unless `P` implements the whole interface, stated as
/// `Reference`'s public declarations (the null provider). Exact-type
/// checks: a type declaration must be the same type (identity, not a
/// structural copy — the wire and request types cross module boundaries),
/// a function must match the reference signature parameter-by-parameter,
/// and any other value (capability flags, the staging-panel locks) must
/// have the reference declaration's type. Signature-only: this never
/// takes a function's address, so asserting conformance does not pull a
/// provider's bodies (or its shim) into a build that elided them.
pub fn assertConforms(comptime Reference: type, comptime P: type) void {
    comptime {
        if (P == Reference) return;
        const who = @typeName(P);
        for (@typeInfo(Reference).@"struct".decls) |decl| {
            const name = decl.name;
            if (!@hasDecl(P, name)) @compileError(who ++ " is missing GPU provider decl `" ++ name ++ "`");
            const RefType = @TypeOf(@field(Reference, name));
            if (RefType == type) {
                if (@field(P, name) != @field(Reference, name))
                    @compileError(who ++ "." ++ name ++ " must re-export the interface type `" ++
                        name ++ "`, found a distinct type " ++ @typeName(@field(P, name)));
            } else if (@typeInfo(RefType) == .@"fn") {
                assertSameFn(who, name, RefType, @TypeOf(@field(P, name)));
            } else if (@TypeOf(@field(P, name)) != RefType) {
                @compileError(who ++ "." ++ name ++ " must be `" ++ @typeName(RefType) ++ "`, found " ++
                    @typeName(@TypeOf(@field(P, name))));
            }
        }
    }
}

fn assertSameFn(comptime who: []const u8, comptime name: []const u8, comptime Want: type, comptime Got: type) void {
    comptime {
        const got_info = @typeInfo(Got);
        if (got_info != .@"fn") @compileError(who ++ "." ++ name ++ " must be a function");
        const got = got_info.@"fn";
        const want = @typeInfo(Want).@"fn";
        if (got.params.len != want.params.len)
            @compileError(std.fmt.comptimePrint("{s}.{s} takes {d} parameters, the GPU provider interface declares {d}", .{
                who, name, got.params.len, want.params.len,
            }));
        for (got.params, want.params, 0..) |g, w, i| {
            if (g.type != w.type)
                @compileError(std.fmt.comptimePrint("{s}.{s} parameter {d} is {s}, the GPU provider interface declares {s}", .{
                    who, name, i, @typeName(g.type orelse void), @typeName(w.type orelse void),
                }));
        }
        if (got.return_type != want.return_type)
            @compileError(std.fmt.comptimePrint("{s}.{s} returns {s}, the GPU provider interface declares {s}", .{
                who, name, @typeName(got.return_type orelse void), @typeName(want.return_type orelse void),
            }));
    }
}
