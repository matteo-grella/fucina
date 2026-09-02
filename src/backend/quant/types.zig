//! Quantized matmul types: the backend-private interleaved block layouts,
//! the RHS container types, and the block-size constants. Every container
//! names its storage format with `pub const dtype: DType`; the block
//! structs and the per-format registry live in `dtype.zig`. A leaf of the
//! quant group: every kernel child imports it, and `quant.zig` forwards the
//! block and RHS types to readers outside the backend.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

// Default W8A8 group length. This matches GGML Q8_0's block length, but the
// container below is not itself GGML Q8_0.
pub const default_i8_group_size: usize = 32;
pub const q1_0_block_size = dtype_mod.q1_0_block_size;
pub const q2_0_block_size = dtype_mod.q2_0_block_size;
pub const q4_0_block_size = dtype_mod.q4_0_block_size;
pub const q4_1_block_size = dtype_mod.q4_1_block_size;
pub const q5_0_block_size = dtype_mod.q5_0_block_size;
pub const q5_1_block_size = dtype_mod.q5_1_block_size;
pub const q8_0_block_size = dtype_mod.q8_0_block_size;
pub const q8_1_block_size = dtype_mod.q8_1_block_size;
pub const qk_k_block_size = dtype_mod.qk_k_block_size;
pub const k_scale_size = dtype_mod.k_scale_size;
pub const iq4_nl_block_size = dtype_mod.iq4_nl_block_size;
pub const mxfp4_block_size = dtype_mod.mxfp4_block_size;
pub const nvfp4_block_size = dtype_mod.nvfp4_block_size;
pub const nvfp4_subblock_size = dtype_mod.nvfp4_subblock_size;

pub const QuantizedFormatError = error{
    InvalidQuantizedLength,
};

/// Column-interleave layout of a quantized matmul RHS container: `.rows`
/// is the compact block-per-output-column layout (GGUF-native);
/// `.x4`/`.x8` interleave 4/8 output columns for the sdot lanes;
/// `.x2mmla` pairs columns for the aarch64 smmla path. Every container
/// below states its layout with `pub const pack`; `backend/ops.zig`
/// re-exports the enum beside `QuantGemm`.
pub const RhsPack = enum { rows, x4, x8, x2mmla };

/// Caller promise about an RHS pointer's lifetime, the gate of the
/// address-keyed accelerator caches.
pub const RhsLifetime = enum {
    /// Ordinary tensor/temporary storage. The backend may still use the GPU,
    /// but it must not cache an address-keyed wrap beyond this dispatch.
    transient,
    /// Caller guarantees the RHS bytes stay mapped at the same address for the
    /// process lifetime, or are registered device-resident storage
    /// (`internal.gpu.allocResidentBytes`) whose owner evicts cached wraps via
    /// `freeResidentBytes` before freeing. A backend may cache address-keyed
    /// wraps.
    stable_process,

    pub fn isCacheable(self: RhsLifetime) bool {
        return self == .stable_process;
    }
};

/// The raw GGUF row blocks behind a lane-packed container, when its
/// loader still holds them: the accelerator's dequant-in-kernel GEMM reads
/// these bytes (`nb01` per RHS row) instead of the CPU lane pack, so the
/// offload decision is made once, in `ExecContext.matmulQuant`, whatever
/// container the caller holds. `lifetime` is the caller's promise about
/// the bytes' address.
pub const RawRhs = struct {
    bytes: []const u8,
    nb01: usize,
    lifetime: RhsLifetime,
};

pub fn checkedProduct(a: usize, b: usize) QuantizedFormatError!usize {
    return std.math.mul(usize, a, b) catch QuantizedFormatError.InvalidQuantizedLength;
}

/// The one whole-block length rule behind every per-format `*BlockCount`
/// spelling: `len` must be a whole number of `block_size`-element blocks.
pub fn blockCountExact(comptime block_size: usize, len: usize) QuantizedFormatError!usize {
    if (len % block_size != 0) return QuantizedFormatError.InvalidQuantizedLength;
    return len / block_size;
}

pub const BlockQ8_0x4 = extern struct {
    d: [4]u16,
    qs: [4 * q8_0_block_size]i8,
};
pub const BlockQ4_Kx4 = extern struct {
    d: [4]u16,
    dmin: [4]u16,
    scales: [8 * 4]u8,
    mins: [8 * 4]u8,
    qs: [qk_k_block_size * 4]i8,
};
pub const BlockQ4_Kx8 = extern struct {
    d: [8]u16,
    dmin: [8]u16,
    scales: [8 * 8]u8,
    mins: [8 * 8]u8,
    qs: [qk_k_block_size * 4]u8,
};
pub const BlockQ4_Kx2Mmla = extern struct {
    d: [2]u16,
    dmin: [2]u16,
    scales: [8 * 2]u8,
    mins: [8 * 2]u8,
    qs: [qk_k_block_size * 2]i8,
};
pub const BlockQ5_Kx8 = extern struct {
    d: [8]u16,
    dmin: [8]u16,
    scales: [8 * 8]u8,
    mins: [8 * 8]u8,
    qs: [qk_k_block_size * 8]i8,
};
pub const BlockQ8_Kx4 = extern struct {
    d: [4]f32,
    qs: [qk_k_block_size * 4]i8,
    bsums: [qk_k_block_size / 4]i16,
};
pub const BlockQ8_Kx2Mmla = extern struct {
    d: [2]f32,
    bsums: [8 * 2]i16,
    qs: [qk_k_block_size * 2]i8,
};
pub const BlockQ6_Kx4 = extern struct {
    d: [4]u16,
    scales: [16 * 4]i8,
    qs: [qk_k_block_size * 4]i8,
};

/// Four TQ2_0 blocks (four RHS rows = output columns) at the same k
/// position, column-interleaved in 4-byte granules so one 16-byte load
/// yields the same 4-element k-group for all four columns — the operand
/// shape of the by-element sdot, where each i32 lane accumulates its own
/// column and the per-block horizontal reduce disappears (the Q8_0x4 /
/// Q4_Kx8 layout family, ternary form). Same bytes as 4 BlockTQ2_0 (264),
/// rearranged: qs[half*128 + fg*16 + col*4 + lane] =
/// src[col].qs[half*32 + fg*4 + lane]; crumb planes ride along untouched
/// (each byte still holds 4 elements at +0/+32/+64/+96 of its 128-half).
pub const BlockTQ2_0x4 = extern struct {
    d: [4]u16, // f16 bits, d[col]
    qs: [4 * 64]u8,
};

/// Four columns' worth of one 256-element block of FOLDED K=2 tie-fitted
/// PTQTP planes: 4-bit codes cu = 3*u1 + u2 in {0..8} (one uniform 9-level
/// quantizer — see ternary.zig's folding section), column-interleaved
/// Q4_0-style: byte(sub, jg, col, j) = cu(col, sub*32 + jg*4 + j) |
/// cu(col, sub*32 + 16 + jg*4 + j) << 4. d[col] is the FINE plane's f16
/// scale; the coarse scale is 3x it, derived in f32 at use.
pub const BlockTQ2_0Foldedx4 = extern struct {
    d: [4]u16,
    qs: [4 * 128]u8,
};

/// Row-major single-column sibling of BlockTQ2_0Foldedx4 for per-row block
/// consumers (the GPU dequant-in-kernel GEMM): 256 elements as 4-bit codes
/// cu in {0..8}, Q4_0-style pairing per 32-element sub-block —
/// byte(s*16 + j) = cu(s*32 + j) | cu(s*32 + 16 + j) << 4 — and the FINE
/// plane's f16 scale (value = d * (cu - 4)).
pub const BlockTQ2_0Folded = extern struct {
    qs: [128]u8,
    d: u16,
};

/// The lane-interleaved block of a packed RHS: which `(dtype, pack)`
/// pairs have a layout is this table, and a pair outside it is a compile
/// error naming it.
pub fn PackedBlock(comptime dt: DType, comptime pack: RhsPack) type {
    return switch (dt) {
        .q8_0 => switch (pack) {
            .x4 => BlockQ8_0x4,
            else => @compileError("no ." ++ @tagName(pack) ++ " pack for q8_0"),
        },
        .q4_k => switch (pack) {
            .x4 => BlockQ4_Kx4,
            .x8 => BlockQ4_Kx8,
            .x2mmla => BlockQ4_Kx2Mmla,
            .rows => @compileError("the .rows layout is CompactRhs, not a lane pack"),
        },
        .q5_k => switch (pack) {
            .x8 => BlockQ5_Kx8,
            else => @compileError("no ." ++ @tagName(pack) ++ " pack for q5_k"),
        },
        .q6_k => switch (pack) {
            .x4 => BlockQ6_Kx4,
            else => @compileError("no ." ++ @tagName(pack) ++ " pack for q6_k"),
        },
        else => @compileError("no lane pack for dtype ." ++ @tagName(dt)),
    };
}

/// The compact (GGUF-native block layout) matmul RHS of one block format:
/// `[n, k]` blocks, `blocks_per_column` per output column. `blocks` may
/// borrow external read-only memory (an mmap'd GGUF, an expert stack kept
/// alive by the model): `allocator` is null then and `deinit` frees
/// nothing. The one container behind every `.rows` kernel request.
pub fn CompactRhs(comptime dt: DType) type {
    return struct {
        allocator: ?Allocator,
        blocks: []const dtype_mod.Storage(dt),
        k: usize,
        n: usize,
        blocks_per_column: usize,

        const Self = @This();
        pub const dtype: DType = dt;
        pub const pack: RhsPack = .rows;

        pub fn deinit(self: *Self) void {
            if (self.allocator) |allocator| allocator.free(self.blocks);
            self.* = undefined;
        }

        pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.Storage(dt) {
            return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
        }
    };
}

/// A lane-packed matmul RHS: the `PackedBlock(dt, lane_pack)` groups of
/// `n / lanes` output columns, `blocks_per_group` per group, built once at
/// load by the format's packer (always owned). `raw` is the loader's
/// statement of the raw GGUF blocks behind the pack, for the accelerator
/// attempt.
pub fn LanePackedRhs(comptime dt: DType, comptime lane_pack: RhsPack) type {
    const Block = PackedBlock(dt, lane_pack);
    return struct {
        allocator: Allocator,
        blocks: []Block,
        k: usize,
        n: usize,
        blocks_per_group: usize,
        /// The loader's raw row blocks for the accelerator attempt (`RawRhs`).
        raw: ?RawRhs = null,

        const Self = @This();
        pub const dtype: DType = dt;
        pub const pack: RhsPack = lane_pack;

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.blocks);
            self.* = undefined;
        }

        pub fn groupBlocks(self: *const Self, column_group: usize) []const Block {
            return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
        }
    };
}

pub fn QuantizedRowsFor(comptime block_dtype: DType) type {
    return struct {
        /// Owning allocator, or null when `blocks` borrows external storage
        /// kept alive by the caller (e.g. packed ES genome blocks); deinit
        /// then frees nothing.
        allocator: ?Allocator,
        blocks: []dtype_mod.Storage(block_dtype),
        rows: usize,
        cols: usize,
        blocks_per_row: usize,

        const Self = @This();
        pub const dtype: DType = block_dtype;

        pub fn deinit(self: *Self) void {
            if (self.allocator) |allocator| allocator.free(self.blocks);
            self.* = undefined;
        }

        pub fn rowBlocks(self: *const Self, row: usize) []const dtype_mod.Storage(block_dtype) {
            return self.blocks[row * self.blocks_per_row ..][0..self.blocks_per_row];
        }
    };
}

/// Every `.rows` container is `CompactRhs`; the name stays for the readers
/// that spell the format-keyed form.
pub const QuantizedMatmulRhsRowsFor = CompactRhs;

pub const QuantizedRowsQ8_1 = QuantizedRowsFor(.q8_1);

pub const QuantizedMatmulRhsQ1_0 = QuantizedMatmulRhsRowsFor(.q1_0);
pub const QuantizedMatmulRhsQ2_0 = QuantizedMatmulRhsRowsFor(.q2_0);
pub const QuantizedMatmulRhsQ4_1 = QuantizedMatmulRhsRowsFor(.q4_1);
pub const QuantizedMatmulRhsQ5_0 = QuantizedMatmulRhsRowsFor(.q5_0);
pub const QuantizedMatmulRhsQ5_1 = QuantizedMatmulRhsRowsFor(.q5_1);
pub const QuantizedMatmulRhsIQ1_S = QuantizedMatmulRhsRowsFor(.iq1_s);
pub const QuantizedMatmulRhsIQ1_M = QuantizedMatmulRhsRowsFor(.iq1_m);
pub const QuantizedMatmulRhsIQ2_XXS = QuantizedMatmulRhsRowsFor(.iq2_xxs);
pub const QuantizedMatmulRhsIQ2_XS = QuantizedMatmulRhsRowsFor(.iq2_xs);
pub const QuantizedMatmulRhsIQ2_S = QuantizedMatmulRhsRowsFor(.iq2_s);
pub const QuantizedMatmulRhsIQ3_XXS = QuantizedMatmulRhsRowsFor(.iq3_xxs);
pub const QuantizedMatmulRhsIQ3_S = QuantizedMatmulRhsRowsFor(.iq3_s);
pub const QuantizedMatmulRhsIQ4_NL = QuantizedMatmulRhsRowsFor(.iq4_nl);
pub const QuantizedMatmulRhsIQ4_XS = QuantizedMatmulRhsRowsFor(.iq4_xs);
pub const QuantizedMatmulRhsTQ1_0 = QuantizedMatmulRhsRowsFor(.tq1_0);
pub const QuantizedMatmulRhsTQ2_0 = QuantizedMatmulRhsRowsFor(.tq2_0);
pub const QuantizedMatmulRhsMXFP4 = QuantizedMatmulRhsRowsFor(.mxfp4);
pub const QuantizedMatmulRhsNVFP4 = QuantizedMatmulRhsRowsFor(.nvfp4);

pub const QuantizedRowsQ8_0 = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF kept alive by the model).
    allocator: ?Allocator,
    blocks: []const dtype_mod.BlockQ8_0,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,

    const Self = @This();
    pub const dtype: DType = .q8_0;

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn rowBlocks(self: *const Self, row: usize) []const dtype_mod.BlockQ8_0 {
        return self.blocks[row * self.blocks_per_row ..][0..self.blocks_per_row];
    }
};

pub const QuantizedRowsQ4_0 = struct {
    allocator: Allocator,
    blocks: []dtype_mod.BlockQ4_0,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,

    const Self = @This();
    pub const dtype: DType = .q4_0;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn rowBlocks(self: *const Self, row: usize) []const dtype_mod.BlockQ4_0 {
        return self.blocks[row * self.blocks_per_row ..][0..self.blocks_per_row];
    }
};

pub const QuantizedMatmulRhsQ8_0 = CompactRhs(.q8_0);

pub const QuantizedMatmulRhsQ8_0x4 = LanePackedRhs(.q8_0, .x4);

pub const QuantizedMatmulRhsQ4_0 = CompactRhs(.q4_0);

pub const QuantizedMatmulRhsQ2_K = CompactRhs(.q2_k);
pub const QuantizedMatmulRhsQ3_K = CompactRhs(.q3_k);
pub const QuantizedMatmulRhsQ4_K = CompactRhs(.q4_k);
pub const QuantizedMatmulRhsQ5_K = CompactRhs(.q5_k);
pub const QuantizedMatmulRhsQ6_K = CompactRhs(.q6_k);
pub const QuantizedMatmulRhsQ4_Kx4 = LanePackedRhs(.q4_k, .x4);
pub const QuantizedMatmulRhsQ4_Kx8 = LanePackedRhs(.q4_k, .x8);
pub const QuantizedMatmulRhsQ4_Kx2Mmla = LanePackedRhs(.q4_k, .x2mmla);
pub const QuantizedMatmulRhsQ5_Kx8 = LanePackedRhs(.q5_k, .x8);
pub const QuantizedMatmulRhsQ6_Kx4 = LanePackedRhs(.q6_k, .x4);

/// The block-quantized dtypes that can be a stored matmul RHS, in registry
/// order (every `block_formats` row that claims `matmul_rhs`).
pub const matmul_rhs_dtypes: []const DType = blk: {
    var count: usize = 0;
    for (dtype_mod.block_formats) |row| count += @intFromBool(row.matmul_rhs);
    var out: [count]DType = undefined;
    var i: usize = 0;
    for (dtype_mod.block_formats) |row| {
        if (row.matmul_rhs) {
            out[i] = row.dtype;
            i += 1;
        }
    }
    const final = out;
    break :blk &final;
};

/// Every stored-RHS format as one runtime-tagged operand of the block
/// dispatch: one arm per matmul-capable registry row, holding
/// `*const CompactRhs(dtype)` under the dtype's own name, plus the W8A8
/// container. Derived from the registry, so a new format never lags here;
/// `@unionInit(AnyQuantizedMatmulRhs, @tagName(dt), &rhs)` is the
/// constructor.
pub const AnyQuantizedMatmulRhs = blk: {
    @setEvalBranchQuota(10_000);
    const dts = matmul_rhs_dtypes;
    var names: [dts.len + 1][:0]const u8 = undefined;
    var field_types: [dts.len + 1]type = undefined;
    for (dts, 0..) |dt, i| {
        names[i] = @tagName(dt);
        field_types[i] = *const CompactRhs(dt);
    }
    names[dts.len] = "fucina_w8a8_rhs";
    field_types[dts.len] = *const QuantizedMatmulRhsI8;
    const Tag = @Enum(u8, .exhaustive, &names, &std.simd.iota(u8, names.len));
    break :blk @Union(.auto, Tag, &names, &field_types, &@splat(.{}));
};

/// Symmetric int8 quantized weights, stored transposed as [n][k] with one
/// f32 scale per (column, group) block along k. A quantized matmul weight
/// container, not a dense TensorOf(.i8) dtype and not a GGML block format:
/// it carries no `DType`, only its group-size policy.
pub const QuantizedMatmulRhsI8 = struct {
    qw: tensor.TensorOf(.i8),
    scales: Tensor,
    k: usize,
    n: usize,
    group_size: usize,
    num_groups: usize,

    const Self = @This();
    pub const default_group_size = default_i8_group_size;

    /// The group length a request resolves to: 0 selects the default.
    pub fn effectiveGroupSize(requested_group_size: usize) usize {
        return if (requested_group_size == 0) default_group_size else requested_group_size;
    }

    /// Groups along k for one column (the last group may be partial).
    pub fn groupCountForSize(k: usize, group_size: usize) usize {
        return (k + group_size - 1) / group_size;
    }

    pub fn deinit(self: *Self) void {
        self.qw.deinit();
        self.scales.deinit();
        self.* = undefined;
    }
};
