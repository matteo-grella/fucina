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

pub fn QuantizedMatmulRhsRowsFor(comptime block_dtype: DType) type {
    return struct {
        rows: QuantizedRowsFor(block_dtype),
        k: usize,
        n: usize,

        const Self = @This();
        pub const dtype: DType = block_dtype;
        pub const pack: RhsPack = .rows;

        pub fn deinit(self: *Self) void {
            self.rows.deinit();
            self.* = undefined;
        }

        pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.Storage(block_dtype) {
            return self.rows.rowBlocks(column);
        }
    };
}

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

pub const QuantizedMatmulRhsQ8_0 = struct {
    rows: QuantizedRowsQ8_0,
    k: usize,
    n: usize,

    const Self = @This();
    pub const dtype: DType = .q8_0;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        self.rows.deinit();
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ8_0 {
        return self.rows.rowBlocks(column);
    }
};

pub const QuantizedMatmulRhsQ8_0x4 = struct {
    allocator: Allocator,
    blocks: []BlockQ8_0x4,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const dtype: DType = .q8_0;
    pub const pack: RhsPack = .x4;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn groupBlocks(self: *const Self, column_group: usize) []const BlockQ8_0x4 {
        return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
    }
};

pub const QuantizedMatmulRhsQ4_0 = struct {
    rows: QuantizedRowsQ4_0,
    k: usize,
    n: usize,

    const Self = @This();
    pub const dtype: DType = .q4_0;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        self.rows.deinit();
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ4_0 {
        return self.rows.rowBlocks(column);
    }
};

pub const QuantizedMatmulRhsQ2_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF expert stack kept alive by the model).
    allocator: ?Allocator,
    blocks: []const dtype_mod.BlockQ2_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const dtype: DType = .q2_k;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ2_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ3_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF expert stack kept alive by the model).
    allocator: ?Allocator,
    blocks: []const dtype_mod.BlockQ3_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const dtype: DType = .q3_k;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ3_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ4_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF kept alive by the model).
    allocator: ?Allocator,
    blocks: []const dtype_mod.BlockQ4_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const dtype: DType = .q4_k;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ4_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ4_Kx4 = struct {
    allocator: Allocator,
    blocks: []BlockQ4_Kx4,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const dtype: DType = .q4_k;
    pub const pack: RhsPack = .x4;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn groupBlocks(self: *const Self, column_group: usize) []const BlockQ4_Kx4 {
        return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
    }
};

pub const QuantizedMatmulRhsQ4_Kx8 = struct {
    allocator: Allocator,
    blocks: []BlockQ4_Kx8,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const dtype: DType = .q4_k;
    pub const pack: RhsPack = .x8;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn groupBlocks(self: *const Self, column_group: usize) []const BlockQ4_Kx8 {
        return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
    }
};

pub const QuantizedMatmulRhsQ4_Kx2Mmla = struct {
    allocator: Allocator,
    blocks: []BlockQ4_Kx2Mmla,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const dtype: DType = .q4_k;
    pub const pack: RhsPack = .x2mmla;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn groupBlocks(self: *const Self, column_group: usize) []const BlockQ4_Kx2Mmla {
        return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
    }
};

pub const QuantizedMatmulRhsQ5_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF kept alive by the model).
    allocator: ?Allocator,
    blocks: []const dtype_mod.BlockQ5_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const dtype: DType = .q5_k;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ5_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ5_Kx8 = struct {
    allocator: Allocator,
    blocks: []BlockQ5_Kx8,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const dtype: DType = .q5_k;
    pub const pack: RhsPack = .x8;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn groupBlocks(self: *const Self, column_group: usize) []const BlockQ5_Kx8 {
        return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
    }
};

pub const QuantizedMatmulRhsQ6_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF kept alive by the model).
    allocator: ?Allocator,
    blocks: []const dtype_mod.BlockQ6_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const dtype: DType = .q6_k;
    pub const pack: RhsPack = .rows;

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.BlockQ6_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ6_Kx4 = struct {
    allocator: Allocator,
    blocks: []BlockQ6_Kx4,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const dtype: DType = .q6_k;
    pub const pack: RhsPack = .x4;

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn groupBlocks(self: *const Self, column_group: usize) []const BlockQ6_Kx4 {
        return self.blocks[column_group * self.blocks_per_group ..][0..self.blocks_per_group];
    }
};

pub const AnyQuantizedMatmulRhs = union(enum) {
    fucina_w8a8_rhs: *const QuantizedMatmulRhsI8,
    q1_0: *const QuantizedMatmulRhsQ1_0,
    q2_0: *const QuantizedMatmulRhsQ2_0,
    q4_0: *const QuantizedMatmulRhsQ4_0,
    q4_1: *const QuantizedMatmulRhsQ4_1,
    q5_0: *const QuantizedMatmulRhsQ5_0,
    q5_1: *const QuantizedMatmulRhsQ5_1,
    q8_0: *const QuantizedMatmulRhsQ8_0,
    q2_k: *const QuantizedMatmulRhsQ2_K,
    q3_k: *const QuantizedMatmulRhsQ3_K,
    q4_k: *const QuantizedMatmulRhsQ4_K,
    q5_k: *const QuantizedMatmulRhsQ5_K,
    q6_k: *const QuantizedMatmulRhsQ6_K,
    iq1_s: *const QuantizedMatmulRhsIQ1_S,
    iq1_m: *const QuantizedMatmulRhsIQ1_M,
    iq2_xxs: *const QuantizedMatmulRhsIQ2_XXS,
    iq2_xs: *const QuantizedMatmulRhsIQ2_XS,
    iq2_s: *const QuantizedMatmulRhsIQ2_S,
    iq3_xxs: *const QuantizedMatmulRhsIQ3_XXS,
    iq3_s: *const QuantizedMatmulRhsIQ3_S,
    iq4_nl: *const QuantizedMatmulRhsIQ4_NL,
    iq4_xs: *const QuantizedMatmulRhsIQ4_XS,
    tq1_0: *const QuantizedMatmulRhsTQ1_0,
    tq2_0: *const QuantizedMatmulRhsTQ2_0,
    mxfp4: *const QuantizedMatmulRhsMXFP4,
    nvfp4: *const QuantizedMatmulRhsNVFP4,

    pub fn innerDim(self: AnyQuantizedMatmulRhs) usize {
        return switch (self) {
            inline else => |rhs| rhs.k,
        };
    }

    pub fn outputDim(self: AnyQuantizedMatmulRhs) usize {
        return switch (self) {
            inline else => |rhs| rhs.n,
        };
    }
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
