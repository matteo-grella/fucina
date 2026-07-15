//! Quant matmul TYPE + format-trait layer relocated out of quant.zig so the
//! quant kernel children import their packed/RHS types + format traits from this
//! child-neutral module instead of the parent barrel — breaking the
//! quant.zig<->children import cycle. quant.zig re-exports all of these so
//! `quant.<sym>` callers are unchanged.

const std = @import("std");
const dtype_mod = @import("../../dtype.zig");
const tensor = @import("../../tensor.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const Tensor = tensor.Tensor;

pub const QuantizedMatmulFormat = enum {
    fucina_w8a8_rhs,
    ggml_q1_0,
    ggml_q2_0,
    ggml_q4_0,
    ggml_q4_1,
    ggml_q5_0,
    ggml_q5_1,
    ggml_q8_0,
    ggml_q8_1,
    ggml_q2_k,
    ggml_q3_k,
    ggml_q4_k,
    ggml_q5_k,
    ggml_q6_k,
    ggml_q8_k,
    ggml_iq1_s,
    ggml_iq1_m,
    ggml_iq2_xxs,
    ggml_iq2_xs,
    ggml_iq2_s,
    ggml_iq3_xxs,
    ggml_iq3_s,
    ggml_iq4_nl,
    ggml_iq4_xs,
    ggml_tq1_0,
    ggml_tq2_0,
    ggml_mxfp4,
    ggml_nvfp4,
};

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
    UnsupportedMatmulKernel,
};

pub fn checkedProduct(a: usize, b: usize) QuantizedFormatError!usize {
    return std.math.mul(usize, a, b) catch QuantizedFormatError.InvalidQuantizedLength;
}

pub const BlockQ1_0 = dtype_mod.BlockQ1_0;
pub const BlockQ2_0 = dtype_mod.BlockQ2_0;
pub const BlockQ8_0 = dtype_mod.BlockQ8_0;
pub const BlockQ8_1 = dtype_mod.BlockQ8_1;
pub const BlockQ8_0x4 = extern struct {
    d: [4]u16,
    qs: [4 * q8_0_block_size]i8,
};
pub const BlockQ4_0 = dtype_mod.BlockQ4_0;
pub const BlockQ4_1 = dtype_mod.BlockQ4_1;
pub const BlockQ5_0 = dtype_mod.BlockQ5_0;
pub const BlockQ5_1 = dtype_mod.BlockQ5_1;
pub const BlockQ2_K = dtype_mod.BlockQ2_K;
pub const BlockQ3_K = dtype_mod.BlockQ3_K;
pub const BlockQ4_K = dtype_mod.BlockQ4_K;
pub const BlockQ5_K = dtype_mod.BlockQ5_K;
pub const BlockQ6_K = dtype_mod.BlockQ6_K;
pub const BlockQ8_K = dtype_mod.BlockQ8_K;
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
pub const BlockIQ1_S = dtype_mod.BlockIQ1_S;
pub const BlockIQ1_M = dtype_mod.BlockIQ1_M;
pub const BlockIQ2_XXS = dtype_mod.BlockIQ2_XXS;
pub const BlockIQ2_XS = dtype_mod.BlockIQ2_XS;
pub const BlockIQ2_S = dtype_mod.BlockIQ2_S;
pub const BlockIQ3_XXS = dtype_mod.BlockIQ3_XXS;
pub const BlockIQ3_S = dtype_mod.BlockIQ3_S;
pub const BlockIQ4_NL = dtype_mod.BlockIQ4_NL;
pub const BlockIQ4_XS = dtype_mod.BlockIQ4_XS;
pub const BlockTQ1_0 = dtype_mod.BlockTQ1_0;
pub const BlockTQ2_0 = dtype_mod.BlockTQ2_0;
pub const BlockMXFP4 = dtype_mod.BlockMXFP4;
pub const BlockNVFP4 = dtype_mod.BlockNVFP4;

pub fn QuantizedRowsFor(comptime dtype: DType) type {
    return struct {
        /// Owning allocator, or null when `blocks` borrows external storage
        /// kept alive by the caller (e.g. packed ES genome blocks); deinit
        /// then frees nothing.
        allocator: ?Allocator,
        blocks: []dtype_mod.Storage(dtype),
        rows: usize,
        cols: usize,
        blocks_per_row: usize,

        const Self = @This();
        pub const format = formatForDType(dtype);
        pub const traits = matmulTraits(format);

        pub fn deinit(self: *Self) void {
            if (self.allocator) |allocator| allocator.free(self.blocks);
            self.* = undefined;
        }

        pub fn rowBlocks(self: *const Self, row: usize) []const dtype_mod.Storage(dtype) {
            return self.blocks[row * self.blocks_per_row ..][0..self.blocks_per_row];
        }
    };
}

pub fn QuantizedMatmulRhsRowsFor(comptime dtype: DType) type {
    return struct {
        rows: QuantizedRowsFor(dtype),
        k: usize,
        n: usize,

        const Self = @This();
        pub const format = formatForDType(dtype);
        pub const traits = matmulTraits(format);

        pub fn deinit(self: *Self) void {
            self.rows.deinit();
            self.* = undefined;
        }

        pub fn columnBlocks(self: *const Self, column: usize) []const dtype_mod.Storage(dtype) {
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
    blocks: []const BlockQ8_0,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q8_0;
    pub const traits = matmulTraits(.ggml_q8_0);

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn rowBlocks(self: *const Self, row: usize) []const BlockQ8_0 {
        return self.blocks[row * self.blocks_per_row ..][0..self.blocks_per_row];
    }
};

pub const QuantizedRowsQ4_0 = struct {
    allocator: Allocator,
    blocks: []BlockQ4_0,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q4_0;
    pub const traits = matmulTraits(.ggml_q4_0);

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn rowBlocks(self: *const Self, row: usize) []const BlockQ4_0 {
        return self.blocks[row * self.blocks_per_row ..][0..self.blocks_per_row];
    }
};

pub const QuantizedMatmulRhsQ8_0 = struct {
    rows: QuantizedRowsQ8_0,
    k: usize,
    n: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q8_0;
    pub const traits = matmulTraits(.ggml_q8_0);

    pub fn deinit(self: *Self) void {
        self.rows.deinit();
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ8_0 {
        return self.rows.rowBlocks(column);
    }
};

/// Comptime layout discriminant carried by every packed (lane-interleaved)
/// matmul RHS container, naming format × grouping. The plain per-format RHS
/// structs carry `format`/`traits`; the packed structs carry `layout` so
/// generic entries dispatch with an exhaustive switch — adding a layout
/// forces edits at every switch site.
pub const PackedRhsLayout = enum { q8_0x4, q6_kx4, q4_kx4, q4_kx8, q4_kx2mmla, q5_kx8 };

/// Packed RHS container type for a given layout.
pub fn PackedRhsFor(comptime rhs_layout: PackedRhsLayout) type {
    return switch (rhs_layout) {
        .q8_0x4 => QuantizedMatmulRhsQ8_0x4,
        .q6_kx4 => QuantizedMatmulRhsQ6_Kx4,
        .q4_kx4 => QuantizedMatmulRhsQ4_Kx4,
        .q4_kx8 => QuantizedMatmulRhsQ4_Kx8,
        .q4_kx2mmla => QuantizedMatmulRhsQ4_Kx2Mmla,
        .q5_kx8 => QuantizedMatmulRhsQ5_Kx8,
    };
}

pub const QuantizedMatmulRhsQ8_0x4 = struct {
    allocator: Allocator,
    blocks: []BlockQ8_0x4,
    k: usize,
    n: usize,
    blocks_per_group: usize,

    const Self = @This();
    pub const layout: PackedRhsLayout = .q8_0x4;

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
    pub const format = QuantizedMatmulFormat.ggml_q4_0;
    pub const traits = matmulTraits(.ggml_q4_0);

    pub fn deinit(self: *Self) void {
        self.rows.deinit();
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ4_0 {
        return self.rows.rowBlocks(column);
    }
};

pub const QuantizedMatmulRhsQ2_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF expert stack kept alive by the model).
    allocator: ?Allocator,
    blocks: []const BlockQ2_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q2_k;
    pub const traits = matmulTraits(.ggml_q2_k);

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ2_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ3_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF expert stack kept alive by the model).
    allocator: ?Allocator,
    blocks: []const BlockQ3_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q3_k;
    pub const traits = matmulTraits(.ggml_q3_k);

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ3_K {
        return self.blocks[column * self.blocks_per_column ..][0..self.blocks_per_column];
    }
};

pub const QuantizedMatmulRhsQ4_K = struct {
    /// Owning allocator, or null when `blocks` borrows external read-only
    /// memory (e.g. an mmap'd GGUF kept alive by the model).
    allocator: ?Allocator,
    blocks: []const BlockQ4_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q4_k;
    pub const traits = matmulTraits(.ggml_q4_k);

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ4_K {
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
    pub const layout: PackedRhsLayout = .q4_kx4;

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
    pub const layout: PackedRhsLayout = .q4_kx8;

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
    pub const layout: PackedRhsLayout = .q4_kx2mmla;

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
    blocks: []const BlockQ5_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q5_k;
    pub const traits = matmulTraits(.ggml_q5_k);

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ5_K {
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
    pub const layout: PackedRhsLayout = .q5_kx8;

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
    blocks: []const BlockQ6_K,
    k: usize,
    n: usize,
    blocks_per_column: usize,

    const Self = @This();
    pub const format = QuantizedMatmulFormat.ggml_q6_k;
    pub const traits = matmulTraits(.ggml_q6_k);

    pub fn deinit(self: *Self) void {
        if (self.allocator) |allocator| allocator.free(self.blocks);
        self.* = undefined;
    }

    pub fn columnBlocks(self: *const Self, column: usize) []const BlockQ6_K {
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
    pub const layout: PackedRhsLayout = .q6_kx4;

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
    ggml_q1_0: *const QuantizedMatmulRhsQ1_0,
    ggml_q2_0: *const QuantizedMatmulRhsQ2_0,
    ggml_q4_0: *const QuantizedMatmulRhsQ4_0,
    ggml_q4_1: *const QuantizedMatmulRhsQ4_1,
    ggml_q5_0: *const QuantizedMatmulRhsQ5_0,
    ggml_q5_1: *const QuantizedMatmulRhsQ5_1,
    ggml_q8_0: *const QuantizedMatmulRhsQ8_0,
    ggml_q2_k: *const QuantizedMatmulRhsQ2_K,
    ggml_q3_k: *const QuantizedMatmulRhsQ3_K,
    ggml_q4_k: *const QuantizedMatmulRhsQ4_K,
    ggml_q5_k: *const QuantizedMatmulRhsQ5_K,
    ggml_q6_k: *const QuantizedMatmulRhsQ6_K,
    ggml_iq1_s: *const QuantizedMatmulRhsIQ1_S,
    ggml_iq1_m: *const QuantizedMatmulRhsIQ1_M,
    ggml_iq2_xxs: *const QuantizedMatmulRhsIQ2_XXS,
    ggml_iq2_xs: *const QuantizedMatmulRhsIQ2_XS,
    ggml_iq2_s: *const QuantizedMatmulRhsIQ2_S,
    ggml_iq3_xxs: *const QuantizedMatmulRhsIQ3_XXS,
    ggml_iq3_s: *const QuantizedMatmulRhsIQ3_S,
    ggml_iq4_nl: *const QuantizedMatmulRhsIQ4_NL,
    ggml_iq4_xs: *const QuantizedMatmulRhsIQ4_XS,
    ggml_tq1_0: *const QuantizedMatmulRhsTQ1_0,
    ggml_tq2_0: *const QuantizedMatmulRhsTQ2_0,
    ggml_mxfp4: *const QuantizedMatmulRhsMXFP4,
    ggml_nvfp4: *const QuantizedMatmulRhsNVFP4,

    pub fn format(self: AnyQuantizedMatmulRhs) QuantizedMatmulFormat {
        return switch (self) {
            .fucina_w8a8_rhs => .fucina_w8a8_rhs,
            .ggml_q1_0 => .ggml_q1_0,
            .ggml_q2_0 => .ggml_q2_0,
            .ggml_q4_0 => .ggml_q4_0,
            .ggml_q4_1 => .ggml_q4_1,
            .ggml_q5_0 => .ggml_q5_0,
            .ggml_q5_1 => .ggml_q5_1,
            .ggml_q8_0 => .ggml_q8_0,
            .ggml_q2_k => .ggml_q2_k,
            .ggml_q3_k => .ggml_q3_k,
            .ggml_q4_k => .ggml_q4_k,
            .ggml_q5_k => .ggml_q5_k,
            .ggml_q6_k => .ggml_q6_k,
            .ggml_iq1_s => .ggml_iq1_s,
            .ggml_iq1_m => .ggml_iq1_m,
            .ggml_iq2_xxs => .ggml_iq2_xxs,
            .ggml_iq2_xs => .ggml_iq2_xs,
            .ggml_iq2_s => .ggml_iq2_s,
            .ggml_iq3_xxs => .ggml_iq3_xxs,
            .ggml_iq3_s => .ggml_iq3_s,
            .ggml_iq4_nl => .ggml_iq4_nl,
            .ggml_iq4_xs => .ggml_iq4_xs,
            .ggml_tq1_0 => .ggml_tq1_0,
            .ggml_tq2_0 => .ggml_tq2_0,
            .ggml_mxfp4 => .ggml_mxfp4,
            .ggml_nvfp4 => .ggml_nvfp4,
        };
    }

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

pub const QuantizedStorageLayout = enum {
    transposed_output_columns,
    ggml_blocks,
};

pub const QuantizedScaleLayout = enum {
    output_column_group,
    inline_block_scale,
};

pub const QuantizedMatmulKernel = enum {
    unsupported,
    fucina_w8a8_f32,
    ggml_q1_0,
    ggml_q2_0,
    ggml_q4_0,
    ggml_q4_1,
    ggml_q5_0,
    ggml_q5_1,
    ggml_q8_0,
    ggml_q2_k,
    ggml_q3_k,
    ggml_q4_k,
    ggml_q5_k,
    ggml_q6_k,
    ggml_iq1_s,
    ggml_iq1_m,
    ggml_iq2_xxs,
    ggml_iq2_xs,
    ggml_iq2_s,
    ggml_iq3_xxs,
    ggml_iq3_s,
    ggml_iq4_nl,
    ggml_iq4_xs,
    ggml_tq1_0,
    ggml_tq2_0,
    ggml_mxfp4,
    ggml_nvfp4,
};

pub const QuantizedMatmulTraits = struct {
    format: QuantizedMatmulFormat,
    source_dtype: DType,
    storage_dtype: DType,
    scale_dtype: DType,
    default_group_size: usize,
    block_size: usize,
    block_byte_size: ?usize,
    storage_layout: QuantizedStorageLayout,
    scale_layout: QuantizedScaleLayout,
    supports_from_float: bool,
    supports_to_float: bool,
    supports_matmul: bool,
    matmul_kernel: QuantizedMatmulKernel,

    pub fn effectiveGroupSize(self: QuantizedMatmulTraits, requested_group_size: usize) usize {
        return if (requested_group_size == 0) self.default_group_size else requested_group_size;
    }

    pub fn groupCountForSize(_: QuantizedMatmulTraits, k: usize, group_size: usize) usize {
        return (k + group_size - 1) / group_size;
    }

    pub fn groupCount(self: QuantizedMatmulTraits, k: usize, requested_group_size: usize) usize {
        return self.groupCountForSize(k, self.effectiveGroupSize(requested_group_size));
    }

    pub fn storageRowSize(self: QuantizedMatmulTraits, k: usize) usize {
        return switch (self.storage_layout) {
            .transposed_output_columns => k,
            .ggml_blocks => self.groupCountForSize(k, self.block_size),
        };
    }

    pub fn storageShape(self: QuantizedMatmulTraits, k: usize, n: usize) [2]usize {
        return switch (self.storage_layout) {
            .transposed_output_columns => .{ n, self.storageRowSize(k) },
            .ggml_blocks => .{ n, self.storageRowSize(k) },
        };
    }

    pub fn scaleShape(self: QuantizedMatmulTraits, k: usize, n: usize, group_size: usize) [2]usize {
        return switch (self.scale_layout) {
            .output_column_group => .{ n, self.groupCountForSize(k, group_size) },
            .inline_block_scale => .{ 0, 0 },
        };
    }

    pub fn storageIndex(self: QuantizedMatmulTraits, output_col: usize, input_index: usize, k: usize) usize {
        return switch (self.storage_layout) {
            .transposed_output_columns => output_col * self.storageRowSize(k) + input_index,
            .ggml_blocks => output_col * self.storageRowSize(k) + input_index / self.block_size,
        };
    }

    pub fn scaleIndex(self: QuantizedMatmulTraits, output_col: usize, group_index: usize, num_groups: usize) usize {
        return switch (self.scale_layout) {
            .output_column_group => output_col * num_groups + group_index,
            .inline_block_scale => self.storageIndex(output_col, group_index * self.block_size, num_groups * self.block_size),
        };
    }
};

fn ggmlBlockTraits(
    comptime format_value: QuantizedMatmulFormat,
    comptime storage_dtype: DType,
    comptime scale_dtype: DType,
    comptime block_size: usize,
    comptime block_byte_size: usize,
    comptime supports_from_float: bool,
    comptime kernel: QuantizedMatmulKernel,
) QuantizedMatmulTraits {
    return .{
        .format = format_value,
        .source_dtype = .f32,
        .storage_dtype = storage_dtype,
        .scale_dtype = scale_dtype,
        .default_group_size = block_size,
        .block_size = block_size,
        .block_byte_size = block_byte_size,
        .storage_layout = .ggml_blocks,
        .scale_layout = .inline_block_scale,
        .supports_from_float = supports_from_float,
        .supports_to_float = true,
        .supports_matmul = true,
        .matmul_kernel = kernel,
    };
}

pub fn matmulTraits(comptime format_value: QuantizedMatmulFormat) QuantizedMatmulTraits {
    return switch (format_value) {
        .fucina_w8a8_rhs => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .i8,
            .scale_dtype = .f32,
            .default_group_size = default_i8_group_size,
            .block_size = default_i8_group_size,
            .block_byte_size = null,
            .storage_layout = .transposed_output_columns,
            .scale_layout = .output_column_group,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .fucina_w8a8_f32,
        },
        .ggml_q1_0 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = q1_0_block_size,
            .block_size = q1_0_block_size,
            .block_byte_size = @sizeOf(BlockQ1_0),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            // No f32 -> Q1_0 encoder exists (decode/matmul only).
            .supports_from_float = false,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q1_0,
        },
        .ggml_q2_0 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = q2_0_block_size,
            .block_size = q2_0_block_size,
            .block_byte_size = @sizeOf(BlockQ2_0),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            // quantize_row_q2_0_ref parity encoder (absmax ternary codes).
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q2_0,
        },
        .ggml_q4_0 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = q4_0_block_size,
            .block_size = q4_0_block_size,
            .block_byte_size = @sizeOf(BlockQ4_0),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q4_0,
        },
        .ggml_q4_1 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = q4_1_block_size,
            .block_size = q4_1_block_size,
            .block_byte_size = @sizeOf(BlockQ4_1),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q4_1,
        },
        .ggml_q5_0 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = q5_0_block_size,
            .block_size = q5_0_block_size,
            .block_byte_size = @sizeOf(BlockQ5_0),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q5_0,
        },
        .ggml_q5_1 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = q5_1_block_size,
            .block_size = q5_1_block_size,
            .block_byte_size = @sizeOf(BlockQ5_1),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q5_1,
        },
        .ggml_q8_0 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .i8,
            .scale_dtype = .f16,
            .default_group_size = q8_0_block_size,
            .block_size = q8_0_block_size,
            .block_byte_size = @sizeOf(BlockQ8_0),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q8_0,
        },
        .ggml_q8_1 => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .i8,
            .scale_dtype = .f16,
            .default_group_size = q8_1_block_size,
            .block_size = q8_1_block_size,
            .block_byte_size = @sizeOf(BlockQ8_1),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = false,
            .matmul_kernel = .unsupported,
        },
        .ggml_q2_k => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = qk_k_block_size,
            .block_size = qk_k_block_size,
            .block_byte_size = @sizeOf(BlockQ2_K),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = false,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q2_k,
        },
        .ggml_q3_k => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = qk_k_block_size,
            .block_size = qk_k_block_size,
            .block_byte_size = @sizeOf(BlockQ3_K),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = false,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q3_k,
        },
        .ggml_q4_k => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = qk_k_block_size,
            .block_size = qk_k_block_size,
            .block_byte_size = @sizeOf(BlockQ4_K),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q4_k,
        },
        .ggml_q5_k => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = qk_k_block_size,
            .block_size = qk_k_block_size,
            .block_byte_size = @sizeOf(BlockQ5_K),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q5_k,
        },
        .ggml_q6_k => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .u8,
            .scale_dtype = .f16,
            .default_group_size = qk_k_block_size,
            .block_size = qk_k_block_size,
            .block_byte_size = @sizeOf(BlockQ6_K),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = true,
            .matmul_kernel = .ggml_q6_k,
        },
        .ggml_q8_k => .{
            .format = format_value,
            .source_dtype = .f32,
            .storage_dtype = .i8,
            .scale_dtype = .f32,
            .default_group_size = qk_k_block_size,
            .block_size = qk_k_block_size,
            .block_byte_size = @sizeOf(BlockQ8_K),
            .storage_layout = .ggml_blocks,
            .scale_layout = .inline_block_scale,
            .supports_from_float = true,
            .supports_to_float = true,
            .supports_matmul = false,
            .matmul_kernel = .unsupported,
        },
        .ggml_iq1_s => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ1_S), false, .ggml_iq1_s),
        .ggml_iq1_m => ggmlBlockTraits(format_value, .u8, .u8, qk_k_block_size, @sizeOf(BlockIQ1_M), false, .ggml_iq1_m),
        .ggml_iq2_xxs => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ2_XXS), false, .ggml_iq2_xxs),
        .ggml_iq2_xs => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ2_XS), false, .ggml_iq2_xs),
        .ggml_iq2_s => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ2_S), false, .ggml_iq2_s),
        .ggml_iq3_xxs => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ3_XXS), false, .ggml_iq3_xxs),
        .ggml_iq3_s => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ3_S), false, .ggml_iq3_s),
        // supports_from_float = false: decode/matmul only, no f32 encoders exist
        // (tq2_0 is the exception — quant/ternary.zig ships its encoders).
        .ggml_iq4_nl => ggmlBlockTraits(format_value, .u8, .f16, iq4_nl_block_size, @sizeOf(BlockIQ4_NL), false, .ggml_iq4_nl),
        .ggml_iq4_xs => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockIQ4_XS), false, .ggml_iq4_xs),
        .ggml_tq1_0 => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockTQ1_0), false, .ggml_tq1_0),
        .ggml_tq2_0 => ggmlBlockTraits(format_value, .u8, .f16, qk_k_block_size, @sizeOf(BlockTQ2_0), true, .ggml_tq2_0),
        .ggml_mxfp4 => ggmlBlockTraits(format_value, .u8, .u8, mxfp4_block_size, @sizeOf(BlockMXFP4), false, .ggml_mxfp4),
        .ggml_nvfp4 => ggmlBlockTraits(format_value, .u8, .u8, nvfp4_block_size, @sizeOf(BlockNVFP4), false, .ggml_nvfp4),
    };
}

pub fn matmulTraitsRuntime(format_value: QuantizedMatmulFormat) QuantizedMatmulTraits {
    return switch (format_value) {
        .fucina_w8a8_rhs => matmulTraits(.fucina_w8a8_rhs),
        .ggml_q1_0 => matmulTraits(.ggml_q1_0),
        .ggml_q2_0 => matmulTraits(.ggml_q2_0),
        .ggml_q4_0 => matmulTraits(.ggml_q4_0),
        .ggml_q4_1 => matmulTraits(.ggml_q4_1),
        .ggml_q5_0 => matmulTraits(.ggml_q5_0),
        .ggml_q5_1 => matmulTraits(.ggml_q5_1),
        .ggml_q8_0 => matmulTraits(.ggml_q8_0),
        .ggml_q8_1 => matmulTraits(.ggml_q8_1),
        .ggml_q2_k => matmulTraits(.ggml_q2_k),
        .ggml_q3_k => matmulTraits(.ggml_q3_k),
        .ggml_q4_k => matmulTraits(.ggml_q4_k),
        .ggml_q5_k => matmulTraits(.ggml_q5_k),
        .ggml_q6_k => matmulTraits(.ggml_q6_k),
        .ggml_q8_k => matmulTraits(.ggml_q8_k),
        .ggml_iq1_s => matmulTraits(.ggml_iq1_s),
        .ggml_iq1_m => matmulTraits(.ggml_iq1_m),
        .ggml_iq2_xxs => matmulTraits(.ggml_iq2_xxs),
        .ggml_iq2_xs => matmulTraits(.ggml_iq2_xs),
        .ggml_iq2_s => matmulTraits(.ggml_iq2_s),
        .ggml_iq3_xxs => matmulTraits(.ggml_iq3_xxs),
        .ggml_iq3_s => matmulTraits(.ggml_iq3_s),
        .ggml_iq4_nl => matmulTraits(.ggml_iq4_nl),
        .ggml_iq4_xs => matmulTraits(.ggml_iq4_xs),
        .ggml_tq1_0 => matmulTraits(.ggml_tq1_0),
        .ggml_tq2_0 => matmulTraits(.ggml_tq2_0),
        .ggml_mxfp4 => matmulTraits(.ggml_mxfp4),
        .ggml_nvfp4 => matmulTraits(.ggml_nvfp4),
    };
}

pub fn formatForDType(comptime tensor_dtype: DType) QuantizedMatmulFormat {
    return switch (tensor_dtype) {
        .q1_0 => .ggml_q1_0,
        .q2_0 => .ggml_q2_0,
        .q4_0 => .ggml_q4_0,
        .q4_1 => .ggml_q4_1,
        .q5_0 => .ggml_q5_0,
        .q5_1 => .ggml_q5_1,
        .q8_0 => .ggml_q8_0,
        .q8_1 => .ggml_q8_1,
        .q2_k => .ggml_q2_k,
        .q3_k => .ggml_q3_k,
        .q4_k => .ggml_q4_k,
        .q5_k => .ggml_q5_k,
        .q6_k => .ggml_q6_k,
        .q8_k => .ggml_q8_k,
        .iq1_s => .ggml_iq1_s,
        .iq1_m => .ggml_iq1_m,
        .iq2_xxs => .ggml_iq2_xxs,
        .iq2_xs => .ggml_iq2_xs,
        .iq2_s => .ggml_iq2_s,
        .iq3_xxs => .ggml_iq3_xxs,
        .iq3_s => .ggml_iq3_s,
        .iq4_nl => .ggml_iq4_nl,
        .iq4_xs => .ggml_iq4_xs,
        .tq1_0 => .ggml_tq1_0,
        .tq2_0 => .ggml_tq2_0,
        .mxfp4 => .ggml_mxfp4,
        .nvfp4 => .ggml_nvfp4,
        else => @compileError("dtype is not a quantized matmul format"),
    };
}

pub fn supportsMatmul(format_value: QuantizedMatmulFormat) bool {
    return matmulTraitsRuntime(format_value).supports_matmul;
}

pub fn QuantizedMatmulRhs(comptime format_value: QuantizedMatmulFormat) type {
    return switch (format_value) {
        // Symmetric int8 quantized weights, stored transposed as [n][k] with one
        // f32 scale per (column, group) block along k. This is a quantized matmul
        // weight container, not a dense TensorOf(.i8) dtype.
        .fucina_w8a8_rhs => struct {
            qw: tensor.TensorOf(.i8),
            scales: Tensor,
            k: usize,
            n: usize,
            group_size: usize,
            num_groups: usize,

            const Self = @This();
            pub const format = format_value;
            pub const traits = matmulTraits(format_value);
            pub const source_dtype = traits.source_dtype;
            pub const packed_dtype = traits.storage_dtype;
            pub const scale_dtype = traits.scale_dtype;

            pub fn deinit(self: *Self) void {
                self.qw.deinit();
                self.scales.deinit();
                self.* = undefined;
            }
        },
        .ggml_q1_0,
        .ggml_q2_0,
        .ggml_q4_0,
        .ggml_q4_1,
        .ggml_q5_0,
        .ggml_q5_1,
        .ggml_q8_0,
        .ggml_q8_1,
        => @compileError("GGML row-block formats use dedicated block structs"),
        .ggml_q2_k,
        .ggml_q3_k,
        .ggml_q4_k,
        .ggml_q5_k,
        .ggml_q6_k,
        .ggml_q8_k,
        .ggml_iq1_s,
        .ggml_iq1_m,
        .ggml_iq2_xxs,
        .ggml_iq2_xs,
        .ggml_iq2_s,
        .ggml_iq3_xxs,
        .ggml_iq3_s,
        .ggml_iq4_nl,
        .ggml_iq4_xs,
        .ggml_tq1_0,
        .ggml_tq2_0,
        .ggml_mxfp4,
        .ggml_nvfp4,
        => @compileError("K-quant matmul RHS containers use dedicated GGML block structs"),
    };
}

pub const QuantizedMatmulRhsI8 = QuantizedMatmulRhs(.fucina_w8a8_rhs);
