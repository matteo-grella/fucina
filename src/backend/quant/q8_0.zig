//! Hot Q8_0 / Q8_0x4 quantized matmul kernels relocated out of quant.zig.
//! See quant.zig for shared type/helper definitions; every shared symbol this
//! module references is aliased from quant.zig (`qm`) below so the moved bodies
//! compile unchanged.

const std = @import("std");
const builtin = @import("builtin");
const tensor = @import("../../tensor.zig");
const q8k_mod = @import("q8k.zig");
const types_mod = @import("types.zig");
const common = @import("common.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;

// Shared symbols defined in quant.zig, aliased here so the moved bodies compile unchanged.
const BlockQ8_0 = types_mod.BlockQ8_0;
const BlockQ8_0x4 = types_mod.BlockQ8_0x4;
const QKV16i8 = common.QKV16i8;
const QKV32i8 = common.QKV32i8;
const QKV32u8 = common.QKV32u8;
const QKV4f32 = common.QKV4f32;
const QKV4i32 = common.QKV4i32;
const QKV8i32 = common.QKV8i32;
const addHalvesI32x8 = common.addHalvesI32x8;
const dotI8GroupsWidenI32x8 = common.dotI8GroupsWidenI32x8;
const dpbusdI32x8 = common.dpbusdI32x8;
const has_x86_vnni_ymm = common.has_x86_vnni_ymm;
const maddubsDotGroupsI32x8 = common.maddubsDotGroupsI32x8;
const psignI8x32 = common.psignI8x32;
const QuantizedFormatError = types_mod.QuantizedFormatError;
const QuantizedMatmulRhsQ8_0 = types_mod.QuantizedMatmulRhsQ8_0;
const QuantizedMatmulRhsQ8_0x4 = types_mod.QuantizedMatmulRhsQ8_0x4;
const checkedProduct = types_mod.checkedProduct;
const dequantizeRowQ8_0Into = q8k_mod.dequantizeRowQ8_0Into;
const dequantizeRowsQ8_0Into = q8k_mod.dequantizeRowsQ8_0Into;
const dotI8x16Portable = common.dotI8x16Portable;
const f16BitsToF32 = common.f16BitsToF32;
const f16x4BitsToF32 = common.f16x4BitsToF32;
const f32ToF16Bits = common.f32ToF16Bits;
const getRowsQ8_0Into = q8k_mod.getRowsQ8_0Into;
const has_x86_avx2 = common.has_x86_avx2;
const i8DotI32 = common.i8DotI32;
const q8_0BlockCount = q8k_mod.q8_0BlockCount;
const q8_0_block_size = types_mod.q8_0_block_size;
const q8_0_row_block = common.q8_0_row_block;
const quantizeRowQ8_0Into = q8k_mod.quantizeRowQ8_0Into;
const quantizeRowsQ8_0 = q8k_mod.quantizeRowsQ8_0;
const quantizeRowsQ8_0Into = q8k_mod.quantizeRowsQ8_0Into;
const roundHalfAwayFromZeroVec4ToI32 = common.roundHalfAwayFromZeroVec4ToI32;
const sdotI8x16 = common.sdotI8x16;
const sdotI8x16Lane = common.sdotI8x16Lane;

pub fn quantizeRowsQ8_0x4Into(blocks: []BlockQ8_0x4, src: *const Tensor) !void {
    return quantizeRowsQ8_0x4IntoImpl(blocks, src, false);
}

pub fn quantizeRowsQ8_0x4PaddedInto(blocks: []BlockQ8_0x4, src: *const Tensor) !void {
    return quantizeRowsQ8_0x4IntoImpl(blocks, src, true);
}

fn quantizeRowsQ8_0x4IntoImpl(blocks: []BlockQ8_0x4, src: *const Tensor, comptime pad_rows: bool) !void {
    const view = try src.rankView(2);
    const rows = view.dim(0);
    const cols = view.dim(1);
    if (!pad_rows and rows % 4 != 0) return tensor.TensorError.InvalidShape;

    const blocks_per_row = try q8_0BlockCount(cols);
    const row_groups = if (pad_rows) (rows + 3) / 4 else rows / 4;
    if (blocks.len != try checkedProduct(row_groups, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const data = try src.dataConstChecked();
    if (!pad_rows) {
        quantizeRowsQ8_0x4GroupsInto(blocks, data, cols, blocks_per_row, 0, row_groups);
        return;
    }
    var row_group: usize = 0;
    while (row_group < row_groups) : (row_group += 1) {
        var block_index: usize = 0;
        while (block_index < blocks_per_row) : (block_index += 1) {
            var dst = &blocks[row_group * blocks_per_row + block_index];
            inline for (0..4) |row_lane| {
                const row = row_group * 4 + row_lane;
                if (row >= rows) {
                    dst.d[row_lane] = 0;
                    inline for (0..8) |feature_group| {
                        inline for (0..4) |lane| {
                            dst.qs[feature_group * 16 + row_lane * 4 + lane] = 0;
                        }
                    }
                } else {
                    const source = data[row * cols + block_index * q8_0_block_size ..][0..q8_0_block_size];

                    var amaxv: QKV4f32 = @splat(0);
                    inline for (0..8) |feature_group| {
                        const v: QKV4f32 = source[feature_group * 4 ..][0..4].*;
                        amaxv = @max(amaxv, @abs(v));
                    }

                    const amax = @reduce(.Max, amaxv);
                    const d = amax / 127.0;
                    const inv_d: f32 = if (d == 0) 0 else 1.0 / d;
                    dst.d[row_lane] = f32ToF16Bits(d);

                    inline for (0..8) |feature_group| {
                        const v: QKV4f32 = source[feature_group * 4 ..][0..4].*;
                        const scaled = v * @as(QKV4f32, @splat(inv_d));
                        const clamped = @max(@as(QKV4f32, @splat(-127.0)), @min(@as(QKV4f32, @splat(127.0)), scaled));
                        const q = roundHalfAwayFromZeroVec4ToI32(clamped);
                        inline for (0..4) |lane| {
                            dst.qs[feature_group * 16 + row_lane * 4 + lane] = @intCast(q[lane]);
                        }
                    }
                }
            }
        }
    }
}

pub fn quantizeRowsQ8_0x4GroupsInto(
    blocks: []BlockQ8_0x4,
    data: []const f32,
    cols: usize,
    blocks_per_row: usize,
    row_group_start: usize,
    row_group_end: usize,
) void {
    var row_group = row_group_start;
    while (row_group < row_group_end) : (row_group += 1) {
        var block_index: usize = 0;
        while (block_index < blocks_per_row) : (block_index += 1) {
            var dst = &blocks[row_group * blocks_per_row + block_index];
            inline for (0..4) |row_lane| {
                const row = row_group * 4 + row_lane;
                const source = data[row * cols + block_index * q8_0_block_size ..][0..q8_0_block_size];

                var amaxv: QKV4f32 = @splat(0);
                inline for (0..8) |feature_group| {
                    const v: QKV4f32 = source[feature_group * 4 ..][0..4].*;
                    amaxv = @max(amaxv, @abs(v));
                }

                const amax = @reduce(.Max, amaxv);
                const d = amax / 127.0;
                const inv_d: f32 = if (d == 0) 0 else 1.0 / d;
                dst.d[row_lane] = f32ToF16Bits(d);

                inline for (0..8) |feature_group| {
                    const v: QKV4f32 = source[feature_group * 4 ..][0..4].*;
                    const scaled = v * @as(QKV4f32, @splat(inv_d));
                    const clamped = @max(@as(QKV4f32, @splat(-127.0)), @min(@as(QKV4f32, @splat(127.0)), scaled));
                    const q = roundHalfAwayFromZeroVec4ToI32(clamped);
                    inline for (0..4) |lane| {
                        dst.qs[feature_group * 16 + row_lane * 4 + lane] = @intCast(q[lane]);
                    }
                }
            }
        }
    }
}

pub fn quantizeSplitSwiGluRowsQ8_0x4PaddedInto(
    blocks: []BlockQ8_0x4,
    data: []const f32,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
) !void {
    if (blocks_per_row != try q8_0BlockCount(cols)) return tensor.TensorError.InvalidShape;
    const row_groups = (rows + 3) / 4;
    if (blocks.len != try checkedProduct(row_groups, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;
    const row_len = std.math.mul(usize, rows, cols) catch return tensor.TensorError.InvalidDataLength;
    const expected_data_len = std.math.mul(usize, row_len, 2) catch return tensor.TensorError.InvalidDataLength;
    if (data.len < expected_data_len) return tensor.TensorError.InvalidDataLength;

    quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto(blocks, data, rows, cols, blocks_per_row, 0, row_groups);
}

pub fn quantizeSplitSwiGluRowsQ8_0x4PaddedGroupsInto(
    blocks: []BlockQ8_0x4,
    data: []const f32,
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    row_group_start: usize,
    row_group_end: usize,
) void {
    const one: QKV4f32 = @splat(1);
    var values: [q8_0_block_size]f32 = undefined;
    var row_group = row_group_start;
    while (row_group < row_group_end) : (row_group += 1) {
        var block_index: usize = 0;
        while (block_index < blocks_per_row) : (block_index += 1) {
            var dst = &blocks[row_group * blocks_per_row + block_index];
            inline for (0..4) |row_lane| {
                const row = row_group * 4 + row_lane;
                if (row >= rows) {
                    dst.d[row_lane] = 0;
                    inline for (0..8) |feature_group| {
                        inline for (0..4) |lane| {
                            dst.qs[feature_group * 16 + row_lane * 4 + lane] = 0;
                        }
                    }
                } else {
                    const row_base = row * cols * 2;
                    const gate = data[row_base + block_index * q8_0_block_size ..][0..q8_0_block_size];
                    const up = data[row_base + cols + block_index * q8_0_block_size ..][0..q8_0_block_size];

                    var amaxv: QKV4f32 = @splat(0);
                    inline for (0..8) |feature_group| {
                        const gate_v: QKV4f32 = gate[feature_group * 4 ..][0..4].*;
                        const up_v: QKV4f32 = up[feature_group * 4 ..][0..4].*;
                        const v = up_v * gate_v * (one / (one + @exp(-gate_v)));
                        values[feature_group * 4 ..][0..4].* = v;
                        amaxv = @max(amaxv, @abs(v));
                    }

                    const amax = @reduce(.Max, amaxv);
                    const d = amax / 127.0;
                    const inv_d: f32 = if (d == 0) 0 else 1.0 / d;
                    dst.d[row_lane] = f32ToF16Bits(d);

                    inline for (0..8) |feature_group| {
                        const v: QKV4f32 = values[feature_group * 4 ..][0..4].*;
                        const scaled = v * @as(QKV4f32, @splat(inv_d));
                        const clamped = @max(@as(QKV4f32, @splat(-127.0)), @min(@as(QKV4f32, @splat(127.0)), scaled));
                        const q = roundHalfAwayFromZeroVec4ToI32(clamped);
                        inline for (0..4) |lane| {
                            dst.qs[feature_group * 16 + row_lane * 4 + lane] = @intCast(q[lane]);
                        }
                    }
                }
            }
        }
    }
}

pub fn packMatmulRhsQ8_0x4(
    allocator: Allocator,
    blocks: []const BlockQ8_0,
    n: usize,
    k: usize,
    blocks_per_row: usize,
) !QuantizedMatmulRhsQ8_0x4 {
    if (n % 4 != 0) return tensor.TensorError.InvalidShape;
    if (blocks_per_row != try q8_0BlockCount(k)) return tensor.TensorError.InvalidShape;
    if (blocks.len != try checkedProduct(n, blocks_per_row)) return QuantizedFormatError.InvalidQuantizedLength;

    const group_count = n / 4;
    const packed_blocks = try allocator.alloc(BlockQ8_0x4, try checkedProduct(group_count, blocks_per_row));
    errdefer allocator.free(packed_blocks);

    for (0..group_count) |group_i| {
        for (0..blocks_per_row) |block_i| {
            const b0 = &blocks[(4 * group_i + 0) * blocks_per_row + block_i];
            const b1 = &blocks[(4 * group_i + 1) * blocks_per_row + block_i];
            const b2 = &blocks[(4 * group_i + 2) * blocks_per_row + block_i];
            const b3 = &blocks[(4 * group_i + 3) * blocks_per_row + block_i];
            var dst = &packed_blocks[group_i * blocks_per_row + block_i];
            dst.d = .{ b0.d, b1.d, b2.d, b3.d };

            for (0..8) |feature_group| {
                const src_offset = feature_group * 4;
                const dst_offset = feature_group * 16;
                inline for (0..4) |lane| {
                    dst.qs[dst_offset + 0 * 4 + lane] = b0.qs[src_offset + lane];
                    dst.qs[dst_offset + 1 * 4 + lane] = b1.qs[src_offset + lane];
                    dst.qs[dst_offset + 2 * 4 + lane] = b2.qs[src_offset + lane];
                    dst.qs[dst_offset + 3 * 4 + lane] = b3.qs[src_offset + lane];
                }
            }
        }
    }

    return .{
        .allocator = allocator,
        .blocks = packed_blocks,
        .k = k,
        .n = n,
        .blocks_per_group = blocks_per_row,
    };
}

pub fn matmulQ8_0RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ8_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return matmulQ8_0RhsTileAarch64(out, lhs_blocks, rhs, n, r0, r1, c0, c1);
    }

    const blocks_per_row = rhs.rows.blocks_per_row;
    var i = r0;
    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;

        while (j + q8_0_col_block <= c1) : (j += q8_0_col_block) {
            var acc = [_]f32{0} ** q8_0_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_block = &lhs_row[block_index];
                inline for (0..q8_0_col_block) |c| {
                    const rhs_block = &rhs.rows.blocks[(j + c) * blocks_per_row + block_index];
                    acc[c] += dotQ8_0Q8_0(lhs_block, rhs_block);
                }
            }
            inline for (0..q8_0_col_block) |c| out[i * n + j + c] = acc[c];
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: f32 = 0;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc += dotQ8_0Q8_0(&lhs_row[block_index], &rhs_col[block_index]);
            }
            out[i * n + j] = acc;
        }
    }
}

fn matmulQ8_0RhsTileAarch64(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ8_0,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.rows.blocks_per_row;
    var i = r0;

    while (i + q8_0_row_block <= r1) : (i += q8_0_row_block) {
        var j = c0;
        while (j + q8_0_aarch64_col_block <= c1) : (j += q8_0_aarch64_col_block) {
            var acc: [q8_0_aarch64_col_block][q8_0_row_block]QKV4f32 = undefined;
            inline for (0..q8_0_aarch64_col_block) |c| {
                inline for (0..q8_0_row_block) |r| acc[c][r] = @splat(0);
            }

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                var lhs_d: [q8_0_row_block]u16 = undefined;
                var lhs_lo: [q8_0_row_block]QKV16i8 = undefined;
                var lhs_hi: [q8_0_row_block]QKV16i8 = undefined;
                inline for (0..q8_0_row_block) |r| {
                    const lhs_block = &lhs_blocks[(i + r) * blocks_per_row + block_index];
                    lhs_d[r] = lhs_block.d;
                    lhs_lo[r] = @bitCast(lhs_block.qs[0..16].*);
                    lhs_hi[r] = @bitCast(lhs_block.qs[16..32].*);
                }

                inline for (0..q8_0_aarch64_col_block) |c| {
                    const rhs_block = &rhs.rows.blocks[(j + c) * blocks_per_row + block_index];
                    const rhs_lo: QKV16i8 = @bitCast(rhs_block.qs[0..16].*);
                    const rhs_hi: QKV16i8 = @bitCast(rhs_block.qs[16..32].*);
                    inline for (0..q8_0_row_block) |r| {
                        acc[c][r] = accumulateQ8_0Aarch64(acc[c][r], lhs_d[r], lhs_lo[r], lhs_hi[r], rhs_block.d, rhs_lo, rhs_hi);
                    }
                }
            }

            inline for (0..q8_0_aarch64_col_block) |c| {
                inline for (0..q8_0_row_block) |r| {
                    out[(i + r) * n + j + c] = @reduce(.Add, acc[c][r]);
                }
            }
        }

        while (j < c1) : (j += 1) {
            var acc: [q8_0_row_block]QKV4f32 = undefined;
            inline for (0..q8_0_row_block) |r| acc[r] = @splat(0);

            const rhs_col = rhs.columnBlocks(j);
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const rhs_block = &rhs_col[block_index];
                const rhs_lo: QKV16i8 = @bitCast(rhs_block.qs[0..16].*);
                const rhs_hi: QKV16i8 = @bitCast(rhs_block.qs[16..32].*);
                inline for (0..q8_0_row_block) |r| {
                    const lhs_block = &lhs_blocks[(i + r) * blocks_per_row + block_index];
                    const lhs_lo: QKV16i8 = @bitCast(lhs_block.qs[0..16].*);
                    const lhs_hi: QKV16i8 = @bitCast(lhs_block.qs[16..32].*);
                    acc[r] = accumulateQ8_0Aarch64(acc[r], lhs_block.d, lhs_lo, lhs_hi, rhs_block.d, rhs_lo, rhs_hi);
                }
            }

            inline for (0..q8_0_row_block) |r| out[(i + r) * n + j] = @reduce(.Add, acc[r]);
        }
    }

    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;

        while (j + q8_0_aarch64_tail_col_block <= c1) : (j += q8_0_aarch64_tail_col_block) {
            var acc = [_]QKV4f32{@splat(0)} ** q8_0_aarch64_tail_col_block;
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_block = &lhs_row[block_index];
                const lhs_lo: QKV16i8 = @bitCast(lhs_block.qs[0..16].*);
                const lhs_hi: QKV16i8 = @bitCast(lhs_block.qs[16..32].*);
                inline for (0..q8_0_aarch64_tail_col_block) |c| {
                    const rhs_block = &rhs.rows.blocks[(j + c) * blocks_per_row + block_index];
                    acc[c] = accumulateQ8_0Aarch64(
                        acc[c],
                        lhs_block.d,
                        lhs_lo,
                        lhs_hi,
                        rhs_block.d,
                        @bitCast(rhs_block.qs[0..16].*),
                        @bitCast(rhs_block.qs[16..32].*),
                    );
                }
            }
            inline for (0..q8_0_aarch64_tail_col_block) |c| out[i * n + j + c] = @reduce(.Add, acc[c]);
        }

        while (j < c1) : (j += 1) {
            const rhs_col = rhs.columnBlocks(j);
            var acc: QKV4f32 = @splat(0);
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const lhs_block = &lhs_row[block_index];
                const lhs_lo: QKV16i8 = @bitCast(lhs_block.qs[0..16].*);
                const lhs_hi: QKV16i8 = @bitCast(lhs_block.qs[16..32].*);
                const rhs_block = &rhs_col[block_index];
                acc = accumulateQ8_0Aarch64(
                    acc,
                    lhs_block.d,
                    lhs_lo,
                    lhs_hi,
                    rhs_block.d,
                    @bitCast(rhs_block.qs[0..16].*),
                    @bitCast(rhs_block.qs[16..32].*),
                );
            }
            out[i * n + j] = @reduce(.Add, acc);
        }
    }
}

pub fn matmulQ8_0RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ8_0,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ8_0RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ8_0x4RhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(c0 % 4 == 0);
    std.debug.assert(c1 % 4 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    var i = r0;
    while (i + q8_0_row_block <= r1) : (i += q8_0_row_block) {
        var j = c0;
        while (j < c1) : (j += 4) {
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: [q8_0_row_block]QKV4f32 = undefined;
            inline for (0..q8_0_row_block) |r| acc[r] = @splat(0);

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                const rhs_block = &rhs_group[block_index];
                accumulateQ8_0x4Rows(lhs_blocks, i, blocks_per_row, block_index, rhs_block, &acc);
            }

            inline for (0..q8_0_row_block) |r| {
                out[(i + r) * n + j + 0] = acc[r][0];
                out[(i + r) * n + j + 1] = acc[r][1];
                out[(i + r) * n + j + 2] = acc[r][2];
                out[(i + r) * n + j + 3] = acc[r][3];
            }
        }
    }

    while (i < r1) : (i += 1) {
        const lhs_row = lhs_blocks[i * blocks_per_row ..][0..blocks_per_row];
        var j = c0;
        while (j < c1) : (j += 4) {
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: QKV4f32 = @splat(0);
            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                acc = accumulateQ8_0x4(&lhs_row[block_index], &rhs_group[block_index], acc);
            }
            out[i * n + j + 0] = acc[0];
            out[i * n + j + 1] = acc[1];
            out[i * n + j + 2] = acc[2];
            out[i * n + j + 3] = acc[3];
        }
    }
}

pub fn matmulQ8_0x4RhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ8_0x4RhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ8_0x4PackedRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0x4,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(r0 % 4 == 0);
    std.debug.assert(r1 % 4 == 0);
    std.debug.assert(c0 % 4 == 0);
    std.debug.assert(c1 % 4 == 0);

    if (r1 - r0 >= 128) {
        matmulQ8_0x4PackedRhsTileColsFirst(out, lhs_blocks, rhs, n, r0, r1, c0, c1);
        return;
    }

    const blocks_per_row = rhs.blocks_per_group;
    var row_group = r0 / 4;
    while (row_group < r1 / 4) : (row_group += 1) {
        var j = c0;
        while (j < c1) : (j += 4) {
            const lhs_group = lhs_blocks[row_group * blocks_per_row ..][0..blocks_per_row];
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: [4]QKV4f32 = undefined;
            inline for (0..4) |r| acc[r] = @splat(0);

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ8_0x4Packed(&lhs_group[block_index], &rhs_group[block_index], &acc);
            }

            inline for (0..4) |r| {
                const row = row_group * 4 + r;
                out[row * n + j + 0] = acc[r][0];
                out[row * n + j + 1] = acc[r][1];
                out[row * n + j + 2] = acc[r][2];
                out[row * n + j + 3] = acc[r][3];
            }
        }
    }
}

fn matmulQ8_0x4PackedRhsTileColsFirst(
    out: []f32,
    lhs_blocks: []const BlockQ8_0x4,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) void {
    const blocks_per_row = rhs.blocks_per_group;
    const row_group_tile = 8;
    var row_group_start = r0 / 4;
    while (row_group_start < r1 / 4) : (row_group_start += row_group_tile) {
        const row_group_end = @min(row_group_start + row_group_tile, r1 / 4);
        var j = c0;
        while (j < c1) : (j += 4) {
            const rhs_group = rhs.groupBlocks(j / 4);
            var row_group = row_group_start;
            // Process two row-groups (8 rows) per pass for sdot-chain ILP.
            while (row_group + 2 <= row_group_end) : (row_group += 2) {
                const lhs_a = lhs_blocks[row_group * blocks_per_row ..][0..blocks_per_row];
                const lhs_b = lhs_blocks[(row_group + 1) * blocks_per_row ..][0..blocks_per_row];
                var acc_a: [4]QKV4f32 = undefined;
                var acc_b: [4]QKV4f32 = undefined;
                inline for (0..4) |r| {
                    acc_a[r] = @splat(0);
                    acc_b[r] = @splat(0);
                }

                var block_index: usize = 0;
                while (block_index < blocks_per_row) : (block_index += 1) {
                    accumulateQ8_0x4PackedDual(&lhs_a[block_index], &lhs_b[block_index], &rhs_group[block_index], &acc_a, &acc_b);
                }

                inline for (0..4) |r| {
                    const row_a = row_group * 4 + r;
                    out[row_a * n + j + 0] = acc_a[r][0];
                    out[row_a * n + j + 1] = acc_a[r][1];
                    out[row_a * n + j + 2] = acc_a[r][2];
                    out[row_a * n + j + 3] = acc_a[r][3];
                    const row_b = (row_group + 1) * 4 + r;
                    out[row_b * n + j + 0] = acc_b[r][0];
                    out[row_b * n + j + 1] = acc_b[r][1];
                    out[row_b * n + j + 2] = acc_b[r][2];
                    out[row_b * n + j + 3] = acc_b[r][3];
                }
            }
            // Odd trailing row-group.
            if (row_group < row_group_end) {
                const lhs_group = lhs_blocks[row_group * blocks_per_row ..][0..blocks_per_row];
                var acc: [4]QKV4f32 = undefined;
                inline for (0..4) |r| acc[r] = @splat(0);

                var block_index: usize = 0;
                while (block_index < blocks_per_row) : (block_index += 1) {
                    accumulateQ8_0x4Packed(&lhs_group[block_index], &rhs_group[block_index], &acc);
                }

                inline for (0..4) |r| {
                    const row = row_group * 4 + r;
                    out[row * n + j + 0] = acc[r][0];
                    out[row * n + j + 1] = acc[r][1];
                    out[row * n + j + 2] = acc[r][2];
                    out[row * n + j + 3] = acc[r][3];
                }
            }
        }
    }
}

pub fn matmulQ8_0x4PackedRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0x4,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
) void {
    _ = m;
    matmulQ8_0x4PackedRhsTile(out, lhs_blocks, rhs, n, row_start, row_end, 0, n);
}

pub fn matmulQ8_0x4PackedPaddedRhsTile(
    out: []f32,
    lhs_blocks: []const BlockQ8_0x4,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
    c0: usize,
    c1: usize,
) void {
    std.debug.assert(c0 % 4 == 0);
    std.debug.assert(c1 % 4 == 0);

    const blocks_per_row = rhs.blocks_per_group;
    const row_groups = (m + 3) / 4;
    var row_group: usize = 0;
    while (row_group < row_groups) : (row_group += 1) {
        var j = c0;
        while (j < c1) : (j += 4) {
            const lhs_group = lhs_blocks[row_group * blocks_per_row ..][0..blocks_per_row];
            const rhs_group = rhs.groupBlocks(j / 4);
            var acc: [4]QKV4f32 = undefined;
            inline for (0..4) |r| acc[r] = @splat(0);

            var block_index: usize = 0;
            while (block_index < blocks_per_row) : (block_index += 1) {
                accumulateQ8_0x4Packed(&lhs_group[block_index], &rhs_group[block_index], &acc);
            }

            inline for (0..4) |r| {
                const row = row_group * 4 + r;
                if (row < m) {
                    out[row * n + j + 0] = acc[r][0];
                    out[row * n + j + 1] = acc[r][1];
                    out[row * n + j + 2] = acc[r][2];
                    out[row * n + j + 3] = acc[r][3];
                }
            }
        }
    }
}

pub fn matmulQ8_0x4PackedPaddedRhsRange(
    out: []f32,
    lhs_blocks: []const BlockQ8_0x4,
    rhs: *const QuantizedMatmulRhsQ8_0x4,
    m: usize,
    n: usize,
) void {
    matmulQ8_0x4PackedPaddedRhsTile(out, lhs_blocks, rhs, m, n, 0, n);
}

fn dotQ8_0Q8_0(a: *const BlockQ8_0, b: *const BlockQ8_0) f32 {
    const d = f16BitsToF32(a.d) * f16BitsToF32(b.d);
    if (comptime has_x86_avx2) {
        // AVX2/VNNI int8 dot over the two 16-byte halves. Operand order is
        // load-bearing on the AVX2 sign-trick path: the SIGN SOURCE (first
        // arg) is b — the RHS/GGUF weight side, which the format permits to
        // hold -128 — while a is the in-engine quantizeToI8 output, clamped to
        // [-127,127]: exactly the sign-trick exactness domain (second operand
        // must not be -128 in lanes where the first is negative). Bit-equal to
        // the i8DotI32 fallback; under VNNI it lowers to vpdpbusd instead.
        const dot = dotI8x16Portable(@bitCast(b.qs[0..16].*), @bitCast(a.qs[0..16].*)) +
            dotI8x16Portable(@bitCast(b.qs[16..32].*), @bitCast(a.qs[16..32].*));
        return @as(f32, @floatFromInt(dot)) * d;
    }
    return @as(f32, @floatFromInt(i8DotI32(&a.qs, &b.qs))) * d;
}

fn accumulateQ8_0Aarch64(
    acc: QKV4f32,
    lhs_d: u16,
    lhs_lo: QKV16i8,
    lhs_hi: QKV16i8,
    rhs_d: u16,
    rhs_lo: QKV16i8,
    rhs_hi: QKV16i8,
) QKV4f32 {
    var dot: QKV4i32 = @splat(0);
    dot = sdotI8x16(dot, lhs_lo, rhs_lo);
    dot = sdotI8x16(dot, lhs_hi, rhs_hi);

    const dot_f: QKV4f32 = @floatFromInt(dot);
    const scale: QKV4f32 = @splat(f16BitsToF32(lhs_d) * f16BitsToF32(rhs_d));
    return acc + dot_f * scale;
}

fn accumulateQ8_0x4(lhs: *const BlockQ8_0, rhs: *const BlockQ8_0x4, acc: QKV4f32) QKV4f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ8_0x4Aarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ8_0x4Vnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ8_0x4Avx2(lhs, rhs, acc);
    }
    return accumulateQ8_0x4Widen(lhs, rhs, acc);
}

fn accumulateQ8_0x4Rows(
    lhs_blocks: []const BlockQ8_0,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ8_0x4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ8_0x4RowsAarch64(lhs_blocks, row_start, blocks_per_row, block_index, rhs, acc);
    }
    inline for (0..q8_0_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        acc[r] = accumulateQ8_0x4(lhs, rhs, acc[r]);
    }
}

fn accumulateQ8_0x4Aarch64(lhs: *const BlockQ8_0, rhs: *const BlockQ8_0x4, acc: QKV4f32) QKV4f32 {
    const lhs_lo: QKV16i8 = @bitCast(lhs.qs[0..16].*);
    const lhs_hi: QKV16i8 = @bitCast(lhs.qs[16..32].*);

    var dot: QKV4i32 = @splat(0);
    dot = sdotI8x16Lane(0, dot, @bitCast(rhs.qs[0..][0..16].*), lhs_lo);
    dot = sdotI8x16Lane(1, dot, @bitCast(rhs.qs[16..][0..16].*), lhs_lo);
    dot = sdotI8x16Lane(2, dot, @bitCast(rhs.qs[32..][0..16].*), lhs_lo);
    dot = sdotI8x16Lane(3, dot, @bitCast(rhs.qs[48..][0..16].*), lhs_lo);
    dot = sdotI8x16Lane(0, dot, @bitCast(rhs.qs[64..][0..16].*), lhs_hi);
    dot = sdotI8x16Lane(1, dot, @bitCast(rhs.qs[80..][0..16].*), lhs_hi);
    dot = sdotI8x16Lane(2, dot, @bitCast(rhs.qs[96..][0..16].*), lhs_hi);
    dot = sdotI8x16Lane(3, dot, @bitCast(rhs.qs[112..][0..16].*), lhs_hi);

    const rhs_scale = f16x4BitsToF32(rhs.d);
    const scale = rhs_scale * @as(QKV4f32, @splat(f16BitsToF32(lhs.d)));
    return acc + @as(QKV4f32, @floatFromInt(dot)) * scale;
}

fn accumulateQ8_0x4RowsAarch64(
    lhs_blocks: []const BlockQ8_0,
    row_start: usize,
    blocks_per_row: usize,
    block_index: usize,
    rhs: *const BlockQ8_0x4,
    acc: *[q8_0_row_block]QKV4f32,
) void {
    const rhs0: QKV16i8 = @bitCast(rhs.qs[0..][0..16].*);
    const rhs1: QKV16i8 = @bitCast(rhs.qs[16..][0..16].*);
    const rhs2: QKV16i8 = @bitCast(rhs.qs[32..][0..16].*);
    const rhs3: QKV16i8 = @bitCast(rhs.qs[48..][0..16].*);
    const rhs4: QKV16i8 = @bitCast(rhs.qs[64..][0..16].*);
    const rhs5: QKV16i8 = @bitCast(rhs.qs[80..][0..16].*);
    const rhs6: QKV16i8 = @bitCast(rhs.qs[96..][0..16].*);
    const rhs7: QKV16i8 = @bitCast(rhs.qs[112..][0..16].*);
    const rhs_scale = f16x4BitsToF32(rhs.d);

    inline for (0..q8_0_row_block) |r| {
        const lhs = &lhs_blocks[(row_start + r) * blocks_per_row + block_index];
        const lhs_lo: QKV16i8 = @bitCast(lhs.qs[0..16].*);
        const lhs_hi: QKV16i8 = @bitCast(lhs.qs[16..32].*);

        var dot: QKV4i32 = @splat(0);
        dot = sdotI8x16Lane(0, dot, rhs0, lhs_lo);
        dot = sdotI8x16Lane(1, dot, rhs1, lhs_lo);
        dot = sdotI8x16Lane(2, dot, rhs2, lhs_lo);
        dot = sdotI8x16Lane(3, dot, rhs3, lhs_lo);
        dot = sdotI8x16Lane(0, dot, rhs4, lhs_hi);
        dot = sdotI8x16Lane(1, dot, rhs5, lhs_hi);
        dot = sdotI8x16Lane(2, dot, rhs6, lhs_hi);
        dot = sdotI8x16Lane(3, dot, rhs7, lhs_hi);

        const scale = rhs_scale * @as(QKV4f32, @splat(f16BitsToF32(lhs.d)));
        acc[r] += @as(QKV4f32, @floatFromInt(dot)) * scale;
    }
}

fn accumulateQ8_0x4Packed(lhs: *const BlockQ8_0x4, rhs: *const BlockQ8_0x4, acc: *[4]QKV4f32) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ8_0x4PackedAarch64(lhs, rhs, acc);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ8_0x4PackedVnni(lhs, rhs, acc);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ8_0x4PackedAvx2(lhs, rhs, acc);
    }
    return accumulateQ8_0x4PackedWiden(lhs, rhs, acc);
}

fn accumulateQ8_0x4PackedAarch64(lhs: *const BlockQ8_0x4, rhs: *const BlockQ8_0x4, acc: *[4]QKV4f32) void {
    var sum0: QKV4i32 = @splat(0);
    var sum1: QKV4i32 = @splat(0);
    var sum2: QKV4i32 = @splat(0);
    var sum3: QKV4i32 = @splat(0);

    inline for (0..2) |group_pair| {
        const base = group_pair * 64;
        const lhs0: QKV16i8 = @bitCast(lhs.qs[base + 0 * 16 ..][0..16].*);
        const lhs1: QKV16i8 = @bitCast(lhs.qs[base + 1 * 16 ..][0..16].*);
        const lhs2: QKV16i8 = @bitCast(lhs.qs[base + 2 * 16 ..][0..16].*);
        const lhs3: QKV16i8 = @bitCast(lhs.qs[base + 3 * 16 ..][0..16].*);
        const rhs0: QKV16i8 = @bitCast(rhs.qs[base + 0 * 16 ..][0..16].*);
        const rhs1: QKV16i8 = @bitCast(rhs.qs[base + 1 * 16 ..][0..16].*);
        const rhs2: QKV16i8 = @bitCast(rhs.qs[base + 2 * 16 ..][0..16].*);
        const rhs3: QKV16i8 = @bitCast(rhs.qs[base + 3 * 16 ..][0..16].*);

        sum0 = sdotI8x16Lane(0, sum0, rhs0, lhs0);
        sum1 = sdotI8x16Lane(1, sum1, rhs0, lhs0);
        sum2 = sdotI8x16Lane(2, sum2, rhs0, lhs0);
        sum3 = sdotI8x16Lane(3, sum3, rhs0, lhs0);
        sum0 = sdotI8x16Lane(0, sum0, rhs1, lhs1);
        sum1 = sdotI8x16Lane(1, sum1, rhs1, lhs1);
        sum2 = sdotI8x16Lane(2, sum2, rhs1, lhs1);
        sum3 = sdotI8x16Lane(3, sum3, rhs1, lhs1);
        sum0 = sdotI8x16Lane(0, sum0, rhs2, lhs2);
        sum1 = sdotI8x16Lane(1, sum1, rhs2, lhs2);
        sum2 = sdotI8x16Lane(2, sum2, rhs2, lhs2);
        sum3 = sdotI8x16Lane(3, sum3, rhs2, lhs2);
        sum0 = sdotI8x16Lane(0, sum0, rhs3, lhs3);
        sum1 = sdotI8x16Lane(1, sum1, rhs3, lhs3);
        sum2 = sdotI8x16Lane(2, sum2, rhs3, lhs3);
        sum3 = sdotI8x16Lane(3, sum3, rhs3, lhs3);
    }

    const rhs_scale = f16x4BitsToF32(rhs.d);
    const lhs_scale = f16x4BitsToF32(lhs.d);
    acc[0] += @as(QKV4f32, @floatFromInt(sum0)) * rhs_scale * @as(QKV4f32, @splat(lhs_scale[0]));
    acc[1] += @as(QKV4f32, @floatFromInt(sum1)) * rhs_scale * @as(QKV4f32, @splat(lhs_scale[1]));
    acc[2] += @as(QKV4f32, @floatFromInt(sum2)) * rhs_scale * @as(QKV4f32, @splat(lhs_scale[2]));
    acc[3] += @as(QKV4f32, @floatFromInt(sum3)) * rhs_scale * @as(QKV4f32, @splat(lhs_scale[3]));
}

// Two row-groups (8 output rows) against one shared RHS column-group, computed
// in one pass. Reuses each RHS sub-vector across both row-groups (one load, two
// uses) and keeps eight independent sdot chains in flight, which hides the
// ~3-cycle sdot latency far better than the 4-accumulator single-row-group
// kernel. The arithmetic per accumulator is identical to
// `accumulateQ8_0x4Packed`, so results are bit-for-bit the same.
fn accumulateQ8_0x4PackedDual(
    lhs_a: *const BlockQ8_0x4,
    lhs_b: *const BlockQ8_0x4,
    rhs: *const BlockQ8_0x4,
    acc_a: *[4]QKV4f32,
    acc_b: *[4]QKV4f32,
) void {
    if (comptime builtin.cpu.arch == .aarch64) {
        return accumulateQ8_0x4PackedDualAarch64(lhs_a, lhs_b, rhs, acc_a, acc_b);
    }
    if (comptime has_x86_vnni_ymm) {
        return accumulateQ8_0x4PackedDualVnni(lhs_a, lhs_b, rhs, acc_a, acc_b);
    }
    if (comptime has_x86_avx2) {
        return accumulateQ8_0x4PackedDualAvx2(lhs_a, lhs_b, rhs, acc_a, acc_b);
    }
    accumulateQ8_0x4PackedWiden(lhs_a, rhs, acc_a);
    accumulateQ8_0x4PackedWiden(lhs_b, rhs, acc_b);
}

fn accumulateQ8_0x4PackedDualAarch64(
    lhs_a: *const BlockQ8_0x4,
    lhs_b: *const BlockQ8_0x4,
    rhs: *const BlockQ8_0x4,
    acc_a: *[4]QKV4f32,
    acc_b: *[4]QKV4f32,
) void {
    var a0: QKV4i32 = @splat(0);
    var a1: QKV4i32 = @splat(0);
    var a2: QKV4i32 = @splat(0);
    var a3: QKV4i32 = @splat(0);
    var b0: QKV4i32 = @splat(0);
    var b1: QKV4i32 = @splat(0);
    var b2: QKV4i32 = @splat(0);
    var b3: QKV4i32 = @splat(0);

    inline for (0..2) |group_pair| {
        const base = group_pair * 64;
        const r0: QKV16i8 = @bitCast(rhs.qs[base + 0 * 16 ..][0..16].*);
        const r1: QKV16i8 = @bitCast(rhs.qs[base + 1 * 16 ..][0..16].*);
        const r2: QKV16i8 = @bitCast(rhs.qs[base + 2 * 16 ..][0..16].*);
        const r3: QKV16i8 = @bitCast(rhs.qs[base + 3 * 16 ..][0..16].*);

        const la0: QKV16i8 = @bitCast(lhs_a.qs[base + 0 * 16 ..][0..16].*);
        const la1: QKV16i8 = @bitCast(lhs_a.qs[base + 1 * 16 ..][0..16].*);
        const la2: QKV16i8 = @bitCast(lhs_a.qs[base + 2 * 16 ..][0..16].*);
        const la3: QKV16i8 = @bitCast(lhs_a.qs[base + 3 * 16 ..][0..16].*);

        const lb0: QKV16i8 = @bitCast(lhs_b.qs[base + 0 * 16 ..][0..16].*);
        const lb1: QKV16i8 = @bitCast(lhs_b.qs[base + 1 * 16 ..][0..16].*);
        const lb2: QKV16i8 = @bitCast(lhs_b.qs[base + 2 * 16 ..][0..16].*);
        const lb3: QKV16i8 = @bitCast(lhs_b.qs[base + 3 * 16 ..][0..16].*);

        a0 = sdotI8x16Lane(0, a0, r0, la0);
        a1 = sdotI8x16Lane(1, a1, r0, la0);
        a2 = sdotI8x16Lane(2, a2, r0, la0);
        a3 = sdotI8x16Lane(3, a3, r0, la0);
        b0 = sdotI8x16Lane(0, b0, r0, lb0);
        b1 = sdotI8x16Lane(1, b1, r0, lb0);
        b2 = sdotI8x16Lane(2, b2, r0, lb0);
        b3 = sdotI8x16Lane(3, b3, r0, lb0);

        a0 = sdotI8x16Lane(0, a0, r1, la1);
        a1 = sdotI8x16Lane(1, a1, r1, la1);
        a2 = sdotI8x16Lane(2, a2, r1, la1);
        a3 = sdotI8x16Lane(3, a3, r1, la1);
        b0 = sdotI8x16Lane(0, b0, r1, lb1);
        b1 = sdotI8x16Lane(1, b1, r1, lb1);
        b2 = sdotI8x16Lane(2, b2, r1, lb1);
        b3 = sdotI8x16Lane(3, b3, r1, lb1);

        a0 = sdotI8x16Lane(0, a0, r2, la2);
        a1 = sdotI8x16Lane(1, a1, r2, la2);
        a2 = sdotI8x16Lane(2, a2, r2, la2);
        a3 = sdotI8x16Lane(3, a3, r2, la2);
        b0 = sdotI8x16Lane(0, b0, r2, lb2);
        b1 = sdotI8x16Lane(1, b1, r2, lb2);
        b2 = sdotI8x16Lane(2, b2, r2, lb2);
        b3 = sdotI8x16Lane(3, b3, r2, lb2);

        a0 = sdotI8x16Lane(0, a0, r3, la3);
        a1 = sdotI8x16Lane(1, a1, r3, la3);
        a2 = sdotI8x16Lane(2, a2, r3, la3);
        a3 = sdotI8x16Lane(3, a3, r3, la3);
        b0 = sdotI8x16Lane(0, b0, r3, lb3);
        b1 = sdotI8x16Lane(1, b1, r3, lb3);
        b2 = sdotI8x16Lane(2, b2, r3, lb3);
        b3 = sdotI8x16Lane(3, b3, r3, lb3);
    }

    const rhs_scale = f16x4BitsToF32(rhs.d);
    const la_scale = f16x4BitsToF32(lhs_a.d);
    const lb_scale = f16x4BitsToF32(lhs_b.d);
    acc_a[0] += @as(QKV4f32, @floatFromInt(a0)) * rhs_scale * @as(QKV4f32, @splat(la_scale[0]));
    acc_a[1] += @as(QKV4f32, @floatFromInt(a1)) * rhs_scale * @as(QKV4f32, @splat(la_scale[1]));
    acc_a[2] += @as(QKV4f32, @floatFromInt(a2)) * rhs_scale * @as(QKV4f32, @splat(la_scale[2]));
    acc_a[3] += @as(QKV4f32, @floatFromInt(a3)) * rhs_scale * @as(QKV4f32, @splat(la_scale[3]));
    acc_b[0] += @as(QKV4f32, @floatFromInt(b0)) * rhs_scale * @as(QKV4f32, @splat(lb_scale[0]));
    acc_b[1] += @as(QKV4f32, @floatFromInt(b1)) * rhs_scale * @as(QKV4f32, @splat(lb_scale[1]));
    acc_b[2] += @as(QKV4f32, @floatFromInt(b2)) * rhs_scale * @as(QKV4f32, @splat(lb_scale[2]));
    acc_b[3] += @as(QKV4f32, @floatFromInt(b3)) * rhs_scale * @as(QKV4f32, @splat(lb_scale[3]));
}

// --- x86 / portable-SIMD arms of the x4 accumulates --------------------------
//
// Every arm below computes the SAME i32 group sums as the scalar arms
// (bit-identical integer accumulation — bounds proven per arm) and applies the
// f16 scales with the scalar arms' exact f32 association
// ((float(sum) · lhs_d) · rhs_d), so results are bit-for-bit equal to the
// scalar reference — asserted by q8_0_tests.zig. The arms are written over the
// comptime-gated primitives in common.zig (vpdpbusd / maddubs+psign / portable
// widening), so each also compiles and runs on any target via the portable
// twins (that is how the aarch64 dev machine exercises them); the real
// instruction attestations live in src/x86dot_check.zig's coverage table.
// They are pub for the sibling exact-parity tests.

// vpshufd-class: broadcast dword `r` within each 128-bit lane — aligns row r's
// 4-byte feature group of both 16-byte chunks of a 32-byte packed-LHS load
// against the column groups of the matching RHS chunks.
inline fn broadcastLaneGroupI32x8(comptime r: comptime_int, v: QKV8i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ r, r, r, r, 4 + r, 4 + r, 4 + r, 4 + r });
}

// vpermd-class (cross-lane): broadcast dword 2g to the low 128-bit lane and
// dword 2g+1 to the high lane — aligns plain-LHS feature groups 2g/2g+1
// against RHS chunks 2g/2g+1 of one 32-byte load.
inline fn broadcastPairGroupsI32x8(comptime g: comptime_int, v: QKV8i32) QKV8i32 {
    return @shuffle(i32, v, undefined, [8]i32{ 2 * g, 2 * g, 2 * g, 2 * g, 2 * g + 1, 2 * g + 1, 2 * g + 1, 2 * g + 1 });
}

// Shared epilogue: fold the ymm accumulators, subtract the (possibly zero)
// bias correction, and apply the scales in the scalar arm's association —
// acc[row] += (float(sums[row]) · lhs_d[row]) · rhs_d — element-for-element
// the scalar arm's f32 expression, so equal integer sums give bit-equal f32.
inline fn applyPackedScales(sums: *const [4]QKV8i32, correction: QKV4i32, lhs_d: [4]u16, rhs_d: [4]u16, acc: *[4]QKV4f32) void {
    const rhs_scale = f16x4BitsToF32(rhs_d);
    inline for (0..4) |r| {
        const t = addHalvesI32x8(sums[r]) - correction;
        acc[r] += @as(QKV4f32, @floatFromInt(t)) * @as(QKV4f32, @splat(f16BitsToF32(lhs_d[r]))) * rhs_scale;
    }
}

/// x86 VNNI arm: vpdpbusd grouped u8·i8 dots over the +128-biased LHS.
/// Per 4-byte group, Σ lhs·rhs = Σ(lhs+128)·rhs − 128·Σrhs: the bias makes the
/// broadcast operand unsigned (vpdpbusd's u8 side) with NO input-domain
/// restriction — exact for all i8 incl. −128 on BOTH sides — and the −128·Σrhs
/// correction costs one dpbusd(1s, rhs) per 32-byte chunk, shared across all
/// four rows (and both row-groups of the Dual arm). No overflow: per lane
/// |Σ(lhs+128)·rhs| ≤ 32·255·128 < 2^21 and |128·Σrhs| ≤ 128·32·128 = 2^19.
pub fn accumulateQ8_0x4PackedVnni(lhs: *const BlockQ8_0x4, rhs: *const BlockQ8_0x4, acc: *[4]QKV4f32) void {
    var sums: [4]QKV8i32 = undefined;
    inline for (0..4) |r| sums[r] = @splat(0);
    var gsum: QKV8i32 = @splat(0);
    const ones: QKV32u8 = @splat(1);
    const bias: QKV32i8 = @splat(-128);

    inline for (0..4) |pair| {
        const base = pair * 32;
        const lhs32: QKV32i8 = @bitCast(lhs.qs[base..][0..32].*);
        const rhs32: QKV32i8 = @bitCast(rhs.qs[base..][0..32].*);
        const lhs_biased: QKV8i32 = @bitCast(lhs32 ^ bias); // lhs+128 as u8, dword view
        gsum = dpbusdI32x8(gsum, ones, rhs32);
        inline for (0..4) |r| {
            const bcast: QKV32u8 = @bitCast(broadcastLaneGroupI32x8(r, lhs_biased));
            sums[r] = dpbusdI32x8(sums[r], bcast, rhs32);
        }
    }

    const correction = addHalvesI32x8(gsum) * @as(QKV4i32, @splat(128));
    applyPackedScales(&sums, correction, lhs.d, rhs.d, acc);
}

/// Dual-row-group VNNI arm: same algebra as accumulateQ8_0x4PackedVnni per
/// accumulator (bit-for-bit identical results), with each RHS chunk loaded
/// once and the shared −128·Σrhs correction computed once for both row-groups;
/// eight independent dpbusd chains keep the VNNI ports fed.
pub fn accumulateQ8_0x4PackedDualVnni(
    lhs_a: *const BlockQ8_0x4,
    lhs_b: *const BlockQ8_0x4,
    rhs: *const BlockQ8_0x4,
    acc_a: *[4]QKV4f32,
    acc_b: *[4]QKV4f32,
) void {
    var sums_a: [4]QKV8i32 = undefined;
    var sums_b: [4]QKV8i32 = undefined;
    inline for (0..4) |r| {
        sums_a[r] = @splat(0);
        sums_b[r] = @splat(0);
    }
    var gsum: QKV8i32 = @splat(0);
    const ones: QKV32u8 = @splat(1);
    const bias: QKV32i8 = @splat(-128);

    inline for (0..4) |pair| {
        const base = pair * 32;
        const rhs32: QKV32i8 = @bitCast(rhs.qs[base..][0..32].*);
        const la_biased: QKV8i32 = @bitCast(@as(QKV32i8, @bitCast(lhs_a.qs[base..][0..32].*)) ^ bias);
        const lb_biased: QKV8i32 = @bitCast(@as(QKV32i8, @bitCast(lhs_b.qs[base..][0..32].*)) ^ bias);
        gsum = dpbusdI32x8(gsum, ones, rhs32);
        inline for (0..4) |r| {
            sums_a[r] = dpbusdI32x8(sums_a[r], @bitCast(broadcastLaneGroupI32x8(r, la_biased)), rhs32);
            sums_b[r] = dpbusdI32x8(sums_b[r], @bitCast(broadcastLaneGroupI32x8(r, lb_biased)), rhs32);
        }
    }

    const correction = addHalvesI32x8(gsum) * @as(QKV4i32, @splat(128));
    applyPackedScales(&sums_a, correction, lhs_a.d, rhs.d, acc_a);
    applyPackedScales(&sums_b, correction, lhs_b.d, rhs.d, acc_b);
}

/// x86 AVX2 (no-VNNI) arm: the ggml sign-transfer trick arranged with the RHS
/// (the GGUF weight side, which the format permits to hold −128) as the SIGN
/// SOURCE and the broadcast LHS as the value operand:
///     lhs·rhs == |rhs| · (sign(rhs)·lhs),   |rhs| on vpmaddubsw's u8 side.
/// EXACTNESS DOMAIN: lhs bytes must be in [−127,127] — guaranteed at every
/// call site because the packed LHS is always in-engine activation
/// quantization (quantizeRowsQ8_0x4*Into / quantizeSplitSwiGluRows* clamp to
/// ±127); rhs is unrestricted (|−128| = 128 is a valid u8).
/// SATURATION PROOF: per lane the product magnitude is ≤ 128·127 = 16256
/// (when rhs = −128, the lhs value side is ≤ 127 in magnitude by the domain),
/// so vpmaddubsw pair sums stay within ±32512 < 32767 — never saturates, the
/// i32 sums are exact.
pub fn accumulateQ8_0x4PackedAvx2(lhs: *const BlockQ8_0x4, rhs: *const BlockQ8_0x4, acc: *[4]QKV4f32) void {
    var sums: [4]QKV8i32 = undefined;
    inline for (0..4) |r| sums[r] = @splat(0);

    inline for (0..4) |pair| {
        const base = pair * 32;
        const lhs32: QKV32i8 = @bitCast(lhs.qs[base..][0..32].*);
        const rhs32: QKV32i8 = @bitCast(rhs.qs[base..][0..32].*);
        const abs_r: QKV32u8 = @bitCast(psignI8x32(rhs32, rhs32)); // |rhs| as u8 (128 ok)
        const lhs_groups: QKV8i32 = @bitCast(lhs32);
        inline for (0..4) |r| {
            const bcast: QKV32i8 = @bitCast(broadcastLaneGroupI32x8(r, lhs_groups));
            sums[r] = maddubsDotGroupsI32x8(sums[r], abs_r, psignI8x32(bcast, rhs32));
        }
    }

    applyPackedScales(&sums, @splat(0), lhs.d, rhs.d, acc);
}

/// Dual-row-group AVX2 arm: same algebra (and domain) as
/// accumulateQ8_0x4PackedAvx2 per accumulator; each RHS chunk and its |rhs|
/// are computed once and shared by both row-groups.
pub fn accumulateQ8_0x4PackedDualAvx2(
    lhs_a: *const BlockQ8_0x4,
    lhs_b: *const BlockQ8_0x4,
    rhs: *const BlockQ8_0x4,
    acc_a: *[4]QKV4f32,
    acc_b: *[4]QKV4f32,
) void {
    var sums_a: [4]QKV8i32 = undefined;
    var sums_b: [4]QKV8i32 = undefined;
    inline for (0..4) |r| {
        sums_a[r] = @splat(0);
        sums_b[r] = @splat(0);
    }

    inline for (0..4) |pair| {
        const base = pair * 32;
        const rhs32: QKV32i8 = @bitCast(rhs.qs[base..][0..32].*);
        const abs_r: QKV32u8 = @bitCast(psignI8x32(rhs32, rhs32));
        const la_groups: QKV8i32 = @bitCast(lhs_a.qs[base..][0..32].*);
        const lb_groups: QKV8i32 = @bitCast(lhs_b.qs[base..][0..32].*);
        inline for (0..4) |r| {
            const bcast_a: QKV32i8 = @bitCast(broadcastLaneGroupI32x8(r, la_groups));
            const bcast_b: QKV32i8 = @bitCast(broadcastLaneGroupI32x8(r, lb_groups));
            sums_a[r] = maddubsDotGroupsI32x8(sums_a[r], abs_r, psignI8x32(bcast_a, rhs32));
            sums_b[r] = maddubsDotGroupsI32x8(sums_b[r], abs_r, psignI8x32(bcast_b, rhs32));
        }
    }

    applyPackedScales(&sums_a, @splat(0), lhs_a.d, rhs.d, acc_a);
    applyPackedScales(&sums_b, @splat(0), lhs_b.d, rhs.d, acc_b);
}

/// Universal portable-SIMD arm (no arch gate, no input-domain restriction):
/// exact widening grouped dot — i8·i8 products in i16, group sums in i32
/// (dotI8GroupsWidenI32x8). The production floor for every ISA that is
/// neither aarch64 (sdot) nor gated x86; the scalar arm survives only as the
/// bit-exactness reference in tests.
pub fn accumulateQ8_0x4PackedWiden(lhs: *const BlockQ8_0x4, rhs: *const BlockQ8_0x4, acc: *[4]QKV4f32) void {
    var sums: [4]QKV8i32 = undefined;
    inline for (0..4) |r| sums[r] = @splat(0);

    inline for (0..4) |pair| {
        const base = pair * 32;
        const lhs32: QKV32i8 = @bitCast(lhs.qs[base..][0..32].*);
        const rhs32: QKV32i8 = @bitCast(rhs.qs[base..][0..32].*);
        const lhs_groups: QKV8i32 = @bitCast(lhs32);
        inline for (0..4) |r| {
            const bcast: QKV32i8 = @bitCast(broadcastLaneGroupI32x8(r, lhs_groups));
            sums[r] = dotI8GroupsWidenI32x8(sums[r], bcast, rhs32);
        }
    }

    applyPackedScales(&sums, @splat(0), lhs.d, rhs.d, acc);
}

/// VNNI arm of accumulateQ8_0x4 (plain BlockQ8_0 LHS row against one packed
/// column group): same +128-bias/correction algebra and bounds as the Packed
/// arm; the LHS feature-group pairs are broadcast cross-lane instead.
pub fn accumulateQ8_0x4Vnni(lhs: *const BlockQ8_0, rhs: *const BlockQ8_0x4, acc: QKV4f32) QKV4f32 {
    var sum: QKV8i32 = @splat(0);
    var gsum: QKV8i32 = @splat(0);
    const ones: QKV32u8 = @splat(1);
    const bias: QKV32i8 = @splat(-128);
    const lhs_biased: QKV8i32 = @bitCast(@as(QKV32i8, @bitCast(lhs.qs)) ^ bias);

    inline for (0..4) |g| {
        const rhs32: QKV32i8 = @bitCast(rhs.qs[g * 32 ..][0..32].*);
        gsum = dpbusdI32x8(gsum, ones, rhs32);
        const bcast: QKV32u8 = @bitCast(broadcastPairGroupsI32x8(g, lhs_biased));
        sum = dpbusdI32x8(sum, bcast, rhs32);
    }

    const t = addHalvesI32x8(sum) - addHalvesI32x8(gsum) * @as(QKV4i32, @splat(128));
    return acc + @as(QKV4f32, @floatFromInt(t)) * @as(QKV4f32, @splat(f16BitsToF32(lhs.d))) * f16x4BitsToF32(rhs.d);
}

/// AVX2 arm of accumulateQ8_0x4: same sign-transfer arrangement, domain, and
/// saturation proof as accumulateQ8_0x4PackedAvx2 (lhs = clamped activations
/// in [−127,127], rhs = weights unrestricted).
pub fn accumulateQ8_0x4Avx2(lhs: *const BlockQ8_0, rhs: *const BlockQ8_0x4, acc: QKV4f32) QKV4f32 {
    var sum: QKV8i32 = @splat(0);
    const lhs_groups: QKV8i32 = @bitCast(lhs.qs);

    inline for (0..4) |g| {
        const rhs32: QKV32i8 = @bitCast(rhs.qs[g * 32 ..][0..32].*);
        const abs_r: QKV32u8 = @bitCast(psignI8x32(rhs32, rhs32));
        const bcast: QKV32i8 = @bitCast(broadcastPairGroupsI32x8(g, lhs_groups));
        sum = maddubsDotGroupsI32x8(sum, abs_r, psignI8x32(bcast, rhs32));
    }

    const t = addHalvesI32x8(sum);
    return acc + @as(QKV4f32, @floatFromInt(t)) * @as(QKV4f32, @splat(f16BitsToF32(lhs.d))) * f16x4BitsToF32(rhs.d);
}

/// Universal portable-SIMD arm of accumulateQ8_0x4 (no arch gate, no domain
/// restriction) — the widening tier of the plain-LHS accumulate.
pub fn accumulateQ8_0x4Widen(lhs: *const BlockQ8_0, rhs: *const BlockQ8_0x4, acc: QKV4f32) QKV4f32 {
    var sum: QKV8i32 = @splat(0);
    const lhs_groups: QKV8i32 = @bitCast(lhs.qs);

    inline for (0..4) |g| {
        const rhs32: QKV32i8 = @bitCast(rhs.qs[g * 32 ..][0..32].*);
        const bcast: QKV32i8 = @bitCast(broadcastPairGroupsI32x8(g, lhs_groups));
        sum = dotI8GroupsWidenI32x8(sum, bcast, rhs32);
    }

    const t = addHalvesI32x8(sum);
    return acc + @as(QKV4f32, @floatFromInt(t)) * @as(QKV4f32, @splat(f16BitsToF32(lhs.d))) * f16x4BitsToF32(rhs.d);
}

// pub: the bit-exactness reference for the SIMD arms above (q8_0_tests.zig).
pub fn accumulateQ8_0x4PackedScalar(lhs: *const BlockQ8_0x4, rhs: *const BlockQ8_0x4, acc: *[4]QKV4f32) void {
    var sums: [4][4]i32 = .{
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
        .{ 0, 0, 0, 0 },
    };

    for (0..8) |feature_group| {
        const offset = feature_group * 16;
        inline for (0..4) |row| {
            inline for (0..4) |col| {
                inline for (0..4) |lane| {
                    sums[row][col] += @as(i32, lhs.qs[offset + row * 4 + lane]) * @as(i32, rhs.qs[offset + col * 4 + lane]);
                }
            }
        }
    }

    inline for (0..4) |row| {
        acc[row] += .{
            @as(f32, @floatFromInt(sums[row][0])) * f16BitsToF32(lhs.d[row]) * f16BitsToF32(rhs.d[0]),
            @as(f32, @floatFromInt(sums[row][1])) * f16BitsToF32(lhs.d[row]) * f16BitsToF32(rhs.d[1]),
            @as(f32, @floatFromInt(sums[row][2])) * f16BitsToF32(lhs.d[row]) * f16BitsToF32(rhs.d[2]),
            @as(f32, @floatFromInt(sums[row][3])) * f16BitsToF32(lhs.d[row]) * f16BitsToF32(rhs.d[3]),
        };
    }
}

// pub: the bit-exactness reference for the SIMD arms above (q8_0_tests.zig).
pub fn accumulateQ8_0x4Scalar(lhs: *const BlockQ8_0, rhs: *const BlockQ8_0x4, acc: QKV4f32) QKV4f32 {
    var sums: [4]i32 = .{ 0, 0, 0, 0 };
    for (0..8) |feature_group| {
        const lhs_offset = feature_group * 4;
        const rhs_offset = feature_group * 16;
        inline for (0..4) |col| {
            inline for (0..4) |lane| {
                sums[col] += @as(i32, lhs.qs[lhs_offset + lane]) * @as(i32, rhs.qs[rhs_offset + col * 4 + lane]);
            }
        }
    }

    const lhs_scale = f16BitsToF32(lhs.d);
    const add: QKV4f32 = .{
        @as(f32, @floatFromInt(sums[0])) * lhs_scale * f16BitsToF32(rhs.d[0]),
        @as(f32, @floatFromInt(sums[1])) * lhs_scale * f16BitsToF32(rhs.d[1]),
        @as(f32, @floatFromInt(sums[2])) * lhs_scale * f16BitsToF32(rhs.d[2]),
        @as(f32, @floatFromInt(sums[3])) * lhs_scale * f16BitsToF32(rhs.d[3]),
    };
    return acc + add;
}

test {
    _ = @import("q8_0_tests.zig");
}

const q8_0_col_block: usize = 3;
const q8_0_aarch64_col_block: usize = 4;
const q8_0_aarch64_tail_col_block: usize = 4;

// Scalar reference replica of dotQ8_0Q8_0: exact i32 integer dot, identical
// f32 expression shape — comparisons against it are BIT-EXACT on every target.
// Mirrored in src/x86dot_check.zig for the cross-ISA attestation runs.
fn refDotQ8_0Q8_0(a: *const BlockQ8_0, b: *const BlockQ8_0) f32 {
    var acc: i32 = 0;
    for (a.qs, b.qs) |x, y| acc += @as(i32, x) * @as(i32, y);
    return @as(f32, @floatFromInt(acc)) * (f16BitsToF32(a.d) * f16BitsToF32(b.d));
}

fn fillRandomBlockQ8_0(block: *BlockQ8_0, random: std.Random, allow_m128: bool) void {
    block.d = f32ToF16Bits(0.25 + random.float(f32));
    for (&block.qs) |*q| {
        if (allow_m128) {
            q.* = @bitCast(random.int(u8)); // full i8 range incl. -128 (GGUF weight side)
        } else {
            q.* = @intCast(@as(i32, random.uintLessThan(u8, 255)) - 127); // quantizeToI8 domain
        }
    }
}

test "ggml_q8_0 randomized blocks: dot kernel matches scalar reference bit-exactly" {
    // Parity for the per-block Q8_0·Q8_0 dot the non-aarch64 matmul reduces
    // to. The AVX2 sign-trick exactness domain is honored exactly as the
    // engine guarantees it: activations (a) are quantizeToI8-clamped to
    // [-127,127]; weights (b) range over ALL of i8 including -128.
    var prng = std.Random.DefaultPrng.init(0x8e8b97cdac5b2e6f);
    const random = prng.random();

    var a: BlockQ8_0 = undefined; // activation side
    var b: BlockQ8_0 = undefined; // weight side
    var iter: usize = 0;
    while (iter < 100) : (iter += 1) {
        fillRandomBlockQ8_0(&a, random, false);
        fillRandomBlockQ8_0(&b, random, true);
        try std.testing.expectEqual(refDotQ8_0Q8_0(&a, &b), dotQ8_0Q8_0(&a, &b));
    }

    // Extremes: every weight byte -128 against ±127 activations (sign-trick
    // stress: |a|=127 lanes against sign-transferred -128 weights), and the
    // alternating saturation pattern maximizing vpmaddubsw pair sums.
    a.d = f32ToF16Bits(1.0);
    b.d = f32ToF16Bits(1.0);
    for (&b.qs) |*q| q.* = -128;
    for (&a.qs, 0..) |*q, i| q.* = if (i % 2 == 0) 127 else -127;
    try std.testing.expectEqual(refDotQ8_0Q8_0(&a, &b), dotQ8_0Q8_0(&a, &b));
    for (&a.qs) |*q| q.* = -127;
    try std.testing.expectEqual(refDotQ8_0Q8_0(&a, &b), dotQ8_0Q8_0(&a, &b));
    for (&b.qs, 0..) |*q, i| q.* = if (i % 2 == 0) 127 else -128;
    for (&a.qs, 0..) |*q, i| q.* = if (i % 2 == 0) -127 else 127;
    try std.testing.expectEqual(refDotQ8_0Q8_0(&a, &b), dotQ8_0Q8_0(&a, &b));
}
