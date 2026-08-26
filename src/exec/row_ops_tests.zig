//! Pinned rowwise kernels: batched quant ops against the m == 1 numerics and
//! fused rms-norm+rope batch rows against single-position calls.
//! Force-imported by `exec.zig`.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const exec = @import("../exec.zig");
const exec_elementwise = @import("elementwise.zig");
const exec_row_ops = @import("row_ops.zig");
const exec_moe_chain = @import("moe_chain.zig");
const dtype_mod = @import("../dtype.zig");
const fpenv = @import("../fpenv.zig");
const parallel = @import("../parallel.zig");
const rng = @import("../rng.zig");
const tensor = @import("../tensor.zig");

const Allocator = std.mem.Allocator;
const Tensor = tensor.Tensor;
const ExecContext = exec.ExecContext;
const CrossEntropyOptions = exec.CrossEntropyOptions;
const Reduction = exec.Reduction;

const util = @import("test_util.zig");
const buildTestMoeRhsQ5K = util.buildTestMoeRhsQ5K;
const f16BitsFromF32 = util.f16BitsFromF32;

test "pinned rowwise kernels: batched quant ops reproduce the m == 1 numerics bitwise" {
    // The lossless-speculation contract (see ExecContext.pin_rowwise_kernels):
    // with the pin ON, every batched quant entry must produce, for each
    // row, exactly the bytes the same entry produces for that row alone —
    // the m-dependent kernel switches (x4-lane prefixes, padded/unpadded
    // q8_0 LHS packing, lane-packed MoE kernels) must all be neutralized.
    const allocator = std.testing.allocator;
    const qm = backend_mod.quantized_matmul;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const k: usize = 512;
    const n: usize = 64;
    const ms = [_]usize{ 2, 3, 4, 5, 8, 12 };
    const max_m: usize = 12;

    // Deterministic activations; the widest buffer (2k cols) also serves
    // as the split-swiglu gate_up input.
    const x_vals = try allocator.alloc(f32, max_m * 2 * k);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13) * 2.0;
    const norm_vals = try allocator.alloc(f32, k);
    defer allocator.free(norm_vals);
    for (norm_vals, 0..) |*v, i| v.* = 0.5 + 0.01 * @as(f32, @floatFromInt(i % 37));
    var norm_w = try ctx.fromSlice(.f32, .{k}, norm_vals);
    defer norm_w.deinit();

    ctx.pinRowwiseKernels(true);
    defer ctx.pinRowwiseKernels(false);

    const expectRowwise = struct {
        fn go(c: *ExecContext, entry: anytype, input: []const f32, cols: usize, m: usize, out_n: usize) !void {
            var batch_in = try c.fromSlice(.f32, .{ m, cols }, input[0 .. m * cols]);
            defer batch_in.deinit();
            var batch_out = try entry.run(c, &batch_in);
            defer batch_out.deinit();
            for (0..m) |r| {
                var row_in = try c.fromSlice(.f32, .{ 1, cols }, input[r * cols ..][0..cols]);
                defer row_in.deinit();
                var row_out = try entry.run(c, &row_in);
                defer row_out.deinit();
                try std.testing.expectEqualSlices(f32, row_out.dataConst(), batch_out.dataConst()[r * out_n ..][0..out_n]);
            }
        }
    }.go;

    // q4_k / q5_k blocks via the real quantizers; q6_k via the K-quant
    // fixture pattern; q8_0 via row quantization.
    {
        const bpc = k / qm.types.qk_k_block_size;
        const q4_blocks = try allocator.alloc(dtype_mod.BlockQ4_K, n * bpc);
        defer allocator.free(q4_blocks);
        const q5_blocks = try allocator.alloc(dtype_mod.BlockQ5_K, n * bpc);
        defer allocator.free(q5_blocks);
        var block_vals: [256]f32 = undefined;
        for (q4_blocks, q5_blocks, 0..) |*b4, *b5, bi| {
            for (&block_vals, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(bi * 256 + i)) * 0.07) * 1.5;
            qm.q4_k.quantizeBlockQ4_KInto(b4, &block_vals);
            qm.q5_k.quantizeBlockQ5_KInto(b5, &block_vals);
        }
        var q4_rhs = try qm.q4_k.packMatmulRhsQ4_Kx8(allocator, q4_blocks, n, k, bpc);
        defer q4_rhs.deinit();
        var q5_rhs = try qm.q5_k.packMatmulRhsQ5_Kx8(allocator, q5_blocks, n, k, bpc);
        defer q5_rhs.deinit();
        const q6_blocks = try allocator.alloc(dtype_mod.BlockQ6_K, n * bpc);
        defer allocator.free(q6_blocks);
        for (q6_blocks, 0..) |*b, block_i| {
            b.d = f16BitsFromF32(0.04 + 0.001 * @as(f32, @floatFromInt(block_i % 5)));
            for (&b.scales, 0..) |*s, i| s.* = @intCast(@as(i32, @intCast((i * 5 + block_i) % 15)) - 7);
            for (&b.ql, 0..) |*q, i| q.* = @intCast((i * 19 + block_i * 7) % 256);
            for (&b.qh, 0..) |*q, i| q.* = @intCast((i * 23 + block_i * 3) % 256);
        }
        var q6_rhs = try qm.q6_k.packMatmulRhsQ6_Kx4(allocator, q6_blocks, n, k, bpc);
        defer q6_rhs.deinit();
        const q8_bpc = try qm.q8k.q8_0BlockCount(k);
        const q8_blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n * q8_bpc);
        defer allocator.free(q8_blocks);
        {
            var row: [512]f32 = undefined;
            for (0..n) |r| {
                for (&row, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(r * 31 + i)) * 0.11) * 1.2;
                try qm.q8k.quantizeRowQ8_0Into(q8_blocks[r * q8_bpc ..][0..q8_bpc], &row);
            }
        }
        var q8_rhs = try qm.q8_0.packMatmulRhsQ8_0x4(allocator, q8_blocks, n, k, q8_bpc);
        defer q8_rhs.deinit();

        for (ms) |m| {
            // Fused split-swiglu (the K-quant use_x4 gate + the q8_0 wrap).
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.splitSwiGluMatmulPacked(in, s.rhs);
                }
            }{ .rhs = &q4_rhs }, x_vals, 2 * k, m, n);
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ5_Kx8,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.splitSwiGluMatmulPacked(in, s.rhs);
                }
            }{ .rhs = &q5_rhs }, x_vals, 2 * k, m, n);
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ6_Kx4,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.splitSwiGluMatmulPacked(in, s.rhs);
                }
            }{ .rhs = &q6_rhs }, x_vals, 2 * k, m, n);
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.splitSwiGluMatmulPacked(in, s.rhs);
                }
            }{ .rhs = &q8_rhs }, x_vals, 2 * k, m, n);
            // Fused rms-norm-mul (gate + wrap) and the plain packed matmuls.
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8,
                w: *const Tensor,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.rmsNormMulMatmulPacked(in, s.w, 1e-6, s.rhs);
                }
            }{ .rhs = &q4_rhs, .w = &norm_w }, x_vals, k, m, n);
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
                w: *const Tensor,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.rmsNormMulMatmulPacked(in, s.w, 1e-6, s.rhs);
                }
            }{ .rhs = &q8_rhs, .w = &norm_w }, x_vals, k, m, n);
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ4_Kx8,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.matmulPacked(in, s.rhs);
                }
            }{ .rhs = &q4_rhs }, x_vals, k, m, n);
            try expectRowwise(&ctx, struct {
                rhs: *const backend_mod.QuantizedMatmulRhsQ8_0x4,
                fn run(s: @This(), c: *ExecContext, in: *const Tensor) !Tensor {
                    return c.matmulPacked(in, s.rhs);
                }
            }{ .rhs = &q8_rhs }, x_vals, k, m, n);
        }
    }

    // The batched MoE op vs the single-token decode op: every row of a
    // pinned verify-shaped batch must equal the decode op on that token —
    // the exact equality deep MTP verification relies on. 64 pairs engage
    // the phased chain; 8 tokens per expert engage the per-expert m >= 4
    // kernel switches under test.
    {
        const seq: usize = 32;
        const top_k: usize = 2;
        const n_expert: usize = 8;
        const hidden: usize = 512;
        const out_pe: usize = 512;
        var gate = try buildTestMoeRhsQ5K(allocator, n_expert * out_pe, hidden, 0);
        defer gate.deinit();
        var up = try buildTestMoeRhsQ5K(allocator, n_expert * out_pe, hidden, 1);
        defer up.deinit();
        var down = try buildTestMoeRhsQ5K(allocator, n_expert * hidden, out_pe, 2);
        defer down.deinit();
        const mx_vals = try allocator.alloc(f32, seq * hidden);
        defer allocator.free(mx_vals);
        for (mx_vals, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 17) % 251)) - 125);
        var mx = try ctx.fromSlice(.f32, .{ seq, hidden }, mx_vals);
        defer mx.deinit();
        var selected: [seq * top_k]usize = undefined;
        var weights: [seq * top_k]f32 = undefined;
        for (&selected, &weights, 0..) |*s, *w, p| {
            s.* = (p * 5) % n_expert;
            w.* = 0.25 + 0.01 * @as(f32, @floatFromInt(p % 13));
        }
        var batch_out = try ctx.moeExpertFfnBatch(&mx, &gate, &up, &down, &selected, &weights, top_k, out_pe, .{ .op = .swiglu }, null, null);
        defer batch_out.deinit();
        for (0..seq) |t| {
            var row = try ctx.fromSlice(.f32, .{ 1, hidden }, mx_vals[t * hidden ..][0..hidden]);
            defer row.deinit();
            var row_out = try ctx.moeExpertFfn(&row, &gate, &up, &down, selected[t * top_k ..][0..top_k], weights[t * top_k ..][0..top_k], out_pe, .{ .op = .swiglu }, null, null);
            defer row_out.deinit();
            try std.testing.expectEqualSlices(f32, row_out.dataConst(), batch_out.dataConst()[t * hidden ..][0..hidden]);
        }
    }
}

test "fused rms-norm+rope: batch rows are bitwise equal to single-position calls at every m" {
    // rmsNormMulRopeWithTable's kernel choice is layout-only by
    // contract: a row's bytes must not depend on how many rows share the
    // call, because k rows land in the KV cache and feed every subsequent
    // token (the lossless-speculation contract, speculative/core.zig).
    // The sweep uses the 16-head/128-d attention shape and spans the row
    // counts where a total-length-gated choice would flip kernels for its
    // q (m = 16) and kv (m = 32) tensors — the exact trap the layout-only
    // rule exists to exclude.
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const heads: usize = 16;
    const d: usize = 128;
    const eps: f32 = 1e-6;
    const max_m: usize = 40;

    const x_vals = try allocator.alloc(f32, max_m * heads * d);
    defer allocator.free(x_vals);
    for (x_vals, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.19) * 1.7;
    const w_vals = try allocator.alloc(f32, d);
    defer allocator.free(w_vals);
    for (w_vals, 0..) |*v, i| v.* = 0.6 + 0.02 * @as(f32, @floatFromInt(i % 23));
    var w = try ctx.fromSlice(.f32, .{d}, w_vals);
    defer w.deinit();

    for ([_]usize{ 1, 2, 15, 16, 17, 31, 32, 33, 40 }) |m| {
        var positions: [max_m]i32 = undefined;
        for (positions[0..m], 0..) |*p, i| p.* = @intCast(i);
        var table = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = positions[0..m] }, .feature_dim = d, .freqs = .{ .theta = .{ .base = 10000.0 } } });
        defer table.deinit();
        var x = try ctx.fromSlice(.f32, .{ m, heads, d }, x_vals[0 .. m * heads * d]);
        defer x.deinit();
        var batch = try ctx.rmsNormMulRopeWithTable(3, &x, &w, 0, 2, eps, &table, .half);
        defer batch.deinit();

        for (0..m) |i| {
            var row_pos = [_]i32{@intCast(i)};
            var row_table = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = &row_pos }, .feature_dim = d, .freqs = .{ .theta = .{ .base = 10000.0 } } });
            defer row_table.deinit();
            var row = try ctx.fromSlice(.f32, .{ 1, heads, d }, x_vals[i * heads * d ..][0 .. heads * d]);
            defer row.deinit();
            var single = try ctx.rmsNormMulRopeWithTable(3, &row, &w, 0, 2, eps, &row_table, .half);
            defer single.deinit();
            try std.testing.expectEqualSlices(
                f32,
                single.dataConst(),
                batch.dataConst()[i * heads * d ..][0 .. heads * d],
            );
        }
    }
}
