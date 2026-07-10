//! Behavioral tests for the Gemma MoE FFN engines (`gemma_moe.zig`):
//! cross-checks the raw GGUF-block CPU/GPU arms against the x4-widened
//! reference path for both batched-prefill and single-token decode, over
//! Q6_K and Q4_K gate_up experts.
const std = @import("std");
const fucina = @import("fucina");
const gemma_moe = @import("moe.zig");

const backend_mod = fucina.internal.backend_mod;
const dtype_mod = backend_mod.dtype_info;

const ExecContext = fucina.ExecContext;

const RawExpertWeights = gemma_moe.RawExpertWeights;
const decodePacked = gemma_moe.decodePacked;
const decodePackedTensor = gemma_moe.decodePackedTensor;
const batchPacked = gemma_moe.batchPacked;
const batchPackedTensor = gemma_moe.batchPackedTensor;
const decodeRaw = gemma_moe.decodeRaw;
const decodeRawTensor = gemma_moe.decodeRawTensor;
const batchRaw = gemma_moe.batchRaw;
const batchRawTensor = gemma_moe.batchRawTensor;

fn expectGemmaMoeClose(cd: []const f32, gd: []const f32, rel: f32, floor_frac: f32, label: []const u8) !void {
    var scale: f32 = 0;
    for (cd) |v| scale = @max(scale, @abs(v));
    try std.testing.expect(scale > 0);
    for (cd, gd, 0..) |cv, gv, i| {
        const tol = @max(rel * @max(@abs(cv), @abs(gv)), floor_frac * scale);
        if (@abs(cv - gv) > tol) {
            std.debug.print(
                "gemma moe {s} mismatch at {d}: ref={e} got={e} scale={e}\n",
                .{ label, i, cv, gv, scale },
            );
            return error.TestUnexpectedResult;
        }
    }
}

test "gemma moe raw cpu + gpu arms match the x4 path (-Dgpu=metal)" {
    if (comptime !backend_mod.gpu_impl.enabled) return error.SkipZigTest;
    const gpu = backend_mod.gpu_impl;
    if (gpu.deviceName() == null) return error.SkipZigTest;

    const qm = backend_mod.quantized_matmul;
    const allocator = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(31);
    const random = prng.random();

    // smallest whole-row shapes: gate_up rows = hidden (one Q6_K block),
    // down rows = out_pe (two Q8_0 blocks)
    const hidden: usize = 256;
    const out_pe: usize = 64;
    const gu_out = 2 * out_pe;
    const n_expert: usize = 4;
    const top_k: usize = 2;
    const seq: usize = 37; // m tiles per expert: partial + multi-tile mixes
    const n_pairs = seq * top_k;
    const bpr_gu = hidden / 256;
    const bpr_dn = out_pe / 32;

    // raw GGUF-layout expert blocks (gate rows first, then up, per expert).
    // device_owned = false below keeps these transient buffers out of the
    // shim's wrap cache (uncached wraps), so they are safely freeable.
    // The Q4_K gate_up arm quantizes the SAME f32 rows, so its outputs
    // differ from the Q6_K arms only by weight quantization error.
    const gu_blocks = try allocator.alloc(dtype_mod.BlockQ6_K, n_expert * gu_out * bpr_gu);
    defer allocator.free(gu_blocks);
    const gu4_blocks = try allocator.alloc(dtype_mod.BlockQ4_K, n_expert * gu_out * bpr_gu);
    defer allocator.free(gu4_blocks);
    const dn_blocks = try allocator.alloc(dtype_mod.BlockQ8_0, n_expert * hidden * bpr_dn);
    defer allocator.free(dn_blocks);
    {
        const row_gu = try allocator.alloc(f32, hidden);
        defer allocator.free(row_gu);
        for (0..n_expert * gu_out) |r| {
            for (row_gu) |*x| x.* = random.floatNorm(f32) * 0.25;
            try qm.quantizeRowQ6_KInto(gu_blocks[r * bpr_gu ..][0..bpr_gu], row_gu);
            try qm.quantizeRowQ4_KInto(gu4_blocks[r * bpr_gu ..][0..bpr_gu], row_gu);
        }
        const row_dn = try allocator.alloc(f32, out_pe);
        defer allocator.free(row_dn);
        for (0..n_expert * hidden) |r| {
            for (row_dn) |*x| x.* = random.floatNorm(f32) * 0.25;
            try qm.quantizeRowQ8_0Into(dn_blocks[r * bpr_dn ..][0..bpr_dn], row_dn);
        }
    }
    const gw = RawExpertWeights{ .gu = .{ .q6_k = gu_blocks }, .dn_blocks = dn_blocks, .device_owned = false };
    const gw4 = RawExpertWeights{ .gu = .{ .q4_k = gu4_blocks }, .dn_blocks = dn_blocks, .device_owned = false };

    // the x4-widened per-expert handles over the same blocks (the reference
    // path; on gpu builds the loader skips these, but the kernels remain)
    const gate = try allocator.alloc(backend_mod.QuantizedMatmulRhsQ6_Kx4, n_expert);
    defer allocator.free(gate);
    const up = try allocator.alloc(backend_mod.QuantizedMatmulRhsQ6_Kx4, n_expert);
    defer allocator.free(up);
    const down = try allocator.alloc(backend_mod.QuantizedMatmulRhsQ8_0x4, n_expert);
    defer allocator.free(down);
    var built: usize = 0;
    defer for (0..built) |e| {
        gate[e].deinit();
        up[e].deinit();
        down[e].deinit();
    };
    for (0..n_expert) |e| {
        const eg = gu_blocks[e * gu_out * bpr_gu ..][0 .. gu_out * bpr_gu];
        gate[e] = try qm.packMatmulRhsQ6_Kx4(allocator, eg[0 .. out_pe * bpr_gu], out_pe, hidden, bpr_gu);
        errdefer gate[e].deinit();
        up[e] = try qm.packMatmulRhsQ6_Kx4(allocator, eg[out_pe * bpr_gu ..], out_pe, hidden, bpr_gu);
        errdefer up[e].deinit();
        const ed = dn_blocks[e * hidden * bpr_dn ..][0 .. hidden * bpr_dn];
        down[e] = try qm.packMatmulRhsQ8_0x4(allocator, ed, hidden, out_pe, bpr_dn);
        built += 1;
    }

    const x_data = try allocator.alloc(f32, seq * hidden);
    defer allocator.free(x_data);
    for (x_data) |*x| x.* = random.floatNorm(f32);
    var x = try ctx.fromSliceRank(2, .{ seq, hidden }, x_data);
    defer x.deinit();
    var xt = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ seq, hidden }, x_data);
    defer xt.deinit();

    const selected = try allocator.alloc(usize, n_pairs);
    defer allocator.free(selected);
    const wgt = try allocator.alloc(f32, n_pairs);
    defer allocator.free(wgt);
    for (selected, wgt) |*s, *w| {
        s.* = random.uintLessThan(usize, n_expert);
        w.* = 0.1 + random.float(f32);
    }

    var x4_out = try batchPacked(&ctx, &x, gate, up, down, selected, wgt, top_k, out_pe, null, null);
    defer x4_out.deinit();
    var x4_tensor = try batchPackedTensor(&ctx, &xt, gate, up, down, selected, wgt, top_k, out_pe, null, null);
    defer x4_tensor.deinit();
    try expectGemmaMoeClose(x4_out.dataConst(), x4_tensor.asRawTensor().dataConst(), 0, 0, "packed tensor batch");

    // raw CPU fallback (GPU refused via an unreachable threshold): same
    // numerics as x4 (Q8_K LHS, LUT GeGLU, Q8_0 requant), different kernel
    // layout — only fp accumulation order differs
    gpu.setMinWorkQMoeForTest(std.math.maxInt(u64));
    var raw_out = try batchRaw(&ctx, &x, gw, n_expert, selected, wgt, top_k, out_pe, null, null);
    defer raw_out.deinit();
    var raw_tensor = try batchRawTensor(&ctx, &xt, gw, n_expert, selected, wgt, top_k, out_pe, null, null);
    defer raw_tensor.deinit();
    try expectGemmaMoeClose(raw_out.dataConst(), raw_tensor.asRawTensor().dataConst(), 0, 0, "raw tensor batch");
    try expectGemmaMoeClose(x4_out.dataConst(), raw_out.dataConst(), 1e-3, 1e-4, "raw cpu batch");

    // GPU arm (unit shapes sit far below the real threshold, so drop it):
    // f16-rounded operands vs the CPU's Q8_K/Q8_0 quantization — loose bar;
    // layout/transposition bugs are O(scale)
    gpu.setMinWorkQMoeForTest(1);
    defer gpu.setMinWorkQMoeForTest(1 << 30);
    var gpu_out = try batchRaw(&ctx, &x, gw, n_expert, selected, wgt, top_k, out_pe, null, null);
    defer gpu_out.deinit();
    try expectGemmaMoeClose(x4_out.dataConst(), gpu_out.dataConst(), 3e-2, 2e-2, "gpu batch");

    // decode (seq == 1): raw plain-block GEMVs vs the x4 decode
    var x1 = try ctx.fromSliceRank(2, .{ 1, hidden }, x_data[0..hidden]);
    defer x1.deinit();
    var x1t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(&ctx, .{ 1, hidden }, x_data[0..hidden]);
    defer x1t.deinit();
    const sel1 = [_]usize{ 1, 3 };
    const wgt1 = [_]f32{ 0.6, 0.4 };
    var dec_x4 = try decodePacked(&ctx, &x1, gate, up, down, &sel1, &wgt1, out_pe, null, null);
    defer dec_x4.deinit();
    var dec_x4_tensor = try decodePackedTensor(&ctx, &x1t, gate, up, down, &sel1, &wgt1, out_pe, null, null);
    defer dec_x4_tensor.deinit();
    try expectGemmaMoeClose(dec_x4.dataConst(), dec_x4_tensor.asRawTensor().dataConst(), 0, 0, "packed tensor decode");
    var dec_raw = try decodeRaw(&ctx, &x1, gw, n_expert, &sel1, &wgt1, out_pe, null, null);
    defer dec_raw.deinit();
    var dec_raw_tensor = try decodeRawTensor(&ctx, &x1t, gw, n_expert, &sel1, &wgt1, out_pe, null, null);
    defer dec_raw_tensor.deinit();
    try expectGemmaMoeClose(dec_raw.dataConst(), dec_raw_tensor.asRawTensor().dataConst(), 0, 0, "raw tensor decode");
    try expectGemmaMoeClose(dec_x4.dataConst(), dec_raw.dataConst(), 1e-3, 1e-4, "raw cpu decode");

    // Q4_K gate_up arm: raw CPU vs GPU agree on the same blocks (independent
    // plumbing: typed-slice views vs nb01/nb02 byte strides + format tag) …
    gpu.setMinWorkQMoeForTest(std.math.maxInt(u64));
    var raw4_out = try batchRaw(&ctx, &x, gw4, n_expert, selected, wgt, top_k, out_pe, null, null);
    defer raw4_out.deinit();
    gpu.setMinWorkQMoeForTest(1);
    var gpu4_out = try batchRaw(&ctx, &x, gw4, n_expert, selected, wgt, top_k, out_pe, null, null);
    defer gpu4_out.deinit();
    try expectGemmaMoeClose(raw4_out.dataConst(), gpu4_out.dataConst(), 3e-2, 2e-2, "q4_k gpu batch");

    // … and stay within weight-quantization distance of the Q6_K x4 output
    // (same source rows; catches catastrophic misinterpretation that a bug
    // shared by both Q4_K engines could otherwise hide). Loose bar: the
    // GeGLU nonlinearity amplifies the q4-vs-q6 weight error at individual
    // coordinates (measured up to ~6% of output scale on this shape), while
    // layout/scale bugs are O(scale).
    try expectGemmaMoeClose(x4_out.dataConst(), raw4_out.dataConst(), 0.5, 0.15, "q4_k vs q6_k sanity");

    // decode over the Q4_K blocks runs the decode-task arm of the same views
    var dec_raw4 = try decodeRaw(&ctx, &x1, gw4, n_expert, &sel1, &wgt1, out_pe, null, null);
    defer dec_raw4.deinit();
    try expectGemmaMoeClose(dec_raw.dataConst(), dec_raw4.dataConst(), 0.5, 0.15, "q4_k raw cpu decode sanity");
}
