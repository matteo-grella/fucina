//! Host-band scalar ops: the ONE definition of the numerics every
//! host-style port shares — f64-accumulated RMS norm, max-subtracted
//! softmax, silu, and the SwiGLU-through-`LinearWeight` FFN. Consumers:
//! the runner's host_reference band, deepseek2, inkling (model + mmproj).
//! Families with genuine variants keep them beside their models
//! (deepseek4's nullable-weight rms norm, inkling's fused gate_up swiglu).
//!
//! MoE routing deliberately stays per-family: the noaux selection SEMANTICS
//! (bias applies to the top-k choice only, mixture weights come from the
//! raw scores) are shared, but the top-k fallback kernels differ — the
//! runner band scans host-side, deepseek2 goes through the core
//! `topKExperts` kernel — and each family's parity anchors pin its own
//! order of operations.

const std = @import("std");
const fucina = @import("fucina");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const LinearWeight = fucina.weights.LinearWeight;

pub fn rmsNormInto(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + eps);
    for (out, x, weight) |*o, v, w| o.* = @floatCast(@as(f64, v) * inv * w);
}

pub fn softmaxInPlace(v: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (v) |x| max = @max(max, x);
    var sum: f32 = 0;
    for (v) |*x| {
        x.* = @exp(x.* - max);
        sum += x.*;
    }
    for (v) |*x| x.* /= sum;
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// SwiGLU FFN over host rows through separate gate/up/down `LinearWeight`s:
/// `down(silu(gate(x)) * up(x))`, returned as a caller-owned f32 slice.
pub fn swigluLinear(ctx: *ExecContext, allocator: Allocator, x: *const fucina.Tensor(.{ .seq, .embed }), gate: *const LinearWeight, up: *const LinearWeight, down: *const LinearWeight) ![]f32 {
    var gate_t = try gate.linearSeq(ctx, x, .embed, .gate_up);
    defer gate_t.deinit();
    var up_t = try up.linearSeq(ctx, x, .embed, .gate_up);
    defer up_t.deinit();
    const width = gate_t.dim(.gate_up);
    const rows = gate_t.dim(.seq);
    var g_t = try fucina.Tensor(.{ .seq, .embed }).empty(ctx, .{ rows, width });
    defer g_t.deinit();
    for (try g_t.data(), try gate_t.dataConst(), try up_t.dataConst()) |*gi, gv, uv| gi.* = silu(gv) * uv;
    var down_t = try down.linearSeq(ctx, &g_t, .embed, .attn);
    defer down_t.deinit();
    return allocator.dupe(f32, try down_t.dataConst());
}
