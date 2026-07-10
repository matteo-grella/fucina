//! MoE router: softmax over expert logits + top-k selection per row.
//!
//! Domain module: receives an explicit `*Runtime`. Negligible-cost route path
//! (called once per MoE layer). Home of `RouterTopKOptions` (re-exported by
//! `exec.zig`).

const std = @import("std");
const tensor = @import("../tensor.zig");

const Runtime = @import("runtime.zig").Runtime;

const Tensor = tensor.Tensor;

pub const RouterTopKOptions = struct {
    normalize_selected: bool = true,
};

/// Router softmax over rank-2 `[row, expert]` logits, followed by top-k expert
/// selection per row. `probs` is per-row scratch of length >= expert count;
/// `selected` and `weights` are row-major `[row, k]` outputs.
pub fn routerTopK(
    rt: *Runtime,
    logits: *const Tensor,
    k: usize,
    options: RouterTopKOptions,
    selected: []usize,
    weights: []f32,
) !void {
    if (k == 0) return tensor.TensorError.InvalidShape;
    const source = try logits.rankView(2);
    const rows = source.shape[0];
    const experts = source.shape[1];
    if (k > experts) return tensor.TensorError.IndexOutOfBounds;
    if (selected.len != rows * k or weights.len != rows * k) return tensor.TensorError.InvalidDataLength;

    var ll = try rt.prepareContiguous(logits);
    defer ll.deinit();
    const input = ll.tensor().dataConst();
    for (0..rows) |row| {
        routerTopKRow(
            input[row * experts ..][0..experts],
            k,
            options.normalize_selected,
            selected[row * k ..][0..k],
            weights[row * k ..][0..k],
        );
    }
}

fn routerTopKRow(logits: []const f32, k: usize, normalize_selected: bool, selected: []usize, weights: []f32) void {
    var max: f32 = logits[0];
    for (0..k) |slot| {
        selected[slot] = 0;
        weights[slot] = -std.math.inf(f32);
    }

    for (logits, 0..) |v, i| {
        if (v > max) max = v;
        if (v <= weights[k - 1]) continue;
        var slot = k - 1;
        while (slot > 0 and v > weights[slot - 1]) : (slot -= 1) {
            weights[slot] = weights[slot - 1];
            selected[slot] = selected[slot - 1];
        }
        weights[slot] = v;
        selected[slot] = i;
    }

    var exp_sum: f32 = 0;
    for (logits) |v| {
        exp_sum += @exp(v - max);
    }
    const inv_sum = 1.0 / exp_sum;

    var total: f32 = 0;
    for (weights[0..k]) |*w| {
        const e = @exp(w.* - max) * inv_sum;
        w.* = e;
        total += e;
    }

    if (normalize_selected and total > 0) {
        const inv_total = 1.0 / total;
        for (weights[0..k]) |*w| w.* *= inv_total;
    }
}
