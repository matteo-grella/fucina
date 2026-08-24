//! Box-frame decode heuristics for MTP parallel box decoding.
//!
//! Verbatim scalar port of refs/locate-anything.cpp/src/mtp.cpp (itself a
//! port of the upstream generate_utils.py sample_tokens / is_valid_box_frame /
//! decode_bbox_avg / decode_ref / handle_pattern). Deliberately scalar host
//! code: the 0.7/0.2/0.1 thresholds ride on softmax probabilities, so the
//! softmax accumulates in f64 exactly like the reference, and top-k
//! reproduces torch's stable ordering (prob desc, index asc on ties) — a
//! vectorized rewrite that reassociates either would move near-threshold
//! decisions.
//!
//! `fast` selects generation_mode='fast' semantics vs 'hybrid':
//!   * decode_bbox_avg: fast skips the abnormal-coord->0 zeroing;
//!   * handle_pattern: fast treats a malformed box as a coord_box (no AR switch).

const std = @import("std");
const config_mod = @import("config.zig");

const Allocator = std.mem.Allocator;

/// Token-id constants, bound from GGUF metadata at engine init.
pub const TokenIds = struct {
    box_start: u32,
    box_end: u32,
    coord_start: u32,
    coord_end: u32,
    ref_start: u32,
    ref_end: u32,
    none: u32,
    null_tok: u32,
    text_mask: u32,
    im_end: u32,

    pub fn fromConfig(config: config_mod.Config) TokenIds {
        return .{
            .box_start = config.tok_box_start,
            .box_end = config.tok_box_end,
            .coord_start = config.tok_coord_start,
            .coord_end = config.tok_coord_end,
            .ref_start = config.tok_ref_start,
            .ref_end = config.tok_ref_end,
            .none = config.tok_none,
            .null_tok = config.tok_null,
            .text_mask = config.tok_text_mask,
            .im_end = config.tok_eos,
        };
    }
};

/// Numerically-stable softmax of one logits row; f64 accumulation like the
/// reference's softmax_row.
pub fn softmaxRow(allocator: Allocator, logits: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, logits.len);
    errdefer allocator.free(out);
    var m = -std.math.inf(f32);
    for (logits) |v| m = @max(m, v);
    var sum: f64 = 0.0;
    for (logits, out) |v, *o| {
        const e = @exp(@as(f64, v) - @as(f64, m));
        o.* = @floatCast(e);
        sum += e;
    }
    const inv: f32 = @floatCast(1.0 / sum);
    for (out) |*o| o.* *= inv;
    return out;
}

/// torch.topk semantics: top-k by descending value, ties broken by ascending
/// index. Insertion into a fixed-size sorted window (k is 4 or 5).
pub fn topk(row: []const f32, ids: []usize, probs: []f32) void {
    const k = ids.len;
    std.debug.assert(probs.len == k);
    var count: usize = 0;
    for (row, 0..) |v, i| {
        if (count == k and v <= probs[k - 1]) continue;
        // Find insertion slot: strictly-greater values move up; equal values
        // keep earlier index first (stable).
        var slot = count;
        while (slot > 0 and v > probs[slot - 1]) : (slot -= 1) {}
        if (slot >= k) continue;
        if (count < k) count += 1;
        var j = count - 1;
        while (j > slot) : (j -= 1) {
            probs[j] = probs[j - 1];
            ids[j] = ids[j - 1];
        }
        probs[slot] = v;
        ids[slot] = i;
    }
}

pub fn argmaxRow(row: []const f32) usize {
    var best: usize = 0;
    for (row[1..], 1..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

const BoxFrameKind = enum { empty_box, legal_box, illegal_box };

/// is_valid_box_frame (generate_utils.py L246-273), start_thresh=0.7,
/// end_thresh=0.2.
fn isValidBoxFrame(t: TokenIds, probs: []const []const f32, start_thresh: f32, end_thresh: f32) BoxFrameKind {
    const p_start = probs[0][t.box_start];
    if (p_start >= start_thresh) {
        if (probs[1][t.none] > 0.2 and
            probs[2][t.box_end] > 0.2 and
            probs[3][t.null_tok] > 0.1 and
            probs[4][t.null_tok] > 0.1)
        {
            return .empty_box;
        }
    }
    const end_score = probs[5][t.box_end] + probs[5][t.null_tok] + probs[5][t.im_end];
    if (end_score >= end_thresh) return .legal_box;
    return .illegal_box;
}

/// decode_bbox_avg (L276-361), keep_k=4. Returns the box 6-tuple, or null for
/// "None" (not a box). fast=true: final coords = first valid ids (no
/// abnormal-coord zeroing).
fn decodeBboxAvg(allocator: Allocator, t: TokenIds, probs: []const []const f32, keep_k: usize, fast: bool) !?[]u32 {
    switch (isValidBoxFrame(t, probs, 0.7, 0.2)) {
        .empty_box => {
            const out = try allocator.alloc(u32, 6);
            out[0] = t.box_start;
            out[1] = t.none;
            out[2] = t.box_end;
            out[3] = t.null_tok;
            out[4] = t.null_tok;
            out[5] = t.null_tok;
            return out;
        },
        .illegal_box => return null,
        .legal_box => {},
    }

    var coords: [4]u32 = .{ 0, 0, 0, 0 };
    var ids_buf: [8]usize = undefined;
    var probs_buf: [8]f32 = undefined;
    for (0..4) |p| {
        const ids = ids_buf[0..keep_k];
        const pr = probs_buf[0..keep_k];
        topk(probs[1 + p], ids, pr);
        var first_valid_idx: ?usize = null;
        var valid_counts: usize = 0;
        var valid_max: i64 = -999999;
        var valid_min: i64 = 999999;
        for (ids, 0..) |id, i| {
            if (id >= t.coord_start and id <= t.coord_end) {
                if (first_valid_idx == null) first_valid_idx = i;
                valid_counts += 1;
                valid_max = @max(valid_max, @as(i64, @intCast(id)));
                valid_min = @min(valid_min, @as(i64, @intCast(id)));
            }
        }
        const fvi = first_valid_idx orelse return null; // not a box
        const first_valid_prob = pr[fvi];
        const first_valid_id: u32 = @intCast(ids[fvi]);
        if (fast) {
            coords[p] = first_valid_id;
        } else {
            const is_abnormal = (first_valid_prob < 0.9) and (valid_counts > 1) and
                ((valid_max - valid_min) > 60);
            coords[p] = if (is_abnormal) 0 else first_valid_id;
        }
    }
    const out = try allocator.alloc(u32, 6);
    out[0] = t.box_start;
    for (coords, 1..) |c, i| out[i] = c;
    out[5] = t.box_end;
    return out;
}

/// decode_ref (L364-405): keep_k=5, start_thresh=0.6 (the sample_tokens call
/// site defaults, NOT decode_bbox_avg's keep_k=4). Null for "None".
fn decodeRef(allocator: Allocator, t: TokenIds, probs: []const []const f32, keep_k: usize, start_thresh: f32) !?[]u32 {
    if (probs[0][t.ref_start] < start_thresh) return null;
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, t.ref_start);
    var ids_buf: [8]usize = undefined;
    var probs_buf: [8]f32 = undefined;
    for (1..probs.len) |p| {
        const ids = ids_buf[0..keep_k];
        topk(probs[p], ids, probs_buf[0..keep_k]);
        var first_valid: ?usize = null;
        for (ids) |id| {
            const is_coord = id >= t.coord_start and id <= t.coord_end;
            if (!is_coord) {
                first_valid = id;
                break;
            }
        }
        const id = first_valid orelse return null;
        try out.append(allocator, @intCast(id));
    }
    return try out.toOwnedSlice(allocator);
}

/// sample_box: decode_bbox_avg -> decode_ref -> {0} fallback (null here).
fn sampleBox(allocator: Allocator, t: TokenIds, probs: []const []const f32, keep_k: usize, fast: bool) !?[]u32 {
    if (try decodeBboxAvg(allocator, t, probs, keep_k, fast)) |box| return box;
    if (try decodeRef(allocator, t, probs, 5, 0.6)) |ref| return ref;
    return null;
}

/// select_new_tokens: softmax the block rows, run sample_box; on the {0}
/// fallback return the per-position argmax of the raw logits instead
/// (reference `is_box_empty ? x0 : box_avg`). `logits_rows` is
/// [block][vocab] position-major. Caller frees the result.
pub fn selectNewTokens(
    allocator: Allocator,
    t: TokenIds,
    logits_rows: []const []const f32,
    keep_k: usize,
    fast: bool,
) ![]u32 {
    const block = logits_rows.len;
    const probs = try allocator.alloc([]f32, block);
    // Zero-init every slot before the fallible fill loop: the cleanup frees
    // all entries, and a mid-loop softmaxRow failure must not hand it
    // undefined slices (freeing a zero-length slice is a no-op).
    for (probs) |*p| p.* = &.{};
    defer {
        for (probs) |p| allocator.free(p);
        allocator.free(probs);
    }
    for (logits_rows, 0..) |row, p| probs[p] = try softmaxRow(allocator, row);

    // []const []f32 -> []const []const f32 for the helpers.
    const probs_const = try allocator.alloc([]const f32, block);
    defer allocator.free(probs_const);
    for (probs, probs_const) |p, *c| c.* = p;

    if (try sampleBox(allocator, t, probs_const, keep_k, fast)) |box| return box;

    const argmax = try allocator.alloc(u32, block);
    for (logits_rows, argmax) |row, *a| a.* = @intCast(argmaxRow(row));
    return argmax;
}

pub const PatternKind = enum { im_end, empty_box, coord_box, point_box, error_box, ref_object };

pub const Pattern = struct {
    kind: PatternKind,
    tokens: []u32,
    terminal: bool,
    need_ar: bool,

    pub fn deinit(self: *Pattern, allocator: Allocator) void {
        allocator.free(self.tokens);
        self.* = undefined;
    }
};

/// handle_pattern (L408-504). fast=true: a malformed box is a coord_box (full
/// x0, no AR switch) instead of error_box.
pub fn handlePattern(allocator: Allocator, t: TokenIds, x0: []const u32, fast: bool) !Pattern {
    if (x0.len == 0 or x0[0] == t.null_tok or x0[0] == t.im_end) {
        const tokens = try allocator.dupe(u32, &.{t.im_end});
        return .{ .kind = .im_end, .tokens = tokens, .terminal = true, .need_ar = false };
    }
    if (x0.len >= 2 and x0[0] == t.box_start and x0[1] == t.none) {
        const tokens = try allocator.dupe(u32, &.{ t.box_start, t.none, t.box_end });
        return .{ .kind = .empty_box, .tokens = tokens, .terminal = false, .need_ar = false };
    }
    if (x0[0] == t.box_start) {
        var coord_ix: usize = 1;
        for (1..@min(5, x0.len)) |i| {
            const c = x0[i];
            if (c >= t.coord_start and c <= t.coord_end) coord_ix += 1 else break;
        }
        if (coord_ix == 5 and x0.len >= 6 and x0[5] == t.box_end) {
            return .{ .kind = .coord_box, .tokens = try allocator.dupe(u32, x0), .terminal = false, .need_ar = false };
        } else if (coord_ix == 3 and x0.len >= 4 and x0[3] == t.box_end) {
            return .{ .kind = .point_box, .tokens = try allocator.dupe(u32, x0[0..4]), .terminal = false, .need_ar = false };
        } else if (fast) {
            return .{ .kind = .coord_box, .tokens = try allocator.dupe(u32, x0), .terminal = false, .need_ar = false };
        } else {
            return .{ .kind = .error_box, .tokens = try allocator.dupe(u32, x0[0..coord_ix]), .terminal = false, .need_ar = true };
        }
    }
    // ref_object: truncate at first null, dedup a trailing ref_end pair.
    var end = x0.len;
    for (x0, 0..) |tok, i| {
        if (tok == t.null_tok) {
            end = i;
            break;
        }
    }
    var toks = try allocator.dupe(u32, x0[0..end]);
    if (toks.len >= 2 and toks[toks.len - 1] == t.ref_end and toks[toks.len - 2] == t.ref_end) {
        toks = try allocator.realloc(toks, toks.len - 1);
    }
    return .{ .kind = .ref_object, .tokens = toks, .terminal = false, .need_ar = false };
}

/// Hybrid MTP<->AR control-flow state (modeling_locateanything.py generate).
pub const HybridState = struct {
    use_mtp: bool = true,
    terminated: bool = false,
};

/// MTP round classification: commit pattern tokens, im_end terminates,
/// error_box drops to AR (hybrid only; fast never yields error_box).
pub fn hybridMtpStep(st: *HybridState, pattern: *const Pattern) void {
    if (pattern.terminal) st.terminated = true else if (pattern.kind == .error_box) st.use_mtp = false;
}

pub const ArKind = enum { box_end_ar, coord_ar, im_end };

/// AR round classification (_sample_token_in_ar): the RAW token is committed
/// in every branch; box_end returns to MTP, coord/none stays AR, anything
/// else terminates.
pub fn hybridArStep(st: *HybridState, t: TokenIds, token: u32) ArKind {
    if (token == t.box_end) {
        st.use_mtp = true;
        return .box_end_ar;
    }
    if ((token >= t.coord_start and token <= t.coord_end) or token == t.none) {
        return .coord_ar;
    }
    st.terminated = true;
    return .im_end;
}

test {
    _ = @import("mtp_tests.zig");
}
