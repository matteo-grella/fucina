//! SCRFD decode + NMS — turns the raw heads (scrfd.zig) into
//! source-pixel detections. distance2bbox/distance2kps with per-stride anchor
//! centers, then insightface greedy NMS (Pascal-VOC IoU with the +1). Mirrors
//! refs/face-detect.cpp `src/detect.cpp`.

const std = @import("std");
const scrfd = @import("scrfd.zig");

pub const Detection = struct {
    score: f32,
    box: [4]f32, // x1,y1,x2,y2 (source pixels)
    kps: [5][2]f32,
};

fn iou(a: Detection, b: Detection) f32 {
    const ix1 = @max(a.box[0], b.box[0]);
    const iy1 = @max(a.box[1], b.box[1]);
    const ix2 = @min(a.box[2], b.box[2]);
    const iy2 = @min(a.box[3], b.box[3]);
    const iw = @max(@as(f32, 0), ix2 - ix1 + 1);
    const ih = @max(@as(f32, 0), iy2 - iy1 + 1);
    const inter = iw * ih;
    const aa = (a.box[2] - a.box[0] + 1) * (a.box[3] - a.box[1] + 1);
    const ab = (b.box[2] - b.box[0] + 1) * (b.box[3] - b.box[1] + 1);
    return inter / (aa + ab - inter);
}

fn scoreDesc(_: void, a: Detection, b: Detection) bool {
    return a.score > b.score;
}

/// Decode all strides above `score_thresh`, then greedy NMS at `nms_thresh`.
/// Returns kept detections in descending-score order (caller frees).
pub fn decode(allocator: std.mem.Allocator, heads: *const scrfd.Heads, det_scale: f32, score_thresh: f32, nms_thresh: f32) ![]Detection {
    const strides = [3]usize{ 8, 16, 32 };
    var cand: std.ArrayList(Detection) = .empty;
    defer cand.deinit(allocator);

    for (0..3) |si| {
        const stride = strides[si];
        const sf: f32 = @floatFromInt(stride);
        const hw = 640 / stride;
        const na: usize = 2;
        const score = heads.score[si];
        const bbox = heads.bbox[si];
        const kps = heads.kps[si];
        for (0..hw) |r| for (0..hw) |col| for (0..na) |a| {
            const idx = (r * hw + col) * na + a;
            const sc = score[idx];
            if (sc < score_thresh) continue;
            const cx: f32 = @floatFromInt(col * stride);
            const cy: f32 = @floatFromInt(r * stride);
            const l = bbox[idx * 4 + 0] * sf;
            const t = bbox[idx * 4 + 1] * sf;
            const rr = bbox[idx * 4 + 2] * sf;
            const bo = bbox[idx * 4 + 3] * sf;
            var d: Detection = undefined;
            d.score = sc;
            d.box = .{ (cx - l) / det_scale, (cy - t) / det_scale, (cx + rr) / det_scale, (cy + bo) / det_scale };
            for (0..5) |k| {
                const px = cx + kps[idx * 10 + k * 2 + 0] * sf;
                const py = cy + kps[idx * 10 + k * 2 + 1] * sf;
                d.kps[k] = .{ px / det_scale, py / det_scale };
            }
            try cand.append(allocator, d);
        };
    }

    std.mem.sort(Detection, cand.items, {}, scoreDesc);

    const suppressed = try allocator.alloc(bool, cand.items.len);
    defer allocator.free(suppressed);
    @memset(suppressed, false);
    var keep: std.ArrayList(Detection) = .empty;
    defer keep.deinit(allocator);
    for (cand.items, 0..) |d, i| {
        if (suppressed[i]) continue;
        try keep.append(allocator, d);
        for (cand.items[i + 1 ..], i + 1..) |e, j| {
            if (!suppressed[j] and iou(d, e) > nms_thresh) suppressed[j] = true;
        }
    }
    return keep.toOwnedSlice(allocator);
}
