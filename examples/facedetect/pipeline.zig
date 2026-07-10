//! End-to-end pipeline: wires the ported pieces into the full embed path —
//! source image → letterbox → SCRFD detect → largest-area face → umeyama align
//! (norm_crop) → ArcFace embed. This is the production path the CLI runs; the
//! whole chain is gated against the reference (cosine ≥ 0.9999).
//!
//! The `*With` variants take caller-held models (weights loaded once — the
//! repeated-forward path the CLI and bench use); the model-less wrappers keep
//! one-shot signatures for the parity tests.

const std = @import("std");
const fucina = @import("fucina");
const image = @import("image.zig");
const preprocess = @import("preprocess.zig");
const scrfd = @import("scrfd.zig");
const detect = @import("detect.zig");
const align_mod = @import("align.zig");
const rec = @import("recognizer.zig");
const genderage = @import("genderage.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

/// Detect + pick the largest-area face → its 5 landmarks. Returns null if none.
pub fn primaryFace(ctx: *ExecContext, al: std.mem.Allocator, file: *const gguf.File, src: *const image.Image) !?detect.Detection {
    var det_model = scrfd.Model.init(al, file);
    defer det_model.deinit();
    return primaryFaceWith(ctx, al, &det_model, src);
}

pub fn primaryFaceWith(ctx: *ExecContext, al: std.mem.Allocator, det_model: *scrfd.Model, src: *const image.Image) !?detect.Detection {
    const dets = try detect_allWith(ctx, al, det_model, src);
    defer al.free(dets);
    if (dets.len == 0) return null;
    var primary = dets[0];
    for (dets) |d| {
        const area = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
        const parea = (primary.box[2] - primary.box[0]) * (primary.box[3] - primary.box[1]);
        if (area > parea) primary = d;
    }
    return primary;
}

/// Detect all faces (source image → letterbox → SCRFD → decode + NMS).
pub fn detect_all(ctx: *ExecContext, al: std.mem.Allocator, file: *const gguf.File, src: *const image.Image) ![]detect.Detection {
    var det_model = scrfd.Model.init(al, file);
    defer det_model.deinit();
    return detect_allWith(ctx, al, &det_model, src);
}

pub fn detect_allWith(ctx: *ExecContext, al: std.mem.Allocator, det_model: *scrfd.Model, src: *const image.Image) ![]detect.Detection {
    var lb = try preprocess.letterbox(al, src, 640);
    defer lb.img.deinit();
    var heads = try scrfd.forwardImageWith(ctx, al, det_model, &lb.img);
    defer heads.deinit(al);
    const st: f32 = @floatCast(det_model.file.getFloat("facedetect.detector.score_thresh") orelse 0.5);
    const nt: f32 = @floatCast(det_model.file.getFloat("facedetect.detector.nms_thresh") orelse 0.4);
    return detect.decode(al, &heads, lb.det_scale, st, nt);
}

/// L2-normalize in place (insightface `normed_embedding`).
pub fn l2normalize(v: []f32) void {
    var ss: f64 = 0;
    for (v) |x| ss += @as(f64, x) * @as(f64, x);
    const inv: f32 = @floatCast(1.0 / @sqrt(if (ss > 0) ss else 1.0));
    for (v) |*x| x.* *= inv;
}

/// Full embed: source image → detect → align → ArcFace → raw 512-d feature vector.
pub fn embed(ctx: *ExecContext, al: std.mem.Allocator, file: *const gguf.File, src: *const image.Image) ![]f32 {
    var det_model = scrfd.Model.init(al, file);
    defer det_model.deinit();
    var rec_model = try rec.Model.load(ctx, al, file);
    defer rec_model.deinit();
    return embedWith(ctx, al, &det_model, &rec_model, src);
}

pub fn embedWith(ctx: *ExecContext, al: std.mem.Allocator, det_model: *scrfd.Model, rec_model: *const rec.Model, src: *const image.Image) ![]f32 {
    const primary = (try primaryFaceWith(ctx, al, det_model, src)) orelse return error.NoFace;
    var lmk: [5][2]f64 = undefined;
    for (0..5) |k| {
        lmk[k][0] = primary.kps[k][0];
        lmk[k][1] = primary.kps[k][1];
    }
    const crop_pixels = try align_mod.normCrop(al, src, lmk, 112);
    defer al.free(crop_pixels);
    var crop_img = image.Image{ .allocator = al, .width = 112, .height = 112, .pixels = crop_pixels };
    return rec_model.embedImage(ctx, al, &crop_img);
}

/// Full analyze: source image → detect → largest face → 96² box crop → genderage.
pub fn analyze(ctx: *ExecContext, al: std.mem.Allocator, file: *const gguf.File, src: *const image.Image) !genderage.Result {
    var det_model = scrfd.Model.init(al, file);
    defer det_model.deinit();
    var ga_model = genderage.Model.init(al, file);
    defer ga_model.deinit();
    return analyzeWith(ctx, al, &det_model, &ga_model, src);
}

pub fn analyzeWith(ctx: *ExecContext, al: std.mem.Allocator, det_model: *scrfd.Model, ga_model: *genderage.Model, src: *const image.Image) !genderage.Result {
    const primary = (try primaryFaceWith(ctx, al, det_model, src)) orelse return error.NoFace;
    const crop_pixels = try align_mod.centerScaleCrop(al, src, primary.box, 96, 1.5);
    defer al.free(crop_pixels);
    var crop_img = image.Image{ .allocator = al, .width = 96, .height = 96, .pixels = crop_pixels };
    return genderage.analyzeImageWith(ctx, al, ga_model, &crop_img);
}

pub fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}
