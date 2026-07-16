//! Inkling multimodal projector (`clip` GGUF, projector_type "inkling"):
//! the hMLP vision stem and the dMel audio tower, plus their exact
//! preprocessing pipelines. Reference: llama.cpp PR #25731 @ 1cb0374
//! (tools/mtmd/models/inkling.cpp, mtmd-image.cpp, mtmd-audio.cpp), with
//! the dMel-width ref patch tools/ref-patches/llama.cpp-inkling-dmel-nembd.patch.
//!
//! Vision: an image is upscaled 2x (capped at a 2048 long edge) with a
//! byte-exact port of Pillow's two-pass fixed-point resampling (Lanczos,
//! support 3), normalized through a bf16 round-trip, split into 40x40
//! patches row-major WITH an always-present extra right-hand column, and
//! each patch runs the hMLP stem: three fold(5,2,4)+linear+rmsnorm+gelu_erf
//! stages, then a temporal-pair linear (the pair is the patch duplicated)
//! and a final rms norm. One patch = one decoder token.
//!
//! Audio: 16 kHz mono samples are left-padded by n_fft-hop (800), right-
//! padded to a whole hop, framed at 1600/800 with a periodic Hann window,
//! transformed with the reference's table-based mixed-radix FFT, reduced to
//! 80 Slaney-normalized mel magnitudes (log10, floors 1e-10), quantized to
//! the nearest of 16 centers over [-7, 2] (ties toward the lower bin), and
//! embedded as the SUM of one row per mel bin from a [80*16, d] table plus
//! a final rms norm. One frame (50 ms) = one decoder token.
//!
//! Embeddings leave the towers final-normed: the decoder must NOT apply its
//! token embedding norm to these rows (see model.zig stepMixed).

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;

pub const Error = weights.Error || error{
    InvalidConfig,
    InvalidImage,
    InvalidAudio,
};

pub const patch_size: usize = 40;
pub const temporal_patch_size: usize = 2;
const spatial_folds = [3]usize{ 5, 2, 4 };

pub const n_mels: usize = 80;
pub const mel_vocab_size: usize = 16;
const audio_n_fft: usize = 1600;
const audio_hop: usize = 800;

pub const MmProj = struct {
    allocator: Allocator,
    // vision
    hmlp_linear: [4]LinearWeight,
    hmlp_in_dims: [4]usize,
    /// x86 only: layer 3 loaded as W3_left + W3_right (f32). The temporal
    /// pair is always the SAME patch duplicated (this port is still-image
    /// only), so W3 x [v; v] == (W3_L + W3_R) x v exactly in structure.
    /// Must become conditional if video (distinct temporal slices) lands.
    s3_folded: bool,
    hmlp_norm: [3][]f32,
    hmlp_dims: [4]usize, // output width of each linear
    hmlp_final_norm: []f32,
    image_mean: [3]f32,
    image_std: [3]f32,
    vision_eps: f32,
    // audio
    dmel_embd: LinearWeight, // [n_mels*mel_vocab_size, n_embd]
    dmel_final_norm: []f32,
    audio_eps: f32,
    /// Decoder hidden width both towers project into.
    n_embd: usize,

    pub fn loadGguf(ctx: *ExecContext, io: std.Io, path: []const u8) !MmProj {
        var file = try gguf.File.loadMmapAuto(ctx.allocator, io, path);
        defer file.deinit();
        return loadGgufFromFile(ctx, &file);
    }

    pub fn loadGgufFromFile(ctx: *ExecContext, file: *gguf.File) !MmProj {
        const allocator = ctx.allocator;
        const arch = file.getString("general.architecture") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, arch, "clip")) return Error.InvalidConfig;
        const vproj = file.getString("clip.vision.projector_type") orelse return Error.InvalidConfig;
        if (!std.mem.eql(u8, vproj, "inkling")) return Error.InvalidConfig;

        const final_norm_info = try file.get("v.hmlp.final_norm.weight");
        if (final_norm_info.n_dims != 1) return Error.InvalidWeightShape;
        const n_embd = final_norm_info.dims[0];

        // The four stem linears; input widths follow the fold chain.
        var hmlp_dims: [4]usize = undefined;
        var hmlp_in_dims: [4]usize = undefined;
        var hmlp_linear: [4]LinearWeight = undefined;
        var name_buf: [64]u8 = undefined;
        var in_dim: usize = 3 * spatial_folds[0] * spatial_folds[0];
        var loaded: usize = 0;
        errdefer for (hmlp_linear[0..loaded]) |*w| w.deinit();
        const fold_s3 = @import("builtin").cpu.arch == .x86_64;
        for (0..4) |l| {
            const name = try std.fmt.bufPrint(&name_buf, "v.hmlp.{d}.linear.weight", .{l});
            const info = try file.get(name);
            if (info.n_dims != 2 or info.dims[0] != in_dim) return Error.InvalidWeightShape;
            const out_dim = info.dims[1];
            if (l == 3 and fold_s3) {
                hmlp_linear[l] = try loadFoldedS3(ctx, info, out_dim, in_dim);
                hmlp_in_dims[l] = in_dim / 2;
            } else {
                hmlp_linear[l] = try loadTowerLinear(ctx, info, out_dim, in_dim);
                hmlp_in_dims[l] = in_dim;
            }
            loaded += 1;
            hmlp_dims[l] = out_dim;
            in_dim = if (l < 2) out_dim * spatial_folds[l + 1] * spatial_folds[l + 1] else out_dim * temporal_patch_size;
        }
        if (hmlp_dims[3] != n_embd) return Error.InvalidWeightShape;

        var hmlp_norm: [3][]f32 = undefined;
        var norms_loaded: usize = 0;
        errdefer for (hmlp_norm[0..norms_loaded]) |n| allocator.free(n);
        for (0..3) |l| {
            const name = try std.fmt.bufPrint(&name_buf, "v.hmlp.{d}.norm.weight", .{l});
            hmlp_norm[l] = try hostVector(allocator, file, name, hmlp_dims[l]);
            norms_loaded += 1;
        }
        const hmlp_final_norm = try hostVector(allocator, file, "v.hmlp.final_norm.weight", n_embd);
        errdefer allocator.free(hmlp_final_norm);

        var dmel_embd = try LinearWeight.load(ctx, try file.get("a.dmel.embedding.weight"), n_mels * mel_vocab_size, n_embd);
        errdefer dmel_embd.deinit();
        const dmel_final_norm = try hostVector(allocator, file, "a.dmel.final_norm.weight", n_embd);
        errdefer allocator.free(dmel_final_norm);

        var image_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
        var image_std = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
        readF32Triple(file, "clip.vision.image_mean", &image_mean);
        readF32Triple(file, "clip.vision.image_std", &image_std);

        const vision_eps: f32 = @floatCast(file.getFloat("clip.vision.attention.layer_norm_epsilon") orelse 1e-6);
        const audio_eps: f32 = @floatCast(file.getFloat("clip.audio.attention.layer_norm_epsilon") orelse 1e-6);

        return .{
            .allocator = allocator,
            .hmlp_linear = hmlp_linear,
            .hmlp_in_dims = hmlp_in_dims,
            .s3_folded = fold_s3,
            .hmlp_norm = hmlp_norm,
            .hmlp_dims = hmlp_dims,
            .hmlp_final_norm = hmlp_final_norm,
            .image_mean = image_mean,
            .image_std = image_std,
            .vision_eps = vision_eps,
            .dmel_embd = dmel_embd,
            .dmel_final_norm = dmel_final_norm,
            .audio_eps = audio_eps,
            .n_embd = n_embd,
        };
    }

    pub fn deinit(self: *MmProj) void {
        self.allocator.free(self.dmel_final_norm);
        self.dmel_embd.deinit();
        self.allocator.free(self.hmlp_final_norm);
        for (&self.hmlp_norm) |n| self.allocator.free(n);
        for (&self.hmlp_linear) |*w| w.deinit();
        self.* = undefined;
    }

    /// Encode preprocessed patches (from `preprocessImage`) into one
    /// embedding row per patch: [n_patches * n_embd], final-normed.
    pub fn visionEncode(self: *const MmProj, ctx: *ExecContext, patches: []const f32, n_patches: usize) ![]f32 {
        const allocator = ctx.allocator;
        const p = patch_size;
        std.debug.assert(patches.len == n_patches * p * p * 3);

        // Stage grids: 40 -> 8 -> 4 -> 1 positions per side.
        var grid: usize = p;
        var chan: usize = 3;
        // Current activations, [n_patches, grid, grid, chan] row-major (y, x).
        var cur = try allocator.dupe(f32, patches);
        // patches layout is [patch][y][x][c] which matches [grid][grid][chan].
        defer allocator.free(cur);

        for (0..3) |l| {
            const s = spatial_folds[l];
            const out_grid = grid / s;
            const in_dim = chan * s * s;
            const n_pos = out_grid * out_grid;
            const rows = n_patches * n_pos;

            // Fold: vector at (x', y') = concat over [h_fold, w_fold, c].
            const folded = try allocator.alloc(f32, rows * in_dim);
            defer allocator.free(folded);
            for (0..n_patches) |pi| {
                const src = cur[pi * grid * grid * chan ..][0 .. grid * grid * chan];
                for (0..out_grid) |gy| {
                    for (0..out_grid) |gx| {
                        const dst = folded[((pi * n_pos) + gy * out_grid + gx) * in_dim ..][0..in_dim];
                        var o: usize = 0;
                        for (0..s) |hf| {
                            for (0..s) |wf| {
                                const base = ((gy * s + hf) * grid + (gx * s + wf)) * chan;
                                @memcpy(dst[o ..][0..chan], src[base..][0..chan]);
                                o += chan;
                            }
                        }
                    }
                }
            }

            const out = try self.stemLinearNormGelu(ctx, folded, rows, in_dim, l);
            allocator.free(cur);
            cur = out;
            grid = out_grid;
            chan = self.hmlp_dims[l];
        }

        // grid == 1: one d2 vector per patch; temporal pair = duplicate.
        std.debug.assert(grid == 1);
        const d2 = chan;
        var cat: []f32 = &.{};
        defer if (cat.len > 0) allocator.free(cat);
        var s3_in: []const f32 = cur;
        var s3_in_dim = d2;
        if (!self.s3_folded) {
            cat = try allocator.alloc(f32, n_patches * d2 * temporal_patch_size);
            for (0..n_patches) |pi| {
                const v = cur[pi * d2 ..][0..d2];
                @memcpy(cat[pi * d2 * 2 ..][0..d2], v);
                @memcpy(cat[pi * d2 * 2 + d2 ..][0..d2], v);
            }
            s3_in = cat;
            s3_in_dim = d2 * temporal_patch_size;
        }

        var out_opt: ?[]f32 = null;
        if (n_patches <= flip_max_rows) {
            switch (self.hmlp_linear[3]) {
                .f32 => |*wt| out_opt = try flippedLinear(ctx, wt, self.n_embd, s3_in_dim, s3_in, n_patches),
                else => {},
            }
        }
        if (out_opt == null) {
            var cat_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ n_patches, s3_in_dim }, s3_in);
            defer cat_t.deinit();
            var out_t = try self.hmlp_linear[3].linearSeq(ctx, &cat_t, .embed, .attn);
            defer out_t.deinit();
            out_opt = try allocator.dupe(f32, try out_t.dataConst());
        }

        const out = out_opt.?;
        errdefer allocator.free(out);
        for (0..n_patches) |pi| {
            const row = out[pi * self.n_embd ..][0..self.n_embd];
            rmsNormInto(row, row, self.hmlp_final_norm, self.vision_eps);
        }
        return out;
    }

    /// Small-batch linear over the worker team: X [rows, in] x W^T, with
    /// the output dimension as the parallel axis. Tasks are blocks of
    /// output units; each streams its weight rows once and feeds one
    /// accumulation chain per input row. Same dot products as `linearSeq`,
    /// kernel-order tier numerics.
    fn flippedLinear(ctx: *ExecContext, w_t: *const weights.WeightF32, out_dim: usize, in_dim: usize, x: []const f32, rows: usize) ![]f32 {
        const allocator = ctx.allocator;
        const w = try w_t.dataConst();
        std.debug.assert(w.len == out_dim * in_dim);
        const out = try allocator.alloc(f32, rows * out_dim);
        errdefer allocator.free(out);

        const block = 32;
        const n_tasks = (out_dim + block - 1) / block;
        const tasks = try allocator.alloc(FlipTask, n_tasks);
        defer allocator.free(tasks);
        for (tasks, 0..) |*t, i| {
            t.* = .{
                .w = w,
                .x = x,
                .out = out,
                .in_dim = in_dim,
                .out_dim = out_dim,
                .rows = rows,
                .j0 = i * block,
                .j1 = @min((i + 1) * block, out_dim),
            };
        }
        if (ctx.workPool()) |pool| {
            pool.parallelChunks(FlipTask, tasks, FlipTask.run);
        } else {
            for (tasks) |*t| FlipTask.run(t);
        }
        return out;
    }

    const FlipTask = struct {
        w: []const f32,
        x: []const f32,
        out: []f32,
        in_dim: usize,
        out_dim: usize,
        rows: usize,
        j0: usize,
        j1: usize,

        fn run(t: *const FlipTask) void {
            var r0: usize = 0;
            while (r0 < t.rows) : (r0 += 8) {
                switch (@min(8, t.rows - r0)) {
                    inline 1, 2, 3, 4, 5, 6, 7, 8 => |rb| t.runBlock(rb, r0),
                    else => unreachable,
                }
            }
        }

        /// RB-row register tile: the weight vector loads once per k-chunk
        /// and feeds RB independent accumulation chains, one per input row.
        fn runBlock(t: *const FlipTask, comptime RB: usize, r0: usize) void {
            const V = @Vector(8, f32);
            var xs: [RB][]const f32 = undefined;
            inline for (0..RB) |ri| xs[ri] = t.x[(r0 + ri) * t.in_dim ..][0..t.in_dim];
            for (t.j0..t.j1) |j| {
                const w_row = t.w[j * t.in_dim ..][0..t.in_dim];
                var acc: [RB]V = @splat(@as(V, @splat(0)));
                var k: usize = 0;
                while (k + 8 <= t.in_dim) : (k += 8) {
                    const wf: V = w_row[k..][0..8].*;
                    inline for (0..RB) |ri| {
                        acc[ri] += wf * @as(V, xs[ri][k..][0..8].*);
                    }
                }
                inline for (0..RB) |ri| {
                    var sum = @reduce(.Add, acc[ri]);
                    var kk = k;
                    while (kk < t.in_dim) : (kk += 1) {
                        sum += w_row[kk] * xs[ri][kk];
                    }
                    t.out[(r0 + ri) * t.out_dim + j] = sum;
                }
            }
        }
    };

    const flip_max_rows: usize = 64;

    fn stemLinearNormGelu(self: *const MmProj, ctx: *ExecContext, rows_in: []const f32, rows: usize, in_dim: usize, l: usize) ![]f32 {
        const allocator = ctx.allocator;
        const d = self.hmlp_dims[l];

        var lin_owned: ?[]f32 = null;
        defer if (lin_owned) |slice| allocator.free(slice);
        var lin_t_opt: ?fucina.Tensor(.{ .seq, .attn }) = null;
        defer if (lin_t_opt) |*t| t.deinit();

        const lin: []const f32 = blk: {
            if (rows <= flip_max_rows) {
                switch (self.hmlp_linear[l]) {
                    .f32 => |*wt| {
                        lin_owned = try flippedLinear(ctx, wt, d, in_dim, rows_in, rows);
                        break :blk lin_owned.?;
                    },
                    else => {},
                }
            }
            var in_t = try fucina.Tensor(.{ .seq, .embed }).fromSlice(ctx, .{ rows, in_dim }, rows_in);
            defer in_t.deinit();
            lin_t_opt = try self.hmlp_linear[l].linearSeq(ctx, &in_t, .embed, .attn);
            break :blk try lin_t_opt.?.dataConst();
        };

        // Per-row rms-norm + gelu_erf, fanned out over the worker team.
        const out = try allocator.alloc(f32, rows * d);
        errdefer allocator.free(out);
        const block = 16;
        const n_tasks = (rows + block - 1) / block;
        const tasks = try allocator.alloc(NormGeluTask, n_tasks);
        defer allocator.free(tasks);
        for (tasks, 0..) |*t, i| {
            t.* = .{
                .lin = lin,
                .out = out,
                .norm_w = self.hmlp_norm[l],
                .eps = self.vision_eps,
                .d = d,
                .r0 = i * block,
                .r1 = @min((i + 1) * block, rows),
            };
        }
        if (ctx.workPool()) |pool| {
            pool.parallelChunks(NormGeluTask, tasks, NormGeluTask.run);
        } else {
            for (tasks) |*t| NormGeluTask.run(t);
        }
        return out;
    }

    /// Row block of the stem epilogue: rms-norm then exact-erf GELU, the
    /// same formula as the fucina `.gelu_erf` unary (vendored-musl erff).
    const NormGeluTask = struct {
        lin: []const f32,
        out: []f32,
        norm_w: []const f32,
        eps: f32,
        d: usize,
        r0: usize,
        r1: usize,

        fn run(t: *const NormGeluTask) void {
            const erff = fucina.internal.backend_mod.ops.erff;
            for (t.r0..t.r1) |r| {
                const dst = t.out[r * t.d ..][0..t.d];
                rmsNormInto(dst, t.lin[r * t.d ..][0..t.d], t.norm_w, t.eps);
                for (dst) |*v| {
                    v.* = 0.5 * v.* * (1 + erff(v.* * 0.70710678118654752440084436210484));
                }
            }
        }
    };

    /// Encode quantized dMel frames (from `preprocessAudio`, one row of
    /// n_mels bin indices per frame) into one embedding row per frame.
    /// All frames' table rows resolve in ONE batched lookup; the per-frame
    /// sum stays in mel-bin order (matches the reference's sequential sum).
    pub fn audioEncode(self: *const MmProj, ctx: *ExecContext, dmel: []const u8, n_frames: usize) ![]f32 {
        const allocator = ctx.allocator;
        std.debug.assert(dmel.len == n_frames * n_mels);

        const ids = try allocator.alloc(usize, n_frames * n_mels);
        defer allocator.free(ids);
        for (0..n_frames) |f| {
            for (0..n_mels) |b| {
                ids[f * n_mels + b] = b * mel_vocab_size + dmel[f * n_mels + b];
            }
        }
        var rows_t = try self.dmel_embd.getRowsAs(ctx, ids, .embed);
        defer rows_t.deinit();
        const rows = try rows_t.dataConst();

        const out = try allocator.alloc(f32, n_frames * self.n_embd);
        errdefer allocator.free(out);
        for (0..n_frames) |f| {
            const dst = out[f * self.n_embd ..][0..self.n_embd];
            const base = f * n_mels;
            @memcpy(dst, rows[base * self.n_embd ..][0..self.n_embd]);
            for (1..n_mels) |b| {
                vecAdd(dst, rows[(base + b) * self.n_embd ..][0..self.n_embd]);
            }
            rmsNormInto(dst, dst, self.dmel_final_norm, self.audio_eps);
        }
        return out;
    }
};

pub const ImagePatches = struct {
    allocator: Allocator,
    /// [n_patches][40][40][3] normalized f32 (bf16-rounded), row-major
    /// patch order (rows outer, the extra right column included).
    data: []f32,
    patch_rows: usize,
    patch_cols: usize,

    pub fn nPatches(self: *const ImagePatches) usize {
        return self.patch_rows * self.patch_cols;
    }

    pub fn deinit(self: *ImagePatches) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

/// Full Inkling image preprocessing on 8-bit RGB pixels (row-major
/// [h][w][3]): 2x Pillow-Lanczos upscale (2048 long-edge cap), bf16-rounded
/// normalization, 40x40 patchify with the extra right column.
pub fn preprocessImage(allocator: Allocator, mm: *const MmProj, rgb: []const u8, width: usize, height: usize) !ImagePatches {
    if (width == 0 or height == 0 or rgb.len != width * height * 3) return Error.InvalidImage;

    // Target size: float math per the reference preprocessor.
    const long_edge: f32 = @floatFromInt(@max(width, height));
    const target_long: f32 = @min(long_edge * 2.0, @max(long_edge, 2048.0));
    const ratio: f32 = target_long / long_edge;
    const out_w: usize = @intCast(@max(1, @as(i64, @intFromFloat(@floor(@as(f32, @floatFromInt(width)) * ratio + 0.5)))));
    const out_h: usize = @intCast(@max(1, @as(i64, @intFromFloat(@floor(@as(f32, @floatFromInt(height)) * ratio + 0.5)))));

    const resized = try pillowResizeLanczos(allocator, rgb, width, height, out_w, out_h);
    defer allocator.free(resized);

    const p = patch_size;
    const patch_rows = (out_h + p - 1) / p;
    const patch_cols = out_w / p + 1; // extra right column, always
    const n_patches = patch_rows * patch_cols;

    // Normalization constants; pad pixels use raw = -1/255.
    var pad_norm: [3]f32 = undefined;
    for (0..3) |c| {
        pad_norm[c] = bf16Round((-1.0 / 255.0 - mm.image_mean[c]) / mm.image_std[c]);
    }

    const data = try allocator.alloc(f32, n_patches * p * p * 3);
    errdefer allocator.free(data);
    for (0..patch_rows) |py| {
        for (0..patch_cols) |px| {
            const patch = data[(py * patch_cols + px) * p * p * 3 ..][0 .. p * p * 3];
            for (0..p) |y| {
                const iy = py * p + y;
                for (0..p) |x| {
                    const ix = px * p + x;
                    const off = (y * p + x) * 3;
                    if (iy < out_h and ix < out_w) {
                        for (0..3) |c| {
                            const raw: f32 = @as(f32, @floatFromInt(resized[(iy * out_w + ix) * 3 + c])) / 255.0;
                            patch[off + c] = bf16Round((raw - mm.image_mean[c]) / mm.image_std[c]);
                        }
                    } else {
                        @memcpy(patch[off..][0..3], &pad_norm);
                    }
                }
            }
        }
    }

    return .{ .allocator = allocator, .data = data, .patch_rows = patch_rows, .patch_cols = patch_cols };
}

/// ggml_fp32_to_bf16 (round-to-nearest-even) then back to f32.
fn bf16Round(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    if ((bits & 0x7fffffff) > 0x7f800000) { // NaN: quiet, truncate
        return @bitCast((bits >> 16 << 16) | 0x00400000 << 0);
    }
    const r: u32 = (bits + (0x7fff + ((bits >> 16) & 1))) >> 16;
    return @bitCast(r << 16);
}

/// Byte-exact port of the reference's Pillow-style two-pass fixed-point
/// resampling with the Lanczos filter (support 3). Input/output are 8-bit
/// RGB row-major.
fn pillowResizeLanczos(allocator: Allocator, src: []const u8, src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) ![]u8 {
    const precision_bits: u5 = 32 - 8 - 2;

    const need_h = dst_w != src_w;
    const need_v = dst_h != src_h;
    if (!need_h and !need_v) return allocator.dupe(u8, src);

    var bounds_h: []i32 = &.{};
    var weights_h: []i32 = &.{};
    var ksize_h: usize = 0;
    defer if (bounds_h.len > 0) allocator.free(bounds_h);
    defer if (weights_h.len > 0) allocator.free(weights_h);
    if (need_h) ksize_h = try precomputePillowWeights(allocator, src_w, dst_w, &bounds_h, &weights_h);

    var bounds_v: []i32 = &.{};
    var weights_v: []i32 = &.{};
    var ksize_v: usize = 0;
    defer if (bounds_v.len > 0) allocator.free(bounds_v);
    defer if (weights_v.len > 0) allocator.free(weights_v);
    if (need_v) ksize_v = try precomputePillowWeights(allocator, src_h, dst_h, &bounds_v, &weights_v);

    const half: i32 = @as(i32, 1) << (precision_bits - 1);

    // Horizontal pass.
    var mid: []u8 = undefined;
    var mid_w: usize = src_w;
    if (need_h) {
        mid = try allocator.alloc(u8, dst_w * src_h * 3);
        mid_w = dst_w;
        for (0..src_h) |yy| {
            for (0..dst_w) |xx| {
                const xmin: usize = @intCast(bounds_h[xx * 2 + 0]);
                const xcnt: usize = @intCast(bounds_h[xx * 2 + 1]);
                var ss = [3]i32{ half, half, half };
                for (0..xcnt) |x| {
                    const px = src[(yy * src_w + x + xmin) * 3 ..][0..3];
                    const w = weights_h[xx * ksize_h + x];
                    for (0..3) |c| ss[c] +%= @as(i32, px[c]) *% w;
                }
                for (0..3) |c| mid[(yy * dst_w + xx) * 3 + c] = clip8(ss[c] >> precision_bits);
            }
        }
    } else {
        mid = try allocator.dupe(u8, src);
    }
    defer if (need_v) allocator.free(mid);

    if (!need_v) return mid;

    // Vertical pass.
    const out = try allocator.alloc(u8, mid_w * dst_h * 3);
    for (0..dst_h) |yy| {
        const ymin: usize = @intCast(bounds_v[yy * 2 + 0]);
        const ycnt: usize = @intCast(bounds_v[yy * 2 + 1]);
        for (0..mid_w) |xx| {
            var ss = [3]i32{ half, half, half };
            for (0..ycnt) |y| {
                const px = mid[((y + ymin) * mid_w + xx) * 3 ..][0..3];
                const w = weights_v[yy * ksize_v + y];
                for (0..3) |c| ss[c] +%= @as(i32, px[c]) *% w;
            }
            for (0..3) |c| out[(yy * mid_w + xx) * 3 + c] = clip8(ss[c] >> precision_bits);
        }
    }
    return out;
}

fn clip8(v: i32) u8 {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return @intCast(v);
}

fn lanczosFilter(x: f64) f64 {
    if (-3.0 <= x and x < 3.0) {
        return sinc(x) * sinc(x / 3.0);
    }
    return 0.0;
}

fn sinc(v: f64) f64 {
    if (v == 0.0) return 1.0;
    const pix = v * 3.141592653589793238462643383279502884;
    return @sin(pix) / pix;
}

fn precomputePillowWeights(allocator: Allocator, in_size: usize, out_size: usize, bounds: *[]i32, weights_out: *[]i32) !usize {
    const precision_bits = 32 - 8 - 2;
    const filter_support: f64 = 3.0; // Lanczos

    const scale: f64 = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale: f64 = @max(scale, 1.0);
    const support = filter_support * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;

    const pre = try allocator.alloc(f64, out_size * ksize);
    defer allocator.free(pre);
    const b = try allocator.alloc(i32, out_size * 2);
    errdefer allocator.free(b);

    for (0..out_size) |xx| {
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        var ww: f64 = 0.0;
        const ss = 1.0 / filterscale;

        var xmin: i64 = @intFromFloat(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        var xmax: i64 = @intFromFloat(center + support + 0.5);
        if (xmax > @as(i64, @intCast(in_size))) xmax = @intCast(in_size);
        xmax -= xmin;

        var x: usize = 0;
        while (x < @as(usize, @intCast(xmax))) : (x += 1) {
            const w = lanczosFilter((@as(f64, @floatFromInt(@as(i64, @intCast(x)) + xmin)) - center + 0.5) * ss);
            pre[xx * ksize + x] = w;
            ww += w;
        }
        x = 0;
        while (x < @as(usize, @intCast(xmax))) : (x += 1) {
            if (ww != 0.0) pre[xx * ksize + x] /= ww;
        }
        x = @intCast(xmax);
        while (x < ksize) : (x += 1) pre[xx * ksize + x] = 0;

        b[xx * 2 + 0] = @intCast(xmin);
        b[xx * 2 + 1] = @intCast(xmax);
    }

    const w_out = try allocator.alloc(i32, out_size * ksize);
    errdefer allocator.free(w_out);
    const fxp_scale = std.math.ldexp(@as(f64, 1.0), precision_bits);
    for (0..out_size * ksize) |i| {
        // Pillow adds +/-0.5 then truncates toward zero.
        const rounded = pre[i] * fxp_scale + (if (pre[i] < 0) @as(f64, -0.5) else @as(f64, 0.5));
        w_out[i] = @intFromFloat(rounded);
    }

    bounds.* = b;
    weights_out.* = w_out;
    return ksize;
}

pub const DmelFrames = struct {
    allocator: Allocator,
    /// [n_frames][n_mels] quantized bin indices (0..15).
    data: []u8,
    n_frames: usize,

    pub fn deinit(self: *DmelFrames) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

/// Full Inkling audio preprocessing on 16 kHz mono f32 samples.
pub fn preprocessAudio(allocator: Allocator, samples: []const f32) !DmelFrames {
    if (samples.len == 0) return Error.InvalidAudio;

    // Left pad by n_fft - hop, right pad to a whole hop.
    const left_pad = audio_n_fft - audio_hop;
    const right_pad = (audio_hop - samples.len % audio_hop) % audio_hop;
    const n_padded = left_pad + samples.len + right_pad;
    const padded = try allocator.alloc(f32, n_padded);
    defer allocator.free(padded);
    @memset(padded[0..left_pad], 0);
    @memcpy(padded[left_pad..][0..samples.len], samples);
    @memset(padded[left_pad + samples.len ..], 0);

    // Reference caches (sin/cos twiddles, periodic Hann, Slaney filterbank)
    // are pure constants — built once into globals.
    ensureDspTables();
    const sin_vals: []const f32 = &dsp_sin;
    const cos_vals: []const f32 = &dsp_cos;
    const hann: []const f32 = &dsp_hann;
    const filters: []const f32 = &dsp_filters;

    const n_fft_bins = audio_n_fft / 2 + 1;
    const n_frames = (n_padded - audio_n_fft) / audio_hop + 1;

    const out = try allocator.alloc(u8, n_frames * n_mels);
    errdefer allocator.free(out);

    const fft_in = try allocator.alloc(f32, audio_n_fft * 2);
    defer allocator.free(fft_in);
    const fft_out = try allocator.alloc(f32, audio_n_fft * 2 * 2 * 2);
    defer allocator.free(fft_out);

    for (0..n_frames) |fi| {
        const offset = fi * audio_hop;
        const valid = @min(audio_n_fft, n_padded - offset);
        for (0..valid) |j| fft_in[j] = hann[j] * padded[offset + j];
        if (valid < audio_n_fft) @memset(fft_in[valid..audio_n_fft], 0);

        fftReal(sin_vals, cos_vals, fft_in, audio_n_fft, fft_out);

        // Magnitude with the power floor (reference clamps |X|^2 before sqrt).
        for (0..n_fft_bins) |j| {
            const re = fft_out[2 * j];
            const im = fft_out[2 * j + 1];
            const power = @max(re * re + im * im, 1e-10);
            fft_out[j] = @sqrt(power);
        }

        for (0..n_mels) |m| {
            // Reference sums groups of four in f32 and accumulates the
            // groups in f64 (the unrolled loop's C arithmetic).
            var sum: f64 = 0.0;
            var k: usize = 0;
            while (k + 3 < n_fft_bins) : (k += 4) {
                const idx = m * n_fft_bins + k;
                const group: f32 =
                    fft_out[k + 0] * filters[idx + 0] +
                    fft_out[k + 1] * filters[idx + 1] +
                    fft_out[k + 2] * filters[idx + 2] +
                    fft_out[k + 3] * filters[idx + 3];
                sum += group;
            }
            while (k < n_fft_bins) : (k += 1) {
                sum += fft_out[k] * filters[m * n_fft_bins + k];
            }
            sum = @max(sum, 1e-10);
            const logmel: f32 = @floatCast(std.math.log10(sum));

            // dMel: nearest of 16 f64 centers over [-7, 2]; strict '<'
            // keeps the lower bin on midpoints (torch.argmin).
            out[fi * n_mels + m] = quantizeDmel(logmel);
        }
    }

    return .{ .allocator = allocator, .data = out, .n_frames = n_frames };
}

// DSP constant tables, built once on first use. The C reference computes
// double angles but calls the FLOAT libm entry points (sinf/cosf receive
// the f32-converted argument); the Hann outer arithmetic is double.
var dsp_state = std.atomic.Value(u8).init(0); // 0 = uninit, 1 = building, 2 = ready
var dsp_sin: [audio_n_fft]f32 = undefined;
var dsp_cos: [audio_n_fft]f32 = undefined;
var dsp_hann: [audio_n_fft]f32 = undefined;
var dsp_filters: [n_mels * (audio_n_fft / 2 + 1)]f32 = undefined;

fn ensureDspTables() void {
    if (dsp_state.load(.acquire) == 2) return;
    if (dsp_state.cmpxchgStrong(0, 1, .acquire, .acquire) != null) {
        // Another thread is building: spin until the tables are ready.
        while (dsp_state.load(.acquire) != 2) std.atomic.spinLoopHint();
        return;
    }
    initDspTables();
    dsp_state.store(2, .release);
}

fn initDspTables() void {
    for (0..audio_n_fft) |i| {
        const theta: f64 = (2.0 * std.math.pi * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(audio_n_fft));
        const theta32: f32 = @floatCast(theta);
        dsp_sin[i] = @sin(theta32);
        dsp_cos[i] = @cos(theta32);
        const c32: f32 = @cos(theta32);
        dsp_hann[i] = @floatCast(0.5 * (1.0 - @as(f64, c32)));
    }
    fillSlaneyFilterbank(&dsp_filters);
}

// Slaney mel scale (librosa default) constants.
const slaney_min_log_hz = 1000.0;
const slaney_lin_slope = 3.0 / 200.0;
const slaney_min_log_mel = slaney_min_log_hz * slaney_lin_slope;

fn slaneyHzToMel(hz: f64) f64 {
    const log_step = @log(6.4) / 27.0;
    return if (hz < slaney_min_log_hz) hz * slaney_lin_slope else slaney_min_log_mel + @log(hz / slaney_min_log_hz) / log_step;
}

fn slaneyMelToHz(m: f64) f64 {
    const log_step = @log(6.4) / 27.0;
    return if (m < slaney_min_log_mel) m / slaney_lin_slope else slaney_min_log_hz * @exp((m - slaney_min_log_mel) * log_step);
}

/// Slaney-scale mel filterbank (librosa default, area-normalized), built in
/// f64 and stored f32 — the reference's fill_mel_filterbank_matrix with
/// fmin 0, fmax sr/2, scale 1, HTK off.
fn slaneyFilterbank(allocator: Allocator) ![]f32 {
    const out = try allocator.alloc(f32, n_mels * (audio_n_fft / 2 + 1));
    fillSlaneyFilterbank(out[0 .. n_mels * (audio_n_fft / 2 + 1)]);
    return out;
}

fn fillSlaneyFilterbank(out: []f32) void {
    const n_fft_bins = audio_n_fft / 2 + 1;
    const sample_rate: f64 = 16000.0;
    const fmax = 0.5 * sample_rate;

    const bin_hz_step = sample_rate / @as(f64, @floatFromInt(audio_n_fft));
    const m_lo = slaneyHzToMel(0.0);
    const m_hi = slaneyHzToMel(fmax);

    var hz_pts: [n_mels + 2]f64 = undefined;
    for (0..n_mels + 2) |i| {
        const mel = m_lo + (m_hi - m_lo) * (@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n_mels + 1)));
        hz_pts[i] = slaneyMelToHz(mel);
    }

    for (0..n_mels) |m| {
        const f_left = hz_pts[m];
        const f_center = hz_pts[m + 1];
        const f_right = hz_pts[m + 2];
        const denom_l = @max(1e-30, f_center - f_left);
        const denom_r = @max(1e-30, f_right - f_center);
        const enorm = 2.0 / @max(1e-30, f_right - f_left); // slaney area norm

        for (0..n_fft_bins) |k| {
            const f = @as(f64, @floatFromInt(k)) * bin_hz_step;
            var w: f64 = 0.0;
            if (f >= f_left and f <= f_center) {
                w = (f - f_left) / denom_l;
            } else if (f > f_center and f <= f_right) {
                w = (f_right - f) / denom_r;
            }
            out[m * n_fft_bins + k] = @floatCast(w * enorm);
        }
    }
}

/// The reference's table-based mixed-radix FFT for real input (whisper.cpp
/// lineage): radix-2 recursion, odd sizes fall back to the O(N^2) DFT, all
/// arithmetic f32 with shared-table twiddles. `in` must have 2*N capacity
/// (the tail past N is recursion scratch, exactly like the C original);
/// `out` must have 8*N capacity (result in the first 2*N floats).
fn fftReal(sin_vals: []const f32, cos_vals: []const f32, in: []f32, n: usize, out: []f32) void {
    const n_vals = sin_vals.len;
    if (n == 1) {
        out[0] = in[0];
        out[1] = 0.0;
        return;
    }
    const half = n / 2;
    if (n % 2 == 1) {
        dftReal(sin_vals, cos_vals, in[0..n], out);
        return;
    }

    // even = in + N (scratch); even_fft = out + 2N; the recursion's own
    // scratch extends past those, capacity-checked by the slice bounds.
    for (0..half) |i| in[n + i] = in[2 * i];
    fftReal(sin_vals, cos_vals, in[n..], half, out[2 * n ..]);

    // odd reuses the same scratch region after the even recursion returns.
    for (0..half) |i| in[n + i] = in[2 * i + 1];
    fftReal(sin_vals, cos_vals, in[n..], half, out[2 * n + n ..]);

    const even_fft = out[2 * n ..];
    const odd_fft = out[2 * n + n ..];

    const step = n_vals / n;
    for (0..half) |k| {
        const idx = k * step;
        const re = cos_vals[idx];
        const im = -sin_vals[idx];

        const re_odd = odd_fft[2 * k];
        const im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

fn dftReal(sin_vals: []const f32, cos_vals: []const f32, in: []const f32, out: []f32) void {
    const n = in.len;
    const n_vals = sin_vals.len;
    const step = n_vals / n;
    for (0..n) |k| {
        var re: f32 = 0;
        var im: f32 = 0;
        for (0..n) |i| {
            const idx = (k * i * step) % n_vals;
            re += in[i] * cos_vals[idx];
            im += -in[i] * sin_vals[idx];
        }
        out[2 * k + 0] = re;
        out[2 * k + 1] = im;
    }
}

/// Fold the temporal-pair halves of the s3 weight into one f32 matrix:
/// out[j][i] = W3[j][i] + W3[j][i + in/2] (exact restructuring for the
/// duplicated still-image pair; see MmProj.s3_folded).
fn loadFoldedS3(ctx: *ExecContext, info: *const gguf.TensorInfo, out_dim: usize, in_dim: usize) !LinearWeight {
    const allocator = ctx.allocator;
    const half = in_dim / 2;
    const folded = try allocator.alloc(f32, out_dim * half);
    defer allocator.free(folded);
    const readW = struct {
        fn f(inf: *const gguf.TensorInfo, idx: usize) f32 {
            return switch (inf.ggml_type) {
                .f32 => @bitCast(std.mem.readInt(u32, inf.data[idx * 4 ..][0..4], .little)),
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, inf.data[idx * 2 ..][0..2], .little)) << 16),
                else => unreachable,
            };
        }
    }.f;
    if (info.ggml_type != .f32 and info.ggml_type != .bf16) return Error.UnsupportedWeightType;
    for (0..out_dim) |j| {
        for (0..half) |i| {
            folded[j * half + i] = readW(info, j * in_dim + i) + readW(info, j * in_dim + half + i);
        }
    }
    var f32_info = info.*;
    f32_info.ggml_type = .f32;
    f32_info.dims[0] = half;
    f32_info.data = std.mem.sliceAsBytes(folded);
    return LinearWeight.load(ctx, &f32_info, out_dim, half);
}

/// Load a tower linear. On x86-64 the bf16 GEMM arm has no AVX-512-BF16
/// path, so bf16 stem weights materialize as f32 up front (an exact
/// conversion) and both the batch path and the flipped small-batch path
/// use the f32 kernels; aarch64 keeps resident bf16.
fn loadTowerLinear(ctx: *ExecContext, info: *const gguf.TensorInfo, out_dim: usize, in_dim: usize) !LinearWeight {
    const materialize_bf16 = @import("builtin").cpu.arch == .x86_64;
    if (info.ggml_type != .bf16 or !materialize_bf16) {
        return LinearWeight.load(ctx, info, out_dim, in_dim);
    }
    const allocator = ctx.allocator;
    const len = out_dim * in_dim;
    if (info.data.len != len * 2) return Error.InvalidWeightShape;
    const f32_vals = try allocator.alloc(f32, len);
    defer allocator.free(f32_vals);
    for (f32_vals, 0..) |*dst, i| {
        const bits: u32 = @as(u32, std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little)) << 16;
        dst.* = @bitCast(bits);
    }
    var f32_info = info.*;
    f32_info.ggml_type = .f32;
    f32_info.data = std.mem.sliceAsBytes(f32_vals);
    return LinearWeight.load(ctx, &f32_info, out_dim, in_dim);
}

fn hostVector(allocator: Allocator, file: *const gguf.File, tensor_name: []const u8, expected: usize) ![]f32 {
    const info = try file.get(tensor_name);
    if (info.n_dims != 1 or info.dims[0] != expected) return Error.InvalidWeightShape;
    const out = try allocator.alloc(f32, expected);
    errdefer allocator.free(out);
    try weights.fillF32(out, info);
    return out;
}

fn readF32Triple(file: *const gguf.File, key: []const u8, out: *[3]f32) void {
    const arr = file.getArray(key) orelse return;
    if (arr.len != 3) return;
    // item_type 6 = f32 in GGUF metadata encoding.
    if (arr.item_type != 6) return;
    for (0..3) |i| {
        const bits = std.mem.readInt(u32, arr.data[i * 4 ..][0..4], .little);
        out[i] = @bitCast(bits);
    }
}

/// acc += v, 8-lane SIMD with scalar tail.
fn vecAdd(acc: []f32, v: []const f32) void {
    const V = @Vector(8, f32);
    var i: usize = 0;
    while (i + 8 <= acc.len) : (i += 8) {
        const r: V = @as(V, acc[i..][0..8].*) + @as(V, v[i..][0..8].*);
        acc[i..][0..8].* = r;
    }
    while (i < acc.len) : (i += 1) acc[i] += v[i];
}

fn rmsNormInto(out: []f32, x: []const f32, weight: []const f32, eps: f32) void {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    const inv = 1.0 / @sqrt(sum / @as(f64, @floatFromInt(x.len)) + eps);
    for (out, x, weight) |*o, v, w| o.* = @floatCast(@as(f64, v) * inv * w);
}

/// dMel quantizer used by preprocessAudio, exposed for tests.
fn quantizeDmel(logmel: f32) u8 {
    const value = @max(-7.0, @min(2.0, @as(f64, logmel)));
    var best: u8 = 0;
    var best_dist = std.math.inf(f64);
    for (0..mel_vocab_size) |bin| {
        const center = -7.0 + 9.0 * @as(f64, @floatFromInt(bin)) / 15.0;
        const dist = @abs(value - center);
        if (dist < best_dist) {
            best = @intCast(bin);
            best_dist = dist;
        }
    }
    return best;
}

const Self = @This();

/// Internal hooks for mmproj_tests.zig only.
pub const testing = struct {
    pub const quantizeDmel = Self.quantizeDmel;
    pub const bf16Round = Self.bf16Round;
    pub const pillowResizeLanczos = Self.pillowResizeLanczos;
    pub const slaneyFilterbank = Self.slaneyFilterbank;
};

test {
    _ = @import("mmproj_tests.zig");
}
