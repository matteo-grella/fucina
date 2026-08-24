//! End-to-end LocateAnything engine: image -> preprocess -> vision tower ->
//! projector -> prompt splice -> LM decode (slow / hybrid / fast) -> boxes.
//!
//! Orchestration mirrors refs/locate-anything.cpp/src/engine.cpp and the
//! decode loops in src/lm.cpp `decode_greedy_resident` / `decode_hybrid`
//! (including the early-stop heuristics, off for the parity gates).

const std = @import("std");
const fucina = @import("fucina");
const config_mod = @import("config.zig");
const tokenizer_mod = @import("tokenizer.zig");
const preproc_mod = @import("preproc.zig");
const vit_mod = @import("vit.zig");
const lm_mod = @import("lm.zig");
const mtp = @import("mtp.zig");
const boxes_mod = @import("boxes.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const Config = config_mod.Config;

pub const Mode = enum { hybrid, slow, fast };

pub const Engine = struct {
    allocator: Allocator,
    ctx: *ExecContext,
    config: Config,
    tok: tokenizer_mod.Tokenizer,
    tok_ids: mtp.TokenIds,
    vit: vit_mod.Vit,
    lm: lm_mod.Lm,

    pub fn load(ctx: *ExecContext, file: *const gguf.File) !Engine {
        const config = try Config.load(file);
        var tok = try tokenizer_mod.Tokenizer.initFromGguf(ctx.allocator, file);
        errdefer tok.deinit();
        var vit = try vit_mod.Vit.load(ctx, file, config);
        errdefer vit.deinit();
        var lm = try lm_mod.Lm.load(ctx, file, config);
        errdefer lm.deinit();
        return .{
            .allocator = ctx.allocator,
            .ctx = ctx,
            .config = config,
            .tok = tok,
            .tok_ids = mtp.TokenIds.fromConfig(config),
            .vit = vit,
            .lm = lm,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.lm.deinit();
        self.vit.deinit();
        self.tok.deinit();
        self.* = undefined;
    }

    pub fn preprocLimits(self: *const Engine) preproc_mod.Limits {
        return .{
            .patch = self.config.vit_patch,
            .merge_h = self.config.vit_merge_h,
            .merge_w = self.config.vit_merge_w,
            .in_token_limit = self.config.in_token_limit,
        };
    }

    /// Vision tower + merge + projector: [n_merged, lm_hidden] host tensor.
    /// Caller frees.
    pub fn projectImage(self: *const Engine, pre: *const preproc_mod.Preprocessed) ![]f32 {
        const ctx = self.ctx;
        var vfinal = try self.vit.forward(ctx, pre.pixel_values, pre.gh, pre.gw, null);
        defer vfinal.deinit();
        var merged = try self.vit.mergePatches(ctx, &vfinal, pre.gh, pre.gw);
        defer merged.deinit();
        var projected = try self.vit.project(ctx, &merged);
        defer projected.deinit();

        const n_merged = projected.dim(.seq);
        const out = try self.allocator.alloc(f32, n_merged * self.config.lm_hidden);
        errdefer self.allocator.free(out);
        try projected.copyTo(out);
        return out;
    }

    /// Reference `embed_and_splice`: token embeds with the <IMG_CONTEXT> rows
    /// overwritten in order by the projected vision tokens.
    pub fn embedAndSplice(self: *const Engine, ids: []const u32, projected: []const f32) ![]f32 {
        const hid = self.config.lm_hidden;
        const out = try self.lm.embedTokens(self.allocator, ids);
        errdefer self.allocator.free(out);
        var vi: usize = 0;
        for (ids, 0..) |id, t| {
            if (id == self.config.tok_image) {
                @memcpy(out[t * hid ..][0..hid], projected[vi * hid ..][0..hid]);
                vi += 1;
            }
        }
        if (vi != projected.len / hid) return error.ImageTokenCountMismatch;
        return out;
    }

    /// Greedy AR decode with the resident cache (reference
    /// `decode_greedy_resident`). Returns generated ids (incl. a trailing
    /// im_end when hit). Caller frees.
    pub fn decodeSlow(self: *const Engine, prompt_ids: []const u32, projected: []const f32, max_new: usize) ![]u32 {
        const allocator = self.allocator;
        const ctx = self.ctx;

        var cache = try self.lm.initCache(ctx, prompt_ids.len + max_new + 32);
        defer cache.deinit();

        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);

        const logits = try allocator.alloc(f32, self.config.lm_vocab);
        defer allocator.free(logits);

        const spliced = try self.embedAndSplice(prompt_ids, projected);
        defer allocator.free(spliced);
        try self.lm.forwardCausal(ctx, &cache, spliced, prompt_ids.len, 0, logits);
        var next: u32 = @intCast(mtp.argmaxRow(logits));
        try out.append(allocator, next);

        var step: usize = 1;
        while (step < max_new and next != self.tok_ids.im_end) : (step += 1) {
            const emb = try self.lm.embedTokens(allocator, &.{next});
            defer allocator.free(emb);
            try self.lm.forwardCausal(ctx, &cache, emb, 1, cache.len, logits);
            next = @intCast(mtp.argmaxRow(logits));
            try out.append(allocator, next);
        }
        return out.toOwnedSlice(allocator);
    }

    pub const HybridOptions = struct {
        fast: bool = false,
        early_stop: bool = false,
        /// When set, each MTP round appends its [block, vocab] logits
        /// (position-major) for the per-round parity gates.
        captured_logits: ?*std.ArrayList([]f32) = null,
    };

    /// Parallel Box Decoding (reference `decode_hybrid`): MTP block rounds
    /// with AR fallback (hybrid) or MTP-only (fast). Returns generated ids.
    pub fn decodeHybrid(self: *const Engine, prompt_ids: []const u32, projected: []const f32, max_new: usize, options: HybridOptions) ![]u32 {
        const allocator = self.allocator;
        const ctx = self.ctx;
        const t = self.tok_ids;
        const block = self.config.lm_block_size;
        const prompt_len = prompt_ids.len;

        var cache = try self.lm.initCache(ctx, prompt_len + max_new + 64);
        defer cache.deinit();

        // ---- prefill: whole prompt at pos0=0 -> cache.len = prompt_len ----
        {
            const logits = try allocator.alloc(f32, self.config.lm_vocab);
            defer allocator.free(logits);
            const spliced = try self.embedAndSplice(prompt_ids, projected);
            defer allocator.free(spliced);
            try self.lm.forwardCausal(ctx, &cache, spliced, prompt_len, 0, logits);
        }

        // `generated` = prompt ++ committed tokens (reference keeps one vector).
        var generated: std.ArrayList(u32) = .empty;
        defer generated.deinit(allocator);
        try generated.appendSlice(allocator, prompt_ids);
        var cached_len: usize = prompt_len;
        var st = mtp.HybridState{};

        const logits_block = try allocator.alloc(f32, block * self.config.lm_vocab);
        defer allocator.free(logits_block);

        while (generated.items.len - prompt_len < max_new and !st.terminated) {
            const n_recompute = generated.items.len - cached_len;
            if (st.use_mtp) {
                // block input: embeds of generated[cached_len:] ++ {last, mask x5}
                var block_ids: std.ArrayList(u32) = .empty;
                defer block_ids.deinit(allocator);
                try block_ids.appendSlice(allocator, generated.items[cached_len..]);
                try block_ids.append(allocator, generated.items[generated.items.len - 1]);
                for (0..block - 1) |_| try block_ids.append(allocator, t.text_mask);

                const x_host = try self.lm.embedTokens(allocator, block_ids.items);
                defer allocator.free(x_host);

                cache.len = cached_len;
                try self.lm.forwardMtpBlock(ctx, &cache, x_host, n_recompute, logits_block);
                if (options.captured_logits) |cap| {
                    try cap.append(allocator, try allocator.dupe(f32, logits_block));
                }

                var rows_buf: [16][]const f32 = undefined;
                const rows = rows_buf[0..block];
                for (rows, 0..) |*row, p| row.* = logits_block[p * self.config.lm_vocab ..][0..self.config.lm_vocab];

                // Principled early-stop (reference comment): once the model's
                // own greedy pick for the next frame start is im_end/null,
                // the real detections are done.
                if (options.early_stop) {
                    const a0 = mtp.argmaxRow(rows[0]);
                    if (a0 == t.im_end or a0 == t.null_tok) {
                        st.terminated = true;
                        break;
                    }
                }

                const new_tokens = try mtp.selectNewTokens(allocator, t, rows, 4, options.fast);
                defer allocator.free(new_tokens);
                var pattern = try mtp.handlePattern(allocator, t, new_tokens, options.fast);
                defer pattern.deinit(allocator);
                mtp.hybridMtpStep(&st, &pattern);
                try generated.appendSlice(allocator, pattern.tokens);
                cached_len += n_recompute;
                cache.len = cached_len;
            } else {
                // AR fallback: forward the uncached committed chunk causally.
                const chunk = generated.items[cached_len..];
                const x_host = try self.lm.embedTokens(allocator, chunk);
                defer allocator.free(x_host);
                const logits = logits_block[0..self.config.lm_vocab];
                try self.lm.forwardCausal(ctx, &cache, x_host, chunk.len, cached_len, logits);
                const ar_token: u32 = @intCast(mtp.argmaxRow(logits));
                _ = mtp.hybridArStep(&st, t, ar_token);
                try generated.append(allocator, ar_token);
                cached_len += n_recompute;
            }
            if (st.terminated) break;

            // Degenerate-tail loop stop (reference `decode_hybrid` early_stop
            // block): an exact repeated box, or an edge-locked sliver march.
            if (options.early_stop) {
                if (self.detectBoxLoop(generated.items[prompt_len..])) |loop_start| {
                    generated.shrinkRetainingCapacity(prompt_len + loop_start);
                    st.terminated = true;
                    break;
                }
            }
        }

        const out = try allocator.dupe(u32, generated.items[prompt_len..]);
        return out;
    }

    /// Scan the committed stream for the degenerate repeated/sliver box tail;
    /// returns the offset (into `committed`) of the last box's start when the
    /// loop pattern fires, else null. Port of the reference's early-stop scan.
    fn detectBoxLoop(self: *const Engine, committed: []const u32) ?usize {
        const t = self.tok_ids;
        var prev_coords: [4]i64 = .{ -2, -2, -2, -2 };
        var last_coords: [4]i64 = .{ -1, -1, -1, -1 };
        var prev_n: usize = 0;
        var last_n: usize = 0;
        var last_start: ?usize = null;

        var i: usize = 0;
        while (i < committed.len) {
            if (committed[i] == t.box_start) {
                var j = i + 1;
                var coords: [4]i64 = .{ 0, 0, 0, 0 };
                var n: usize = 0;
                while (j < committed.len and committed[j] != t.box_end) : (j += 1) {
                    if (committed[j] >= t.coord_start and committed[j] <= t.coord_end and n < 4) {
                        coords[n] = @intCast(committed[j]);
                        n += 1;
                    }
                }
                if (n == 4) {
                    prev_coords = last_coords;
                    last_coords = coords;
                    prev_n = last_n;
                    last_n = 4;
                    last_start = i;
                }
                i = j + 1;
            } else {
                i += 1;
            }
        }
        if (prev_n == 4 and last_n == 4) {
            const lc = last_coords;
            const pc = prev_coords;
            const dup = lc[0] == pc[0] and lc[1] == pc[1] and lc[2] == pc[2] and lc[3] == pc[3];
            const xedge = lc[0] == pc[0] and lc[2] == pc[2];
            const yedge = lc[1] == pc[1] and lc[3] == pc[3];
            const sliver = (lc[2] - lc[0] < 20) or (lc[3] - lc[1] < 20);
            if (dup or ((xedge or yedge) and sliver)) return last_start;
        }
        return null;
    }

    pub const LocateResult = struct {
        allocator: Allocator,
        boxes: []boxes_mod.Box,
        /// The preprocessed target size the boxes denormalize against.
        target_w: usize,
        target_h: usize,
        generated: []u32,

        pub fn deinit(self: *LocateResult) void {
            boxes_mod.freeBoxes(self.allocator, self.boxes);
            self.allocator.free(self.generated);
            self.* = undefined;
        }
    };

    /// Full pixels -> labeled-boxes pipeline on an RGB8 image.
    pub fn locate(
        self: *Engine,
        rgb: []const u8,
        img_w: usize,
        img_h: usize,
        query: []const u8,
        mode: Mode,
        max_new: usize,
        early_stop: bool,
    ) !LocateResult {
        const allocator = self.allocator;

        var pre = try preproc_mod.preprocess(allocator, rgb, img_w, img_h, self.preprocLimits());
        defer pre.deinit();

        const n_image_tokens = (pre.gh / self.config.vit_merge_h) * (pre.gw / self.config.vit_merge_w);
        const prompt_ids = try tokenizer_mod.buildPrompt(allocator, &self.tok, n_image_tokens, query);
        defer allocator.free(prompt_ids);

        const projected = try self.projectImage(&pre);
        defer allocator.free(projected);

        const generated = switch (mode) {
            .slow => try self.decodeSlow(prompt_ids, projected, max_new),
            .hybrid => try self.decodeHybrid(prompt_ids, projected, max_new, .{ .early_stop = early_stop }),
            .fast => try self.decodeHybrid(prompt_ids, projected, max_new, .{ .fast = true, .early_stop = early_stop }),
        };
        errdefer allocator.free(generated);

        const LabelDecoder = struct {
            tok: *const tokenizer_mod.Tokenizer,
            pub fn decode(d: @This(), a: Allocator, ids: []const u32) ![]u8 {
                return d.tok.decode(a, ids);
            }
        };
        const boxes = try boxes_mod.parseBoxes(allocator, self.tok_ids, generated, pre.target_w, pre.target_h, LabelDecoder{ .tok = &self.tok });
        return .{
            .allocator = allocator,
            .boxes = boxes,
            .target_w = pre.target_w,
            .target_h = pre.target_h,
            .generated = generated,
        };
    }
};
