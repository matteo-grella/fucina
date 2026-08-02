//! Qwen3-TTS talker + MTP code predictor (qwentts.cpp talker-forward.h /
//! code-predictor-forward.h): two Qwen3-shaped decoder stacks over the codec
//! token space.
//!
//! Talker: 28 layers (0.6B), hidden 1024, 16Q/8KV heads, head_dim 128,
//! per-head QK-RMSNorm, NEOX RoPE θ=1e6 (the checkpoint's mrope collapses to
//! plain 1-D rope in TTS mode — all three sections share position ids),
//! SwiGLU FFN, `codec_head` → 3072 codec logits. Input is EMBEDDINGS only —
//! the prompt builder assembles summed text/codec rows host-side, and each
//! decode step feeds the 16-embedding sum + text overlay.
//!
//! Code predictor: 5 layers, same block shape, own 16-slot KV cache RESET per
//! frame; prefill = [talker last hidden, talker codec-embed of c0], then 14
//! single-token decodes; codebook g samples from `code_pred.lm_head.{g-1}` and
//! feeds `code_pred.codec_embd.{g-1}`.
//!
//! KV caches are host-side f32 (matching the reference's f32 cache exactly);
//! K/V tensors are rebuilt per step from the host rows — sequences here are
//! hundreds of rows, so the copies are noise next to the matmuls.

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("../weights.zig");

const ptqtp_gguf = @import("../ptqtp_gguf.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;
const Allocator = std.mem.Allocator;
const LinearWeight = weights.LinearWeight;

pub const Error = error{
    MissingMetadata,
    BadShape,
    NotF32,
    KvOverflow,
    UnknownLanguage,
    UnknownSpeaker,
};

const VecEmbed = fucina.Tensor(.{.embed});
const VecD = fucina.Tensor(.{.d});
const Rows = fucina.Tensor(.{ .seq, .embed });

pub const Config = struct {
    hidden: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize, // codec vocab (3072)
    text_vocab: usize,
    text_hidden: usize,
    rope_theta: f32,
    rms_eps: f32,
    max_seq: usize,

    fn sub(file: *const gguf.File, comptime prefix: []const u8, comptime key: []const u8) ?i64 {
        return file.getInt("qwen3-tts." ++ prefix ++ "." ++ key);
    }

    fn load(file: *const gguf.File, comptime prefix: []const u8, max_seq: usize) !Config {
        const hidden: usize = @intCast(sub(file, prefix, "embedding_length") orelse return Error.MissingMetadata);
        return .{
            .hidden = hidden,
            .n_layers = @intCast(sub(file, prefix, "block_count") orelse return Error.MissingMetadata),
            .n_heads = @intCast(sub(file, prefix, "attention.head_count") orelse return Error.MissingMetadata),
            .n_kv_heads = @intCast(sub(file, prefix, "attention.head_count_kv") orelse return Error.MissingMetadata),
            .head_dim = @intCast(sub(file, prefix, "attention.key_length") orelse return Error.MissingMetadata),
            .ffn = @intCast(sub(file, prefix, "feed_forward_length") orelse return Error.MissingMetadata),
            .vocab = @intCast(sub(file, prefix, "vocab_size") orelse return Error.MissingMetadata),
            .text_vocab = @intCast(file.getInt("qwen3-tts.talker.text_vocab_size") orelse 0),
            .text_hidden = @intCast(file.getInt("qwen3-tts.talker.text_hidden_size") orelse 0),
            .rope_theta = @floatCast(file.getFloat("qwen3-tts." ++ prefix ++ ".rope.freq_base") orelse 1e6),
            .rms_eps = @floatCast(file.getFloat("qwen3-tts." ++ prefix ++ ".attention.layer_norm_rms_epsilon") orelse 1e-6),
            .max_seq = max_seq,
        };
    }
};

/// Special token ids and voice tables read from the talker GGUF metadata.
pub const Specials = struct {
    codec_bos: usize,
    codec_eos: usize,
    codec_pad: usize,
    nothink: usize,
    think: usize,
    think_bos: usize,
    think_eos: usize,
    tts_bos: usize,
    tts_eos: usize,
    tts_pad: usize,
    num_code_groups: usize,

    fn id(file: *const gguf.File, comptime key: []const u8) !usize {
        return @intCast(file.getInt(key) orelse return Error.MissingMetadata);
    }

    pub fn fromGguf(file: *const gguf.File) !Specials {
        return .{
            .codec_bos = try id(file, "qwen3-tts.codec.bos_id"),
            .codec_eos = try id(file, "qwen3-tts.codec.eos_id"),
            .codec_pad = try id(file, "qwen3-tts.codec.pad_id"),
            .nothink = try id(file, "qwen3-tts.codec.nothink_id"),
            .think = try id(file, "qwen3-tts.codec.think_id"),
            .think_bos = try id(file, "qwen3-tts.codec.think_bos_id"),
            .think_eos = try id(file, "qwen3-tts.codec.think_eos_id"),
            .tts_bos = try id(file, "qwen3-tts.text.tts_bos_id"),
            .tts_eos = try id(file, "qwen3-tts.text.tts_eos_id"),
            .tts_pad = try id(file, "qwen3-tts.text.tts_pad_id"),
            .num_code_groups = try id(file, "qwen3-tts.num_code_groups"),
        };
    }
};

fn f32Data(t: *const gguf.TensorInfo) ![]const f32 {
    if (t.ggml_type != .f32) return Error.NotF32;
    if (!std.mem.isAligned(@intFromPtr(t.data.ptr), @alignOf(f32))) return Error.BadShape;
    return @as([*]const f32, @ptrCast(@alignCast(t.data.ptr)))[0 .. t.data.len / 4];
}


fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// Linear load with PTQTP pair-detection: `<name>.ptqtpK` plane tensors (if
/// the GGUF carries them) serve as ternary-plane matmuls; otherwise the
/// plain (possibly quantized) tensor loads as usual.
fn loadLinear(ctx: *ExecContext, file: *const gguf.File, name: []const u8, rows: usize, cols: usize) !LinearWeight {
    if (try ptqtp_gguf.maybeLoadPlanes(ctx, file, name, rows, cols)) |w| return w;
    return LinearWeight.load(ctx, try file.get(name), rows, cols);
}

const Layer = struct {
    attn_norm: VecEmbed,
    q: LinearWeight,
    k: LinearWeight,
    v: LinearWeight,
    o: LinearWeight,
    q_norm: VecD,
    k_norm: VecD,
    ffn_norm: VecEmbed,
    gate: LinearWeight,
    up: LinearWeight,
    down: LinearWeight,

    fn deinit(self: *Layer) void {
        inline for (@typeInfo(Layer).@"struct".fields) |f| @field(self, f.name).deinit();
        self.* = undefined;
    }
};

fn loadLayer(ctx: *ExecContext, file: *const gguf.File, comptime prefix: []const u8, i: usize, cfg: *const Config) !Layer {
    var buf: [96]u8 = undefined;
    const qdim = cfg.n_heads * cfg.head_dim;
    const kvdim = cfg.n_kv_heads * cfg.head_dim;
    var l: Layer = undefined;
    l.attn_norm = try loadVec(VecEmbed, ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_norm.weight", .{i}), cfg.hidden);
    l.q = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_q.weight", .{i}), qdim, cfg.hidden);
    l.k = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_k.weight", .{i}), kvdim, cfg.hidden);
    l.v = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_v.weight", .{i}), kvdim, cfg.hidden);
    l.o = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_output.weight", .{i}), cfg.hidden, qdim);
    l.q_norm = try loadVec(VecD, ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_q_norm.weight", .{i}), cfg.head_dim);
    l.k_norm = try loadVec(VecD, ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.attn_k_norm.weight", .{i}), cfg.head_dim);
    l.ffn_norm = try loadVec(VecEmbed, ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.ffn_norm.weight", .{i}), cfg.hidden);
    l.gate = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.ffn_gate.weight", .{i}), cfg.ffn, cfg.hidden);
    l.up = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.ffn_up.weight", .{i}), cfg.ffn, cfg.hidden);
    l.down = try loadLinear(ctx, file, fmtName(&buf, prefix ++ ".blk.{d}.ffn_down.weight", .{i}), cfg.hidden, cfg.ffn);
    return l;
}

fn loadVec(comptime T: type, ctx: *ExecContext, file: *const gguf.File, name: []const u8, n: usize) !T {
    const t = try file.get(name);
    const data = try f32Data(t);
    if (data.len != n) return Error.BadShape;
    return T.fromSlice(ctx, .{n}, data);
}

/// Host-side f32 KV cache: per layer, appended rows of K and V in
/// `[row, kv_heads * head_dim]` layout (row-major, matching the reference's
/// f32 cache numerics — no f16 round-trip).
pub const Kv = struct {
    allocator: Allocator,
    k: [][]f32,
    v: [][]f32,
    len: usize,
    capacity: usize,
    kv_dim: usize,

    pub fn init(allocator: Allocator, n_layers: usize, kv_dim: usize, capacity: usize) !Kv {
        const k = try allocator.alloc([]f32, n_layers);
        var built: usize = 0;
        errdefer {
            for (k[0..built]) |b| allocator.free(b);
            allocator.free(k);
        }
        for (k) |*b| {
            b.* = try allocator.alloc(f32, capacity * kv_dim);
            built += 1;
        }
        const v = try allocator.alloc([]f32, n_layers);
        var vbuilt: usize = 0;
        errdefer {
            for (v[0..vbuilt]) |b| allocator.free(b);
            allocator.free(v);
        }
        for (v) |*b| {
            b.* = try allocator.alloc(f32, capacity * kv_dim);
            vbuilt += 1;
        }
        return .{ .allocator = allocator, .k = k, .v = v, .len = 0, .capacity = capacity, .kv_dim = kv_dim };
    }

    pub fn deinit(self: *Kv) void {
        for (self.k) |b| self.allocator.free(b);
        self.allocator.free(self.k);
        for (self.v) |b| self.allocator.free(b);
        self.allocator.free(self.v);
        self.* = undefined;
    }

    pub fn reset(self: *Kv) void {
        self.len = 0;
    }
};

/// One decoder stack (used for both the talker and the code predictor).
pub const Stack = struct {
    cfg: Config,
    layers: []Layer,
    output_norm: VecEmbed,
    kv_head_for_head: []usize,
    allocator: Allocator,

    fn load(ctx: *ExecContext, file: *const gguf.File, comptime prefix: []const u8, max_seq: usize) !Stack {
        const cfg = try Config.load(file, prefix, max_seq);
        const layers = try ctx.allocator.alloc(Layer, cfg.n_layers);
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*l| l.deinit();
            ctx.allocator.free(layers);
        }
        for (layers, 0..) |*l, i| {
            l.* = try loadLayer(ctx, file, prefix, i, &cfg);
            built += 1;
        }
        var output_norm = try loadVec(VecEmbed, ctx, file, prefix ++ ".output_norm.weight", cfg.hidden);
        errdefer output_norm.deinit();
        const map = try ctx.allocator.alloc(usize, cfg.n_heads);
        const group = cfg.n_heads / cfg.n_kv_heads;
        for (map, 0..) |*m, h| m.* = h / group;
        return .{ .cfg = cfg, .layers = layers, .output_norm = output_norm, .kv_head_for_head = map, .allocator = ctx.allocator };
    }

    fn deinit(self: *Stack) void {
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.output_norm.deinit();
        self.allocator.free(self.kv_head_for_head);
        self.* = undefined;
    }

    pub fn newKv(self: *const Stack, allocator: Allocator) !Kv {
        return Kv.init(allocator, self.cfg.n_layers, self.cfg.n_kv_heads * self.cfg.head_dim, self.cfg.max_seq);
    }

    /// Forward `rows` ([T, hidden] embeddings) at positions
    /// `kv.len .. kv.len+T`, appending K/V, and return the post-final-norm
    /// hidden of every position (`[T, hidden]`). `taps`, when set, receives
    /// a copy of the post-layer hidden for the listed layer indices.
    pub fn forward(self: *const Stack, ctx: *ExecContext, kv: *Kv, rows: *const Rows, taps: ?*StackTaps) !Rows {
        const cfg = &self.cfg;
        const t = rows.dim(.seq);
        if (kv.len + t > kv.capacity) return Error.KvOverflow;
        const n_past = kv.len;

        const positions = try ctx.allocator.alloc(i32, t);
        defer ctx.allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(n_past + i);
        var rope_table = try ctx.prepareRopeTable(positions, cfg.head_dim, cfg.rope_theta, false);
        defer rope_table.deinit();

        var x = try rows.withTags(ctx, .{ .seq, .embed });
        errdefer x.deinit();

        for (self.layers, 0..) |*layer, li| {
            const cached_len = n_past + t;

            var attn_in = try x.rmsNormMul(ctx, .embed, &layer.attn_norm, cfg.rms_eps);
            defer attn_in.deinit();
            var q = try layer.q.linearSeq(ctx, &attn_in, .embed, .q);
            defer q.deinit();
            var k = try layer.k.linearSeq(ctx, &attn_in, .embed, .k);
            defer k.deinit();
            var v = try layer.v.linearSeq(ctx, &attn_in, .embed, .k);
            defer v.deinit();

            var q3 = try q.split(ctx, .q, .{ .head, .d }, .{ cfg.n_heads, cfg.head_dim });
            defer q3.deinit();
            var k3 = try k.split(ctx, .k, .{ .kv_head, .d }, .{ cfg.n_kv_heads, cfg.head_dim });
            defer k3.deinit();
            var v3 = try v.split(ctx, .k, .{ .kv_head, .d }, .{ cfg.n_kv_heads, cfg.head_dim });
            defer v3.deinit();

            var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, cfg.rms_eps, &rope_table);
            defer q_rope.deinit();
            var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, cfg.rms_eps, &rope_table);
            defer k_rope.deinit();

            // Append this step's K/V rows to the host cache, then rebuild
            // the full-context views.
            {
                const kd = try k_rope.dataConst();
                const vd = try v3.dataConst();
                @memcpy(kv.k[li][n_past * kv.kv_dim ..][0 .. t * kv.kv_dim], kd);
                @memcpy(kv.v[li][n_past * kv.kv_dim ..][0 .. t * kv.kv_dim], vd);
            }
            var k_all = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ cached_len, cfg.n_kv_heads, cfg.head_dim }, kv.k[li][0 .. cached_len * kv.kv_dim]);
            defer k_all.deinit();
            var v_all = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(ctx, .{ cached_len, cfg.n_kv_heads, cfg.head_dim }, kv.v[li][0 .. cached_len * kv.kv_dim]);
            defer v_all.deinit();

            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
            var attn = try q_rope.groupedAttention(ctx, &k_all, &v_all, self.kv_head_for_head, .attn, scale, .{});
            defer attn.deinit();

            var attn_out = try layer.o.linearSeq(ctx, &attn, .attn, .embed);
            defer attn_out.deinit();
            const res1 = try x.add(ctx, &attn_out);
            x.deinit();
            x = res1;

            var ffn_in = try x.rmsNormMul(ctx, .embed, &layer.ffn_norm, cfg.rms_eps);
            defer ffn_in.deinit();
            var gate = try layer.gate.linearSeq(ctx, &ffn_in, .embed, .ffn);
            defer gate.deinit();
            var up = try layer.up.linearSeq(ctx, &ffn_in, .embed, .ffn);
            defer up.deinit();
            var gated = try up.swiglu(ctx, &gate);
            defer gated.deinit();
            var down = try layer.down.linearSeq(ctx, &gated, .ffn, .embed);
            defer down.deinit();
            const res2 = try x.add(ctx, &down);
            x.deinit();
            x = res2;

            if (taps) |tp| try tp.capture(ctx, li, &x);
        }
        kv.len = n_past + t;

        const normed = try x.rmsNormMul(ctx, .embed, &self.output_norm, cfg.rms_eps);
        x.deinit();
        return normed;
    }
};

/// Optional per-layer prefill taps (parity harness).
pub const StackTaps = struct {
    allocator: Allocator,
    layer_set: []const usize,
    captured: std.AutoHashMap(usize, []f32),

    pub fn init(allocator: Allocator, layer_set: []const usize) StackTaps {
        return .{ .allocator = allocator, .layer_set = layer_set, .captured = std.AutoHashMap(usize, []f32).init(allocator) };
    }

    pub fn deinit(self: *StackTaps) void {
        var it = self.captured.valueIterator();
        while (it.next()) |v| self.allocator.free(v.*);
        self.captured.deinit();
        self.* = undefined;
    }

    fn capture(self: *StackTaps, ctx: *ExecContext, layer: usize, x: *const fucina.Tensor(.{ .seq, .embed })) !void {
        _ = ctx;
        for (self.layer_set) |want| {
            if (want == layer) {
                const copy = try self.allocator.dupe(f32, try x.dataConst());
                try self.captured.put(layer, copy);
            }
        }
    }
};

pub const Model = struct {
    allocator: Allocator,
    talker: Stack,
    predictor: Stack,
    specials: Specials,
    codec_head: LinearWeight,
    /// Borrowed mmap rows: `talker.codec_embd.weight` `[vocab][hidden]`.
    codec_embd: []const f32, // owned (dequantized at load; 3k rows)
    /// Borrowed mmap rows: `talker.text_embd.weight` `[text_vocab][text_hidden]`.
    text_embd: gguf.RowTable,
    text_scratch: []f32,
    pred_scratch: []f32,
    ctx: *ExecContext,
    text_fc1: LinearWeight, // [text_hidden, text_hidden]
    text_fc1_b: []const f32,
    text_fc2: LinearWeight, // [hidden, text_hidden]
    text_fc2_b: []const f32,
    /// Per-codebook predictor tables, borrowed: `[15][2048][pred_hidden]`.
    pred_embd: []gguf.RowTable,
    // 1.7B: talker hidden (2048) != predictor hidden (1024); every predictor
    // input row projects through mtp_proj. Absent (null) at 0.6B = identity.
    mtp_proj: ?LinearWeight,
    mtp_proj_b: ?[]const f32,
    pred_heads: []LinearWeight,
    languages: std.StringHashMap(usize),
    speakers: std.StringHashMap(usize),
    /// Speaker → dialect language name (empty entries omitted); overrides the
    /// language id when the requested language is chinese or auto.
    speaker_dialects: std.StringHashMap([]const u8),

    pub fn load(ctx: *ExecContext, file: *const gguf.File) !Model {
        const allocator = ctx.allocator;
        var talker = try Stack.load(ctx, file, "talker", 4096);
        errdefer talker.deinit();
        var predictor = try Stack.load(ctx, file, "code_pred", 16);
        errdefer predictor.deinit();
        const specials = try Specials.fromGguf(file);

        var codec_head = try loadLinear(ctx, file, "talker.codec_head.weight", talker.cfg.vocab, talker.cfg.hidden);
        errdefer codec_head.deinit();

        // Small tables materialize to f32 once; the huge text table and the
        // 15 MTP codebooks stay quantized with per-row reads.
        const codec_embd = try gguf.decodeAllocF32(allocator, try file.get("talker.codec_embd.weight"));
        errdefer allocator.free(codec_embd);
        const text_embd = try gguf.RowTable.init(allocator, try file.get("talker.text_embd.weight"));
        var fc1 = try loadLinear(ctx, file, "talker.text_proj.fc1.weight", talker.cfg.text_hidden, talker.cfg.text_hidden);
        errdefer fc1.deinit();
        const fc1_b = try gguf.decodeAllocF32(allocator, try file.get("talker.text_proj.fc1.bias"));
        errdefer allocator.free(fc1_b);
        var fc2 = try loadLinear(ctx, file, "talker.text_proj.fc2.weight", talker.cfg.hidden, talker.cfg.text_hidden);
        errdefer fc2.deinit();
        const fc2_b = try gguf.decodeAllocF32(allocator, try file.get("talker.text_proj.fc2.bias"));
        errdefer allocator.free(fc2_b);
        const text_scratch = try allocator.alloc(f32, talker.cfg.text_hidden);
        errdefer allocator.free(text_scratch);
        // Codebook tables live in TALKER space (projected into the predictor
        // when the spaces differ), so the row scratch is talker-width.
        const pred_scratch = try allocator.alloc(f32, @max(talker.cfg.hidden, predictor.cfg.hidden));
        errdefer allocator.free(pred_scratch);
        var mtp_proj: ?LinearWeight = if (file.get("code_pred.mtp_proj.weight")) |_|
            try loadLinear(ctx, file, "code_pred.mtp_proj.weight", predictor.cfg.hidden, talker.cfg.hidden)
        else |_|
            null;
        errdefer if (mtp_proj) |*w| w.deinit();
        const mtp_proj_b: ?[]const f32 = if (file.get("code_pred.mtp_proj.bias")) |t|
            try gguf.decodeAllocF32(allocator, t)
        else |_|
            null;
        errdefer if (mtp_proj_b) |b| allocator.free(b);

        const n_cb = specials.num_code_groups - 1;
        var buf: [64]u8 = undefined;
        const pred_embd = try allocator.alloc(gguf.RowTable, n_cb);
        errdefer allocator.free(pred_embd);
        for (pred_embd, 0..) |*e, i| {
            e.* = try gguf.RowTable.init(allocator, try file.get(fmtName(&buf, "code_pred.codec_embd.{d}.weight", .{i})));
        }
        const pred_heads = try allocator.alloc(LinearWeight, n_cb);
        var heads_built: usize = 0;
        errdefer {
            for (pred_heads[0..heads_built]) |*h| h.deinit();
            allocator.free(pred_heads);
        }
        for (pred_heads, 0..) |*h, i| {
            h.* = try loadLinear(ctx, file, fmtName(&buf, "code_pred.lm_head.{d}.weight", .{i}), predictor.cfg.vocab, predictor.cfg.hidden);
            heads_built += 1;
        }

        var languages = std.StringHashMap(usize).init(allocator);
        errdefer freeNameTable(&languages, allocator);
        try loadNameTable(&languages, file, "qwen3-tts.codec.language_names", "qwen3-tts.codec.language_ids", allocator);
        var speakers = std.StringHashMap(usize).init(allocator);
        errdefer freeNameTable(&speakers, allocator);
        try loadNameTable(&speakers, file, "qwen3-tts.codec.speaker_names", "qwen3-tts.codec.speaker_ids", allocator);
        var speaker_dialects = std.StringHashMap([]const u8).init(allocator);
        errdefer freeStrTable(&speaker_dialects, allocator);
        try loadStrTable(&speaker_dialects, file, "qwen3-tts.codec.speaker_names", "qwen3-tts.codec.speaker_dialects", allocator);

        return .{
            .allocator = allocator,
            .talker = talker,
            .predictor = predictor,
            .specials = specials,
            .codec_head = codec_head,
            .codec_embd = codec_embd,
            .text_embd = text_embd,
            .text_scratch = text_scratch,
            .pred_scratch = pred_scratch,
            .ctx = ctx,
            .text_fc1 = fc1,
            .text_fc1_b = fc1_b,
            .text_fc2 = fc2,
            .text_fc2_b = fc2_b,
            .pred_embd = pred_embd,
            .mtp_proj = mtp_proj,
            .mtp_proj_b = mtp_proj_b,
            .pred_heads = pred_heads,
            .languages = languages,
            .speakers = speakers,
            .speaker_dialects = speaker_dialects,
        };
    }

    pub fn deinit(self: *Model) void {
        self.talker.deinit();
        self.predictor.deinit();
        self.codec_head.deinit();
        self.allocator.free(self.codec_embd);
        self.text_fc1.deinit();
        self.allocator.free(@constCast(self.text_fc1_b));
        self.text_fc2.deinit();
        self.allocator.free(@constCast(self.text_fc2_b));
        self.allocator.free(self.text_scratch);
        self.allocator.free(self.pred_scratch);
        self.allocator.free(self.pred_embd);
        if (self.mtp_proj) |*w| w.deinit();
        if (self.mtp_proj_b) |b| self.allocator.free(b);
        for (self.pred_heads) |*h| h.deinit();
        self.allocator.free(self.pred_heads);
        freeNameTable(&self.languages, self.allocator);
        freeNameTable(&self.speakers, self.allocator);
        freeStrTable(&self.speaker_dialects, self.allocator);
        self.* = undefined;
    }

    /// Row `id` of the talker codec embedding table.
    pub fn codecRow(self: *const Model, id: usize) []const f32 {
        const h = self.talker.cfg.hidden;
        return self.codec_embd[id * h ..][0..h];
    }

    /// Row `id` of predictor codebook table `g` (0-based, codebooks 1..15).
    /// Talker-space width (== talker hidden).
    pub fn predRow(self: *const Model, g: usize, id: usize) []const f32 {
        return self.pred_embd[g].row(id, self.pred_scratch);
    }

    /// Talker-space row → predictor input space (mtp_proj at 1.7B; identity
    /// at 0.6B where the spaces coincide). Core tensor ops.
    fn mtpProject(self: *const Model, src: []const f32, out: []f32) !void {
        const w = &(self.mtp_proj orelse {
            @memcpy(out, src);
            return;
        });
        const ctx = self.ctx;
        var row = try Rows.fromSlice(ctx, .{ 1, src.len }, src);
        defer row.deinit();
        var y = try w.linearSeq(ctx, &row, .embed, .d);
        defer y.deinit();
        if (self.mtp_proj_b) |b| try y.addAxisVectorInPlace(ctx, b, .d);
        @memcpy(out, try y.dataConst());
    }

    /// `text_proj(text_embd[id])` → `out` ([hidden]) on core tensor ops
    /// (fc1 → SiLU → fc2, the reference builder's MLP).
    pub fn textProject(self: *const Model, id: usize, out: []f32) !void {
        const ctx = self.ctx;
        const th = self.talker.cfg.text_hidden;
        const e = self.text_embd.row(id, self.text_scratch);
        var row = try Rows.fromSlice(ctx, .{ 1, th }, e);
        defer row.deinit();
        var h1 = try self.text_fc1.linearSeq(ctx, &row, .embed, .d);
        defer h1.deinit();
        try h1.addAxisVectorInPlace(ctx, self.text_fc1_b, .d);
        var act = try h1.silu(ctx);
        defer act.deinit();
        var h2 = try self.text_fc2.linearSeq(ctx, &act, .d, .embed);
        defer h2.deinit();
        try h2.addAxisVectorInPlace(ctx, self.text_fc2_b, .embed);
        @memcpy(out, try h2.dataConst());
    }

    /// One code-predictor frame: prefill `[talker_hidden_last, codec_embd[c0]]`
    /// then 14 decodes; returns codes[1..16] into `out_codes[1..]` (slot 0 is
    /// the caller's c0). Sampling per codebook via `sampler`.
    pub fn predictFrame(
        self: *const Model,
        ctx: *ExecContext,
        kv: *Kv,
        talker_hidden_last: []const f32,
        c0: usize,
        out_codes: []i32,
        samplerFn: anytype,
    ) !void {
        const h = self.predictor.cfg.hidden;
        kv.reset();

        var two_rows = try self.allocator.alloc(f32, 2 * h);
        defer self.allocator.free(two_rows);
        try self.mtpProject(talker_hidden_last, two_rows[0..h]);
        try self.mtpProject(self.codecRow(c0), two_rows[h..]);
        const step_buf = try self.allocator.alloc(f32, h);
        defer self.allocator.free(step_buf);
        var rows = try Rows.fromSlice(ctx, .{ 2, h }, two_rows);
        defer rows.deinit();

        var hidden = try self.predictor.forward(ctx, kv, &rows, null);
        var hidden_live = true;
        errdefer if (hidden_live) hidden.deinit();
        for (0..self.specials.num_code_groups - 1) |g| {
            var last = try hidden.narrow(ctx, .seq, hidden.dim(.seq) - 1, 1);
            defer last.deinit();
            var logits = try self.pred_heads[g].linearSeq(ctx, &last, .embed, .vocab);
            defer logits.deinit();
            const ld = try logits.dataConst();
            const sampled: usize = samplerFn.sample(g, ld);
            out_codes[g + 1] = @intCast(sampled);
            hidden.deinit();
            hidden_live = false;
            if (g + 1 < self.specials.num_code_groups - 1) {
                try self.mtpProject(self.predRow(g, sampled), step_buf);
                var step_rows = try Rows.fromSlice(ctx, .{ 1, h }, step_buf);
                defer step_rows.deinit();
                hidden = try self.predictor.forward(ctx, kv, &step_rows, null);
                hidden_live = true;
            }
        }
    }
};

fn loadNameTable(map: *std.StringHashMap(usize), file: *const gguf.File, comptime names_key: []const u8, comptime ids_key: []const u8, allocator: Allocator) !void {
    const names_arr = file.getArray(names_key) orelse return;
    const ids_arr = file.getArray(ids_key) orelse return;
    const names = try names_arr.stringSlices(allocator);
    defer allocator.free(names);
    if (ids_arr.item_type != 5 or ids_arr.data.len != names.len * 4) return Error.BadShape;
    for (names, 0..) |name, i| {
        const id = std.mem.readInt(i32, ids_arr.data[i * 4 ..][0..4], .little);
        if (name.len == 0 or map.contains(name)) continue;
        const owned = try allocator.dupe(u8, name);
        errdefer allocator.free(owned);
        try map.put(owned, @intCast(id));
    }
}

/// Pair two string arrays (keys, values) into an owned map, skipping empty
/// values and duplicate keys.
fn loadStrTable(map: *std.StringHashMap([]const u8), file: *const gguf.File, comptime keys_key: []const u8, comptime vals_key: []const u8, allocator: Allocator) !void {
    const keys_arr = file.getArray(keys_key) orelse return;
    const vals_arr = file.getArray(vals_key) orelse return;
    const keys = try keys_arr.stringSlices(allocator);
    defer allocator.free(keys);
    const vals = try vals_arr.stringSlices(allocator);
    defer allocator.free(vals);
    if (keys.len != vals.len) return Error.BadShape;
    for (keys, vals) |key, val| {
        if (key.len == 0 or val.len == 0 or map.contains(key)) continue;
        const owned_key = try allocator.dupe(u8, key);
        errdefer allocator.free(owned_key);
        const owned_val = try allocator.dupe(u8, val);
        errdefer allocator.free(owned_val);
        try map.put(owned_key, owned_val);
    }
}

fn freeStrTable(map: *std.StringHashMap([]const u8), allocator: Allocator) void {
    var it = map.iterator();
    while (it.next()) |e| {
        allocator.free(e.key_ptr.*);
        allocator.free(e.value_ptr.*);
    }
    map.deinit();
}

fn freeNameTable(map: *std.StringHashMap(usize), allocator: Allocator) void {
    var it = map.keyIterator();
    while (it.next()) |k| allocator.free(k.*);
    map.deinit();
}

test {
    _ = @import("talker_tests.zig");
}
