//! Prompt builder (qwentts.cpp prompt-builder.h): assembles the talker's
//! prefill as HOST-SIDE embedding rows `[T_ctx, hidden]` — the talker never
//! sees token ids.
//!
//! Standard (CustomVoice / no-ICL) layout:
//!   [instruct rows] +
//!   text_proj(text_embd(role ids[0:3])) +
//!   { tts_pad×(n−1), tts_bos } ⊕ codec_embd([prefill…, codec_pad]) +
//!   { text_proj(utterance ids[3:3+N_text]) ⊕ codec_pad } +
//!   { tts_eos ⊕ codec_pad } + { tts_pad ⊕ codec_bos }
//! where the codec prefill list is [think, think_bos, lang, think_eos] (or
//! the nothink variant for language "auto"), followed by the CustomVoice
//! speaker id row or the x-vector row (voice clone). ICL clone appends the
//! aligned ref-text/ref-codes block instead of the trailing utterance rows.
//! The trailing text overlay collapses to one tts_pad row in non-ICL mode.

const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const tokenizer_mod = @import("../text/tokenizer.zig");

const Allocator = std.mem.Allocator;

pub const Error = error{
    PromptTooShort,
    UnknownLanguage,
    UnknownSpeaker,
    RefTextRequired,
    InvalidRefCodes,
};

pub const Output = struct {
    allocator: Allocator,
    /// `[t_ctx][hidden]` prefill embedding rows.
    input_embed: []f32,
    t_ctx: usize,
    prompt_ids: []u32,
    /// Per-step text overlay rows (`[t_trailing][hidden]`), then tts_pad.
    trailing_text_hidden: []f32,
    t_trailing: usize,
    tts_pad_embed: []f32,

    pub fn deinit(self: *Output) void {
        self.allocator.free(self.input_embed);
        self.allocator.free(self.prompt_ids);
        self.allocator.free(self.trailing_text_hidden);
        self.allocator.free(self.tts_pad_embed);
        self.* = undefined;
    }
};

pub const Request = struct {
    text: []const u8,
    language: []const u8 = "english",
    speaker: ?[]const u8 = null,
    instruct: ?[]const u8 = null,
    /// Voice clone: 2048-dim x-vector row (talker hidden), mutually
    /// exclusive with `speaker`.
    ref_spk_emb: ?[]const f32 = null,
    /// ICL clone: reference transcript + codes `[16, ref_t]` (T fastest).
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const i32 = null,
    ref_codes_t: usize = 0,
};

fn lowerAscii(allocator: Allocator, s: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, s.len);
    for (out, s) |*d, c| d.* = std.ascii.toLower(c);
    return out;
}

pub fn build(
    allocator: Allocator,
    model: *const model_mod.Model,
    tok: *const tokenizer_mod.Tokenizer,
    req: Request,
) !Output {
    const hidden = model.talker.cfg.hidden;
    const sp = &model.specials;

    // Chat-templated utterance.
    const full_text = try std.fmt.allocPrint(allocator, "<|im_start|>assistant\n{s}<|im_end|>\n<|im_start|>assistant\n", .{req.text});
    defer allocator.free(full_text);
    const ids = try tok.encode(allocator, full_text);
    errdefer allocator.free(ids);
    if (ids.len < 8) return Error.PromptTooShort;
    const n_text = ids.len - 3 - 5;
    if (n_text == 0) return Error.PromptTooShort;

    // Language id ("auto" → none), speaker id + dialect override.
    var language_id: ?usize = null;
    {
        const lang_lc = try lowerAscii(allocator, req.language);
        defer allocator.free(lang_lc);
        if (!std.mem.eql(u8, lang_lc, "auto")) {
            language_id = model.languages.get(lang_lc) orelse return Error.UnknownLanguage;
        }
    }
    var speaker_id: ?usize = null;
    if (req.speaker) |name| {
        if (req.ref_spk_emb != null) return Error.UnknownSpeaker;
        const spk_lc = try lowerAscii(allocator, name);
        defer allocator.free(spk_lc);
        speaker_id = model.speakers.get(spk_lc) orelse return Error.UnknownSpeaker;
        // Dialect override (upstream modeling_qwen3_tts 2118-2122): a
        // dialect-carrying speaker overrides the language id when the
        // requested language is chinese or auto.
        if (model.speaker_dialects.get(spk_lc)) |dialect| {
            const lang_lc2 = try lowerAscii(allocator, req.language);
            defer allocator.free(lang_lc2);
            if (std.mem.eql(u8, lang_lc2, "chinese") or std.mem.eql(u8, lang_lc2, "auto")) {
                language_id = model.languages.get(dialect) orelse return Error.UnknownLanguage;
            }
        }
    }
    const icl = req.ref_text != null and req.ref_text.?.len > 0 and req.ref_codes != null and req.ref_codes_t > 0;
    if (icl and req.ref_spk_emb == null) return Error.RefTextRequired;

    // Special text embeds.
    const tts_bos_emb = try allocator.alloc(f32, hidden);
    defer allocator.free(tts_bos_emb);
    const tts_eos_emb = try allocator.alloc(f32, hidden);
    defer allocator.free(tts_eos_emb);
    const tts_pad_emb = try allocator.alloc(f32, hidden);
    errdefer allocator.free(tts_pad_emb);
    try model.textProject(sp.tts_bos, tts_bos_emb);
    try model.textProject(sp.tts_eos, tts_eos_emb);
    try model.textProject(sp.tts_pad, tts_pad_emb);

    // Codec prefill list; -2 marks the x-vector slot.
    var codec_prefill_buf: [6]i64 = undefined;
    var n_prefill: usize = 0;
    if (language_id) |lid| {
        codec_prefill_buf[0] = @intCast(sp.think);
        codec_prefill_buf[1] = @intCast(sp.think_bos);
        codec_prefill_buf[2] = @intCast(lid);
        codec_prefill_buf[3] = @intCast(sp.think_eos);
        n_prefill = 4;
    } else {
        codec_prefill_buf[0] = @intCast(sp.nothink);
        codec_prefill_buf[1] = @intCast(sp.think_bos);
        codec_prefill_buf[2] = @intCast(sp.think_eos);
        n_prefill = 3;
    }
    if (speaker_id) |sid| {
        codec_prefill_buf[n_prefill] = @intCast(sid);
        n_prefill += 1;
    } else if (req.ref_spk_emb != null) {
        codec_prefill_buf[n_prefill] = -2;
        n_prefill += 1;
    }
    const n_pad_pre = n_prefill; // codec_left = prefill + codec_pad; n-1 pads then bos

    // Instruct segment.
    var instruct_ids: []u32 = &.{};
    defer if (instruct_ids.len > 0) allocator.free(instruct_ids);
    if (req.instruct) |txt| {
        const wrapped = try std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n", .{txt});
        defer allocator.free(wrapped);
        instruct_ids = try tok.encode(allocator, wrapped);
    }

    // ICL reference tokenization.
    var ref_ids: []u32 = &.{};
    defer if (ref_ids.len > 0) allocator.free(ref_ids);
    var n_ref_text: usize = 0;
    if (icl) {
        const ref_full = try std.fmt.allocPrint(allocator, "<|im_start|>assistant\n{s}<|im_end|>\n<|im_start|>assistant\n", .{req.ref_text.?});
        defer allocator.free(ref_full);
        ref_ids = try tok.encode(allocator, ref_full);
        if (ref_ids.len < 9) return Error.PromptTooShort; // >= 1 ref body token
        n_ref_text = ref_ids.len - 3 - 5;
        // Reference codes must be valid table indices (the ref throws too).
        const rc = req.ref_codes.?;
        for (rc[0 .. model.specials.num_code_groups * req.ref_codes_t]) |c| {
            if (c < 0 or c >= 2048) return Error.InvalidRefCodes;
        }
    }

    const text_lens_icl = if (icl) n_ref_text + n_text + 1 else 0;
    const codec_lens_icl = if (icl) 1 + req.ref_codes_t else 0;
    const icl_t = codec_lens_icl;

    const t_ctx = if (icl)
        instruct_ids.len + 3 + (n_pad_pre + 1) + icl_t
    else
        instruct_ids.len + 3 + (n_pad_pre + 1) + n_text + 1 + 1;

    const input_embed = try allocator.alloc(f32, t_ctx * hidden);
    errdefer allocator.free(input_embed);
    @memset(input_embed, 0);

    var row: usize = 0;
    const rowAt = struct {
        fn f(buf: []f32, h: usize, r: usize) []f32 {
            return buf[r * h ..][0..h];
        }
    }.f;

    // Instruct rows (standalone text projections).
    for (instruct_ids) |tid| {
        try model.textProject(tid, rowAt(input_embed, hidden, row));
        row += 1;
    }
    // Role rows ids[0:3].
    for (ids[0..3]) |tid| {
        try model.textProject(tid, rowAt(input_embed, hidden, row));
        row += 1;
    }
    // Codec prefix rows: text = tts_pad ×(n−1) then tts_bos; codec = the
    // prefill list + codec_pad (dropping codec_bos per upstream [:, :-1]).
    {
        var i: usize = 0;
        while (i <= n_prefill) : (i += 1) { // prefill list + trailing codec_pad
            const r = rowAt(input_embed, hidden, row);
            const text_vec = if (i == n_prefill) tts_bos_emb else tts_pad_emb;
            @memcpy(r, text_vec);
            const code: i64 = if (i == n_prefill) @intCast(sp.codec_pad) else codec_prefill_buf[i];
            if (code == -2) {
                for (r, req.ref_spk_emb.?) |*d, s| d.* += s;
            } else {
                for (r, model.codecRow(@intCast(code))) |*d, s| d.* += s;
            }
            row += 1;
        }
    }

    var trailing: []f32 = undefined;
    var t_trailing: usize = 1;
    if (!icl) {
        // Utterance text rows ⊕ codec_pad.
        const codec_pad_row = model.codecRow(sp.codec_pad);
        for (ids[3 .. 3 + n_text]) |tid| {
            const r = rowAt(input_embed, hidden, row);
            try model.textProject(tid, r);
            for (r, codec_pad_row) |*d, s| d.* += s;
            row += 1;
        }
        { // tts_eos ⊕ codec_pad
            const r = rowAt(input_embed, hidden, row);
            @memcpy(r, tts_eos_emb);
            for (r, codec_pad_row) |*d, s| d.* += s;
            row += 1;
        }
        { // tts_pad ⊕ codec_bos
            const r = rowAt(input_embed, hidden, row);
            @memcpy(r, tts_pad_emb);
            for (r, model.codecRow(sp.codec_bos)) |*d, s| d.* += s;
            row += 1;
        }
        trailing = try allocator.dupe(f32, tts_pad_emb);
    } else {
        // ICL block: aligned text ⊕ codec streams.
        const text_stream = try allocator.alloc(f32, text_lens_icl * hidden);
        defer allocator.free(text_stream);
        for (ref_ids[3 .. 3 + n_ref_text], 0..) |tid, i| {
            try model.textProject(tid, rowAt(text_stream, hidden, i));
        }
        for (ids[3 .. 3 + n_text], 0..) |tid, i| {
            try model.textProject(tid, rowAt(text_stream, hidden, n_ref_text + i));
        }
        @memcpy(rowAt(text_stream, hidden, text_lens_icl - 1), tts_eos_emb);

        // Codec stream: codec_bos row then per-frame 16-codebook sums.
        const ref_codes = req.ref_codes.?;
        const rt = req.ref_codes_t;
        for (0..icl_t) |i| {
            const r = rowAt(input_embed, hidden, row + i);
            // text side (truncate/pad to icl_t)
            if (i < text_lens_icl) {
                @memcpy(r, rowAt(text_stream, hidden, i));
            } else {
                @memcpy(r, tts_pad_emb);
            }
            // codec side
            if (i == 0) {
                for (r, model.codecRow(sp.codec_bos)) |*d, s| d.* += s;
            } else {
                const t = i - 1;
                for (r, model.codecRow(@intCast(ref_codes[0 * rt + t]))) |*d, s| d.* += s;
                for (0..sp.num_code_groups - 1) |g| {
                    const code: usize = @intCast(ref_codes[(g + 1) * rt + t]);
                    for (r, model.predRow(g, code)) |*d, s| d.* += s;
                }
            }
        }
        row += icl_t;

        if (text_lens_icl >= icl_t) {
            const n_trail = text_lens_icl - icl_t;
            if (n_trail > 0) {
                t_trailing = n_trail;
                trailing = try allocator.dupe(f32, text_stream[icl_t * hidden ..][0 .. n_trail * hidden]);
            } else {
                trailing = try allocator.dupe(f32, tts_pad_emb);
            }
        } else {
            trailing = try allocator.dupe(f32, tts_pad_emb);
        }
    }
    errdefer allocator.free(trailing);

    std.debug.assert(row == t_ctx);

    return .{
        .allocator = allocator,
        .input_embed = input_embed,
        .t_ctx = t_ctx,
        .prompt_ids = ids,
        .trailing_text_hidden = trailing,
        .t_trailing = t_trailing,
        .tts_pad_embed = tts_pad_emb,
    };
}
