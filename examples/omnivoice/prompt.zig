//! OmniVoice TTS prompt builder: the `_combine_text` five-step normalisation,
//! the non-verbal-tag tokenizer, and the cond + uncond CFG buffer fill.
//! Ports refs/omnivoice.cpp/src/prompt-tts.h.
//!
//! Special tokens (`<|denoise|>`, `<|lang_start|>`, ... `<|text_end|>`) are
//! spliced programmatically from `omnivoice.special.*` ids around
//! `encodeRaw`-encoded plain-text spans — equivalent to the reference's
//! `bpe_encode`, which partitions on registered specials first, so segments
//! between specials are BPE-encoded standalone in both implementations.
//!
//! The [S, S] attention matrices of the reference PromptTTS are NOT built
//! here: the cond row's all-ones bias is softmax-shift-invariant (plain
//! bidirectional attention), and the uncond row's mixed bias is realised in
//! the forward (lm.buildUncondBias + Model.forwardUncondPadded).

const std = @import("std");
const llm = @import("fucina_llm");

const langmap = @import("langmap.zig");
const lm = @import("lm.zig");

const Allocator = std.mem.Allocator;
const Tokenizer = llm.tokenizer.Tokenizer;

pub const Error = error{ InvalidTargetTokens, InvalidRefTokens };

/// The 13 non-verbal tags of the reference NONVERBAL_PATTERN, in the
/// reference scan order (prompt-tts.h:39-44).
pub const nonverbal_tags = [_][]const u8{
    "[laughter]",    "[sigh]",        "[confirmation-en]",     "[question-en]", "[question-ah]",
    "[question-oh]", "[question-ei]", "[question-yi]",         "[surprise-ah]", "[surprise-oh]",
    "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
};

/// Decode one UTF-8 sequence, mirroring the reference prompt_tts_utf8_decode
/// exactly: no continuation-byte validation, 0 on a malformed LEAD byte (the
/// caller then skips one byte).
fn utf8Decode(bytes: []const u8, cp: *u32) usize {
    if (bytes.len == 0) return 0;
    const c = bytes[0];
    if (c < 0x80) {
        cp.* = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 and bytes.len >= 2) {
        cp.* = (@as(u32, c & 0x1F) << 6) | (bytes[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 and bytes.len >= 3) {
        cp.* = (@as(u32, c & 0x0F) << 12) | (@as(u32, bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 and bytes.len >= 4) {
        cp.* = (@as(u32, c & 0x07) << 18) | (@as(u32, bytes[1] & 0x3F) << 12) |
            (@as(u32, bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
        return 4;
    }
    return 0;
}

fn utf8Append(allocator: Allocator, out: *std.ArrayList(u8), cp: u32) !void {
    if (cp < 0x80) {
        try out.append(allocator, @intCast(cp));
    } else if (cp < 0x800) {
        try out.append(allocator, @intCast(0xC0 | (cp >> 6)));
        try out.append(allocator, @intCast(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        try out.append(allocator, @intCast(0xE0 | (cp >> 12)));
        try out.append(allocator, @intCast(0x80 | ((cp >> 6) & 0x3F)));
        try out.append(allocator, @intCast(0x80 | (cp & 0x3F)));
    } else {
        try out.append(allocator, @intCast(0xF0 | (cp >> 18)));
        try out.append(allocator, @intCast(0x80 | ((cp >> 12) & 0x3F)));
        try out.append(allocator, @intCast(0x80 | ((cp >> 6) & 0x3F)));
        try out.append(allocator, @intCast(0x80 | (cp & 0x3F)));
    }
}

/// CJK Unified Ideographs main range ([一-鿿] of the reference).
fn isCjk(cp: u32) bool {
    return cp >= 0x4E00 and cp <= 0x9FFF;
}

/// Strip leading/trailing spaces and tabs ONLY (the reference lambda).
fn stripSpacesTabs(s: []const u8) []const u8 {
    var a: usize = 0;
    var b: usize = s.len;
    while (a < b and (s[a] == ' ' or s[a] == '\t')) a += 1;
    while (b > a and (s[b - 1] == ' ' or s[b - 1] == '\t')) b -= 1;
    return s[a..b];
}

/// `_combine_text`, five steps pixel-perfect with prompt_tts_combine_text:
///   1. strip(ref_text) + " " + strip(text) (just strip(text) if ref empty)
///   2. drop CR / LF
///   3. U+FF08 / U+FF09 -> ASCII '(' / ')'
///   4. collapse [ \t]+ runs into a single ' '
///   5. drop any ' ' adjacent (either side) to a CJK ideograph U+4E00..9FFF
/// Malformed UTF-8 lead bytes are skipped one byte at a time, like the
/// reference decoder. Caller frees the result.
pub fn combineText(allocator: Allocator, text: []const u8, ref_text: []const u8) ![]u8 {
    const text_stripped = stripSpacesTabs(text);
    const ref_stripped = stripSpacesTabs(ref_text);

    var raw: std.ArrayList(u8) = .empty;
    defer raw.deinit(allocator);
    if (ref_stripped.len == 0) {
        try raw.appendSlice(allocator, text_stripped);
    } else {
        try raw.appendSlice(allocator, ref_stripped);
        try raw.append(allocator, ' ');
        try raw.appendSlice(allocator, text_stripped);
    }

    // Steps 2 + 3 inline while decoding to codepoints.
    var cps: std.ArrayList(u32) = .empty;
    defer cps.deinit(allocator);
    var i: usize = 0;
    while (i < raw.items.len) {
        var cp: u32 = 0;
        const n = utf8Decode(raw.items[i..], &cp);
        if (n == 0) {
            i += 1;
            continue;
        }
        i += n;
        if (cp == '\r' or cp == '\n') continue;
        if (cp == 0xFF08) {
            cp = '(';
        } else if (cp == 0xFF09) {
            cp = ')';
        }
        try cps.append(allocator, cp);
    }

    // Step 4: collapse space/tab runs into one space.
    var collapsed: std.ArrayList(u32) = .empty;
    defer collapsed.deinit(allocator);
    var in_space = false;
    for (cps.items) |cp| {
        const is_ws = cp == ' ' or cp == '\t';
        if (is_ws) {
            if (!in_space) {
                try collapsed.append(allocator, ' ');
                in_space = true;
            }
        } else {
            try collapsed.append(allocator, cp);
            in_space = false;
        }
    }

    // Step 5: drop spaces adjacent to CJK ideographs, encode back to UTF-8.
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (collapsed.items, 0..) |cp, j| {
        if (cp == ' ') {
            const prev_cjk = j > 0 and isCjk(collapsed.items[j - 1]);
            const next_cjk = j + 1 < collapsed.items.len and isCjk(collapsed.items[j + 1]);
            if (prev_cjk or next_cjk) continue;
        }
        try utf8Append(allocator, &out, cp);
    }
    return out.toOwnedSlice(allocator);
}

fn appendEncoded(allocator: Allocator, tok: *const Tokenizer, span: []const u8, ids: *std.ArrayList(i32)) !void {
    const encoded = try tok.encodeRaw(allocator, span);
    defer allocator.free(encoded);
    for (encoded) |id| try ids.append(allocator, @intCast(id));
}

/// prompt_tts_tokenize_nonverbal: scan for the LEFTMOST of the 13 non-verbal
/// tags; text between tags and each tag itself are BPE-encoded independently
/// (encodeRaw, no BOS/EOS) and concatenated, so tag id streams stay stable
/// across surrounding context. Caller frees the result.
pub fn tokenizeNonverbal(allocator: Allocator, tok: *const Tokenizer, text: []const u8) ![]i32 {
    var ids: std.ArrayList(i32) = .empty;
    errdefer ids.deinit(allocator);
    var pos: usize = 0;
    while (pos < text.len) {
        var best_pos: ?usize = null;
        var best_tag: []const u8 = undefined;
        for (&nonverbal_tags) |tag| {
            if (std.mem.indexOfPos(u8, text, pos, tag)) |p| {
                if (best_pos == null or p < best_pos.?) {
                    best_pos = p;
                    best_tag = tag;
                }
            }
        }
        const tag_pos = best_pos orelse {
            try appendEncoded(allocator, tok, text[pos..], &ids);
            break;
        };
        if (tag_pos > pos) {
            try appendEncoded(allocator, tok, text[pos..tag_pos], &ids);
        }
        try appendEncoded(allocator, tok, best_tag, &ids);
        pos = tag_pos + best_tag.len;
    }
    return ids.toOwnedSlice(allocator);
}

/// The built prompt + CFG batch (reference PromptTTS, minus the attention
/// matrices). Row 0 = cond, row 1 = uncond; B is always 1.
pub const Prompt = struct {
    allocator: Allocator,
    /// [B'=2, K, S_max] i32, b slow / k mid / s fast.
    input_ids: []i32,
    /// [2, S_max] 0/1 i32.
    audio_mask: []i32,
    num_codebooks: usize,
    c_len: usize,
    u_len: usize,
    s_max: usize,

    pub fn deinit(self: *Prompt) void {
        self.allocator.free(self.input_ids);
        self.allocator.free(self.audio_mask);
        self.* = undefined;
    }

    /// Cond row ids [K, S_max] (k slow). Mutated in place during decoding.
    pub fn condIds(self: *const Prompt) []i32 {
        return self.input_ids[0 .. self.num_codebooks * self.s_max];
    }

    /// Uncond row ids [K, S_max] (k slow).
    pub fn uncondIds(self: *const Prompt) []i32 {
        return self.input_ids[self.num_codebooks * self.s_max .. 2 * self.num_codebooks * self.s_max];
    }

    pub fn condAudioMask(self: *const Prompt) []const i32 {
        return self.audio_mask[0..self.s_max];
    }

    pub fn uncondAudioMask(self: *const Prompt) []const i32 {
        return self.audio_mask[self.s_max .. 2 * self.s_max];
    }
};

pub const BuildOptions = struct {
    text: []const u8,
    lang: []const u8 = "",
    /// Already resolved/normalised (pipeline_tts_resolve_instruct is the
    /// caller's job — voicedesign.normalize with use_zh = hasCjk(text)).
    instruct: []const u8 = "",
    num_target_tokens: usize,
    denoise: bool = true,
    ref_text: []const u8 = "",
    /// [K, ref_len] i32, k slow / t fast; null on the no-reference paths.
    ref_audio_tokens: ?[]const i32 = null,
    ref_len: usize = 0,
};

/// prompt_tts_build: assemble style + text token ids and fill the CFG batch
/// buffers. `<|denoise|>` is emitted iff denoise AND ref_audio_tokens != null.
pub fn build(allocator: Allocator, tok: *const Tokenizer, config: *const lm.Config, options: BuildOptions) !Prompt {
    if (options.num_target_tokens == 0) return Error.InvalidTargetTokens;
    const has_ref = options.ref_audio_tokens != null;
    if (has_ref) {
        if (options.ref_len == 0) return Error.InvalidRefTokens;
        if (options.ref_audio_tokens.?.len != config.num_audio_codebook * options.ref_len) return Error.InvalidRefTokens;
    }

    const specials = config.specials;

    // Style segment: [denoise?] lang_start enc(lang) lang_end
    //                instruct_start enc(instruct) instruct_end.
    var style_ids: std.ArrayList(i32) = .empty;
    defer style_ids.deinit(allocator);
    if (options.denoise and has_ref) try style_ids.append(allocator, @intCast(specials.denoise));
    try style_ids.append(allocator, @intCast(specials.lang_start));
    const lang_resolved = langmap.resolveLanguage(options.lang);
    const lang_str = if (lang_resolved.len == 0) "None" else lang_resolved;
    try appendEncoded(allocator, tok, lang_str, &style_ids);
    try style_ids.append(allocator, @intCast(specials.lang_end));
    try style_ids.append(allocator, @intCast(specials.instruct_start));
    const instruct_str = if (options.instruct.len == 0) "None" else options.instruct;
    try appendEncoded(allocator, tok, instruct_str, &style_ids);
    try style_ids.append(allocator, @intCast(specials.instruct_end));

    // Text segment: text_start ++ tokenize_nonverbal(_combine_text) ++ text_end.
    const full_text = try combineText(allocator, options.text, options.ref_text);
    defer allocator.free(full_text);
    var text_ids: std.ArrayList(i32) = .empty;
    defer text_ids.deinit(allocator);
    try text_ids.append(allocator, @intCast(specials.text_start));
    {
        const inner = try tokenizeNonverbal(allocator, tok, full_text);
        defer allocator.free(inner);
        try text_ids.appendSlice(allocator, inner);
    }
    try text_ids.append(allocator, @intCast(specials.text_end));

    return fillFromIds(
        allocator,
        config.num_audio_codebook,
        @intCast(config.audio_mask_id),
        style_ids.items,
        text_ids.items,
        options.ref_audio_tokens,
        options.ref_len,
        options.num_target_tokens,
    );
}

/// The buffer fill of prompt_tts_build (:259-343), split out so the exact
/// offsets are unit-testable without a tokenizer. All input_ids start at
/// mask_id; cond rows get style/text ids duplicated across K, per-k ref codes
/// on [N1+N2, N1+N2+Sref); the target window keeps mask. The uncond row is a
/// copy of the cond tail (all mask at build), padded with mask. audio_mask:
/// cond 1 on [N1+N2, c_len), uncond 1 on [0, u_len).
pub fn fillFromIds(
    allocator: Allocator,
    num_codebooks: usize,
    mask_id: i32,
    style_ids: []const i32,
    text_ids: []const i32,
    ref_audio_tokens: ?[]const i32,
    ref_len: usize,
    num_target_tokens: usize,
) !Prompt {
    const n1 = style_ids.len;
    const n2 = text_ids.len;
    const s_ref = if (ref_audio_tokens != null) ref_len else 0;
    const c_len = n1 + n2 + s_ref + num_target_tokens;
    const u_len = num_target_tokens;
    const num_k = num_codebooks;

    const input_ids = try allocator.alloc(i32, 2 * num_k * c_len);
    errdefer allocator.free(input_ids);
    @memset(input_ids, mask_id);
    const audio_mask = try allocator.alloc(i32, 2 * c_len);
    errdefer allocator.free(audio_mask);
    @memset(audio_mask, 0);

    for (0..num_k) |k| {
        const cond_row = input_ids[k * c_len ..][0..c_len];
        @memcpy(cond_row[0..n1], style_ids);
        @memcpy(cond_row[n1..][0..n2], text_ids);
        if (ref_audio_tokens) |ref| {
            @memcpy(cond_row[n1 + n2 ..][0..s_ref], ref[k * ref_len ..][0..s_ref]);
        }
        // [c_len - Stgt, c_len) keeps mask_id from the init.
    }
    for (audio_mask[n1 + n2 .. c_len]) |*m| m.* = 1;

    for (0..num_k) |k| {
        const cond_row = input_ids[k * c_len ..][0..c_len];
        const uncond_row = input_ids[(num_k + k) * c_len ..][0..c_len];
        @memcpy(uncond_row[0..u_len], cond_row[c_len - num_target_tokens ..][0..u_len]);
    }
    for (audio_mask[c_len .. c_len + u_len]) |*m| m.* = 1;

    return .{
        .allocator = allocator,
        .input_ids = input_ids,
        .audio_mask = audio_mask,
        .num_codebooks = num_k,
        .c_len = c_len,
        .u_len = u_len,
        .s_max = c_len,
    };
}

test {
    _ = @import("prompt_tests.zig");
}
