//! LocateAnything tokenizer: the shared Qwen2 byte-level BPE from
//! `fucina_llm.tokenizer` plus the reference's special-token handling.
//!
//! The reference (refs/locate-anything.cpp/src/tokenizer.cpp `encode`,
//! L259-299) treats every `token_type == 4` vocab entry (the HF
//! `added_tokens.json` set: `<|im_start|>`, `<IMG_CONTEXT>`, `</c>`, `<box>`,
//! coordinate tokens, ...) as an atomic special, greedily matched
//! longest-first at every byte position; the text runs between matches go
//! through the qwen2 pretokenizer + BPE. That set does not follow the
//! `<|...|>` marker shape, so the base tokenizer's marker resolution is
//! bypassed (`encodePlainAppend`) and the splitting happens here.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const config_mod = @import("config.zig");

const Allocator = std.mem.Allocator;
const gguf = fucina.gguf;

pub const arch = config_mod.arch;

pub const Tokenizer = struct {
    allocator: Allocator,
    base: llm.tokenizer.Tokenizer,
    /// Special (atomic) token bytes -> id; slices into `special_blob`.
    special_ids: std.StringHashMap(u32),
    special_blob: []u8,
    /// Distinct special-token byte lengths, descending (longest match first).
    special_lens: []usize,

    pub fn initFromGguf(allocator: Allocator, file: *const gguf.File) !Tokenizer {
        const tokens_arr = file.getArray(arch ++ ".tokenizer.tokens") orelse return error.NoTokenizerVocab;
        const merges_arr = file.getArray(arch ++ ".tokenizer.merges") orelse return error.NoTokenizerVocab;
        const types_arr = file.getArray(arch ++ ".tokenizer.token_types") orelse return error.NoTokenizerVocab;

        const token_strings = try tokens_arr.stringSlices(allocator);
        defer allocator.free(token_strings);
        const merge_strings = try merges_arr.stringSlices(allocator);
        defer allocator.free(merge_strings);
        const types_view = try config_mod.i32ArrayValues(types_arr);
        if (types_view.len != token_strings.len) return error.InvalidMetadata;

        const types = try allocator.alloc(i32, types_view.len);
        defer allocator.free(types);
        for (types, 0..) |*t, i| t.* = types_view.get(i);

        return initFromArrays(allocator, token_strings, merge_strings, types);
    }

    /// Build from raw arrays (the GGUF-independent core; also the unit-test
    /// entry). `types[i] == 4` marks token i as an atomic special.
    pub fn initFromArrays(
        allocator: Allocator,
        token_strings: []const []const u8,
        merge_strings: []const []const u8,
        types: []const i32,
    ) !Tokenizer {
        var base = try llm.tokenizer.Tokenizer.initFromParts(allocator, token_strings, merge_strings, .{});
        errdefer base.deinit();

        // Collect the atomic specials (token_type == 4, HF "control").
        var blob_len: usize = 0;
        for (token_strings, 0..) |s, i| {
            if (types[i] == 4) blob_len += s.len;
        }
        const special_blob = try allocator.alloc(u8, blob_len);
        errdefer allocator.free(special_blob);

        var special_ids = std.StringHashMap(u32).init(allocator);
        errdefer special_ids.deinit();

        var lens_set: std.AutoArrayHashMapUnmanaged(usize, void) = .empty;
        defer lens_set.deinit(allocator);

        var off: usize = 0;
        for (token_strings, 0..) |s, i| {
            if (types[i] != 4 or s.len == 0) continue;
            @memcpy(special_blob[off..][0..s.len], s);
            const stored = special_blob[off..][0..s.len];
            off += s.len;
            // First writer wins on duplicate bytes (matches lowest id).
            if (!special_ids.contains(stored)) try special_ids.put(stored, @intCast(i));
            try lens_set.put(allocator, s.len, {});
        }

        const special_lens = try allocator.dupe(usize, lens_set.keys());
        errdefer allocator.free(special_lens);
        std.mem.sort(usize, special_lens, {}, std.sort.desc(usize));

        return .{
            .allocator = allocator,
            .base = base,
            .special_ids = special_ids,
            .special_blob = special_blob,
            .special_lens = special_lens,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.allocator.free(self.special_lens);
        self.special_ids.deinit();
        self.allocator.free(self.special_blob);
        self.base.deinit();
        self.* = undefined;
    }

    /// Reference `Tokenizer::encode`: longest atomic special match at every
    /// position; the runs in between go through pretokenize + BPE only.
    pub fn encode(self: *const Tokenizer, allocator: Allocator, text: []const u8) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(allocator);

        var run_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            var matched = false;
            for (self.special_lens) |len| {
                if (i + len > text.len) continue;
                if (self.special_ids.get(text[i..][0..len])) |id| {
                    try self.base.encodePlainAppend(allocator, text[run_start..i], &out);
                    try out.append(allocator, id);
                    i += len;
                    run_start = i;
                    matched = true;
                    break;
                }
            }
            if (!matched) i += 1;
        }
        try self.base.encodePlainAppend(allocator, text[run_start..], &out);
        return out.toOwnedSlice(allocator);
    }

    /// Byte-decode a token-id span (labels round-trip through the GPT-2
    /// byte-level map, e.g. "trafficĠlight" -> "traffic light").
    pub fn decode(self: *const Tokenizer, allocator: Allocator, ids: []const u32) ![]u8 {
        return self.base.decode(allocator, ids);
    }

    pub fn tokenId(self: *const Tokenizer, piece: []const u8) ?u32 {
        return self.base.tokenId(piece);
    }
};

/// Reference `build_prompt` (src/prompt.cpp): the HF chat template with the
/// `<image-1>` placeholder expanded to `<image 1><img>{<IMG_CONTEXT> x N}</img>`,
/// N = (gh/merge_h)*(gw/merge_w) merged vision tokens.
pub fn buildPrompt(
    allocator: Allocator,
    tok: *const Tokenizer,
    n_image_tokens: usize,
    query: []const u8,
) ![]u32 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);

    try text.appendSlice(allocator, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    try text.appendSlice(allocator, "<|im_start|>user\n");
    try text.appendSlice(allocator, "<image 1><img>");
    for (0..n_image_tokens) |_| try text.appendSlice(allocator, "<IMG_CONTEXT>");
    try text.appendSlice(allocator, "</img>");
    try text.appendSlice(allocator, query);
    try text.appendSlice(allocator, "<|im_end|>\n<|im_start|>assistant\n");

    return tok.encode(allocator, text.items);
}

test {
    _ = @import("tokenizer_tests.zig");
}
