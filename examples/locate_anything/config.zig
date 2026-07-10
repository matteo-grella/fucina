//! LocateAnything-3B GGUF configuration: the `locateanything.*` metadata
//! schema written by the reference converter
//! (refs/locate-anything.cpp/scripts/gguf_keys.py). Every hyperparameter is
//! read from the file; nothing is hardcoded except the schema itself.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const gguf = fucina.gguf;
const gguf_meta = llm.gguf_meta;

pub const arch = "locateanything";

pub const Config = struct {
    // --- language model (Qwen2.5-3B) ---
    lm_hidden: usize,
    lm_n_layers: usize,
    lm_n_heads: usize,
    lm_n_kv_heads: usize,
    lm_head_dim: usize,
    lm_intermediate: usize,
    lm_vocab: usize,
    lm_rope_theta: f32,
    lm_rms_eps: f32,
    /// MTP / parallel-box-decoding block size (6).
    lm_block_size: usize,

    // --- vision tower (MoonViT) ---
    vit_hidden: usize,
    vit_n_layers: usize,
    vit_n_heads: usize,
    vit_head_dim: usize,
    vit_intermediate: usize,
    vit_patch: usize,
    vit_merge_h: usize,
    vit_merge_w: usize,
    vit_pos_emb_hw: usize,
    vit_rope_theta: f32,

    // --- special token ids ---
    tok_image: u32,
    tok_box_start: u32,
    tok_box_end: u32,
    tok_coord_start: u32,
    tok_coord_end: u32,
    tok_ref_start: u32,
    tok_ref_end: u32,
    tok_none: u32,
    tok_null: u32,
    tok_text_mask: u32,
    tok_eos: u32,
    tok_bos: u32,

    // --- image preprocessing ---
    in_token_limit: usize,

    pub fn load(file: *const gguf.File) !Config {
        const merge = file.getArray(arch ++ ".vit.merge_kernel_size") orelse return error.MissingMetadata;
        const merge_vals = try i32ArrayValues(merge);
        if (merge_vals.len != 2) return error.InvalidMetadata;
        // Range-validate like every metaInt sibling (reject_zero): a negative
        // value would trap in the @intCast below, and a zero would divide by
        // zero in the preprocessing grid math.
        for (0..2) |i| {
            const v = merge_vals.get(i);
            if (v < 1 or v > 16) return error.InvalidMetadata;
        }

        return .{
            .lm_hidden = try gguf_meta.metaInt(file, arch, "lm.hidden_size", .reject_zero),
            .lm_n_layers = try gguf_meta.metaInt(file, arch, "lm.n_layers", .reject_zero),
            .lm_n_heads = try gguf_meta.metaInt(file, arch, "lm.n_heads", .reject_zero),
            .lm_n_kv_heads = try gguf_meta.metaInt(file, arch, "lm.n_kv_heads", .reject_zero),
            .lm_head_dim = try gguf_meta.metaInt(file, arch, "lm.head_dim", .reject_zero),
            .lm_intermediate = try gguf_meta.metaInt(file, arch, "lm.intermediate_size", .reject_zero),
            .lm_vocab = try gguf_meta.metaInt(file, arch, "lm.vocab_size", .reject_zero),
            .lm_rope_theta = try gguf_meta.metaFloat(file, arch, "lm.rope_theta"),
            .lm_rms_eps = try gguf_meta.metaFloat(file, arch, "lm.rms_norm_eps"),
            .lm_block_size = try gguf_meta.metaInt(file, arch, "lm.block_size", .reject_zero),
            .vit_hidden = try gguf_meta.metaInt(file, arch, "vit.hidden_size", .reject_zero),
            .vit_n_layers = try gguf_meta.metaInt(file, arch, "vit.n_layers", .reject_zero),
            .vit_n_heads = try gguf_meta.metaInt(file, arch, "vit.n_heads", .reject_zero),
            .vit_head_dim = try gguf_meta.metaInt(file, arch, "vit.head_dim", .reject_zero),
            .vit_intermediate = try gguf_meta.metaInt(file, arch, "vit.intermediate_size", .reject_zero),
            .vit_patch = try gguf_meta.metaInt(file, arch, "vit.patch_size", .reject_zero),
            .vit_merge_h = @intCast(merge_vals.get(0)),
            .vit_merge_w = @intCast(merge_vals.get(1)),
            .vit_pos_emb_hw = try gguf_meta.metaInt(file, arch, "vit.init_pos_emb_hw", .reject_zero),
            .vit_rope_theta = try gguf_meta.metaFloat(file, arch, "vit.rope_theta"),
            .tok_image = try tokenId(file, "image"),
            .tok_box_start = try tokenId(file, "box_start"),
            .tok_box_end = try tokenId(file, "box_end"),
            .tok_coord_start = try tokenId(file, "coord_start"),
            .tok_coord_end = try tokenId(file, "coord_end"),
            .tok_ref_start = try tokenId(file, "ref_start"),
            .tok_ref_end = try tokenId(file, "ref_end"),
            .tok_none = try tokenId(file, "none"),
            .tok_null = try tokenId(file, "null"),
            .tok_text_mask = try tokenId(file, "text_mask"),
            .tok_eos = try tokenId(file, "eos"),
            .tok_bos = try tokenId(file, "bos"),
            .in_token_limit = try gguf_meta.metaInt(file, arch, "image.in_token_limit", .reject_zero),
        };
    }

    fn tokenId(file: *const gguf.File, comptime name: []const u8) !u32 {
        const value = file.getInt(arch ++ ".token." ++ name) orelse return error.MissingMetadata;
        if (value < 0 or value > std.math.maxInt(u32)) return error.InvalidMetadata;
        return @intCast(value);
    }
};

/// Typed view over a fixed-width-integer GGUF metadata array (the file bytes
/// are little-endian and may be unaligned, so elements are decoded per access).
pub const I32Array = struct {
    data: []const u8,
    len: usize,

    pub fn get(self: I32Array, index: usize) i32 {
        return std.mem.readInt(i32, self.data[index * 4 ..][0..4], .little);
    }
};

pub fn i32ArrayValues(array: gguf.Array) !I32Array {
    // GGUF metadata type 5 = INT32 (see gguf.MetaType).
    if (array.item_type != 5) return error.InvalidMetadata;
    if (array.data.len < array.len * 4) return error.InvalidMetadata;
    return .{ .data = array.data, .len = array.len };
}
