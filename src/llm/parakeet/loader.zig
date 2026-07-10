//! Parakeet GGUF loader — NeMo FastConformer ASR.
//!
//! Parses the `parakeet.*` metadata into a `Config`, resolves every expected
//! tensor by its verbatim NeMo `state_dict` name and validates shape + dtype
//! class, and borrows the featurizer (mel filterbank + STFT window) and the
//! tokenizer pieces from the mapped GGUF. Weight materialization into the model
//! engine happens in the sibling modules; this file is parse + validate +
//! zero-copy borrow only.
//!
//! Tensor-name scheme and GGUF metadata keys follow parakeet.cpp's
//! `scripts/convert_parakeet_to_gguf.py` (the authoritative name/dtype map).
//! Dim order in GGUF is ggml `ne[]` (innermost/fastest axis first); validation
//! below uses that order verbatim.
const std = @import("std");
const fucina = @import("fucina");
const gguf = fucina.gguf;
const pweights = @import("weights.zig");

const Allocator = std.mem.Allocator;
const TensorInfo = gguf.TensorInfo;

pub const Error = error{
    NotParakeet,
    UnsupportedArch,
    UnsupportedConvNorm,
    InvalidConfig,
    MissingMetadata,
    TensorNotFound,
    TensorShapeMismatch,
    TensorDtypeMismatch,
};

/// Decoder family (`parakeet.arch`). The hybrid variants carry BOTH a transducer
/// (RNNT/TDT predictor + joint) and a CTC head; pure `ctc` has neither predictor
/// nor joint. The CLI decoder choice is separate — this only describes which
/// weights exist.
pub const DecoderArch = enum {
    ctc,
    rnnt,
    tdt,
    hybrid_tdt_ctc,
    hybrid_rnnt_ctc,

    pub fn fromStr(s: []const u8) Error!DecoderArch {
        const map = .{
            .{ "ctc", DecoderArch.ctc },
            .{ "rnnt", DecoderArch.rnnt },
            .{ "tdt", DecoderArch.tdt },
            .{ "hybrid_tdt_ctc", DecoderArch.hybrid_tdt_ctc },
            .{ "hybrid_rnnt_ctc", DecoderArch.hybrid_rnnt_ctc },
        };
        inline for (map) |entry| {
            if (std.mem.eql(u8, s, entry[0])) return entry[1];
        }
        return Error.UnsupportedArch;
    }

    /// True when a CTC head (`ctc_decoder.*`) is present.
    pub fn hasCtc(self: DecoderArch) bool {
        return self == .ctc or self == .hybrid_tdt_ctc or self == .hybrid_rnnt_ctc;
    }

    /// True when a transducer (LSTM predictor + joint network) is present.
    pub fn hasTransducer(self: DecoderArch) bool {
        return self != .ctc;
    }

    /// True when the joint carries a TDT duration head (`tdt.durations`).
    pub fn isTdt(self: DecoderArch) bool {
        return self == .tdt or self == .hybrid_tdt_ctc;
    }
};

pub const ConvNorm = enum { batch_norm, layer_norm };
/// Mel per-feature normalization. `per_feature` = NeMo per-feature z-score over the
/// whole clip (offline models). `na` = none (the streaming `realtime_eou` model —
/// streaming-friendly: no whole-clip statistic needed).
pub const Normalize = enum { per_feature, na };
pub const JointActivation = enum { relu, tanh };

/// Largest TDT duration table we accept (real models use 5: [0,1,2,3,4]).
pub const max_durations = 16;

pub const Config = struct {
    arch: DecoderArch,

    // --- FastConformer encoder ---
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize, // derived: d_model / n_heads
    ff_dim: usize,
    feat_in: usize, // encoder input feature dim (== n_mels)
    conv_kernel: usize, // conformer conv-module depthwise kernel (k=9)
    conv_norm: ConvNorm,
    subsampling_factor: usize,
    subsampling_stages: usize, // derived: log2(subsampling_factor)
    subsampling_conv_channels: usize,
    pos_emb_max_len: usize,
    xscaling: bool,

    // --- Mel front-end (preprocessor) ---
    sample_rate: usize,
    n_mels: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    preemph: f32,
    mag_power: f32,
    log_zero_guard: f32,
    normalize: Normalize,

    // --- Vocabulary ---
    vocab_size: usize,
    blank_id: usize,

    // --- Decoder (LSTM predictor) ---
    pred_hidden: usize,
    pred_rnn_layers: usize,
    max_symbols: usize,

    // --- Joint network ---
    joint_hidden: usize,
    joint_activation: JointActivation,

    // --- TDT duration head ---
    num_durations: usize,
    durations: [max_durations]i32,

    /// Joint-network output width = vocab + blank + durations. Equals the joint's
    /// `joint_net.2` output dim and the CTC vocab is `vocab_size + 1` (blank).
    pub fn vPlus(self: Config) usize {
        return self.vocab_size + 1 + self.num_durations;
    }

    pub fn checkedVPlus(self: Config) Error!usize {
        const vocab_plus_blank = std.math.add(usize, self.vocab_size, 1) catch return Error.InvalidConfig;
        return std.math.add(usize, vocab_plus_blank, self.num_durations) catch return Error.InvalidConfig;
    }

    /// Subsampled frequency bins fed to the encoder linear proj: ceil-halve the
    /// mel axis once per stride-2 subsampling stage.
    pub fn subsampledFreq(self: Config) usize {
        var f = self.n_mels;
        var i: usize = 0;
        while (i < self.subsampling_stages) : (i += 1) f = (f + 1) / 2;
        return f;
    }

    pub fn checkedSubsampledFreq(self: Config) Error!usize {
        var f = self.n_mels;
        var i: usize = 0;
        while (i < self.subsampling_stages) : (i += 1) {
            f = (std.math.add(usize, f, 1) catch return Error.InvalidConfig) / 2;
        }
        return f;
    }

    pub fn durationsSlice(self: *const Config) []const i32 {
        return self.durations[0..self.num_durations];
    }

    pub fn fromGguf(file: *const gguf.File) Error!Config {
        const general_arch = file.getString("general.architecture") orelse return Error.NotParakeet;
        if (!std.mem.eql(u8, general_arch, "parakeet")) return Error.NotParakeet;

        const arch = try DecoderArch.fromStr(metaStr(file, "arch") orelse return Error.MissingMetadata);

        const d_model = try metaInt(file, "encoder.d_model");
        const n_heads = try metaInt(file, "encoder.n_heads");
        if (n_heads == 0 or d_model % n_heads != 0) return Error.InvalidConfig;

        const subsampling_factor = try metaInt(file, "encoder.subsampling_factor");
        if (subsampling_factor == 0 or (subsampling_factor & (subsampling_factor - 1)) != 0)
            return Error.InvalidConfig; // must be a power of two

        const conv_norm = try parseConvNorm(metaStr(file, "encoder.conv_norm_type") orelse "batch_norm");
        const normalize = try parseNormalize(metaStr(file, "preprocessor.normalize") orelse "per_feature");
        const joint_activation = try parseJointActivation(metaStr(file, "joint.activation") orelse "relu");

        var cfg = Config{
            .arch = arch,
            .d_model = d_model,
            .n_layers = try metaInt(file, "encoder.n_layers"),
            .n_heads = n_heads,
            .head_dim = d_model / n_heads,
            .ff_dim = try metaInt(file, "encoder.ff_dim"),
            .feat_in = try metaInt(file, "encoder.feat_in"),
            .conv_kernel = try metaInt(file, "encoder.conv_kernel"),
            .conv_norm = conv_norm,
            .subsampling_factor = subsampling_factor,
            .subsampling_stages = std.math.log2_int(usize, subsampling_factor),
            .subsampling_conv_channels = try metaInt(file, "encoder.subsampling_conv_channels"),
            .pos_emb_max_len = metaIntOpt(file, "encoder.pos_emb_max_len") orelse 5000,
            .xscaling = metaBoolOpt(file, "encoder.xscaling") orelse false,

            .sample_rate = metaIntOpt(file, "preprocessor.sample_rate") orelse 16000,
            .n_mels = try metaInt(file, "preprocessor.n_mels"),
            .n_fft = try metaInt(file, "preprocessor.n_fft"),
            .win_length = try metaInt(file, "preprocessor.win_length"),
            .hop_length = try metaInt(file, "preprocessor.hop_length"),
            .preemph = metaFloatOpt(file, "preprocessor.preemph") orelse 0.97,
            .mag_power = metaFloatOpt(file, "preprocessor.mag_power") orelse 2.0,
            .log_zero_guard = metaFloatOpt(file, "preprocessor.log_zero_guard") orelse 5.9604645e-8,
            .normalize = normalize,

            .vocab_size = try metaInt(file, "vocab_size"),
            .blank_id = try metaInt(file, "blank_id"),

            .pred_hidden = metaIntOpt(file, "decoder.pred_hidden") orelse 640,
            .pred_rnn_layers = metaIntOpt(file, "decoder.pred_rnn_layers") orelse 1,
            .max_symbols = metaIntOpt(file, "decoding.max_symbols") orelse 10,

            .joint_hidden = metaIntOpt(file, "joint.joint_hidden") orelse 640,
            .joint_activation = joint_activation,

            .num_durations = 0,
            .durations = [_]i32{0} ** max_durations,
        };

        // TDT durations (a small int array). Absent on pure-CTC models.
        if (file.getArray("parakeet.tdt.durations")) |arr| {
            if (arr.len > max_durations) return Error.InvalidConfig;
            const esz: usize = switch (arr.item_type) {
                4, 5 => 4,
                10, 11 => 8,
                else => return Error.InvalidConfig,
            };
            const min_len = std.math.mul(usize, arr.len, esz) catch return Error.InvalidConfig;
            if (arr.data.len < min_len) return Error.InvalidConfig;
            for (0..arr.len) |i| {
                cfg.durations[i] = intArrAt(arr, i) orelse return Error.InvalidConfig;
                if (cfg.durations[i] < 0) return Error.InvalidConfig;
            }
            cfg.num_durations = arr.len;
        }

        if (cfg.arch.isTdt() and cfg.num_durations == 0) return Error.InvalidConfig;
        if (!cfg.arch.isTdt() and cfg.num_durations != 0) return Error.InvalidConfig;
        try cfg.validate();
        return cfg;
    }

    fn validate(self: Config) Error!void {
        if (self.d_model == 0 or self.n_layers == 0 or self.n_heads == 0 or self.head_dim == 0) return Error.InvalidConfig;
        if (self.ff_dim == 0 or self.feat_in == 0 or self.conv_kernel == 0) return Error.InvalidConfig;
        if (self.subsampling_factor == 0 or self.subsampling_conv_channels == 0) return Error.InvalidConfig;
        if (self.sample_rate == 0 or self.n_mels == 0 or self.n_fft == 0 or self.win_length == 0 or self.hop_length == 0) return Error.InvalidConfig;
        if (self.win_length > self.n_fft) return Error.InvalidConfig;
        if (self.vocab_size == 0 or self.blank_id > self.vocab_size) return Error.InvalidConfig;
        if (self.pred_hidden == 0 or self.pred_rnn_layers == 0 or self.max_symbols == 0) return Error.InvalidConfig;
        if (self.joint_hidden == 0) return Error.InvalidConfig;
        _ = try self.checkedVPlus();
        _ = try self.checkedSubsampledFreq();
    }
};

/// Encoder attention windowing (`parakeet.encoder.att_context_style`). `regular`
/// is full/bidirectional attention (offline models); `chunked_limited` is the
/// streaming windowed attention (left/right context bounded).
pub const AttContextStyle = enum {
    regular,
    chunked_limited,

    pub fn fromStr(s: []const u8) AttContextStyle {
        if (std.mem.eql(u8, s, "chunked_limited")) return .chunked_limited;
        return .regular;
    }
};

/// Cache-aware streaming metadata (`parakeet.streaming.*` + the encoder
/// `att_context_*` keys). Present only on streaming-variant models (e.g.
/// `realtime_eou_120m-v1`); `fromGguf` returns null for offline models. The list
/// fields are `[step0, step≥1]` (the first chunk vs subsequent chunks); a length-1
/// GGUF array broadcasts to both. Mirrors `refs/parakeet.cpp` `src/model_loader.cpp:142-167`.
pub const StreamingConfig = struct {
    att_context_left: i32, // left attention context in encoder frames (-1 = unlimited)
    att_context_right: i32, // right context / lookahead in encoder frames
    att_context_style: AttContextStyle,

    chunk_size: [2]i32, // mel frames consumed per step (e.g. [9, 16])
    shift_size: [2]i32, // step advance in mel frames (e.g. [9, 16])
    pre_encode_cache_size: [2]i32, // mel-frame overlap prepended to step≥1 (e.g. [0, 9])
    cache_drop_size: i32,
    last_channel_cache_size: i32, // attention K/V cache depth (== att_context_left, e.g. 70)
    valid_out_len: i32, // encoder frames dropped from each chunk's tail (kept only on the last)
    drop_extra_pre_encoded: i32,

    /// Returns null when the model has no streaming metadata (offline models),
    /// detected by the absence of the `streaming.chunk_size` schedule.
    pub fn fromGguf(file: *const gguf.File) Error!?StreamingConfig {
        const chunk = (try metaIntArr2(file, "streaming.chunk_size")) orelse return null;
        return StreamingConfig{
            .att_context_left = metaI32Opt(file, "encoder.att_context_left") orelse -1,
            .att_context_right = metaI32Opt(file, "encoder.att_context_right") orelse -1,
            .att_context_style = AttContextStyle.fromStr(metaStr(file, "encoder.att_context_style") orelse "regular"),
            .chunk_size = chunk,
            .shift_size = (try metaIntArr2(file, "streaming.shift_size")) orelse chunk,
            .pre_encode_cache_size = (try metaIntArr2(file, "streaming.pre_encode_cache_size")) orelse .{ 0, 0 },
            .cache_drop_size = metaI32Opt(file, "streaming.cache_drop_size") orelse 0,
            .last_channel_cache_size = metaI32Opt(file, "streaming.last_channel_cache_size") orelse 0,
            .valid_out_len = metaI32Opt(file, "streaming.valid_out_len") orelse 0,
            .drop_extra_pre_encoded = metaI32Opt(file, "streaming.drop_extra_pre_encoded") orelse 0,
        };
    }

    /// Schedule index for a step: step 0 uses `[*][0]`, steps ≥ 1 use `[*][1]`.
    pub fn stepIdx(step: usize) usize {
        return if (step == 0) 0 else 1;
    }
};

/// Language-prompt conditioning config for the multilingual streaming model
/// (`nemotron-3.5-asr-streaming-0.6b`): a per-utterance locale one-hot is
/// concatenated to the encoder output and projected through `prompt_kernel`.
/// `default_lang` is a slice into the GGUF bytes (valid while the `File`
/// lives). Mirrors parakeet.cpp `PromptCfg`.
pub const PromptConfig = struct {
    present: bool,
    num_prompts: i32, // one-hot width (the prompt_kernel input is d_model + num_prompts)
    default_lang: []const u8, // resolved for an empty `--lang` (e.g. "auto")

    /// Returns null when the model is NOT prompt-conditioned (`prompt.present`
    /// absent/false) — i.e. the English models, which skip the prompt path.
    pub fn fromGguf(file: *const gguf.File) ?PromptConfig {
        if (!(metaBoolOpt(file, "prompt.present") orelse false)) return null;
        return .{
            .present = true,
            .num_prompts = metaI32Opt(file, "prompt.num_prompts") orelse 0,
            .default_lang = metaStr(file, "prompt.default_lang") orelse "",
        };
    }

    /// Resolve a target locale to its prompt one-hot index via the model's
    /// `dictionary.keys`→`values` map; an empty `target_lang` uses `default_lang`.
    /// Returns null if the (resolved) locale is unknown. Mirrors parakeet.cpp
    /// `PromptCfg::lang_to_index`. `allocator` backs only the transient key-slice
    /// array (freed before return).
    pub fn resolveLang(self: PromptConfig, file: *const gguf.File, allocator: Allocator, target_lang: []const u8) !?i32 {
        const lang = if (target_lang.len == 0) self.default_lang else target_lang;
        const keys = file.getArray("parakeet.prompt.dictionary.keys") orelse return null;
        const vals = file.getArray("parakeet.prompt.dictionary.values") orelse return null;
        const ks = try keys.stringSlices(allocator);
        defer allocator.free(ks);
        for (ks, 0..) |k, i| {
            if (std.mem.eql(u8, k, lang)) return intArrAt(vals, i);
        }
        return null;
    }
};

/// Read the i-th element of a ggml int array (item types 4=i32/5=u32/10=i64/11=u64)
/// as i32; null if out of range or a non-int item type.
fn intArrAt(arr: gguf.Array, i: usize) ?i32 {
    if (i >= arr.len) return null;
    const esz: usize = switch (arr.item_type) {
        4, 5 => 4,
        10, 11 => 8,
        else => return null,
    };
    const next = std.math.add(usize, i, 1) catch return null;
    const needed = std.math.mul(usize, next, esz) catch return null;
    if (arr.data.len < needed) return null;
    const off = std.math.mul(usize, i, esz) catch return null;
    return switch (arr.item_type) {
        4 => std.mem.readInt(i32, arr.data[off..][0..4], .little),
        5 => std.math.cast(i32, std.mem.readInt(u32, arr.data[off..][0..4], .little)),
        10 => std.math.cast(i32, std.mem.readInt(i64, arr.data[off..][0..8], .little)),
        11 => std.math.cast(i32, std.mem.readInt(u64, arr.data[off..][0..8], .little)),
        else => null,
    };
}

// --- metadata helpers (all keys are `parakeet.<suffix>`) ---

fn metaKey(buf: []u8, suffix: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "parakeet.{s}", .{suffix}) catch null;
}

fn metaIntOpt(file: *const gguf.File, suffix: []const u8) ?usize {
    var buf: [192]u8 = undefined;
    const key = metaKey(&buf, suffix) orelse return null;
    const v = file.getInt(key) orelse return null;
    if (v < 0) return null;
    return @intCast(v);
}

fn metaInt(file: *const gguf.File, suffix: []const u8) Error!usize {
    return metaIntOpt(file, suffix) orelse Error.MissingMetadata;
}

/// Signed scalar read (unlike `metaIntOpt`, allows negatives — e.g. att_context = -1).
fn metaI32Opt(file: *const gguf.File, suffix: []const u8) ?i32 {
    var buf: [192]u8 = undefined;
    const key = metaKey(&buf, suffix) orelse return null;
    const v = file.getInt(key) orelse return null;
    return std.math.cast(i32, v);
}

/// Parse a `parakeet.<suffix>` int array into `[step0, step≥1]` (length-1 broadcasts
/// to both). Handles ggml int item types 4=i32 / 5=u32 / 10=i64 / 11=u64. Returns
/// null if the key is absent.
fn metaIntArr2(file: *const gguf.File, suffix: []const u8) Error!?[2]i32 {
    var buf: [192]u8 = undefined;
    const key = metaKey(&buf, suffix) orelse return null;
    const arr = file.getArray(key) orelse return null;
    if (arr.len == 0) return Error.InvalidConfig;
    const esz: usize = switch (arr.item_type) {
        4, 5 => 4,
        10, 11 => 8,
        else => return Error.InvalidConfig,
    };
    const min_len = std.math.mul(usize, arr.len, esz) catch return Error.InvalidConfig;
    if (arr.data.len < min_len) return Error.InvalidConfig;
    var out: [2]i32 = undefined;
    for (0..2) |i| {
        const idx = if (i < arr.len) i else arr.len - 1; // broadcast a length-1 array
        out[i] = intArrAt(arr, idx) orelse return Error.InvalidConfig;
    }
    return out;
}

fn metaFloatOpt(file: *const gguf.File, suffix: []const u8) ?f32 {
    var buf: [192]u8 = undefined;
    const key = metaKey(&buf, suffix) orelse return null;
    const v = file.getFloat(key) orelse return null;
    return @floatCast(v);
}

fn metaBoolOpt(file: *const gguf.File, suffix: []const u8) ?bool {
    var buf: [192]u8 = undefined;
    const key = metaKey(&buf, suffix) orelse return null;
    return file.getBool(key);
}

fn metaStr(file: *const gguf.File, suffix: []const u8) ?[]const u8 {
    var buf: [192]u8 = undefined;
    const key = metaKey(&buf, suffix) orelse return null;
    return file.getString(key);
}

fn parseConvNorm(s: []const u8) Error!ConvNorm {
    if (std.mem.eql(u8, s, "batch_norm")) return .batch_norm;
    if (std.mem.eql(u8, s, "layer_norm")) return .layer_norm;
    return Error.UnsupportedConvNorm;
}

fn parseNormalize(s: []const u8) Error!Normalize {
    if (std.mem.eql(u8, s, "per_feature")) return .per_feature;
    if (std.mem.eql(u8, s, "NA") or std.mem.eql(u8, s, "none")) return .na;
    return Error.InvalidConfig;
}

fn parseJointActivation(s: []const u8) Error!JointActivation {
    if (std.mem.eql(u8, s, "relu")) return .relu;
    if (std.mem.eql(u8, s, "tanh")) return .tanh;
    return Error.InvalidConfig;
}

// --- tensor resolution + validation ---

/// Dtype expectation. `f32_required` tensors stay f32 across every quant format
/// (featurizer, convs, batch-norm, all norms/biases, embeddings, LSTM, the final
/// joint/ctc output layers). `quantizable` weights may be f16 OR a block-quant
/// (k-quant / q8_0) OR f32 (the selective-quant set: encoder FFN/attn linears,
/// the subsampling output proj, `joint.enc`/`joint.pred`) — only the shape is
/// checked. Verified against the f16 and q4_k 110m GGUFs.
pub const TensorClass = enum { f32_required, quantizable };

/// Resolve a tensor by verbatim name and validate ggml `ne[]` dims + dtype class.
pub fn expectTensor(
    file: *const gguf.File,
    name: []const u8,
    class: TensorClass,
    dims: []const usize,
) Error!*const TensorInfo {
    const t = file.maybeGet(name) orelse return Error.TensorNotFound;
    if (t.n_dims != dims.len) return Error.TensorShapeMismatch;
    for (dims, 0..) |d, i| {
        if (t.dims[i] != d) return Error.TensorShapeMismatch;
    }
    switch (class) {
        .f32_required => if (t.ggml_type != .f32) return Error.TensorDtypeMismatch,
        .quantizable => {},
    }
    return t;
}

/// The mel filterbank `preprocessor.featurizer.fb`, validated f32 with logical
/// shape [n_mels, n_fft/2+1]. GGUF stores it as ggml `ne=[bins, mels, 1]` in the
/// f16/q8_0 models but the k-quant (q4_k/q5_k/q6_k) conversion squeezes the
/// trailing singleton to `ne=[bins, mels]` — accept both.
fn fbTensor(file: *const gguf.File, cfg: Config) Error!*const TensorInfo {
    const bins = cfg.n_fft / 2 + 1;
    const t = file.maybeGet("preprocessor.featurizer.fb") orelse return Error.TensorNotFound;
    if (t.ggml_type != .f32) return Error.TensorDtypeMismatch;
    const ok2 = t.n_dims == 2 and t.dims[0] == bins and t.dims[1] == cfg.n_mels;
    const ok3 = t.n_dims == 3 and t.dims[0] == bins and t.dims[1] == cfg.n_mels and t.dims[2] == 1;
    if (!ok2 and !ok3) return Error.TensorShapeMismatch;
    return t;
}

fn layerName(buf: []u8, il: usize, suffix: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "encoder.layers.{d}.{s}", .{ il, suffix }) catch unreachable;
}

/// Resolve and validate EVERY tensor the configured architecture requires.
/// Returns the number of tensors validated. Errors on the first missing or
/// mismatched tensor (the "every expected tensor resolves" load-time gate).
pub fn validateTensors(file: *const gguf.File, cfg: Config) Error!usize {
    var n: usize = 0;
    var nb: [256]u8 = undefined;

    const d = cfg.d_model;
    const ff = cfg.ff_dim;
    const hd = cfg.head_dim;
    const H = cfg.n_heads;
    const cc = cfg.subsampling_conv_channels;
    const ck = cfg.conv_kernel;
    const ph = cfg.pred_hidden;
    const jh = cfg.joint_hidden;
    const v1 = std.math.add(usize, cfg.vocab_size, 1) catch return Error.InvalidConfig; // incl. blank
    const vp = try cfg.checkedVPlus();
    const subsampled_freq = try cfg.checkedSubsampledFreq();
    const pre_encode_out_in = std.math.mul(usize, cc, subsampled_freq) catch return Error.InvalidConfig;
    const two_d = std.math.mul(usize, 2, d) catch return Error.InvalidConfig;
    const four_ph = std.math.mul(usize, 4, ph) catch return Error.InvalidConfig;

    // Featurizer (always f32).
    _ = try expectTensor(file, "preprocessor.featurizer.window", .f32_required, &.{cfg.win_length});
    n += 1;
    _ = try fbTensor(file, cfg);
    n += 1;

    // Subsampling stem: 3×3 depthwise + 1×1 pointwise conv2d stack (f32 kernels),
    // then the linear proj to d_model (quantizable). conv sub-indices 1/4 are
    // activations (no weights).
    n += try expectConv(file, "encoder.pre_encode.conv.0", &.{ 3, 3, 1, cc }, cc);
    n += try expectConv(file, "encoder.pre_encode.conv.2", &.{ 3, 3, 1, cc }, cc);
    n += try expectConv(file, "encoder.pre_encode.conv.3", &.{ 1, 1, cc, cc }, cc);
    n += try expectConv(file, "encoder.pre_encode.conv.5", &.{ 3, 3, 1, cc }, cc);
    n += try expectConv(file, "encoder.pre_encode.conv.6", &.{ 1, 1, cc, cc }, cc);
    _ = try expectTensor(file, "encoder.pre_encode.out.weight", .quantizable, &.{ pre_encode_out_in, d });
    n += 1;
    _ = try expectTensor(file, "encoder.pre_encode.out.bias", .f32_required, &.{d});
    n += 1;

    // Conformer blocks.
    for (0..cfg.n_layers) |il| {
        // macaron FFN #1
        n += try expectNorm(file, layerName(&nb, il, "norm_feed_forward1"), d);
        n += try expectLinear(file, layerName(&nb, il, "feed_forward1.linear1"), &.{ d, ff }, ff);
        n += try expectLinear(file, layerName(&nb, il, "feed_forward1.linear2"), &.{ ff, d }, d);
        // relpos self-attention
        n += try expectNorm(file, layerName(&nb, il, "norm_self_att"), d);
        n += try expectLinear(file, layerName(&nb, il, "self_attn.linear_q"), &.{ d, d }, d);
        n += try expectLinear(file, layerName(&nb, il, "self_attn.linear_k"), &.{ d, d }, d);
        n += try expectLinear(file, layerName(&nb, il, "self_attn.linear_v"), &.{ d, d }, d);
        n += try expectLinear(file, layerName(&nb, il, "self_attn.linear_out"), &.{ d, d }, d);
        _ = try expectTensor(file, layerName(&nb, il, "self_attn.linear_pos.weight"), .quantizable, &.{ d, d });
        n += 1;
        _ = try expectTensor(file, layerName(&nb, il, "self_attn.pos_bias_u"), .f32_required, &.{ hd, H });
        n += 1;
        _ = try expectTensor(file, layerName(&nb, il, "self_attn.pos_bias_v"), .f32_required, &.{ hd, H });
        n += 1;
        // conv module
        n += try expectNorm(file, layerName(&nb, il, "norm_conv"), d);
        _ = try expectTensor(file, layerName(&nb, il, "conv.pointwise_conv1.weight"), .f32_required, &.{ 1, d, two_d });
        n += 1;
        _ = try expectTensor(file, layerName(&nb, il, "conv.pointwise_conv1.bias"), .f32_required, &.{two_d});
        n += 1;
        _ = try expectTensor(file, layerName(&nb, il, "conv.depthwise_conv.weight"), .f32_required, &.{ ck, 1, d });
        n += 1;
        _ = try expectTensor(file, layerName(&nb, il, "conv.depthwise_conv.bias"), .f32_required, &.{d});
        n += 1;
        if (cfg.conv_norm == .batch_norm) {
            _ = try expectTensor(file, layerName(&nb, il, "conv.batch_norm.weight"), .f32_required, &.{d});
            n += 1;
            _ = try expectTensor(file, layerName(&nb, il, "conv.batch_norm.bias"), .f32_required, &.{d});
            n += 1;
            _ = try expectTensor(file, layerName(&nb, il, "conv.batch_norm.running_mean"), .f32_required, &.{d});
            n += 1;
            _ = try expectTensor(file, layerName(&nb, il, "conv.batch_norm.running_var"), .f32_required, &.{d});
            n += 1;
        } else {
            n += try expectNorm(file, layerName(&nb, il, "conv.layer_norm"), d);
        }
        _ = try expectTensor(file, layerName(&nb, il, "conv.pointwise_conv2.weight"), .f32_required, &.{ 1, d, d });
        n += 1;
        _ = try expectTensor(file, layerName(&nb, il, "conv.pointwise_conv2.bias"), .f32_required, &.{d});
        n += 1;
        // macaron FFN #2
        n += try expectNorm(file, layerName(&nb, il, "norm_feed_forward2"), d);
        n += try expectLinear(file, layerName(&nb, il, "feed_forward2.linear1"), &.{ d, ff }, ff);
        n += try expectLinear(file, layerName(&nb, il, "feed_forward2.linear2"), &.{ ff, d }, d);
        // final block norm
        n += try expectNorm(file, layerName(&nb, il, "norm_out"), d);
    }

    // Transducer: LSTM predictor + joint network.
    if (cfg.arch.hasTransducer()) {
        _ = try expectTensor(file, "decoder.prediction.embed.weight", .f32_required, &.{ ph, v1 });
        n += 1;
        for (0..cfg.pred_rnn_layers) |l| {
            const in_dim: usize = if (l == 0) ph else ph; // single hidden size throughout
            _ = try expectTensor(file, lstmName(&nb, "weight_ih_l", l), .f32_required, &.{ in_dim, four_ph });
            n += 1;
            _ = try expectTensor(file, lstmName(&nb, "weight_hh_l", l), .f32_required, &.{ ph, four_ph });
            n += 1;
            _ = try expectTensor(file, lstmName(&nb, "bias_ih_l", l), .f32_required, &.{four_ph});
            n += 1;
            _ = try expectTensor(file, lstmName(&nb, "bias_hh_l", l), .f32_required, &.{four_ph});
            n += 1;
        }
        _ = try expectTensor(file, "joint.enc.weight", .quantizable, &.{ d, jh });
        n += 1;
        _ = try expectTensor(file, "joint.enc.bias", .f32_required, &.{jh});
        n += 1;
        _ = try expectTensor(file, "joint.pred.weight", .quantizable, &.{ ph, jh });
        n += 1;
        _ = try expectTensor(file, "joint.pred.bias", .f32_required, &.{jh});
        n += 1;
        _ = try expectTensor(file, "joint.joint_net.2.weight", .f32_required, &.{ jh, vp });
        n += 1;
        _ = try expectTensor(file, "joint.joint_net.2.bias", .f32_required, &.{vp});
        n += 1;
    }

    // CTC head.
    if (cfg.arch.hasCtc()) {
        _ = try expectTensor(file, "ctc_decoder.decoder_layers.0.weight", .f32_required, &.{ 1, d, v1 });
        n += 1;
        _ = try expectTensor(file, "ctc_decoder.decoder_layers.0.bias", .f32_required, &.{v1});
        n += 1;
    }

    return n;
}

fn lstmName(buf: []u8, stem: []const u8, l: usize) []const u8 {
    return std.fmt.bufPrint(buf, "decoder.prediction.dec_rnn.lstm.{s}{d}", .{ stem, l }) catch unreachable;
}

/// A `<base>.weight` (quantizable) + `<base>.bias` (f32) pair. Returns 2.
fn expectLinear(file: *const gguf.File, base: []const u8, w_dims: []const usize, bias_dim: usize) Error!usize {
    var buf: [288]u8 = undefined;
    _ = try expectTensor(file, std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch unreachable, .quantizable, w_dims);
    _ = try expectTensor(file, std.fmt.bufPrint(&buf, "{s}.bias", .{base}) catch unreachable, .f32_required, &.{bias_dim});
    return 2;
}

/// A `<base>.weight` + `<base>.bias`, both f32, both shape `[dim]` (a LayerNorm).
fn expectNorm(file: *const gguf.File, base: []const u8, dim: usize) Error!usize {
    var buf: [288]u8 = undefined;
    _ = try expectTensor(file, std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch unreachable, .f32_required, &.{dim});
    _ = try expectTensor(file, std.fmt.bufPrint(&buf, "{s}.bias", .{base}) catch unreachable, .f32_required, &.{dim});
    return 2;
}

/// A conv `<base>.weight` (f32 kernel, given dims) + `<base>.bias` (f32 [out_ch]).
fn expectConv(file: *const gguf.File, base: []const u8, w_dims: []const usize, out_ch: usize) Error!usize {
    var buf: [288]u8 = undefined;
    _ = try expectTensor(file, std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch unreachable, .f32_required, w_dims);
    _ = try expectTensor(file, std.fmt.bufPrint(&buf, "{s}.bias", .{base}) catch unreachable, .f32_required, &.{out_ch});
    return 2;
}

// --- featurizer + tokenizer accessors ---

/// Mel filterbank + STFT window, borrowed (zero-copy) from the mapped GGUF.
/// Valid only while the source `gguf.File` mapping is alive. Both are f32 on
/// every model. `fb` is row-major over ggml `ne=[n_fft/2+1, n_mels, 1]`, i.e.
/// `fb[m * fb_bins + k]` (mel m, freq bin k).
pub const Featurizer = struct {
    fb: []const f32,
    fb_bins: usize, // n_fft/2+1
    fb_mels: usize, // n_mels
    window: []const f32, // [win_length]
};

// Inferred error union: the loader's own `Error` plus the alignment guard's
// `weights.Error.InvalidWeightShape` from `borrowF32`.
pub fn loadFeaturizer(file: *const gguf.File, cfg: Config) !Featurizer {
    const fb_t = try fbTensor(file, cfg);
    const win_t = try expectTensor(file, "preprocessor.featurizer.window", .f32_required, &.{cfg.win_length});
    return .{
        .fb = try borrowF32(fb_t),
        .fb_bins = cfg.n_fft / 2 + 1,
        .fb_mels = cfg.n_mels,
        .window = try borrowF32(win_t),
    };
}

fn borrowF32(t: *const TensorInfo) ![]const f32 {
    return pweights.borrowF32(t.data);
}

/// Decode the SentencePiece-BPE piece table. The outer slice is allocated (caller
/// frees); each piece borrows the mapped GGUF bytes. Full detokenize lives in
/// `tokenizer.zig`.
pub fn loadPieces(file: *const gguf.File, allocator: Allocator) ![][]const u8 {
    const arr = file.getArray("parakeet.tokenizer.pieces") orelse return Error.MissingMetadata;
    return arr.stringSlices(allocator);
}

test {
    _ = @import("loader_tests.zig");
}
