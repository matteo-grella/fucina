//! Qwen3 model surface. The family's decoder is the descriptor runner
//! (`models/qwen3/runner.zig`) driven by the qwen3-shaped `Descriptor`; this module
//! is the family-named surface over it. `Config` IS `runner.Descriptor`
//! (qwen3/qwen3moe GGUF metadata fills the `.fused` block style), `Model`
//! IS `runner.Model` — including the batched decode entries
//! (`forwardStepBatch`, `forwardStepBatchSpans`) and the
//! `attention_override` research seam. Family-specific code lives beside
//! this file:
//! training (`train.zig`), PTQTP decoration (`ptqtp.zig`), SHINE
//! (`shine.zig`); greedy generation is the shared `models.text.generate` loop.
//!
//! Correctness anchors: the real-model golden suites (chat, SHINE, TTS
//! talker, training goldens) plus `runner_tests.zig`'s recorded-logits
//! gates on real Qwen3-0.6B GGUFs and the synthetic fixtures.

const std = @import("std");
const fucina = @import("fucina");
const chat = @import("../text/chat.zig");
const model_common = @import("../model_common.zig");
const runner = @import("runner.zig");
const tokenizer_mod = @import("../text/tokenizer.zig");

pub const Config = runner.Descriptor;
pub const Model = runner.Model;
pub const Error = runner.Error;
pub const ForwardProfile = runner.ForwardProfile;
pub const LoadOptions = runner.LoadOptions;
pub const MoeStreamOptions = runner.MoeStreamOptions;

// Layer internals train.zig and shine.zig address directly (differentiable
// forwards re-drive the same weights).
pub const Layer = runner.Layer;
pub const DenseFfn = runner.DenseFfn;
pub const QkvProjection = runner.QkvProjection;
pub const GateUpProjection = runner.GateUpProjection;
pub const splitQkv = runner.splitQkv;
pub const splitGateUp = runner.splitGateUp;
pub const applyExpertTopP = runner.applyExpertTopP;

/// Registry surface (`models.registry`): what `serving.open` needs to load
/// and serve this family (qwen3 and qwen3moe share it).
pub const Family = struct {
    pub const Model = runner.Model;
    /// The tokenizer MODULE (byte-level BPE) and its `Tokenizer` type.
    pub const Tok = tokenizer_mod;
    pub const Tokenizer = tokenizer_mod.Tokenizer;

    pub fn load(ctx: *fucina.ExecContext, file: *fucina.gguf.File, options: model_common.FamilyLoadOptions) !runner.Model {
        _ = options;
        return runner.Model.loadGgufFromFile(ctx, file, try Config.fromGguf(file));
    }

    pub fn tokenizer(allocator: std.mem.Allocator, file: *const fucina.gguf.File) !Tokenizer {
        return Tokenizer.initFromGguf(allocator, file, .{});
    }

    /// Chat-template fallback when the GGUF carries none (null = the GGUF
    /// must carry a recognizable template).
    pub const template_fallback: ?chat.Format = null;
};

test {
    _ = @import("model_tests.zig");
}
