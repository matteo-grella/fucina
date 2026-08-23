//! Qwen3 model surface. The family's decoder is the descriptor runner
//! (`llm/runner.zig`) driven by the qwen3-shaped `Descriptor`; this module
//! is the family-named surface over it. `Config` IS `runner.Descriptor`
//! (qwen3/qwen3moe GGUF metadata fills the `.fused` block style), `Model`
//! IS `runner.Model` — including the batched decode entries
//! (`forwardStepBatch`, `forwardStepBatchSpans`) and the
//! `attention_override` research seam. Family-specific code lives beside
//! this file:
//! training (`train.zig`), PTQTP decoration (`ptqtp.zig`), greedy
//! generation (`generate.zig`), SHINE (`shine.zig`).
//!
//! Correctness anchors: the real-model golden suites (chat, SHINE, TTS
//! talker, training goldens) plus `runner_tests.zig`'s recorded-logits
//! gates on real Qwen3-0.6B GGUFs and the synthetic fixtures.

const runner = @import("../runner.zig");

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

test {
    _ = @import("model_tests.zig");
}
