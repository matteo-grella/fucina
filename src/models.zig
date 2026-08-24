//! Model-band module root. One criterion: anything with a loader and a
//! forward lives in `models/<family>/` and is exposed as a namespace
//! (`models.parakeet.decoder`, `models.gemma.model`, …); the
//! modality-agnostic text runtime lives in `models/text/`; training helpers
//! in `models/train/`; research in `models/research/`.
//! Model I/O (weights containers, PTQTP sidecars, GGUF metadata) is core:
//! `fucina.weights` / `fucina.ptqtp_gguf` / `fucina.gguf_meta`.

/// Qwen3 dense + LoRA fine-tuning. Files in `models/qwen3/`.
pub const qwen3 = struct {
    pub const model = @import("models/qwen3/model.zig");
    pub const train = @import("models/qwen3/train.zig");
    pub const ptqtp = @import("models/qwen3/ptqtp.zig");
    /// The descriptor runner: the qwen3-family decoder driven by a runtime
    /// `Descriptor`; the glm4moe trunk imports it by relative path.
    /// Recorded-logits gates in `models/qwen3/runner_tests.zig` pin its
    /// numerics.
    pub const runner = @import("models/qwen3/runner.zig");
    /// SHINE adapter-fleet serving (`text.serving.open`'s counterpart for
    /// the research adapters in `research.shine`).
    pub const shine_serving = @import("models/qwen3/shine_serving.zig");
};

/// Qwen3.5 Gated-DeltaNet hybrid. Files in `models/qwen35/`.
pub const qwen35 = struct {
    pub const model = @import("models/qwen35/model.zig");
    pub const chat = @import("models/qwen35/chat.zig");
    /// The family's serving adapter (`text.serving.open` dispatches here).
    pub const serving = @import("models/qwen35/serving.zig");
};
/// Gemma 4 (text) + MoE + LoRA fine-tuning. Files in `models/gemma/`.
pub const gemma = struct {
    pub const model = @import("models/gemma/model.zig");
    pub const train = @import("models/gemma/train.zig");
    pub const moe = @import("models/gemma/moe.zig");
};
/// DiffusionGemma block text-diffusion (gemma4 backbone). Files in
/// `models/diffusion_gemma/`.
pub const diffusion_gemma = struct {
    pub const model = @import("models/diffusion_gemma/model.zig");
};
/// Research tier: evaluators and memory modules that install into the
/// production forwards through typed seams (`qwen3.runner.AttentionOverride`,
/// `qwen3.train.ResidualHook`), plus the model ports kept for study.
/// Files in `models/research/` (shine rides `models/qwen3/`).
/// Stability: experimental (CHANGELOG.md tiers).
pub const research = struct {
    /// SubQ attention: decode-path evaluator (docs/SUBQUADRATIC-ATTENTION.md);
    /// installs through `qwen3.runner.AttentionOverride`.
    pub const subq = @import("models/research/subq.zig");
    /// Engram conditional n-gram memory (docs/ENGRAM.md); grafts through
    /// the qwen3 trainer's `residual_hook` seam (`ResidualGraft`).
    pub const engram = @import("models/research/engram.zig");
    /// SHINE context-to-LoRA hypernetwork and its trainer (qwen3 dense);
    /// served through `qwen3.shine_serving`.
    pub const shine = @import("models/qwen3/shine.zig");
    pub const shine_train = @import("models/qwen3/shine_train.zig");
    /// Kimi-K3 (Kimi-Linear lineage: KDA + Gated-MLA-NoPE hybrid, latent
    /// MoE, attention residuals, SiTU). Files in `models/research/kimi3/`.
    pub const kimi3 = struct {
        pub const model = @import("models/research/kimi3/model.zig");
    };
};
/// Parakeet ASR (NeMo FastConformer/RNN-T). Files in `models/parakeet/`.
pub const parakeet = struct {
    pub const loader = @import("models/parakeet/loader.zig");
    pub const frontend = @import("models/parakeet/frontend.zig");
    pub const subsampling = @import("models/parakeet/subsampling.zig");
    pub const encoder = @import("models/parakeet/encoder.zig");
    pub const weights = @import("models/parakeet/weights.zig");
    pub const decoder = @import("models/parakeet/decoder.zig");
    pub const tokenizer = @import("models/parakeet/tokenizer.zig");
    pub const streaming = @import("models/parakeet/streaming.zig");
    pub const transcription = @import("models/parakeet/transcription.zig");
};
/// DeepSeek-V2 MLA + fine-grained MoE with shared experts. Files in
/// `models/deepseek2/`.
pub const deepseek2 = struct {
    pub const model = @import("models/deepseek2/model.zig");
};
/// GLM-4.5 MoE with native MTP (`nextn`) self-speculation. Files in
/// `models/glm4moe/`.
pub const glm4moe = struct {
    pub const model = @import("models/glm4moe/model.zig");
};
/// DeepSeek V4 Flash (hyper-connections, compressed-KV MQA, streamed experts,
/// MTP). Files in `models/deepseek4/`.
pub const deepseek4 = struct {
    pub const model = @import("models/deepseek4/model.zig");
    /// The family's serving adapter (`text.serving.open` dispatches here).
    pub const serving = @import("models/deepseek4/serving.zig");
};
/// Qwen3-TTS 12.5 Hz two-stack TTS (talker + MTP code predictor + codec).
/// Files in `models/qwen3tts/`.
pub const qwen3tts = struct {
    pub const codec = @import("models/qwen3tts/codec.zig");
    pub const sampling = @import("models/qwen3tts/sampling.zig");
    pub const model = @import("models/qwen3tts/model.zig");
    pub const prompt = @import("models/qwen3tts/prompt.zig");
    pub const pipeline = @import("models/qwen3tts/pipeline.zig");
};
/// Pocket TTS (kyutai): continuous-latent flow-matching prefix-LM + VAE-Mimi
/// streaming decoder. Files in `models/pockettts/`.
pub const pockettts = struct {
    pub const model = @import("models/pockettts/model.zig");
};
/// Inkling (hybrid SWA/global rel-bias attention, shortconv sites, sink-shared
/// MoE; hMLP vision + dMel audio towers). Files in `models/inkling/`.
pub const inkling = struct {
    pub const model = @import("models/inkling/model.zig");
    pub const mmproj = @import("models/inkling/mmproj.zig");
    pub const chat = @import("models/inkling/chat.zig");
    /// The family's serving adapter (`text.serving.open` dispatches here).
    pub const serving = @import("models/inkling/serving.zig");
};

/// The modality-agnostic text runtime: tokenizers, sampling, chat, the
/// generation loop, KV caches, SFT data, cartridges, speculative decoding,
/// and serving. Files in `models/text/`.
pub const text = struct {
    pub const tokenizer = @import("models/text/tokenizer.zig");
    pub const spm_tokenizer = @import("models/text/spm_tokenizer.zig");
    /// Generated \p{L}/\p{N}/\s tables (the byte-BPE pretokenizer's).
    /// Re-exported so out-of-module consumers (nanochat's example-local
    /// tokenizer) share the file instead of rooting it as a second module:
    /// a file may belong to only one module per compilation.
    pub const unicode_categories = @import("models/text/unicode_categories.zig");
    pub const sampler = @import("models/text/sampler.zig");
    pub const logit_processor = @import("models/text/logit_processor.zig");
    pub const llguidance = @import("models/text/llguidance.zig");
    pub const chat = @import("models/text/chat.zig");
    /// The reference generation loop over the decoder contract; the family
    /// chat engines and `gemma.Model.generate` call it.
    pub const generate = @import("models/text/generate.zig");
    pub const kv_cache = @import("models/text/kv_cache.zig");
    pub const kv_persist = @import("models/text/kv_persist.zig");
    pub const data = @import("models/text/data.zig");
    pub const cartridge = @import("models/text/cartridge.zig");
    pub const cartridge_fleet = @import("models/text/cartridge_fleet.zig");
    /// Lossless draft-model-free speculative decoding. Files in
    /// `models/text/speculative/`.
    pub const speculative = struct {
        pub const core = @import("models/text/speculative/core.zig");
        /// Native-MTP drafting behind the `DraftSource` vtable (glm4moe).
        pub const mtp = @import("models/text/speculative/mtp.zig");
        pub const sam_index = @import("models/text/speculative/sam_index.zig");
        pub const recycling = @import("models/text/speculative/recycling.zig");
        pub const cascade = @import("models/text/speculative/cascade.zig");
        pub const constrained = @import("models/text/speculative/constrained.zig");
    };
    /// The serving band: the contract (`GenerateRequest`/`GenerateResult`,
    /// capability flags, the per-model-family `Backend` vtable) plus the
    /// transport (HTTP server, scheduler, OpenAI/Anthropic dialects), the
    /// generic GGUF chat engine, and the `text.serving.open` load-and-serve
    /// entry (`apps/lmserve` is the CLI front end built on it). Files in
    /// `models/text/serving/`.
    pub const serving = @import("models/text/serving.zig");
};

/// Training helpers shared by the family trainers. Files in `models/train/`.
pub const train = struct {
    /// The family-independent half of a LoRA trainer: target selection, the
    /// per-layer adapter set, its A/B tuple, and the dropout seed stream. The
    /// qwen3 and gemma trainers instantiate it and add their own forward.
    pub const lora_trainer = @import("models/train/lora_trainer.zig");
    /// The trainers' checkpoint resume state; the save/load frame is core
    /// (`fucina.training_checkpoint`, generic over the state struct).
    pub const trainer_state = @import("models/train/trainer_state.zig");
};

// === Band-level helpers (flat in `models/`) ===
/// The autoregressive text-decoder contract (`Caps`, `assertDecoder`): the
/// comptime surface the generic layers (chat, speculative, serving,
/// generate) are written against.
pub const decoder = @import("models/decoder.zig");
/// The architecture registry: GGUF `general.architecture` to family
/// module (`Family` decls in each family's `model.zig`); `text.serving.open`
/// dispatches over it.
pub const registry = @import("models/registry.zig");
/// Shared GGUF-family load helpers: PTQTP-aware projection loading, the
/// dense-FFN weight containers, the MoE expert-trio loader, and the
/// embed/output-norm/lm-head trio; only load-band semantics identical
/// across families live here.
pub const model_common = @import("models/model_common.zig");
/// Host-band scalar ops: the one definition of the numerics the host-style
/// ports share (f64-accumulated RMS norm, max-subtracted softmax, silu, and
/// the SwiGLU-through-`LinearWeight` FFN).
pub const host_ops = @import("models/host_ops.zig");
/// Cache-aware expert routing policy shared by the streamed-MoE decoders.
pub const moe_router = @import("models/moe_router.zig");
/// Shared `--moe-*` argv parser and exit-time report for the streamed-MoE
/// runner CLIs; the options struct it fills is core
/// (`fucina.weights.MoeStreamOptions`).
pub const moe_stream_cli = @import("models/moe_stream_cli.zig");
/// Shared gates for model- and fixture-dependent tests: the native-only
/// guard and open-or-skip file loading (`FUCINA_TEST_REQUIRE_MODELS=1`
/// turns a missing model into a failure).
pub const test_support = @import("models/test_support.zig");

test {
    _ = decoder;
    _ = text.generate;
    _ = registry;
    _ = qwen3.runner;
    _ = research.kimi3.model;
    _ = qwen3.model;
    _ = qwen3.train;
    _ = qwen3.ptqtp;
    _ = qwen3.shine_serving;
    _ = research.shine;
    _ = research.shine_train;
    _ = qwen35.model;
    _ = qwen35.chat;
    _ = qwen35.serving;
    _ = inkling.serving;
    _ = deepseek4.serving;
    _ = gemma.model;
    _ = gemma.train;
    _ = gemma.moe;
    _ = diffusion_gemma.model;
    _ = parakeet.loader;
    _ = parakeet.frontend;
    _ = parakeet.subsampling;
    _ = parakeet.encoder;
    _ = parakeet.weights;
    _ = parakeet.decoder;
    _ = parakeet.tokenizer;
    _ = parakeet.streaming;
    _ = parakeet.transcription;
    _ = text.speculative.core;
    _ = text.speculative.sam_index;
    _ = text.speculative.recycling;
    _ = text.speculative.cascade;
    _ = text.speculative.constrained;
    _ = deepseek2.model;
    _ = glm4moe.model;
    _ = deepseek4.model;
    _ = qwen3tts.codec;
    _ = pockettts.model;
    _ = qwen3tts.sampling;
    _ = qwen3tts.model;
    _ = qwen3tts.prompt;
    _ = qwen3tts.pipeline;
    _ = inkling.model;
    _ = inkling.mmproj;
    _ = inkling.chat;
    _ = text.cartridge;
    _ = text.cartridge_fleet;
    _ = research.engram;
    _ = train.lora_trainer;
    _ = moe_stream_cli;
    _ = train.trainer_state;
    _ = moe_router;
    _ = text.kv_cache;
    _ = text.kv_persist;
    _ = text.tokenizer;
    _ = text.spm_tokenizer;
    _ = text.sampler;
    _ = text.logit_processor;
    _ = text.llguidance;
    _ = text.chat;
    _ = text.serving.gguf_chat;
    _ = text.serving.scheduler;
    _ = text.serving.openai;
    _ = text.serving.anthropic;
    _ = text.serving.emitter;
    _ = text.serving.http;
    _ = text.serving.toolcall;
    _ = text.data;
    _ = research.subq;
}
