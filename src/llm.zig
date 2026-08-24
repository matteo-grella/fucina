//! LLM/ASR module root. Model families live in subdirectories (`llm/<family>/`)
//! and are exposed as namespaces (`llm.parakeet.decoder`, `llm.gemma.model`, …);
//! generic/shared helpers (kv_cache, tokenizers, sampler, chat) stay flat.
//! Model I/O (weights containers, PTQTP sidecars, GGUF metadata) is core:
//! `fucina.weights` / `fucina.ptqtp_gguf` / `fucina.gguf_meta` (aliased here).

/// Qwen3 dense + LoRA fine-tuning. Files in `llm/qwen3/`.
pub const qwen3 = struct {
    pub const model = @import("llm/qwen3/model.zig");
    pub const train = @import("llm/qwen3/train.zig");
    pub const ptqtp = @import("llm/qwen3/ptqtp.zig");
    /// SHINE adapter-fleet serving (`serving.open`'s counterpart for the
    /// research adapters in `research.shine`).
    pub const shine_serving = @import("llm/qwen3/shine_serving.zig");
};

/// Qwen3.5 Gated-DeltaNet hybrid. Files in `llm/qwen35/`.
pub const qwen35 = struct {
    pub const model = @import("llm/qwen35/model.zig");
    pub const chat = @import("llm/qwen35/chat.zig");
    /// The family's serving adapter (`serving.open` dispatches here).
    pub const serving = @import("llm/qwen35/serving.zig");
};
/// Gemma 4 (text) + MoE + LoRA fine-tuning. Files in `llm/gemma/`.
pub const gemma = struct {
    pub const model = @import("llm/gemma/model.zig");
    pub const train = @import("llm/gemma/train.zig");
    pub const moe = @import("llm/gemma/moe.zig");
};
/// DiffusionGemma block text-diffusion (gemma4 backbone). Files in `llm/diffusion_gemma/`.
pub const diffusion_gemma = struct {
    pub const model = @import("llm/diffusion_gemma/model.zig");
};
/// Research tier: evaluators and memory modules that install into the
/// production forwards through typed seams (`runner.AttentionOverride`,
/// `qwen3.train.ResidualHook`), plus the model ports kept for study.
/// Stability: experimental (CHANGELOG.md tiers).
pub const research = struct {
    /// SubQ attention: decode-path evaluator (docs/SUBQUADRATIC-ATTENTION.md);
    /// installs through `runner.AttentionOverride`.
    pub const subq = @import("llm/subq.zig");
    /// Engram conditional n-gram memory (docs/ENGRAM.md); grafts through
    /// the qwen3 trainer's `residual_hook` seam (`ResidualGraft`).
    pub const engram = @import("llm/engram.zig");
    /// SHINE context-to-LoRA hypernetwork and its trainer (qwen3 dense);
    /// served through `qwen3.shine_serving`.
    pub const shine = @import("llm/qwen3/shine.zig");
    pub const shine_train = @import("llm/qwen3/shine_train.zig");
    /// Kimi-K3 (Kimi-Linear lineage: KDA + Gated-MLA-NoPE hybrid, latent
    /// MoE, attention residuals, SiTU). Files in `llm/kimi3/`.
    pub const kimi3 = struct {
        pub const model = @import("llm/kimi3/model.zig");
    };
};
/// Parakeet ASR (NeMo FastConformer/RNN-T). Files in `llm/parakeet/`.
pub const parakeet = struct {
    pub const loader = @import("llm/parakeet/loader.zig");
    pub const frontend = @import("llm/parakeet/frontend.zig");
    pub const subsampling = @import("llm/parakeet/subsampling.zig");
    pub const encoder = @import("llm/parakeet/encoder.zig");
    pub const weights = @import("llm/parakeet/weights.zig");
    pub const decoder = @import("llm/parakeet/decoder.zig");
    pub const tokenizer = @import("llm/parakeet/tokenizer.zig");
    pub const streaming = @import("llm/parakeet/streaming.zig");
    pub const transcription = @import("llm/parakeet/transcription.zig");
};
/// Lossless draft-model-free speculative decoding. Files in `llm/speculative/`.
pub const speculative = struct {
    pub const core = @import("llm/speculative/core.zig");
    /// Native-MTP drafting behind the `DraftSource` vtable (glm4moe).
    pub const mtp = @import("llm/speculative/mtp.zig");
    pub const sam_index = @import("llm/speculative/sam_index.zig");
    pub const recycling = @import("llm/speculative/recycling.zig");
    pub const cascade = @import("llm/speculative/cascade.zig");
    pub const constrained = @import("llm/speculative/constrained.zig");
};
/// DeepSeek-V2 MLA + fine-grained MoE with shared experts. Files in `llm/deepseek2/`.
pub const deepseek2 = struct {
    pub const model = @import("llm/deepseek2/model.zig");
};
/// GLM-4.5 MoE with native MTP (`nextn`) self-speculation. Files in `llm/glm4moe/`.
pub const glm4moe = struct {
    pub const model = @import("llm/glm4moe/model.zig");
};
/// DeepSeek V4 Flash (hyper-connections, compressed-KV MQA, streamed experts,
/// MTP). Files in `llm/deepseek4/`.
pub const deepseek4 = struct {
    pub const model = @import("llm/deepseek4/model.zig");
    /// The family's serving adapter (`serving.open` dispatches here).
    pub const serving = @import("llm/deepseek4/serving.zig");
};
/// Qwen3-TTS 12.5 Hz two-stack TTS (talker + MTP code predictor + codec).
/// Files in `llm/qwen3tts/`.
pub const qwen3tts = struct {
    pub const codec = @import("llm/qwen3tts/codec.zig");
    pub const sampling = @import("llm/qwen3tts/sampling.zig");
    pub const model = @import("llm/qwen3tts/model.zig");
    pub const prompt = @import("llm/qwen3tts/prompt.zig");
    pub const pipeline = @import("llm/qwen3tts/pipeline.zig");
};
/// Pocket TTS (kyutai): continuous-latent flow-matching prefix-LM + VAE-Mimi
/// streaming decoder. Files in `llm/pockettts/`.
pub const pockettts = struct {
    pub const model = @import("llm/pockettts/model.zig");
};
/// Inkling (hybrid SWA/global rel-bias attention, shortconv sites, sink-shared
/// MoE; hMLP vision + dMel audio towers). Files in `llm/inkling/`.
pub const inkling = struct {
    pub const model = @import("llm/inkling/model.zig");
    pub const mmproj = @import("llm/inkling/mmproj.zig");
    pub const chat = @import("llm/inkling/chat.zig");
    /// The family's serving adapter (`serving.open` dispatches here).
    pub const serving = @import("llm/inkling/serving.zig");
};

// === Generic / shared helpers (stay flat) ===
// Model I/O (weights containers, PTQTP sidecars, GGUF metadata) lives in
// the core module — it is model-agnostic (`fucina.weights` /
// `fucina.ptqtp_gguf` / `fucina.gguf_meta`).
pub const cartridge = @import("llm/cartridge.zig");
pub const cartridge_fleet = @import("llm/cartridge_fleet.zig");
/// The family-independent half of a LoRA trainer: target selection, the
/// per-layer adapter set, its A/B tuple, and the dropout seed stream. The
/// qwen3 and gemma trainers instantiate it and add their own forward.
pub const lora_trainer = @import("llm/lora_trainer.zig");
/// Shared `--moe-*` argv parser and exit-time report for the streamed-MoE
/// runner CLIs; the options struct it fills is core
/// (`fucina.weights.MoeStreamOptions`).
pub const moe_stream_cli = @import("llm/moe_stream_cli.zig");
/// The LLM trainers' checkpoint resume state; the save/load frame is core
/// (`fucina.training_checkpoint`, generic over the state struct).
pub const trainer_state = @import("llm/trainer_state.zig");
/// Cache-aware expert routing policy shared by the streamed-MoE decoders.
pub const moe_router = @import("llm/moe_router.zig");
/// The autoregressive text-decoder contract (`Caps`, `assertDecoder`): the
/// comptime surface the generic layers (chat, speculative, serving,
/// generate) are written against.
pub const decoder = @import("llm/decoder.zig");
/// The architecture registry: GGUF `general.architecture` to family
/// module (`Family` decls in each family's `model.zig`); `serving.open`
/// dispatches over it.
pub const registry = @import("llm/registry.zig");
pub const kv_cache = @import("llm/kv_cache.zig");
/// The reference generation loop over the decoder contract; the family
/// chat engines and `gemma.Model.generate` call it.
pub const generate = @import("llm/generate.zig");
pub const kv_persist = @import("llm/kv_persist.zig");
pub const tokenizer = @import("llm/tokenizer.zig");
pub const spm_tokenizer = @import("llm/spm_tokenizer.zig");
pub const sampler = @import("llm/sampler.zig");
pub const logit_processor = @import("llm/logit_processor.zig");
pub const llguidance = @import("llm/llguidance.zig");
pub const chat = @import("llm/chat.zig");
/// The serving band: the contract (`GenerateRequest`/`GenerateResult`,
/// capability flags, the per-model-family `Backend` vtable) plus the
/// transport (HTTP server, scheduler, OpenAI/Anthropic dialects), the
/// generic GGUF chat engine, and the `serving.open` load-and-serve entry
/// (`examples/lmserve` is the CLI front end built on it). Files in
/// `llm/serving/`.
pub const serving = @import("llm/serving.zig");
/// The descriptor runner: one family-independent decoder driven by a
/// runtime `Descriptor` (Level 0 of the universal checkpoint runner).
/// The qwen3 family and the glm4moe trunk run on it; recorded-logits
/// gates in `llm/runner_tests.zig` pin its numerics.
pub const runner = @import("llm/runner.zig");
pub const data = @import("llm/data.zig");
/// Generated \p{L}/\p{N}/\s tables (the byte-BPE pretokenizer's). Re-exported
/// so out-of-module consumers (nanochat's example-local tokenizer) share the
/// file instead of rooting it as a second module — a file may belong to only
/// one module per compilation.
pub const unicode_categories = @import("llm/unicode_categories.zig");

test {
    _ = decoder;
    _ = generate;
    _ = registry;
    _ = runner;
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
    _ = speculative.core;
    _ = speculative.sam_index;
    _ = speculative.recycling;
    _ = speculative.cascade;
    _ = speculative.constrained;
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
    _ = cartridge;
    _ = cartridge_fleet;
    _ = research.engram;
    _ = lora_trainer;
    _ = moe_stream_cli;
    _ = trainer_state;
    _ = moe_router;
    _ = kv_cache;
    _ = kv_persist;
    _ = tokenizer;
    _ = spm_tokenizer;
    _ = sampler;
    _ = logit_processor;
    _ = llguidance;
    _ = chat;
    _ = serving.gguf_chat;
    _ = serving.scheduler;
    _ = serving.openai;
    _ = serving.anthropic;
    _ = serving.emitter;
    _ = serving.http;
    _ = serving.toolcall;
    _ = data;
    _ = research.subq;
}
