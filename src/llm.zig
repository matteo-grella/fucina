//! LLM/ASR module root. Model families live in subdirectories (`llm/<family>/`)
//! and are exposed as namespaces (`llm.parakeet.decoder`, `llm.gemma.gemma4`, …);
//! generic/shared helpers (kv_cache, tokenizers, sampler, chat) stay flat.
//! Model I/O (weights containers, PTQTP sidecars, GGUF metadata) is core:
//! `fucina.weights` / `fucina.ptqtp_gguf` / `fucina.gguf_meta` (aliased here).

/// SubQ attention: research decode-path evaluator (docs/SUBQUADRATIC-ATTENTION.md).
pub const subq = @import("llm/subq.zig");

/// Qwen3 dense + LoRA fine-tuning. Files in `llm/qwen3/`.
pub const qwen3 = struct {
    pub const model = @import("llm/qwen3/model.zig");
    pub const train = @import("llm/qwen3/train.zig");
    pub const ptqtp = @import("llm/qwen3/ptqtp.zig");
    pub const generate = @import("llm/qwen3/generate.zig");
    pub const shine = @import("llm/qwen3/shine.zig");
    pub const shine_train = @import("llm/qwen3/shine_train.zig");
};
/// Kimi-K3 (Kimi-Linear lineage: KDA + Gated-MLA-NoPE hybrid, latent MoE,
/// attention residuals, SiTU). Files in `llm/kimi3/`.
pub const kimi3 = struct {
    pub const model = @import("llm/kimi3/model.zig");
};

/// Qwen3.5 Gated-DeltaNet hybrid. Files in `llm/qwen35/`.
pub const qwen35 = struct {
    pub const model = @import("llm/qwen35/model.zig");
    pub const chat = @import("llm/qwen35/chat.zig");
};
/// Gemma 4 (text) + MoE + LoRA fine-tuning. Files in `llm/gemma/`.
pub const gemma = struct {
    pub const gemma4 = @import("llm/gemma/gemma4.zig");
    pub const gemma4_train = @import("llm/gemma/gemma4_train.zig");
    pub const moe = @import("llm/gemma/moe.zig");
    pub const moe_route = @import("llm/gemma/moe_route.zig");
    pub const moe_route_tensor = @import("llm/gemma/moe_route_tensor.zig");
};
/// DiffusionGemma block text-diffusion (gemma4 backbone). Files in `llm/diffusion_gemma/`.
pub const diffusion_gemma = struct {
    pub const model = @import("llm/diffusion_gemma/model.zig");
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
    pub const pocket = @import("llm/pockettts/pocket.zig");
};
/// Inkling (hybrid SWA/global rel-bias attention, shortconv sites, sink-shared
/// MoE; hMLP vision + dMel audio towers). Files in `llm/inkling/`.
pub const inkling = struct {
    pub const model = @import("llm/inkling/model.zig");
    pub const mmproj = @import("llm/inkling/mmproj.zig");
    pub const chat = @import("llm/inkling/chat.zig");
};

// === Generic / shared helpers (stay flat) ===
// Model I/O (weights containers, PTQTP sidecars, GGUF metadata) lives in
// the core module — it is model-agnostic (`fucina.weights` et al.).
/// Deprecated: use `fucina.weights`. Removal per docs/DEVELOPMENT.md §6.
pub const weights = @import("fucina").weights;
/// Deprecated: use `fucina.ptqtp_gguf`. Removal per docs/DEVELOPMENT.md §6.
pub const ptqtp_gguf = @import("fucina").ptqtp_gguf;
/// Deprecated: use `fucina.gguf_meta`. Removal per docs/DEVELOPMENT.md §6.
pub const gguf_meta = @import("fucina").gguf_meta;
pub const cartridge = @import("llm/cartridge.zig");
pub const cartridge_fleet = @import("llm/cartridge_fleet.zig");
pub const engram = @import("llm/engram.zig");
pub const kv_cache = @import("llm/kv_cache.zig");
pub const kv_persist = @import("llm/kv_persist.zig");
pub const tokenizer = @import("llm/tokenizer.zig");
pub const spm_tokenizer = @import("llm/spm_tokenizer.zig");
pub const sampler = @import("llm/sampler.zig");
pub const logit_processor = @import("llm/logit_processor.zig");
pub const llguidance = @import("llm/llguidance.zig");
pub const chat = @import("llm/chat.zig");
/// The serving contract: `GenerateRequest`/`GenerateResult`, capability
/// flags, and the per-model-family `Backend` vtable a server consumes
/// (`examples/lmserve` is the in-tree server built on it).
pub const serving = @import("llm/serving.zig");
/// The descriptor runner: one family-independent decoder driven by a
/// runtime `Descriptor` (Level 0 of the universal checkpoint runner);
/// `qwen3` remains the hand-written parity oracle.
pub const runner = @import("llm/runner.zig");
pub const data = @import("llm/data.zig");
/// Generated \p{L}/\p{N}/\s tables (the byte-BPE pretokenizer's). Re-exported
/// so out-of-module consumers (nanochat's example-local tokenizer) share the
/// file instead of rooting it as a second module — a file may belong to only
/// one module per compilation.
pub const unicode_categories = @import("llm/unicode_categories.zig");

test {
    _ = runner;
    _ = @import("llm/runner_tests.zig");
    _ = kimi3.model;
    _ = qwen3.model;
    _ = qwen3.train;
    _ = qwen3.ptqtp;
    _ = qwen3.generate;
    _ = qwen3.shine;
    _ = qwen3.shine_train;
    _ = qwen35.model;
    _ = qwen35.chat;
    _ = gemma.gemma4;
    _ = gemma.gemma4_train;
    _ = gemma.moe;
    _ = gemma.moe_route;
    _ = gemma.moe_route_tensor;
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
    _ = pockettts.pocket;
    _ = qwen3tts.sampling;
    _ = qwen3tts.model;
    _ = qwen3tts.prompt;
    _ = qwen3tts.pipeline;
    _ = inkling.model;
    _ = inkling.mmproj;
    _ = inkling.chat;
    _ = cartridge;
    _ = cartridge_fleet;
    _ = engram;
    _ = kv_cache;
    _ = kv_persist;
    _ = tokenizer;
    _ = spm_tokenizer;
    _ = sampler;
    _ = logit_processor;
    _ = llguidance;
    _ = chat;
    _ = data;
    _ = subq;
}
