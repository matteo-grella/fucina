//! The serving contract: the model-agnostic boundary between an HTTP (or
//! any other) serving layer and a hosted model family. A server parses its
//! wire dialect into one chat-shaped `GenerateRequest`; a `Backend` — one
//! per hosted family — turns it into streamed reply bytes plus a
//! `GenerateResult`. Everything model-family-specific (template, tokenizer,
//! cache type, generation paradigm) stays behind the `Backend` vtable, so a
//! new family integrates by writing one adapter and a new server by
//! consuming this band (the in-tree HTTP front end sits in the
//! `fucina_serving` module, `src/serving/http.zig`, consumed by
//! `apps/lmserve`). Every name here is re-exported flat on
//! `models.text.serving` (`../serving.zig`, the band index).

const std = @import("std");
const fucina = @import("fucina");
const chat = @import("../chat.zig");
const sampler = @import("../sampler.zig");

/// A grammar constraint requested by the client (`response_format` /
/// `text.format` JSON schema, or the llama.cpp-style `regex` / `lark`
/// extension fields). Source text is borrowed from the request arena.
pub const ConstraintSpec = union(enum) {
    json_schema: []const u8,
    regex: []const u8,
    lark: []const u8,

    /// Stable cache-key prefix per grammar kind.
    pub fn kindByte(self: ConstraintSpec) u8 {
        return switch (self) {
            .json_schema => 'j',
            .regex => 'r',
            .lark => 'l',
        };
    }

    pub fn source(self: ConstraintSpec) []const u8 {
        return switch (self) {
            .json_schema => |s| s,
            .regex => |s| s,
            .lark => |s| s,
        };
    }
};

/// One normalized generation request. All slices are owned by the request's
/// arena (`scheduler.Job`), alive until the job finishes.
pub const GenerateRequest = struct {
    /// Full message history, already normalized by the OpenAI layer
    /// (developer -> system, content parts flattened to text).
    messages: []const chat.Message,
    /// Fully resolved sampling config (model defaults + client overrides).
    sampling: sampler.Config,
    /// Per-reply generation cap. Always bounded: an unbounded budget plus a
    /// grammar with an open-ended field can loop forever (see
    /// docs/CONSTRAINED-DECODING.md §7).
    max_tokens: usize,
    /// Client stop strings (OpenAI `stop`).
    stop: []const []const u8 = &.{},
    constraint: ?ConstraintSpec = null,
    /// Reasoning enabled (`reasoning.effort` != "none"). Only offered when
    /// the backend reports `caps.think`; a constraint forces it off — the
    /// grammar governs the reply from token 0.
    think: bool = false,
};

pub const FinishReason = enum { stop, length };

pub const GenerateResult = struct {
    prompt_tokens: usize,
    completion_tokens: usize,
    /// Prompt tokens whose KV rows were reused from the previous request
    /// (cross-request prefix cache) instead of being prefilled; a subset of
    /// `prompt_tokens`. Backends without the reuse slot report 0.
    cached_tokens: usize = 0,
    /// When a client text stop sequence ended the reply: its index into the
    /// request's `stop` list (the Anthropic dialect reports the sequence
    /// itself as `stop_sequence`). Null when the turn ended any other way.
    stop_sequence: ?usize = null,
    finish: FinishReason,
};

/// What a backend can honor; the OpenAI layer rejects (400) requests that
/// need an absent capability instead of silently dropping the field.
pub const Caps = struct {
    /// JSON-schema / regex / Lark constraints (llguidance built in AND the
    /// backend's tokenizer is bridged).
    grammar: bool = false,
    /// Reasoning channel can be toggled per request.
    think: bool = false,
    /// Text stop sequences honored.
    stop_sequences: bool = true,
};

/// The reply's reasoning-block delimiters (the OpenAI layer routes the
/// enclosed text to `reasoning_content` / a reasoning item instead of the
/// message content).
pub const ThinkMarkers = struct { open: []const u8, close: []const u8 };

/// The tool-calling convention the family's chat template speaks. `hermes`
/// is the Qwen3 shape: declarations as JSON inside `<tools>` in the system
/// block, calls emitted as `<tool_call>{"name",…,"arguments":{…}}</tool_call>`,
/// results returned inside `<tool_response>` sections of a user turn
/// (`toolcall.zig` renders and scans it). `none` backends reject tool
/// fields at parse time.
pub const ToolStyle = enum { none, hermes };

pub const Info = struct {
    /// Model id echoed by `GET /v1/models` and in responses (file basename).
    model_id: []const u8,
    /// Per-request context budget (prompt + reply tokens).
    context_len: usize,
    caps: Caps = .{},
    /// Present when `caps.think`.
    think_markers: ?ThinkMarkers = null,
    tool_style: ToolStyle = .none,
    default_sampling: sampler.Config = .{},
};

/// Errors the OpenAI layer maps to specific HTTP responses (anything else is
/// a 500). Backends surface them from `validate`/`generate`.
pub const RequestError = error{
    /// Prompt alone exceeds the context budget (400).
    PromptTooLong,
    /// Message list shape the template cannot render (400): empty, trailing
    /// assistant message, or a mid-conversation system message on a
    /// single-system-slot template.
    EmptyMessages,
    TrailingAssistantMessage,
    SystemMidConversation,
    /// The model's chat protocol has no system role at all (nanochat).
    NoSystemRole,
    /// Grammar rejected by llguidance (400).
    InvalidGrammar,
    /// Built without -Dllguidance=true (501).
    LlguidanceNotEnabled,
};

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    info: Info,

    pub const VTable = struct {
        /// Cheap pre-queue validation on the CONNECTION thread: message
        /// shape, rendered prompt length vs context. Must not touch worker
        /// state (the constraint cache is worker-only).
        validate: *const fn (ptr: *anyopaque, req: *const GenerateRequest) anyerror!void,
        /// Run one generation, streaming reply bytes to `sink` (flushed per
        /// token). WORKER THREAD ONLY — backends are single-threaded by
        /// contract (one ExecContext). A sink write failure (client gone,
        /// job cancelled) aborts generation and propagates.
        generate: *const fn (ptr: *anyopaque, req: *const GenerateRequest, sink: *std.Io.Writer) anyerror!GenerateResult,
        /// Run several generations in one lockstep batched decode (lmserve
        /// `--batch`); null when the family has no batch forward. WORKER
        /// THREAD ONLY, like `generate`. Per-request failures — a dropped
        /// client's sink, a request-level setup error — land in `errs[i]`
        /// (with `results[i]` undefined) while the other requests keep
        /// decoding; a returned error is batch-fatal and applies to every
        /// request whose `errs[i]` is still null.
        generate_batch: ?*const fn (
            ptr: *anyopaque,
            reqs: []const *const GenerateRequest,
            sinks: []const *std.Io.Writer,
            results: []GenerateResult,
            errs: []?anyerror,
        ) anyerror!void = null,
    };

    pub fn validate(self: Backend, req: *const GenerateRequest) anyerror!void {
        return self.vtable.validate(self.ptr, req);
    }

    pub fn generate(self: Backend, req: *const GenerateRequest, sink: *std.Io.Writer) anyerror!GenerateResult {
        return self.vtable.generate(self.ptr, req, sink);
    }

    pub fn supportsBatch(self: Backend) bool {
        return self.vtable.generate_batch != null;
    }

    pub fn generateBatch(
        self: Backend,
        reqs: []const *const GenerateRequest,
        sinks: []const *std.Io.Writer,
        results: []GenerateResult,
        errs: []?anyerror,
    ) anyerror!void {
        return self.vtable.generate_batch.?(self.ptr, reqs, sinks, results, errs);
    }
};

/// Engine options for `serving.open`/`openFromFile`: the CLI-independent
/// form of lmserve's flags. Exclusions an engine cannot host are rejected
/// with `error.InvalidOptions`: `fleet_dir` excludes `cartridge_path`,
/// `kv_cache_dir` and `batch > 1`; the engine-hosted families (qwen35,
/// inkling, deepseek4) reject cartridges, fleets and the KV disk tier.
/// SHINE adapter fleets are served by `models.qwen3.shine_serving`, which
/// shares this options surface.
pub const OpenOptions = struct {
    /// Per-request context budget in tokens (prompt + reply).
    context_len: usize = 4096,
    /// Speculative decoding for solo generations (qwen3/qwen3moe).
    spec: bool = false,
    /// Lockstep batch width the host intends to drive (sizes the constraint
    /// cache and raises the KV slot pool to one slot per stream).
    batch: usize = 1,
    /// Zero-copy MoE expert load (gemma4).
    experts_borrow: bool = false,
    /// Disk-streaming tier for MoE expert weights (deepseek4). Borrowed:
    /// the options' buffers must outlive the open call.
    moe_stream: ?fucina.weights.MoeStreamOptions = null,
    /// Resident cross-request KV reuse slots (each a full `context_len`
    /// cache; the RAM guard prices the total at load).
    kv_slots: usize = 1,
    /// Keep the requested `kv_slots` even when the RAM guard would clamp.
    kv_slots_force: bool = false,
    /// Evict-to-disk KV tier directory (must exist; null = off).
    kv_cache_dir: ?[]const u8 = null,
    /// Max sidecar files under `kv_cache_dir`.
    kv_disk_slots: usize = 8,
    /// Trained KV-prefix cartridge (safetensors path; docs/CARTRIDGES.md).
    cartridge_path: ?[]const u8 = null,
    /// Per-document cartridge fleet directory (qwen3 dense, gemma4).
    fleet_dir: ?[]const u8 = null,
    /// Fleet: documents composed per request.
    rag_docs: usize = 2,
    /// Fleet: cosine top-N chunks scanned per selection.
    rag_chunks: usize = 8,
    /// Fleet: decisive-margin knowledge-base switching for continuing
    /// conversations (default fully sticky).
    rag_adaptive: bool = false,
    /// Fleet: the adaptive switch margin (cosine units).
    rag_margin: f32 = 0.05,
};

/// A loaded serving engine: the model, tokenizer, optional cartridge/fleet
/// state, and the adapter behind `backend`, heap-owned behind one handle.
/// `backend` stays valid until `deinit`.
pub const Opened = struct {
    ptr: *anyopaque,
    destroyFn: *const fn (ptr: *anyopaque) void,
    backend: Backend,
    /// The model's streamed-expert store when one is armed (`moe_stream`):
    /// the host reads its exit-time report before `deinit`.
    expert_store: ?*fucina.ExpertStore = null,

    pub fn deinit(self: *Opened) void {
        self.destroyFn(self.ptr);
        self.* = undefined;
    }
};

/// GGUF-recommended sampling (`general.sampling.*`), as the gemma4 chat
/// harness reads it.
pub fn samplingFromGguf(file: *const fucina.gguf.File) sampler.Config {
    return .{
        .temperature = if (file.getFloat("general.sampling.temp")) |v| @floatCast(v) else 1.0,
        .top_k = if (file.getInt("general.sampling.top_k")) |v| @intCast(@max(@as(i64, 0), v)) else 64,
        .top_p = if (file.getFloat("general.sampling.top_p")) |v| @floatCast(v) else 0.95,
        .min_p = if (file.getFloat("general.sampling.min_p")) |v| @floatCast(v) else 0.0,
        .repeat_penalty = if (file.getFloat("general.sampling.penalty_repeat")) |v| @floatCast(v) else 1.0,
        .freq_penalty = if (file.getFloat("general.sampling.penalty_freq")) |v| @floatCast(v) else 0.0,
        .presence_penalty = if (file.getFloat("general.sampling.penalty_present")) |v| @floatCast(v) else 0.0,
        .repeat_last_n = if (file.getInt("general.sampling.penalty_last_n")) |v| @intCast(@max(@as(i64, 0), v)) else 64,
    };
}
