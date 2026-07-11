//! The lmserve example's internal request/result contract: the OpenAI layer
//! (`openai.zig`) parses either wire dialect into one chat-shaped
//! `GenerateRequest`; a `Backend` (one per hosted model family —
//! `backend.zig` adapters) turns it into streamed reply bytes plus a
//! `GenerateResult`. Everything model-family-specific (template, tokenizer,
//! cache type, generation paradigm) stays behind the `Backend` vtable, so a
//! new family integrates by writing one adapter.

const std = @import("std");
const llm = @import("fucina_llm");

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
    messages: []const llm.chat.Message,
    /// Fully resolved sampling config (model defaults + client overrides).
    sampling: llm.sampler.Config,
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

pub const Info = struct {
    /// Model id echoed by `GET /v1/models` and in responses (file basename).
    model_id: []const u8,
    /// Per-request context budget (prompt + reply tokens).
    context_len: usize,
    caps: Caps = .{},
    /// Present when `caps.think`.
    think_markers: ?ThinkMarkers = null,
    default_sampling: llm.sampler.Config = .{},
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
    };

    pub fn validate(self: Backend, req: *const GenerateRequest) anyerror!void {
        return self.vtable.validate(self.ptr, req);
    }

    pub fn generate(self: Backend, req: *const GenerateRequest, sink: *std.Io.Writer) anyerror!GenerateResult {
        return self.vtable.generate(self.ptr, req, sink);
    }
};
