//! Serving band index. The contract (`Backend` vtable, `GenerateRequest`/
//! `GenerateResult`, `Caps`) lives in `serving/contract.zig` and is
//! re-exported flat here; the transport and engine sub-modules follow the
//! `models.zig` family-namespace pattern. `open`/`openFromFile` load a GGUF
//! and return a ready `Backend` for every registry-served family: the
//! `Conversation`-hosted set (qwen3, qwen3moe, gemma4) through the generic
//! engine box, and the engine-hosted set (qwen35, qwen35moe, inkling,
//! deepseek4) through the family serving adapters. `apps/lmserve` is
//! the CLI front end.

const contract = @import("serving/contract.zig");

// === The serving contract (flat re-exports; `serving/contract.zig`) ===
pub const ConstraintSpec = contract.ConstraintSpec;
pub const GenerateRequest = contract.GenerateRequest;
pub const FinishReason = contract.FinishReason;
pub const GenerateResult = contract.GenerateResult;
pub const Caps = contract.Caps;
pub const ThinkMarkers = contract.ThinkMarkers;
pub const ToolStyle = contract.ToolStyle;
pub const Info = contract.Info;
pub const RequestError = contract.RequestError;
pub const Backend = contract.Backend;

// === Transport: HTTP front end + request scheduling ===
/// HTTP server: accept loop, per-connection threads, routing, SSE plumbing
/// (needs libc on Linux for the `std.c.recv` hang-up probe).
pub const http = @import("serving/http.zig");
/// Bounded FIFO queue + the single inference worker.
pub const scheduler = @import("serving/scheduler.zig");
/// Per-request response emission (SSE frames, final bodies, reasoning split).
pub const emitter = @import("serving/emitter.zig");
/// OpenAI Chat Completions + Responses request parsing and JSON shapes.
pub const openai = @import("serving/openai.zig");
/// Anthropic Messages API request parsing and event framing.
pub const anthropic = @import("serving/anthropic.zig");
/// Hermes-style `<tool_call>` rendering and reply scanning.
pub const toolcall = @import("serving/toolcall.zig");

// === Engine: the generic GGUF chat backend + load-and-serve entry ===
/// `GgufChatBackend` (any `models.text.chat.Conversation` family), the constraint
/// cache, KV reuse slots/disk tier, and the KV RAM guard.
pub const gguf_chat = @import("serving/gguf_chat.zig");

const opener = @import("serving/open.zig");
/// Load a GGUF and return a ready `Backend` (arch-dispatched).
pub const open = opener.open;
/// `open` over an already-loaded `fucina.gguf.File`.
pub const openFromFile = opener.openFromFile;
pub const OpenOptions = opener.OpenOptions;
pub const Opened = opener.Opened;
/// GGUF-recommended sampling (`general.sampling.*`) with gemma-shaped
/// fallbacks.
pub const samplingFromGguf = opener.samplingFromGguf;
