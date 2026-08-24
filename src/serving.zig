//! The transport band (`fucina_serving`): the model-free half of serving.
//! HTTP front end, request scheduling, response emission, and the wire
//! dialects. Everything here is written against the serving contract
//! (`fucina_models.text.serving`: `GenerateRequest`/`GenerateResult`, the
//! `Backend` vtable), which stays in the models band together with the
//! generic GGUF chat engine and the `open` load-and-serve entry.
//! `apps/lmserve` is the CLI front end; the voice agent hosts the same
//! stack in-process.

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

test {
    _ = http;
    _ = scheduler;
    _ = emitter;
    _ = openai;
    _ = anthropic;
    _ = toolcall;
    _ = @import("serving/wire_json.zig");
}
