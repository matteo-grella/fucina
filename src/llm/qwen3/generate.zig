//! Qwen3 greedy generation: the family-named surface over the shared
//! driver in `llm/generate.zig` (the runner and the examples own the
//! richer sampling loops).
const generate = @import("../generate.zig");

pub const Options = generate.Options;
pub const greedy = generate.greedy;
