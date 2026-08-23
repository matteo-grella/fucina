//! The streamed-experts configuration band, public as
//! `fucina.weights.MoeStreamOptions`: the options every MoE loader takes
//! (`LoadOptions.moe_stream`). Split from the weight containers so the
//! user-facing configuration is one file; a leaf (the store types it
//! configures are named in doc comments only). The shared `--moe-*` argv
//! parser and the exit-time report live with the runners
//! (`llm.moe_stream_cli`).

/// Opt-in disk streaming for the MoE expert stacks, shared by every MoE
/// loader (`LoadOptions.moe_stream`): experts stay on disk and are `pread`
/// on demand through a tiered store (pinned + LRU + working set), so a
/// mixture model loads with only its dense weights resident. Decode then
/// pays disk reads for expert-cache misses — the explicit trade that lets a
/// bigger-than-RAM model run at all (docs: out-of-core MoE).
pub const MoeStreamOptions = struct {
    /// Path of the same GGUF being loaded; the store opens its own read fds
    /// (every part of a split GGUF), so the load-time mmap can be released
    /// after load — resident memory stays dense weights + expert cache.
    gguf_path: []const u8,
    /// Total RAM budget for the streamed tiers across all layers. Default:
    /// half of available memory at load time.
    cache_bytes: ?usize = null,
    /// Fixed LRU slots per layer; wins over `cache_bytes` when set.
    cache_slots_per_layer: ?usize = null,
    /// OS readahead hints for miss batches.
    readahead: bool = true,
    /// The learning cache: pin the hottest experts from the persisted usage
    /// sidecar (`<gguf>.experts`) at load; save updated counts with
    /// `ExpertStore.saveUsage` at generation/turn boundaries.
    auto_pin: bool = true,
    /// RAM for the pinned tier (default: half the budget when history
    /// qualifies).
    pin_bytes: ?usize = null,
    /// Router-lookahead prefetch: predict each next layer's experts from the
    /// current post-attention state and readahead them from a background I/O
    /// thread while the current layer computes. Honored by the models that
    /// implement the lookahead hook (qwen3, deepseek2); never changes
    /// output. Prediction recall is measured in `ExpertStore.Stats`.
    pilot: bool = false,
    /// Extra full copies of the model, one path per copy (part-1 path for
    /// split GGUFs; siblings resolve like `gguf_path`), typically each on
    /// its own drive. Expert reads split across every copy by a
    /// deterministic weighted hash, so aggregate streaming bandwidth
    /// scales with the drives holding one — the lever for disk-bound
    /// decode. Output is unchanged; a mirror read error falls back to the
    /// primary.
    mirror_paths: []const []const u8 = &.{},
    /// Read share per mirror relative to the primary's 1 (parallel to
    /// `mirror_paths`); null = 1 each, an even split across all copies.
    mirror_weights: ?[]const f32 = null,
    /// Demand-miss reads fan out across this many persistent I/O worker
    /// threads (the acquiring thread participates too); 0 = sequential.
    /// Parallelism is what lets disk queue depth — and mirror copies on
    /// separate drives — add bandwidth within one acquire.
    io_workers: usize = 8,
    /// Uncached streamed reads (macOS F_NOCACHE; Linux no-op for now):
    /// stops expert streaming from churning the page cache that backs the
    /// mmap'd dense weights. Opt-in (`--moe-uncached`).
    uncached: bool = false,
    /// Cache-aware routing (`ExpertStore.cacheRouteTopK`), default off:
    /// near-tie expert selection prefers already-resident experts, trading
    /// exact top-k routing for fewer disk fetches. QUALITY-AFFECTING —
    /// opt-in only; the swap fraction is reported at exit. Wired into the
    /// deepseek2, glm4moe, and deepseek4 selection paths (qwen3moe routes
    /// through the fused `routerTopK` core op, which has no selection
    /// override yet).
    cache_route: bool = false,
    /// True top ranks always taken (resident or not) under cache routing.
    route_sacred: usize = 2,
    /// Max-rank window the resident-preferring fill may draw from.
    route_window: usize = 12,
    /// Record the routed (layer, expert) sequence and write it to this path
    /// at store teardown (`ExpertStore.saveTrace` format) — the input of
    /// `tools/replay_experts.zig`. Borrowed; must outlive the model.
    trace_path: ?[]const u8 = null,
    /// L2 expert tier on a faster drive (`ExpertStore.Options.l2_path`):
    /// sparse per-part files + `.idx` presence sidecar. Borrowed.
    l2_path: ?[]const u8 = null,
    /// (Re)build the tier at load up to this many payload bytes
    /// (`ExpertStore.Options.l2_build_bytes`). Requires `l2_path`.
    l2_build_bytes: ?u64 = null,
};
