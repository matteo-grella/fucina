//! The streamed-experts configuration band, public as
//! `fucina.weights.MoeStreamOptions` / `MoeStreamCli` / `parseMirrorWeights`:
//! the options every MoE loader takes (`LoadOptions.moe_stream`) and the
//! shared `--moe-*` argv parser the MoE example mains speak. Split from
//! the weight containers so the user-facing configuration is one file;
//! a std-only leaf (the store types it configures are named in doc
//! comments only).

const std = @import("std");

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

/// The runners' shared `--moe-mirror-weights=` comma list, parsed into
/// `buf` and validated against the number of `--moe-mirror` flags given.
/// A null argument returns null (even split).
pub fn parseMirrorWeights(arg: ?[]const u8, n_mirrors: usize, buf: []f32) !?[]const f32 {
    const list = arg orelse return null;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |tok| {
        if (n >= buf.len) return error.TooManyMirrors;
        buf[n] = try std.fmt.parseFloat(f32, tok);
        n += 1;
    }
    if (n != n_mirrors) return error.MirrorWeightsMismatch;
    return buf[0..n];
}

/// Shared runner CLI for the streamed-experts flags: every MoE runner
/// (qwen3, deepseek2, deepseek4, glm4moe) speaks the same six `--moe-*`
/// flags, so their parsing and the `MoeStreamOptions` assembly live here —
/// a new shared streaming lever lands in every runner at once.
/// Family-specific levers (`--moe-pilot`, the cache-route trio, the
/// pinned-tier knobs) stay in the runners: they arm streaming through
/// `armed` and set their fields on the returned options.
pub const MoeStreamCli = struct {
    armed: bool = false,
    cache_mb: ?usize = null,
    mirror_n: usize = 0,
    mirror_buf: [8][]const u8 = undefined,
    mirror_weights_arg: ?[]const u8 = null,
    mirror_weights_buf: [8]f32 = undefined,
    uncached: bool = false,
    io_threads: ?usize = null,
    trace_path: ?[]const u8 = null,
    l2_path: ?[]const u8 = null,
    l2_build_gb: ?u64 = null,

    /// Consume one argv entry; false = not a shared `--moe-*` flag (the
    /// caller keeps its family-specific flags and unknown-flag error).
    /// The stored slices borrow `arg`.
    pub fn tryParse(self: *MoeStreamCli, arg: []const u8) !bool {
        if (std.mem.eql(u8, arg, "--moe-stream")) {
            self.armed = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-cache-mb=")) {
            self.armed = true;
            self.cache_mb = try std.fmt.parseInt(usize, arg["--moe-cache-mb=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-mirror=")) {
            // Another full copy of the model, typically on another drive
            // (repeatable): expert reads split across every copy, so
            // miss-bound streaming gets each drive's bandwidth.
            self.armed = true;
            if (self.mirror_n >= self.mirror_buf.len) return error.TooManyMirrors;
            self.mirror_buf[self.mirror_n] = arg["--moe-mirror=".len..];
            self.mirror_n += 1;
        } else if (std.mem.startsWith(u8, arg, "--moe-mirror-weights=")) {
            // Per-mirror read share relative to the primary's 1, comma
            // list in --moe-mirror order (default 1 each: even split).
            // Arms like every sibling so that weights WITHOUT --moe-mirror
            // reach `options`' parseMirrorWeights and abort with
            // MirrorWeightsMismatch instead of silently running unmirrored.
            self.armed = true;
            self.mirror_weights_arg = arg["--moe-mirror-weights=".len..];
        } else if (std.mem.eql(u8, arg, "--moe-uncached")) {
            // Uncached streamed reads (macOS F_NOCACHE): expert streaming
            // stops churning the page cache behind the mmap'd dense weights.
            self.armed = true;
            self.uncached = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-io-threads=")) {
            // Demand-miss read fan-out (default 8; 0 = sequential reads).
            self.armed = true;
            self.io_threads = try std.fmt.parseInt(usize, arg["--moe-io-threads=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-trace=")) {
            // Record the routed (layer, expert) sequence and write it here
            // at store teardown — the input of tools/replay_experts.zig
            // (offline LRU/Belady/pinned cache replay across capacities).
            self.armed = true;
            self.trace_path = arg["--moe-trace=".len..];
        } else if (std.mem.startsWith(u8, arg, "--moe-l2=")) {
            // L2 expert tier on a faster drive (sparse partial mirror +
            // presence index). Same bytes, same kernels — only the drive
            // serving a miss changes.
            self.armed = true;
            self.l2_path = arg["--moe-l2=".len..];
        } else if (std.mem.startsWith(u8, arg, "--moe-l2-build-gb=")) {
            // (Re)build the tier at load: hottest non-pinned experts by
            // usage history until this many GB. Requires --moe-l2=.
            self.armed = true;
            self.l2_build_gb = try std.fmt.parseInt(u64, arg["--moe-l2-build-gb=".len..], 10);
        } else {
            return false;
        }
        return true;
    }

    /// Assemble the shared `MoeStreamOptions`, null when nothing armed
    /// streaming. The result borrows `self`'s mirror buffers, so `self`
    /// must outlive the model load. The caller sets its family-specific
    /// fields on the returned value afterwards.
    pub fn options(self: *MoeStreamCli, gguf_path: []const u8) !?MoeStreamOptions {
        if (!self.armed) return null;
        if (self.l2_build_gb != null and self.l2_path == null) return error.L2BuildWithoutPath;
        return .{
            .gguf_path = gguf_path,
            .cache_bytes = if (self.cache_mb) |mb| mb << 20 else null,
            .mirror_paths = self.mirror_buf[0..self.mirror_n],
            .mirror_weights = try parseMirrorWeights(self.mirror_weights_arg, self.mirror_n, &self.mirror_weights_buf),
            .io_workers = self.io_threads orelse 8,
            .uncached = self.uncached,
            .trace_path = self.trace_path,
            .l2_path = self.l2_path,
            .l2_build_bytes = if (self.l2_build_gb) |gb| gb << 30 else null,
        };
    }
};
