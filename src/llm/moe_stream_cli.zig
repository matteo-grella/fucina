//! Shared runner CLI for the streamed-experts flags, plus the exit-time
//! report: every MoE runner (qwen3, deepseek2, deepseek4, glm4moe) speaks
//! the same `--moe-*` flags, so their parsing, the `MoeStreamOptions`
//! assembly, and the stats report live here. Public as
//! `llm.moe_stream_cli`. The options struct the parser fills stays core
//! (`fucina.weights.MoeStreamOptions`, consumed by the loaders).

const std = @import("std");
const fucina = @import("fucina");

const MoeStreamOptions = fucina.weights.MoeStreamOptions;
const ExpertStore = fucina.ExpertStore;

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

/// Exit-time streamed-tier report shared by the MoE runners: print the
/// stats line(s) and persist the usage histogram (the learning cache)
/// unless `learn` is false. Failures lose only the report/learning.
pub fn reportAndSaveMoeStream(store: *ExpertStore, learn: bool, writer: anytype) void {
    if (learn) store.saveUsage() catch {};
    const s = store.stats;
    writer.print(
        "moe stream: {d} acquires, hits {d} / misses {d} ({d:.1}% hit, {d} pin hits), {d:.2} GB read in {d:.2}s, cap {d} slots/layer, pinned {d} experts ({d:.2} GB)\n",
        .{ s.acquires, s.hits, s.misses, s.hitRate() * 100, s.pin_hits, @as(f64, @floatFromInt(s.bytes_read)) / 1e9, @as(f64, @floatFromInt(s.read_ns)) / 1e9, store.cap, store.pinned_experts, @as(f64, @floatFromInt(store.pinned_bytes)) / 1e9 },
    ) catch {};
    if (store.l2_fds.len > 0) writer.print(
        "moe l2: {d} expert reads served ({d:.2} GB in {d:.2}s thread-aggregate, {d:.2} GB/s), {d} fallbacks\n",
        .{
            store.l2_expert_hits.load(.monotonic),
            @as(f64, @floatFromInt(store.l2_bytes.load(.monotonic))) / 1e9,
            @as(f64, @floatFromInt(store.l2_read_ns.load(.monotonic))) / 1e9,
            @as(f64, @floatFromInt(store.l2_bytes.load(.monotonic))) / @max(1.0, @as(f64, @floatFromInt(store.l2_read_ns.load(.monotonic)))),
            store.l2_fallbacks.load(.monotonic),
        },
    ) catch {};
    if (s.pilot_recall_total > 0) writer.print(
        "moe pilot: recall {d:.1}% ({d}/{d} routed experts predicted), {d} experts hinted\n",
        .{ s.pilotRecall() * 100, s.pilot_recall_hits, s.pilot_recall_total, s.pilot_ranges },
    ) catch {};
    if (s.staged_loads > 0) writer.print(
        "moe prefetch: staged {d} loads ({d:.2} GB), consumed {d}, wasted {d}\n",
        .{ s.staged_loads, @as(f64, @floatFromInt(s.staged_bytes)) / 1e9, s.staged_consumed, s.staged_wasted },
    ) catch {};
    if (s.route_slots > 0) writer.print(
        "moe cache-route: {d} of {d} slots swapped to resident experts ({d:.1}%)\n",
        .{ s.route_swaps, s.route_slots, s.routeSwapRate() * 100 },
    ) catch {};
    if (store.mirrors.len > 0) {
        var total: u64 = 0;
        for (store.copy_bytes) |*b| total += b.load(.monotonic);
        writer.print("moe mirror: {d} copies, reads", .{store.mirrors.len + 1}) catch {};
        for (store.copy_bytes, 0..) |*b, i| {
            const bytes = b.load(.monotonic);
            const pct = if (total == 0) 0 else @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(total)) * 100;
            writer.print("{s}{d:.1}% ({d:.2} GB)", .{ if (i == 0) " " else " / ", pct, @as(f64, @floatFromInt(bytes)) / 1e9 }) catch {};
        }
        const fallbacks = store.mirror_fallbacks.load(.monotonic);
        if (fallbacks > 0) {
            writer.print(", {d} mirror reads fell back to the primary\n", .{fallbacks}) catch {};
        } else {
            writer.print("\n", .{}) catch {};
        }
    }
}

test "MoeStreamCli: mirror weights without mirrors abort instead of arming nothing" {
    var cli: MoeStreamCli = .{};
    try std.testing.expect(try cli.tryParse("--moe-mirror-weights=2,1"));
    try std.testing.expect(cli.armed);
    try std.testing.expectError(error.MirrorWeightsMismatch, cli.options("model.gguf"));

    // The valid pairing is unaffected: one mirror, one weight.
    var ok: MoeStreamCli = .{};
    try std.testing.expect(try ok.tryParse("--moe-mirror=/alt/model.gguf"));
    try std.testing.expect(try ok.tryParse("--moe-mirror-weights=2"));
    const opts = (try ok.options("model.gguf")).?;
    try std.testing.expectEqual(@as(usize, 1), opts.mirror_paths.len);
    try std.testing.expectEqualSlices(f32, &.{2}, opts.mirror_weights.?);

    // Unrelated flags never arm.
    var off: MoeStreamCli = .{};
    try std.testing.expect(!try off.tryParse("--prompt"));
    try std.testing.expect((try off.options("model.gguf")) == null);
}
