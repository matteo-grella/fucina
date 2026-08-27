//! Streamed-expert policy: mirror routing (`MirrorSet`, `routeCopy`),
//! cache-aware expert selection (`cacheRouteTopK`), the live pinned-tier
//! adaptation pass (`repinPass`), the LRU-vs-heat eviction A/B gate
//! (`envPlainLru`), and the memory-budget probe (`memAvailableBytes`).
//! Store-parameterized functions take the owning `ExpertStore` as
//! `anytype` (see tiers.zig for the cycle rationale) and follow the
//! facade's locking contract.
const std = @import("std");
const builtin = @import("builtin");
const tuning = @import("../tuning.zig");
const io = @import("io.zig");
const tiers = @import("tiers.zig");

/// `FUCINA_MOE_LRU=1` forces the pure-LRU victim scan (A/B on one binary);
/// read once, cached (`tuning.Table.moe_lru`).
pub fn envPlainLru() bool {
    return tuning.get().moe_lru;
}

/// One extra full copy of the model (same split parts, another drive) and
/// its relative read weight (the primary's weight is 1).
pub const MirrorSet = struct { fds: []io.fd_t, weight: f32 };

/// Currently-available physical memory, best effort: Linux reads
/// `MemAvailable` (free + reclaimable page cache); macOS sums the free,
/// speculative, and purgeable page pools — deliberately NOT the inactive or
/// external (file-cache) pools, so the number still understates what the OS
/// could reclaim but never counts page cache that consumers (streamed
/// experts, the lmserve KV guard) exist to protect. `null` when
/// undeterminable.
pub fn memAvailableBytes() ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const open_rc = linux.openat(linux.AT.FDCWD, "/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(open_rc) != .SUCCESS) return null;
            const fd: io.fd_t = @intCast(open_rc);
            defer io.closeFd(fd);
            var buf: [4096]u8 = undefined;
            const read_rc = linux.read(fd, &buf, buf.len);
            if (linux.errno(read_rc) != .SUCCESS) return null;
            const n = read_rc;
            var it = std.mem.splitScalar(u8, buf[0..n], '\n');
            while (it.next()) |line| {
                const prefix = "MemAvailable:";
                if (!std.mem.startsWith(u8, line, prefix)) continue;
                const rest = std.mem.trim(u8, line[prefix.len..], " ");
                const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
                const kb = std.fmt.parseInt(u64, rest[0..end], 10) catch return null;
                return kb * 1024;
            }
            return null;
        },
        .macos => {
            var total_pages: u64 = 0;
            inline for (.{ "vm.page_free_count", "vm.page_speculative_count", "vm.page_purgeable_count" }) |name| {
                var pages: c_int = 0;
                var len: usize = @sizeOf(c_int);
                if (std.c.sysctlbyname(name, &pages, &len, null, 0) != 0) return null;
                if (pages < 0) return null;
                total_pages += @intCast(pages);
            }
            var page_size: c_int = 0;
            var len: usize = @sizeOf(c_int);
            if (std.c.sysctlbyname("hw.pagesize", &page_size, &len, null, 0) != 0) return null;
            if (page_size <= 0) return null;
            return total_pages * @as(u64, @intCast(page_size));
        },
        else => return null,
    }
}

/// Copy index (0 = primary) serving an expert's bytes: a splitmix64
/// finalizer over (layer, expert) cut against the cumulative weight
/// table. No mirrors = primary, always.
pub fn routeCopy(self: anytype, layer_i: usize, eid: u32) usize {
    if (self.route_cum.len == 0) return 0;
    var h = (@as(u64, @intCast(layer_i)) << 32) | eid;
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    const cut: u32 = @intCast(h & 0xFFFF);
    for (self.route_cum, 0..) |threshold, i| {
        if (cut < threshold) return i;
    }
    return 0;
}

/// Largest max-rank window `cacheRouteTopK` supports (stack scratch).
pub const max_route_window = 64;

/// Cache-aware top-k expert selection (max-rank, arXiv:2412.00099 —
/// colibri's CACHE_ROUTE), active only when `Options.cache_route` was
/// set: rank the bias-augmented `choice` scores, always take the true
/// top-`sacred` ranks, fill the remaining `sel` slots preferring
/// experts whose blocks already sit in RAM (pinned, LRU, or staged)
/// among the top-`window` ranks, and complete in true rank order. Ties
/// resolve to the lowest expert id, like the plain top-k it replaces.
/// Returns false when cache routing is off or the shape doesn't apply
/// — the caller keeps its plain selection. WORKER THREAD ONLY (shares
/// the acquire scratch).
pub fn cacheRouteTopK(self: anytype, layer_i: usize, choice: []const f32, sel: []usize) bool {
    const cr = self.options.cache_route orelse return false;
    if (!self.finalized or layer_i >= self.layers.len or !self.registered[layer_i]) return false;
    const ls = &self.layers[layer_i];
    const k = sel.len;
    if (k == 0 or k > max_route_window or choice.len != ls.n_expert or k > ls.n_expert) return false;

    var rank_buf: [max_route_window]u32 = undefined;
    const window: usize = @min(@max(cr.window, k), @min(ls.n_expert, max_route_window));
    const sacred: usize = @min(cr.sacred, k);

    // Rank the top-`window` scores (strict > keeps ties on the lowest
    // id, the plain top-k's order).
    const taken = self.seen[0..ls.n_expert];
    @memset(taken, false);
    for (rank_buf[0..window]) |*slot| {
        var best: usize = 0;
        var best_c = -std.math.inf(f32);
        for (choice, 0..) |c, e| {
            if (!taken[e] and c > best_c) {
                best_c = c;
                best = e;
            }
        }
        taken[best] = true;
        slot.* = @intCast(best);
    }

    var chosen: usize = 0;
    for (rank_buf[0..sacred]) |e| {
        sel[chosen] = e;
        chosen += 1;
    }
    if (chosen < k) {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (rank_buf[sacred..window]) |e| {
            if (chosen == k) break;
            if (residentLocked(self, ls, layer_i, e)) {
                sel[chosen] = e;
                chosen += 1;
            }
        }
    }
    // Remainder in true rank order, skipping what's already chosen.
    outer: for (rank_buf[0..window]) |e| {
        if (chosen == k) break;
        for (sel[0..chosen]) |s| {
            if (s == e) continue :outer;
        }
        sel[chosen] = e;
        chosen += 1;
    }

    // Swap accounting vs the true top-k (rank_buf[0..k]).
    self.stats.route_slots += @intCast(k);
    outer2: for (sel[0..k]) |s| {
        for (rank_buf[0..k]) |t| {
            if (t == s) continue :outer2;
        }
        self.stats.route_swaps += 1;
    }
    return true;
}

/// Residency probe for cache-aware routing: any tier whose blocks are
/// already in RAM. Caller holds `mutex`; the staging peek takes
/// `stage_mutex` in the same order the worker's `stageConsume` does.
fn residentLocked(self: anytype, ls: *tiers.LayerState, layer_i: usize, eid: u32) bool {
    if (tiers.findPinned(ls, eid) != null) return true;
    if (self.findCached(ls, eid) != null) return true;
    if (self.stage_meta.len > 0) {
        self.stage_mutex.lock();
        defer self.stage_mutex.unlock();
        for (self.stage_meta) |*m| {
            if (m.state == .ready and m.layer_i == @as(u32, @intCast(layer_i)) and m.eid == eid) return true;
        }
    }
    return false;
}

// ---- live tier adaptation ----------------------------------------------

/// One adaptation pass over the pinned tier: per layer, replace up to
/// `max_swaps_per_layer` of the coldest pinned experts with hotter
/// streamed ones (25% + fixed hysteresis — see `tierPickSwap`), reading
/// the replacement into the existing pinned slab; then halve the heat so
/// the signal follows the current workload. Call at safe boundaries
/// (between generations / chat turns), never inside an acquire. Returns
/// the number of swaps performed.
pub fn repinPass(self: anytype, max_swaps_per_layer: usize) usize {
    self.mutex.lock();
    defer self.mutex.unlock();
    std.debug.assert(self.acquired_layer == null);
    if (!self.finalized) return 0;

    var swaps: usize = 0;
    for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
        if (!reg) continue;
        if (ls.pinned.len > 0) {
            var done: usize = 0;
            while (done < max_swaps_per_layer) : (done += 1) {
                const pick = tiers.tierPickSwap(ls) orelse break;
                const slot = &ls.pinned[pick.slot];
                // Invalidate first: a failed read leaves the slot skipped
                // by lookups instead of resolving to stale bytes.
                slot.eid = tiers.invalid_eid;
                slot.ensureCapacity(self.allocator, ls.slab_bytes) catch break;
                self.readExpert(ls, layer_i, pick.eid, slot) catch break;
                slot.eid = pick.eid;
                swaps += 1;
            }
        }
        for (ls.heat) |*h| h.* >>= 1;
    }
    return swaps;
}
