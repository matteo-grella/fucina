//! The expert cache tiers: the slab slot and per-layer LRU+heat state
//! (`Slot`, `LayerState`, the heat-vs-LRU pick helpers), the pinned hot
//! tier loader (`selectAndLoadPins`), the pilot staging ring (SPSC
//! prefetch worker: `pilotStart`/`pilotEnqueue`/`pilotWorker`/`stageLoad`/
//! `stageConsume`), and the striped L2 tier with its CL2F offset index
//! (`l2Build`/`l2Open`/`l2Offset`). Store-parameterized functions take the
//! owning `ExpertStore` as `anytype` (the facade imports this file; an
//! `ExpertStore` import here would be an import cycle) and follow the
//! facade's locking contract, stated on `expert_store.zig`.
const std = @import("std");
const tuning = @import("../tuning.zig");
const io = @import("io.zig");
const geometry = @import("geometry.zig");

const Allocator = std.mem.Allocator;
const Error = io.Error;
const ProjGeometry = geometry.ProjGeometry;

/// Block slabs start page-aligned (16 KiB covers Apple Silicon — a superset
/// of every kernel-load alignment need, K-quant blocks themselves only being
/// 2-byte aligned): uncached reads (F_NOCACHE) into sub-page-aligned buffers
/// take the kernel's bounce path; page-aligned slabs keep the DMA fast path.
pub const slab_align = 16384;
pub const invalid_eid = std.math.maxInt(u32);

/// One expert prefetch request for the pilot's I/O thread (SPSC ring
/// entry). The worker LOADS the expert into a staging slot — true
/// asynchronous prefetch, not kernel advice; a lost request is not an
/// error (staging is advisory, `acquire` always falls back to its own
/// synchronous read).
pub const PilotReq = struct { layer_i: u32, eid: u32 };
pub const pilot_ring_cap = 4096;

/// Staging-slot lifecycle (`stage_mutex` guards every transition; the slab
/// is worker-owned while `.loading`, so neither consume nor reclaim may
/// touch it until `.ready`).
pub const StageState = enum(u8) { empty, loading, ready };
pub const StageMeta = struct { layer_i: u32 = 0, eid: u32 = invalid_eid, state: StageState = .empty, stamp: u64 = 0 };

/// One cached expert: a single slab holding its gate+up+down blocks (loaded
/// with one coalesced logical fetch), stamped for
/// LRU. Slots inside a layer all share that layer's slab size; the shared
/// working-set slots are re-checked per use because layers may differ.
pub const Slot = struct {
    eid: u32 = invalid_eid,
    used: u64 = 0,
    slab: []align(slab_align) u8 = &.{},

    pub fn ensureCapacity(self: *Slot, allocator: Allocator, bytes: usize) !void {
        if (self.slab.len >= bytes) return;
        if (self.slab.len > 0) allocator.free(self.slab);
        self.slab = &.{};
        self.slab = try allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(slab_align), bytes);
    }

    pub fn deinit(self: *Slot, allocator: Allocator) void {
        if (self.slab.len > 0) allocator.free(self.slab);
        self.* = undefined;
    }
};

pub const LayerState = struct {
    projs: [3]ProjGeometry,
    n_expert: usize,
    /// Per-expert slab layout: projection `p`'s blocks start at `proj_off[p]`.
    proj_off: [3]usize,
    slab_bytes: usize,
    /// Slab-native layer (`addLayerSlab`): the on-disk record IS the RAM
    /// slab, byte for byte — expert `e` occupies
    /// `[slab_file_offset + e*slab_bytes, +slab_bytes)` in `slab_part` and
    /// a miss is ONE contiguous pread (three per-projection reads
    /// otherwise). The per-projection ProjGeometry stays authoritative for
    /// serving; only the read path changes.
    slab_file_offset: ?u64 = null,
    slab_part: u16 = 0,
    /// LRU tier (`cap` entries; slabs allocated on first promotion).
    slots: []Slot = &.{},
    n_slots: usize = 0,
    /// Pinned hot tier: experts selected from the persistent usage history at
    /// finalize (and adapted by `repinPass`); checked before the LRU, never
    /// evicted by promotions.
    pinned: []Slot = &.{},
    /// Resolved pointer per (expert, projection), valid between acquire and
    /// release for the acquired experts only.
    resolved: [][3]?[*]const u8 = &.{},
    /// Persistent routing histogram (one count per routed pair), the raw
    /// signal for usage-driven pinning. Never decayed.
    usage: []u64 = &.{},
    /// Recent-routing heat for live pin adaptation; halved every
    /// `repinPass` so the pinned tier follows the current workload while
    /// `usage` keeps the long-term history.
    heat: []u32 = &.{},
    /// Pilot prediction marks (epoch-stamped): `pred_marks[e] == pred_epoch`
    /// means expert `e` was predicted for this layer's next acquire; the
    /// acquire scores recall and bumps `pred_scored`.
    pred_marks: []u32 = &.{},
    pred_epoch: u32 = 0,
    pred_scored: u32 = 0,

    pub fn expertSlabOffsets(projs: [3]ProjGeometry) struct { off: [3]usize, total: usize } {
        var off: [3]usize = undefined;
        var at: usize = 0;
        for (projs, 0..) |g, p| {
            at = std.mem.alignForward(usize, at, slab_align);
            off[p] = at;
            at += g.expert_bytes;
        }
        return .{ .off = off, .total = std.mem.alignForward(usize, at, slab_align) };
    }

    pub fn isPinned(self: *const LayerState, eid: u32) bool {
        for (self.pinned) |*slot| {
            if (slot.eid == eid) return true;
        }
        return false;
    }
};

/// Pick one pinned slot to replace from recent routing heat: the coldest
/// pinned expert vs the hottest unpinned one. The fixed +4 margin handles
/// tiny samples; the 25% margin prevents ping-pong (tier hysteresis).
pub fn tierPickSwap(ls: *const LayerState) ?struct { slot: usize, eid: u32 } {
    if (ls.pinned.len == 0 or ls.heat.len == 0) return null;
    var cold: usize = 0;
    for (ls.pinned, 0..) |*slot, i| {
        if (slot.eid == invalid_eid) return .{ .slot = i, .eid = hottestUnpinned(ls) orelse return null };
        if (ls.heat[slot.eid] < ls.heat[ls.pinned[cold].eid]) cold = i;
    }
    const hot = hottestUnpinned(ls) orelse return null;
    const fc = ls.heat[ls.pinned[cold].eid];
    const fh = ls.heat[hot];
    if (fh <= fc + (fc >> 2) + 4) return null;
    return .{ .slot = cold, .eid = hot };
}

pub fn hottestUnpinned(ls: *const LayerState) ?u32 {
    var best: ?u32 = null;
    var best_heat: u32 = 0;
    for (ls.heat, 0..) |h, e| {
        const eid: u32 = @intCast(e);
        if (h == 0 or ls.isPinned(eid)) continue;
        if (best == null or h > best_heat) {
            best = eid;
            best_heat = h;
        }
    }
    return best;
}

pub fn findPinned(ls: *LayerState, eid: u32) ?*Slot {
    for (ls.pinned) |*slot| {
        if (slot.eid == eid) return slot;
    }
    return null;
}

pub fn expertBytes(ls: *const LayerState) u64 {
    var total: u64 = 0;
    for (&ls.projs) |*g| total += g.expert_bytes;
    return total;
}

/// Greedy hottest-first pin selection over every registered layer's
/// usage histogram, then one sequential read pass to load the picks.
/// Returns the bytes spent. Experts with zero recorded usage are never
/// pinned.
pub fn selectAndLoadPins(self: anytype, pin_budget: usize) Error!usize {
    const Cand = struct { count: u64, layer: u32, eid: u32 };
    var cands: std.ArrayList(Cand) = .empty;
    defer cands.deinit(self.allocator);
    for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
        if (!reg) continue;
        for (ls.usage, 0..) |count, e| {
            if (count == 0) continue;
            try cands.append(self.allocator, .{ .count = count, .layer = @intCast(layer_i), .eid = @intCast(e) });
        }
    }
    std.mem.sort(Cand, cands.items, {}, struct {
        fn hotter(_: void, a: Cand, b: Cand) bool {
            if (a.count != b.count) return a.count > b.count;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.eid < b.eid;
        }
    }.hotter);

    // Pass 1: greedy pick under the budget.
    const pick_counts = try self.allocator.alloc(usize, self.layers.len);
    defer self.allocator.free(pick_counts);
    @memset(pick_counts, 0);
    var picks: std.ArrayList(Cand) = .empty;
    defer picks.deinit(self.allocator);
    var spent: usize = 0;
    for (cands.items) |cand| {
        const ls = &self.layers[cand.layer];
        if (spent + ls.slab_bytes > pin_budget) continue;
        spent += ls.slab_bytes;
        pick_counts[cand.layer] += 1;
        try picks.append(self.allocator, cand);
    }
    if (picks.items.len == 0) return 0;

    // Flatness guard (see Options.auto_pin_min_advantage): compare the
    // traffic share the greedy picks would retain against what the same
    // slot count retains under a flat histogram. Skipped when every
    // used expert fits — a fully-pinned working set wins at any skew.
    if (picks.items.len < cands.items.len) {
        var picked_traffic: u64 = 0;
        for (picks.items) |cand| picked_traffic += cand.count;
        var total_traffic: u64 = 0;
        for (cands.items) |cand| total_traffic += cand.count;
        const coverage = @as(f64, @floatFromInt(picked_traffic)) / @as(f64, @floatFromInt(total_traffic));
        const flat = @as(f64, @floatFromInt(picks.items.len)) / @as(f64, @floatFromInt(cands.items.len));
        if (coverage < flat * self.options.auto_pin_min_advantage) {
            self.pins_declined_flat = true;
            return 0;
        }
    }

    // Pass 2: allocate the pinned tiers, hint the whole pick set, then
    // read it sequentially (the hints let the kernel batch the reads).
    for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
        if (!reg or pick_counts[layer_i] == 0) continue;
        ls.pinned = try self.allocator.alloc(Slot, pick_counts[layer_i]);
        @memset(ls.pinned, .{});
    }
    if (self.options.readahead) {
        for (picks.items) |cand| {
            const ls = &self.layers[cand.layer];
            const copy = self.routeCopy(cand.layer, cand.eid);
            for (&ls.projs) |*g| {
                for (0..g.plane_count) |plane| io.hintWillNeed(self.copyFd(copy, g.part), g.planeFileOffset(cand.eid, plane), g.plane_bytes);
            }
        }
    }
    const fill = try self.allocator.alloc(usize, self.layers.len);
    defer self.allocator.free(fill);
    @memset(fill, 0);
    for (picks.items) |cand| {
        const ls = &self.layers[cand.layer];
        const slot = &ls.pinned[fill[cand.layer]];
        fill[cand.layer] += 1;
        try slot.ensureCapacity(self.allocator, ls.slab_bytes);
        try self.readExpert(ls, cand.layer, cand.eid, slot);
        slot.eid = cand.eid;
    }
    self.pinned_experts = picks.items.len;
    self.pinned_bytes = spent;
    return spent;
}

pub const l2_absent: u64 = std.math.maxInt(u64);

/// Striped-tier prefix fraction: the bandwidth-balance point between
/// the fast tier drive and the slower primary. Reading prefix and
/// suffix concurrently makes every covered miss cost
/// max(P/fast, (S-P)/slow) — minimized when the split matches the
/// drives' bandwidth ratio, and always cheaper than the whole slab on
/// either device alone. Capacity then decides COVERAGE (how many
/// experts stripe), not depth.
const l2_stripe_num: u64 = 11;
const l2_stripe_den: u64 = 16;

/// Build/refresh the striped L2 tier: rank every registered
/// non-pinned expert by loaded usage (no-history falls back to natural
/// order) and, until `budget_bytes`, write each covered expert's slab
/// PREFIX (the bandwidth-balance fraction, 16 KiB-rounded) to the tier
/// file; the suffix stays on the primary and the two halves are read
/// concurrently per miss. Uniform miss cost for every covered expert,
/// both devices busy by construction, and coverage (not selection
/// quality) is the only histogram-sensitive part. Folded layers are
/// not striped (their slab bytes are not primary file bytes). The
/// tier file is TRUNCATED first: a build is a fresh snapshot.
pub fn l2Build(self: anytype, base: []const u8, budget_bytes: u64) Error!void {
    const allocator = self.allocator;
    const Entry = struct { layer: u32, eid: u32, usage: u64 };
    var list: std.ArrayList(Entry) = .empty;
    defer list.deinit(allocator);
    var max_slab: usize = 0;
    for (self.layers, self.registered, 0..) |*ls, reg, li| {
        if (!reg) continue;
        var folded = false;
        for (&ls.projs) |*g| folded = folded or g.fold;
        if (folded or l2PrefixBytes(ls.slab_bytes) == 0) continue;
        max_slab = @max(max_slab, ls.slab_bytes);
        for (0..ls.n_expert) |e| {
            const eid: u32 = @intCast(e);
            if (findPinned(ls, eid) != null) continue;
            list.append(allocator, .{ .layer = @intCast(li), .eid = eid, .usage = ls.usage[e] }) catch return Error.UsageFileWriteFailed;
        }
    }
    // Refuse before truncating: a build over a store whose every layer
    // is fold-served (an all-tied-K2 PTQTP model without
    // FUCINA_PTQTP_FOLD=0) would wipe the existing tier and record
    // zero coverage — an easy-to-miss silent perf cliff.
    if (max_slab == 0) return Error.L2NoStripeableLayers;

    const S = struct {
        fn hotter(_: void, a: Entry, b: Entry) bool {
            if (a.usage != b.usage) return a.usage > b.usage;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.eid < b.eid;
        }
    };
    std.sort.pdq(Entry, list.items, {}, S.hotter);

    const fd = try io.openWriteTrunc(allocator, base);
    defer io.closeFd(fd);

    const offsets = try allocator.alloc([]u64, self.layers.len);
    var built_layers: usize = 0;
    defer {
        for (offsets[0..built_layers]) |p| if (p.len > 0) allocator.free(p);
        allocator.free(offsets);
    }
    for (self.layers, self.registered) |*ls, reg| {
        offsets[built_layers] = if (reg) blk: {
            const p = try allocator.alloc(u64, ls.n_expert);
            @memset(p, l2_absent);
            break :blk p;
        } else &.{};
        built_layers += 1;
    }

    var slab = Slot{};
    defer slab.deinit(allocator);
    try slab.ensureCapacity(allocator, max_slab);
    var write_off: u64 = 0;
    for (list.items) |e| {
        const ls = &self.layers[e.layer];
        const prefix = l2PrefixBytes(ls.slab_bytes);
        if (write_off + prefix > budget_bytes) break;
        try self.readExpert(ls, e.layer, e.eid, &slab);
        try io.pwriteFullFd(fd, slab.slab[0..@intCast(prefix)], write_off);
        offsets[e.layer][e.eid] = write_off;
        write_off += prefix;
    }

    // Offset index: magic, version, layer count, then per registered
    // layer its expert count, one u64 slab offset per expert, and (v4)
    // the slab_bytes + content fingerprint the open-side must match.
    const idx_path = std.fmt.allocPrint(allocator, "{s}.idx", .{base}) catch return Error.UsageFileWriteFailed;
    defer allocator.free(idx_path);
    const idx_fd = try io.openWriteTrunc(allocator, idx_path);
    defer io.closeFd(idx_fd);
    var off: u64 = 0;
    var head: [12]u8 = undefined;
    std.mem.writeInt(u32, head[0..4], 0x4632_4C43, .little); // "CL2F"
    std.mem.writeInt(u32, head[4..8], 4, .little);
    std.mem.writeInt(u32, head[8..12], @intCast(self.layers.len), .little);
    try io.pwriteFullFd(idx_fd, &head, off);
    off += head.len;
    for (self.layers, offsets) |*ls, p| {
        var n: [4]u8 = undefined;
        std.mem.writeInt(u32, &n, @intCast(p.len), .little);
        try io.pwriteFullFd(idx_fd, &n, off);
        off += n.len;
        var pb: [8]u8 = undefined;
        std.mem.writeInt(u64, &pb, if (p.len > 0) l2PrefixBytes(ls.slab_bytes) else 0, .little);
        try io.pwriteFullFd(idx_fd, &pb, off);
        off += pb.len;
        if (p.len > 0) {
            try io.pwriteFullFd(idx_fd, std.mem.sliceAsBytes(p), off);
            off += p.len * 8;
        }
        var ident: [16]u8 = undefined;
        std.mem.writeInt(u64, ident[0..8], if (p.len > 0) ls.slab_bytes else 0, .little);
        std.mem.writeInt(u64, ident[8..16], if (p.len > 0) try l2LayerFingerprint(self, ls) else 0, .little);
        try io.pwriteFullFd(idx_fd, &ident, off);
        off += ident.len;
    }
}

/// Content fingerprint for a registered layer: Wyhash of the first
/// min(4 KiB, one plane) of the primary-file expert bytes this layer's
/// tier slabs mirror. Ties a tier to the EXPERT BYTES it holds — model
/// files that share expert stacks (trunk-only variants) keep matching,
/// while a file with different expert content is refused at open
/// instead of silently served another model's weights.
pub fn l2LayerFingerprint(self: anytype, ls: *const LayerState) Error!u64 {
    var buf: [4096]u8 = undefined;
    const g = &ls.projs[0];
    var part: usize = g.part;
    var offset: u64 = g.plane_offsets[0];
    if (ls.slab_file_offset) |so| {
        part = ls.slab_part;
        offset = so;
    }
    const len = @min(buf.len, g.plane_bytes);
    try io.preadFullFd(self.fds[part], buf[0..len], offset);
    return std.hash.Wyhash.hash(0x4632_4c46, buf[0..len]);
}

/// Striped prefix length for a slab: the bandwidth-balance fraction,
/// 16 KiB-rounded (uncached fast-drive reads want page-aligned
/// lengths). 0 = slab too small to stripe usefully.
pub fn l2PrefixBytes(slab_bytes: usize) u64 {
    const raw = @as(u64, slab_bytes) * l2_stripe_num / l2_stripe_den;
    const rounded = (raw / 16384) * 16384;
    if (rounded == 0 or rounded >= slab_bytes) return 0;
    return rounded;
}

/// Open the L2 tier for serving: load + validate the offset index
/// against the registered geometry, open the tier file read-only.
/// The tier fd is uncached by default, like the primary (see the
/// F_NOCACHE note at the open below; `FUCINA_MOE_L2_CACHED=1` keeps
/// it page-cached instead).
pub fn l2Open(self: anytype, base: []const u8) Error!void {
    const allocator = self.allocator;
    const idx_path = std.fmt.allocPrint(allocator, "{s}.idx", .{base}) catch return Error.ExpertFileReadFailed;
    defer allocator.free(idx_path);
    const idx_fd = try io.openReadOnly(allocator, idx_path);
    defer io.closeFd(idx_fd);
    var head: [12]u8 = undefined;
    if ((try io.preadOnce(idx_fd, &head, 0)) != head.len) return Error.InvalidExpertGeometry;
    if (std.mem.readInt(u32, head[0..4], .little) != 0x4632_4C43) return Error.InvalidExpertGeometry;
    // v3 tiers (no fingerprint) stay openable and are trusted as-is;
    // rebuilding upgrades them to v4.
    const idx_version = std.mem.readInt(u32, head[4..8], .little);
    if (idx_version != 3 and idx_version != 4) return Error.InvalidExpertGeometry;
    if (std.mem.readInt(u32, head[8..12], .little) != self.layers.len) return Error.InvalidExpertGeometry;

    const offsets = try allocator.alloc([]u64, self.layers.len);
    var built: usize = 0;
    errdefer {
        for (offsets[0..built]) |p| if (p.len > 0) allocator.free(p);
        allocator.free(offsets);
    }
    const prefixes = try allocator.alloc(u64, self.layers.len);
    errdefer allocator.free(prefixes);
    var off: u64 = head.len;
    for (self.layers, self.registered) |*ls, reg| {
        var nb: [4]u8 = undefined;
        if ((try io.preadOnce(idx_fd, &nb, off)) != nb.len) return Error.InvalidExpertGeometry;
        off += nb.len;
        const n = std.mem.readInt(u32, &nb, .little);
        const want: u32 = if (reg) @intCast(ls.n_expert) else 0;
        if (n != want) return Error.InvalidExpertGeometry;
        var pb: [8]u8 = undefined;
        if ((try io.preadOnce(idx_fd, &pb, off)) != pb.len) return Error.InvalidExpertGeometry;
        off += pb.len;
        const prefix = std.mem.readInt(u64, &pb, .little);
        if (prefix % 16384 != 0) return Error.InvalidExpertGeometry;
        if (reg and prefix >= ls.slab_bytes) return Error.InvalidExpertGeometry;
        prefixes[built] = prefix;
        offsets[built] = if (n > 0) blk: {
            const p = try allocator.alloc(u64, n);
            const bytes = std.mem.sliceAsBytes(p);
            if ((try io.preadOnce(idx_fd, bytes, off)) != bytes.len) return Error.InvalidExpertGeometry;
            off += bytes.len;
            break :blk p;
        } else &.{};
        // Tier offsets locate slab PREFIXES of primary-file bytes. A
        // layer served folded rebuilds its slab section at fill time
        // (the 4-bit pack is not primary bytes), so coverage recorded
        // for it — a tier built under FUCINA_PTQTP_FOLD=0, opened
        // without it — must never be served: `readExpertPrefix` would
        // overwrite the pack's head with unfolded plane bytes. Drop
        // that layer's coverage instead (the tier adds speed, never
        // correctness; the drop is visible as zero l2 reads in the
        // exit stats).
        // Count the layer as built NOW: the fingerprint checks below
        // can error, and the errdefer frees offsets[0..built] only.
        const cur = built;
        built += 1;
        if (reg and offsets[cur].len > 0) {
            var folded = false;
            for (&ls.projs) |*g| folded = folded or g.fold;
            if (folded) @memset(offsets[cur], l2_absent);
        }
        if (idx_version >= 4) {
            var ident: [16]u8 = undefined;
            if ((try io.preadOnce(idx_fd, &ident, off)) != ident.len) return Error.InvalidExpertGeometry;
            off += ident.len;
            if (n > 0) {
                // The tier's slabs are primary-file bytes of the model
                // it was built from: refuse any file whose expert
                // content differs (same-trunk-different-experts served
                // silently wrong is exactly the failure this catches).
                if (std.mem.readInt(u64, ident[0..8], .little) != ls.slab_bytes) return Error.InvalidExpertGeometry;
                if (std.mem.readInt(u64, ident[8..16], .little) != try l2LayerFingerprint(self, ls)) return Error.InvalidExpertGeometry;
            }
        }
    }

    const fds = try allocator.alloc(io.fd_t, 1);
    errdefer allocator.free(fds);
    fds[0] = try io.openReadOnly(allocator, base);
    // Uncached by default, like the primary: with page-aligned slabs
    // AND page-cache headroom (a RAM budget well under free memory)
    // uncached slab preads run at the fast drive's full speed, but
    // F_NOCACHE throughput collapses under memory pressure — so the
    // budget knob governs tier speed, not just capacity.
    // FUCINA_MOE_L2_CACHED=1 keeps the tier page-cached instead (the
    // FUCINA_MOE_LRU-style A/B on one binary).
    if (!tuning.get().moe_l2_cached) io.setUncached(fds[0]);
    self.l2_fds = fds;
    self.l2_present = offsets;
    self.l2_prefix = prefixes;
}

/// Slab-prefix offset of an expert in the L2 tier, null when absent.
pub fn l2Offset(self: anytype, layer_i: usize, eid: u32) ?u64 {
    if (self.l2_present.len == 0) return null;
    const p = self.l2_present[layer_i];
    if (eid >= p.len or p[eid] == l2_absent) return null;
    return p[eid];
}

pub fn pilotStart(self: anytype) !void {
    if (self.pilot_ring.len == 0) {
        self.pilot_ring = try self.allocator.alloc(PilotReq, pilot_ring_cap);
    }
    if (self.stage_meta.len == 0 and self.options.prefetch_stage_slots > 0) {
        self.stage_slots = try self.allocator.alloc(Slot, self.options.prefetch_stage_slots);
        for (self.stage_slots) |*s| s.* = .{};
        self.stage_meta = try self.allocator.alloc(StageMeta, self.options.prefetch_stage_slots);
        for (self.stage_meta) |*m| m.* = .{};
    }
    self.pilot_thread = try std.Thread.spawn(.{}, pilotWorker, .{self});
}

pub fn pilotEnqueue(self: anytype, range: PilotReq) void {
    const w = self.pilot_w.load(.monotonic);
    const r = self.pilot_r.load(.acquire);
    if (w -% r >= pilot_ring_cap) return; // full: drop the hint
    self.pilot_ring[w % pilot_ring_cap] = range;
    self.pilot_w.store(w +% 1, .release);
    self.stats.pilot_ranges += 1;
}

pub fn pilotWorker(self: anytype) void {
    io.setIoThreadQos();
    while (!self.pilot_stop.load(.acquire)) {
        const r = self.pilot_r.load(.monotonic);
        const w = self.pilot_w.load(.acquire);
        if (r == w) {
            io.sleepMicros(200);
            continue;
        }
        const req = self.pilot_ring[r % pilot_ring_cap];
        self.pilot_r.store(r +% 1, .release);
        stageLoad(self, req);
    }
}

/// Load one predicted expert into a staging slot: claim under
/// `stage_mutex`, pread outside the lock, publish ready. Duplicates,
/// full tiers, and read failures degrade to no-ops — staging is advice.
pub fn stageLoad(self: anytype, req: PilotReq) void {
    const ls = &self.layers[req.layer_i];
    var slot_i: usize = 0;
    {
        self.stage_mutex.lock();
        defer self.stage_mutex.unlock();
        for (self.stage_meta) |*m| {
            if (m.state != .empty and m.layer_i == req.layer_i and m.eid == req.eid) return;
        }
        var oldest: ?usize = null;
        var found = false;
        for (self.stage_meta, 0..) |*m, i| {
            if (m.state == .empty) {
                slot_i = i;
                found = true;
                break;
            }
            if (m.state == .ready and (oldest == null or m.stamp < self.stage_meta[oldest.?].stamp)) oldest = i;
        }
        if (!found) {
            slot_i = oldest orelse return; // every slot mid-load: drop
            self.stats.staged_wasted += 1;
        }
        self.stage_stamp += 1;
        self.stage_meta[slot_i] = .{ .layer_i = req.layer_i, .eid = req.eid, .state = .loading, .stamp = self.stage_stamp };
    }
    const slot = &self.stage_slots[slot_i];
    var ok = true;
    slot.ensureCapacity(self.allocator, ls.slab_bytes) catch {
        ok = false;
    };
    if (ok) self.readExpert(ls, req.layer_i, req.eid, slot) catch {
        ok = false;
    };
    self.stage_mutex.lock();
    defer self.stage_mutex.unlock();
    if (!ok) {
        self.stage_meta[slot_i].state = .empty;
        return;
    }
    self.stage_meta[slot_i].state = .ready;
    self.stats.staged_loads += 1;
    self.stats.staged_bytes += expertBytes(ls);
}

/// Consume a ready staged expert into `dst` by slab swap (zero copy).
/// False = not staged; the caller does its synchronous read as always.
pub fn stageConsume(self: anytype, layer_i: usize, eid: u32, dst: *Slot) bool {
    if (self.stage_meta.len == 0) return false;
    self.stage_mutex.lock();
    defer self.stage_mutex.unlock();
    for (self.stage_meta, 0..) |*m, i| {
        if (m.state == .ready and m.layer_i == @as(u32, @intCast(layer_i)) and m.eid == eid) {
            const tmp = dst.slab;
            dst.slab = self.stage_slots[i].slab;
            self.stage_slots[i].slab = tmp;
            m.* = .{};
            self.stats.staged_consumed += 1;
            return true;
        }
    }
    return false;
}
