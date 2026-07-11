//! Disk-streamed MoE expert tier: resolves one layer's routed experts to
//! their quantized weight blocks through a pinned set, a per-layer LRU cache
//! in RAM, and `pread` from the GGUF file — so a mixture model whose expert
//! stacks dwarf physical RAM still decodes, paying disk reads only for cache
//! misses. The design follows measured out-of-core MoE lessons: the
//! streamed tier reads with `pread` into store-owned
//! buffers rather than mmap, so resident memory is exactly dense weights +
//! this cache (mmap'd expert pages inflate RSS and let page-cache pressure
//! evict semi-randomly instead of by routing recency); misses issue
//! OS readahead hints (`POSIX_FADV_WILLNEED` / `F_RDADVISE`) for the whole
//! batch before the first synchronous read; and miss loads land in reusable
//! working-set slots that are promoted into the LRU by buffer swap after the
//! layer computes, so the cache capacity is independent of how many experts
//! one batched prefill touches.
//!
//! Concurrency contract: `acquire` locks the store until the matching
//! `release` (one MoE layer op at a time — the model forward is sequential
//! over ops). Between the two, worker threads read resolved expert pointers
//! through `StreamedMoeRhs.expertBytes` without further synchronization:
//! the pointer table is written before the op's tasks are spawned and
//! nothing evicts or swaps slots until `release`.
const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../backend.zig");
const thread = @import("../thread.zig");

const Allocator = std.mem.Allocator;
const qm = backend_mod.quantized_matmul;

pub const Error = error{
    LayerAlreadyRegistered,
    LayerNotRegistered,
    InvalidExpertGeometry,
    StoreNotFinalized,
    ExpertFileOpenFailed,
    ExpertFileReadFailed,
    UsageFileWriteFailed,
    UnexpectedEndOfFile,
} || Allocator.Error;

// ---- platform I/O shims -------------------------------------------------
// The streamed tier wants positional reads and readahead advice without an
// `std.Io` handle (the MoE ops run with `io == null` in production decode).
// Linux goes straight to the syscall layer (no libc requirement); everything
// else uses libc, which is always linked on the Apple targets.

const fd_t = std.posix.fd_t;

fn openReadOnly(allocator: Allocator, path: []const u8) Error!fd_t {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.openat(linux.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(rc) != .SUCCESS) return Error.ExpertFileOpenFailed;
            return @intCast(rc);
        },
        else => {
            const rc = std.c.open(path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
            if (rc < 0) return Error.ExpertFileOpenFailed;
            return rc;
        },
    }
}

fn closeFd(fd: fd_t) void {
    switch (builtin.os.tag) {
        .linux => _ = std.os.linux.close(fd),
        else => _ = std.c.close(fd),
    }
}

/// Monotonic nanoseconds for the read-time stat (the exec ops' clock rides
/// on `std.Io`, which the MoE decode path legitimately runs without).
fn monotonicNanos() ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            var ts: std.os.linux.timespec = undefined;
            if (std.os.linux.clock_gettime(.MONOTONIC, &ts) != 0) return null;
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        else => {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return null;
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
    }
}

/// Short sleep for the pilot thread's empty-ring wait (no std.Thread.sleep
/// in this std; the syscall detour is fine at a 200 µs cadence).
fn sleepMicros(us: u64) void {
    switch (builtin.os.tag) {
        .linux => {
            var req = std.os.linux.timespec{ .sec = @intCast(us / 1_000_000), .nsec = @intCast((us % 1_000_000) * 1000) };
            _ = std.os.linux.nanosleep(&req, null);
        },
        else => {
            const c = struct {
                extern "c" fn nanosleep(req: *const std.c.timespec, rem: ?*std.c.timespec) c_int;
            };
            var req = std.c.timespec{ .sec = @intCast(us / 1_000_000), .nsec = @intCast((us % 1_000_000) * 1000) };
            _ = c.nanosleep(&req, null);
        },
    }
}

fn preadOnce(fd: fd_t, buf: []u8, offset: u64) Error!usize {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.pread(fd, buf.ptr, buf.len, @intCast(offset));
            if (linux.errno(rc) != .SUCCESS) return Error.ExpertFileReadFailed;
            return rc;
        },
        else => {
            const rc = std.c.pread(fd, buf.ptr, buf.len, @intCast(offset));
            if (rc < 0) return Error.ExpertFileReadFailed;
            return @intCast(rc);
        },
    }
}

fn openWriteTrunc(allocator: Allocator, path: []const u8) Error!fd_t {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.openat(linux.AT.FDCWD, path_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, 0o644);
            if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
            return @intCast(rc);
        },
        else => {
            const rc = std.c.open(path_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o644));
            if (rc < 0) return Error.UsageFileWriteFailed;
            return rc;
        },
    }
}

fn writeFull(fd: fd_t, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        switch (builtin.os.tag) {
            .linux => {
                const linux = std.os.linux;
                const rc = linux.write(fd, bytes.ptr + done, bytes.len - done);
                if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
                done += rc;
            },
            else => {
                const rc = std.c.write(fd, bytes.ptr + done, bytes.len - done);
                if (rc < 0) return Error.UsageFileWriteFailed;
                done += @intCast(rc);
            },
        }
    }
}

fn renamePath(allocator: Allocator, old: []const u8, new: []const u8) Error!void {
    const old_z = try allocator.dupeZ(u8, old);
    defer allocator.free(old_z);
    const new_z = try allocator.dupeZ(u8, new);
    defer allocator.free(new_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.renameat(linux.AT.FDCWD, old_z, linux.AT.FDCWD, new_z);
            if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
        },
        else => {
            if (std.c.rename(old_z, new_z) != 0) return Error.UsageFileWriteFailed;
        },
    }
}

/// Whole small file into an allocated buffer, or null on any failure. Capped
/// (histograms are tens of KB) so a bogus path can't balloon memory.
fn readWholeFile(allocator: Allocator, path: []const u8) ?[]u8 {
    const max_bytes = 16 << 20;
    const fd = openReadOnly(allocator, path) catch return null;
    defer closeFd(fd);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(allocator);
    var chunk: [65536]u8 = undefined;
    while (bytes.items.len <= max_bytes) {
        const n = preadOnce(fd, &chunk, bytes.items.len) catch return null;
        if (n == 0) return bytes.toOwnedSlice(allocator) catch null;
        bytes.appendSlice(allocator, chunk[0..n]) catch return null;
    }
    return null;
}

/// Quantized formats an expert stack may stream in — the K-quant family
/// every real MoE GGUF uses (matching `MoeRhs`'s resident arms).
pub const StreamedQuant = enum {
    q4_k,
    q5_k,
    q6_k,
    q8_0,

    pub fn blockSize(self: StreamedQuant) usize {
        return switch (self) {
            .q4_k => @sizeOf(qm.BlockQ4_K),
            .q5_k => @sizeOf(qm.BlockQ5_K),
            .q6_k => @sizeOf(qm.BlockQ6_K),
            .q8_0 => @sizeOf(qm.BlockQ8_0),
        };
    }

    /// Weight blocks per row for a row of `in_dim` inputs.
    pub fn blocksPerColumn(self: StreamedQuant, in_dim: usize) Error!usize {
        switch (self) {
            .q8_0 => {
                if (in_dim == 0 or in_dim % 32 != 0) return Error.InvalidExpertGeometry;
                return in_dim / 32;
            },
            else => return qm.qkBlockCount(in_dim) catch Error.InvalidExpertGeometry,
        }
    }
};

/// The three projections of one MoE FFN layer, in slab order.
pub const Proj = enum(u2) { gate = 0, up = 1, down = 2 };

/// One projection's stacked expert tensor as it sits in the GGUF file:
/// expert-major contiguous, so expert `e` occupies the byte range
/// `[file_offset + e*expert_bytes, +expert_bytes)`.
pub const ProjSpec = struct {
    quant: StreamedQuant,
    /// Which split part (file) holds the tensor; 0 for single-file GGUFs.
    part: u16 = 0,
    /// Absolute offset of the tensor's data within its part on disk
    /// (`gguf.File.partDataOffset(part) + TensorInfo.offset`).
    file_offset: u64,
    /// Total tensor bytes (validated against `n_expert * expert_bytes`).
    byte_len: usize,
    in_dim: usize,
    out_dim: usize,
};

const ProjGeometry = struct {
    quant: StreamedQuant,
    part: u16,
    file_offset: u64,
    in_dim: usize,
    out_dim: usize,
    blocks_per_column: usize,
    expert_bytes: usize,

    fn init(spec: ProjSpec, n_expert: usize) Error!ProjGeometry {
        const bpc = try spec.quant.blocksPerColumn(spec.in_dim);
        const row_bytes = std.math.mul(usize, bpc, spec.quant.blockSize()) catch return Error.InvalidExpertGeometry;
        const expert_bytes = std.math.mul(usize, spec.out_dim, row_bytes) catch return Error.InvalidExpertGeometry;
        const total = std.math.mul(usize, expert_bytes, n_expert) catch return Error.InvalidExpertGeometry;
        if (total != spec.byte_len or expert_bytes == 0) return Error.InvalidExpertGeometry;
        return .{
            .quant = spec.quant,
            .part = spec.part,
            .file_offset = spec.file_offset,
            .in_dim = spec.in_dim,
            .out_dim = spec.out_dim,
            .blocks_per_column = bpc,
            .expert_bytes = expert_bytes,
        };
    }

    fn expertFileOffset(self: *const ProjGeometry, eid: usize) u64 {
        return self.file_offset + @as(u64, eid) * self.expert_bytes;
    }
};

/// Block slabs start 64-byte aligned so the widest kernel loads stay aligned
/// regardless of the K-quant block's own (2-byte) alignment.
const slab_align = 64;
const invalid_eid = std.math.maxInt(u32);

/// One readahead range for the pilot's I/O thread (SPSC ring entry).
const PilotRange = struct { offset: u64, len: u32, part: u16 };
const pilot_ring_cap = 4096;

/// One cached expert: a single slab holding its gate+up+down blocks (loaded
/// with one coalesced logical fetch), stamped for
/// LRU. Slots inside a layer all share that layer's slab size; the shared
/// working-set slots are re-checked per use because layers may differ.
const Slot = struct {
    eid: u32 = invalid_eid,
    used: u64 = 0,
    slab: []align(slab_align) u8 = &.{},

    fn ensureCapacity(self: *Slot, allocator: Allocator, bytes: usize) !void {
        if (self.slab.len >= bytes) return;
        if (self.slab.len > 0) allocator.free(self.slab);
        self.slab = &.{};
        self.slab = try allocator.alignedAlloc(u8, comptime std.mem.Alignment.fromByteUnits(slab_align), bytes);
    }

    fn deinit(self: *Slot, allocator: Allocator) void {
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

    fn expertSlabOffsets(projs: [3]ProjGeometry) struct { off: [3]usize, total: usize } {
        var off: [3]usize = undefined;
        var at: usize = 0;
        for (projs, 0..) |g, p| {
            at = std.mem.alignForward(usize, at, slab_align);
            off[p] = at;
            at += g.expert_bytes;
        }
        return .{ .off = off, .total = std.mem.alignForward(usize, at, slab_align) };
    }

    fn isPinned(self: *const LayerState, eid: u32) bool {
        for (self.pinned) |*slot| {
            if (slot.eid == eid) return true;
        }
        return false;
    }
};

/// Pick one pinned slot to replace from recent routing heat: the coldest
/// pinned expert vs the hottest unpinned one. The fixed +4 margin handles
/// tiny samples; the 25% margin prevents ping-pong (tier hysteresis).
fn tierPickSwap(ls: *const LayerState) ?struct { slot: usize, eid: u32 } {
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

fn hottestUnpinned(ls: *const LayerState) ?u32 {
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

/// Best-effort async readahead hint for a file range: tells the kernel to
/// start reading in the background so the synchronous `pread` that follows
/// (or follows several other hinted reads) finds the page cache warm. A
/// failed hint is not an error — it only costs the overlap.
fn hintWillNeed(fd: fd_t, offset: u64, len: usize) void {
    switch (builtin.os.tag) {
        .linux => {
            _ = std.os.linux.fadvise(fd, @intCast(offset), @intCast(len), std.os.linux.POSIX_FADV.WILLNEED);
        },
        .macos, .ios => {
            // struct radvisory + F_RDADVISE (fcntl.h): macOS's readahead advice.
            const radvisory = extern struct { ra_offset: i64, ra_count: c_int };
            var adv = radvisory{
                .ra_offset = @intCast(offset),
                .ra_count = @intCast(@min(len, std.math.maxInt(c_int))),
            };
            _ = std.c.fcntl(fd, 44, @intFromPtr(&adv));
        },
        else => {},
    }
}

/// Currently-available physical memory, best effort: Linux reads
/// `MemAvailable` (free + reclaimable page cache); macOS counts only the
/// free-page pool, which understates what the OS could reclaim — acceptable
/// for a conservative default budget. `null` when undeterminable.
pub fn memAvailableBytes() ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const open_rc = linux.openat(linux.AT.FDCWD, "/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(open_rc) != .SUCCESS) return null;
            const fd: fd_t = @intCast(open_rc);
            defer closeFd(fd);
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
            var pages: c_int = 0;
            var page_size: c_int = 0;
            var len: usize = @sizeOf(c_int);
            if (std.c.sysctlbyname("vm.page_free_count", &pages, &len, null, 0) != 0) return null;
            len = @sizeOf(c_int);
            if (std.c.sysctlbyname("hw.pagesize", &page_size, &len, null, 0) != 0) return null;
            if (pages < 0 or page_size <= 0) return null;
            return @as(u64, @intCast(pages)) * @as(u64, @intCast(page_size));
        },
        else => return null,
    }
}

pub const ExpertStore = struct {
    pub const Options = struct {
        /// Fixed LRU slots per layer; wins over `cache_bytes` when set.
        cache_slots_per_layer: ?usize = null,
        /// Total RAM budget for the streamed tiers (pinned + LRU) across all
        /// layers. Default: half of the available memory at finalize time
        /// (8 GiB fallback).
        cache_bytes: ?usize = null,
        /// Issue WILLNEED-style readahead for the whole miss set before the
        /// first synchronous read.
        readahead: bool = true,
        /// The learning cache: when the persisted usage histogram (sidecar
        /// `<gguf>.experts` file) carries enough history, pin the hottest
        /// experts in RAM at finalize — they are read once at startup and
        /// never evicted. The engine gets faster the more it is used.
        auto_pin: bool = true,
        /// RAM for the pinned tier; default: half of the total budget when
        /// history qualifies (the LRU gets the remainder).
        pin_bytes: ?usize = null,
        /// Minimum recorded routed pairs before auto-pin trusts the history.
        auto_pin_min_history: u64 = 5000,
    };

    pub const Stats = struct {
        hits: u64 = 0,
        pin_hits: u64 = 0,
        misses: u64 = 0,
        bytes_read: u64 = 0,
        read_ns: u64 = 0,
        acquires: u64 = 0,
        /// Pilot (router lookahead): ranges enqueued to the I/O thread, and
        /// prediction recall over the acquires that followed a prediction.
        pilot_ranges: u64 = 0,
        pilot_recall_hits: u64 = 0,
        pilot_recall_total: u64 = 0,

        pub fn hitRate(self: Stats) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }

        pub fn pilotRecall(self: Stats) f64 {
            if (self.pilot_recall_total == 0) return 0;
            return @as(f64, @floatFromInt(self.pilot_recall_hits)) / @as(f64, @floatFromInt(self.pilot_recall_total));
        }
    };

    allocator: Allocator,
    /// One read fd per split part (single-file GGUFs: one entry).
    fds: []fd_t,
    layers: []LayerState,
    registered: []bool,
    options: Options,
    /// Sidecar path of the persistent usage histogram (`<gguf>.experts`).
    usage_path: []u8,
    /// LRU slots per layer, fixed at `finalize`.
    cap: usize = 0,
    /// Pinned-tier summary, fixed at `finalize` (adapted by `repinPass`).
    pinned_experts: usize = 0,
    pinned_bytes: usize = 0,
    finalized: bool = false,
    clock: u64 = 0,
    stats: Stats = .{},
    mutex: thread.Mutex = .{},
    // ---- per-acquire state (valid while the mutex is held) ----
    acquired_layer: ?usize = null,
    /// Unique experts of the current acquire / their miss subset (parallel
    /// to `work[0..n_miss]`).
    active: []u32 = &.{},
    n_active: usize = 0,
    miss_eids: []u32 = &.{},
    n_miss: usize = 0,
    /// Working-set slots for the current acquire's misses; promoted into the
    /// layer LRU (by slab swap) at release. Grown to the largest miss set.
    work: []Slot = &.{},
    seen: []bool = &.{},
    // ---- pilot (router-lookahead prefetch) ----
    // A dedicated I/O thread drains an SPSC ring of file ranges and issues
    // the readahead advice there: with a saturated disk queue the advice
    // call itself BLOCKS (measured ~0.5 ms each upstream), so hinting inline
    // would cost the forward thread more than the overlap earns. Ring full
    // = drop: a lost hint is not an error.
    pilot_ring: []PilotRange = &.{},
    pilot_w: std.atomic.Value(u32) = .init(0),
    pilot_r: std.atomic.Value(u32) = .init(0),
    pilot_stop: std.atomic.Value(bool) = .init(false),
    pilot_thread: ?std.Thread = null,

    /// The store is heap-allocated so `StreamedMoeRhs` values and the owning
    /// model can hold stable pointers while the model struct moves by value.
    /// `gguf_paths` lists every part of a split GGUF (single-file: one
    /// entry); `ProjSpec.part` indexes into it.
    pub fn create(allocator: Allocator, gguf_paths: []const []const u8, n_layers: usize, options: Options) Error!*ExpertStore {
        std.debug.assert(gguf_paths.len > 0);
        const fds = try allocator.alloc(fd_t, gguf_paths.len);
        errdefer allocator.free(fds);
        var n_open: usize = 0;
        errdefer for (fds[0..n_open]) |fd| closeFd(fd);
        for (gguf_paths) |path| {
            fds[n_open] = try openReadOnly(allocator, path);
            n_open += 1;
        }

        const self = try allocator.create(ExpertStore);
        errdefer allocator.destroy(self);
        const layers = try allocator.alloc(LayerState, n_layers);
        errdefer allocator.free(layers);
        const registered = try allocator.alloc(bool, n_layers);
        errdefer allocator.free(registered);
        @memset(registered, false);
        const usage_path = try std.fmt.allocPrint(allocator, "{s}.experts", .{gguf_paths[0]});

        self.* = .{
            .allocator = allocator,
            .fds = fds,
            .layers = layers,
            .registered = registered,
            .options = options,
            .usage_path = usage_path,
        };
        return self;
    }

    pub fn destroy(self: *ExpertStore) void {
        const allocator = self.allocator;
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            for (ls.slots) |*slot| slot.deinit(allocator);
            allocator.free(ls.slots);
            for (ls.pinned) |*slot| slot.deinit(allocator);
            allocator.free(ls.pinned);
            allocator.free(ls.resolved);
            allocator.free(ls.usage);
            allocator.free(ls.heat);
            allocator.free(ls.pred_marks);
        }
        if (self.pilot_thread) |t| {
            self.pilot_stop.store(true, .release);
            t.join();
        }
        allocator.free(self.pilot_ring);
        allocator.free(self.usage_path);
        for (self.work) |*slot| slot.deinit(allocator);
        allocator.free(self.work);
        allocator.free(self.active);
        allocator.free(self.miss_eids);
        allocator.free(self.seen);
        allocator.free(self.layers);
        allocator.free(self.registered);
        for (self.fds) |fd| closeFd(fd);
        allocator.free(self.fds);
        allocator.destroy(self);
    }

    /// Register one MoE layer's three stacked expert tensors. Call once per
    /// layer during model load, then `finalize`.
    pub fn addLayer(self: *ExpertStore, layer_i: usize, specs: [3]ProjSpec, n_expert: usize) Error!void {
        if (layer_i >= self.layers.len) return Error.InvalidExpertGeometry;
        if (self.registered[layer_i]) return Error.LayerAlreadyRegistered;
        if (n_expert == 0 or n_expert >= invalid_eid) return Error.InvalidExpertGeometry;

        var projs: [3]ProjGeometry = undefined;
        for (specs, 0..) |spec, p| {
            if (spec.part >= self.fds.len) return Error.InvalidExpertGeometry;
            projs[p] = try ProjGeometry.init(spec, n_expert);
        }
        const layout = LayerState.expertSlabOffsets(projs);

        const resolved = try self.allocator.alloc([3]?[*]const u8, n_expert);
        errdefer self.allocator.free(resolved);
        @memset(resolved, .{ null, null, null });
        const usage = try self.allocator.alloc(u64, n_expert);
        errdefer self.allocator.free(usage);
        @memset(usage, 0);
        const heat = try self.allocator.alloc(u32, n_expert);
        @memset(heat, 0);
        const pred_marks = try self.allocator.alloc(u32, n_expert);
        @memset(pred_marks, 0);

        self.layers[layer_i] = .{
            .projs = projs,
            .n_expert = n_expert,
            .proj_off = layout.off,
            .slab_bytes = layout.total,
            .resolved = resolved,
            .usage = usage,
            .heat = heat,
            .pred_marks = pred_marks,
        };
        self.registered[layer_i] = true;
    }

    /// Fix the tier layout from the configured budget: load the persisted
    /// usage history, carve the pinned tier out of the budget when the
    /// history qualifies (auto-pin — the learning cache), give the LRU the
    /// remainder, and allocate the bookkeeping sized to the largest
    /// registered layer. LRU slot slabs are allocated lazily on first
    /// promotion; pinned slabs are read from disk here, once.
    pub fn finalize(self: *ExpertStore) Error!void {
        var total_slab: usize = 0;
        var max_expert: usize = 0;
        var n_registered: usize = 0;
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            n_registered += 1;
            total_slab += ls.slab_bytes;
            max_expert = @max(max_expert, ls.n_expert);
        }
        if (n_registered == 0) return Error.LayerNotRegistered;

        const history_pairs = self.loadUsage();

        var budget: usize = self.options.cache_bytes orelse blk: {
            const avail = memAvailableBytes() orelse (8 << 30);
            break :blk @max(avail / 2, 512 << 20);
        };
        if (self.options.auto_pin and history_pairs >= self.options.auto_pin_min_history) {
            const pin_budget = self.options.pin_bytes orelse budget / 2;
            const spent = try self.selectAndLoadPins(@min(pin_budget, budget));
            budget -= @min(spent, budget);
        }

        var cap: usize = undefined;
        if (self.options.cache_slots_per_layer) |slots| {
            cap = @max(1, slots);
        } else {
            cap = @max(1, budget / total_slab);
        }
        cap = @min(cap, max_expert);
        self.cap = cap;

        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            const layer_cap = @min(cap, ls.n_expert);
            ls.slots = try self.allocator.alloc(Slot, layer_cap);
            @memset(ls.slots, .{});
        }
        self.active = try self.allocator.alloc(u32, max_expert);
        self.miss_eids = try self.allocator.alloc(u32, max_expert);
        self.work = try self.allocator.alloc(Slot, max_expert);
        @memset(self.work, .{});
        self.seen = try self.allocator.alloc(bool, max_expert);
        self.finalized = true;
    }

    /// Greedy hottest-first pin selection over every registered layer's
    /// usage histogram, then one sequential read pass to load the picks.
    /// Returns the bytes spent. Experts with zero recorded usage are never
    /// pinned.
    fn selectAndLoadPins(self: *ExpertStore, pin_budget: usize) Error!usize {
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
                for (&ls.projs) |*g| hintWillNeed(self.fds[g.part], g.expertFileOffset(cand.eid), g.expert_bytes);
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
            try self.readExpert(ls, cand.eid, slot);
            slot.eid = cand.eid;
        }
        self.pinned_experts = picks.items.len;
        self.pinned_bytes = spent;
        return spent;
    }

    /// Total bytes one layer's LRU tier may hold; times registered layers =
    /// the tier's peak footprint (plus one working set).
    pub fn perLayerCacheBytes(self: *const ExpertStore) usize {
        var max_bytes: usize = 0;
        for (self.layers, self.registered) |*ls, reg| {
            if (reg) max_bytes = @max(max_bytes, ls.slots.len * ls.slab_bytes);
        }
        return max_bytes;
    }

    /// A `MoeRhs.streamed` arm for one registered projection.
    pub fn streamedRhs(self: *ExpertStore, layer_i: usize, proj: Proj) StreamedMoeRhs {
        std.debug.assert(self.registered[layer_i]);
        const ls = &self.layers[layer_i];
        const g = &ls.projs[@intFromEnum(proj)];
        return .{
            .store = self,
            .layer_state = ls,
            .layer = layer_i,
            .proj = proj,
            .quant = g.quant,
            .k = g.in_dim,
            .out_dim = g.out_dim,
            .n_expert = ls.n_expert,
            .blocks_per_column = g.blocks_per_column,
        };
    }

    /// Resolve every expert in `selected` (dupes fine — it is the raw routed
    /// pair list) for one layer op: pinned/cached experts resolve to their
    /// slabs; misses are read from disk into working-set slots, readahead
    /// hints for the whole miss set going out before the first synchronous
    /// read. Locks the store until `release`.
    pub fn acquire(self: *ExpertStore, layer_i: usize, selected: []const usize) Error!void {
        self.mutex.lock();
        errdefer self.mutex.unlock();
        std.debug.assert(self.acquired_layer == null);
        if (!self.finalized) return Error.StoreNotFinalized;
        if (layer_i >= self.layers.len or !self.registered[layer_i]) return Error.LayerNotRegistered;
        const ls = &self.layers[layer_i];

        // Unique active set + usage/heat histograms (every routed pair
        // counts; `usage` persists across sessions, `heat` decays per
        // repin pass).
        @memset(self.seen[0..ls.n_expert], false);
        self.n_active = 0;
        for (selected) |e| {
            std.debug.assert(e < ls.n_expert);
            ls.usage[e] += 1;
            ls.heat[e] +|= 1;
            if (self.seen[e]) continue;
            self.seen[e] = true;
            self.active[self.n_active] = @intCast(e);
            self.n_active += 1;
        }

        // Score the pilot's prediction for this layer, once per prediction:
        // recall = predicted ∩ routed over routed (the measurement that says
        // whether router lookahead is worth its prefetch bandwidth).
        if (ls.pred_epoch != ls.pred_scored) {
            ls.pred_scored = ls.pred_epoch;
            for (self.active[0..self.n_active]) |eid| {
                if (ls.pred_marks[eid] == ls.pred_epoch) self.stats.pilot_recall_hits += 1;
            }
            self.stats.pilot_recall_total += self.n_active;
        }

        // Pinned and LRU hits resolve in place (pin first — a pinned expert
        // may transiently also sit in the LRU after a repin); misses collect.
        self.n_miss = 0;
        for (self.active[0..self.n_active]) |eid| {
            if (findPinned(ls, eid)) |slot| {
                self.resolveSlot(ls, eid, slot);
                self.stats.hits += 1;
                self.stats.pin_hits += 1;
            } else if (self.findCached(ls, eid)) |slot| {
                self.clock += 1;
                slot.used = self.clock;
                self.resolveSlot(ls, eid, slot);
                self.stats.hits += 1;
            } else {
                self.miss_eids[self.n_miss] = eid;
                self.n_miss += 1;
                self.stats.misses += 1;
            }
        }

        // Hint the whole miss set first: the kernel reads ahead while we
        // pread the earlier misses.
        if (self.options.readahead and self.n_miss > 1) {
            for (self.miss_eids[0..self.n_miss]) |eid| {
                for (&ls.projs) |*g| hintWillNeed(self.fds[g.part], g.expertFileOffset(eid), g.expert_bytes);
            }
        }

        const read_start = if (self.n_miss > 0) monotonicNanos() else null;
        for (self.miss_eids[0..self.n_miss], 0..) |eid, w| {
            const slot = &self.work[w];
            try slot.ensureCapacity(self.allocator, ls.slab_bytes);
            try self.readExpert(ls, eid, slot);
            slot.eid = eid;
            self.resolveSlot(ls, eid, slot);
        }
        if (read_start) |t0| {
            if (monotonicNanos()) |t1| self.stats.read_ns += t1 -| t0;
        }

        self.stats.acquires += 1;
        self.acquired_layer = layer_i;
    }

    /// End the layer op: promote this acquire's misses into the layer LRU
    /// (slab swap — no copy; the displaced victim slab becomes the working
    /// buffer), clear the resolved pointers, unlock.
    pub fn release(self: *ExpertStore, layer_i: usize) void {
        std.debug.assert(self.acquired_layer == layer_i);
        const ls = &self.layers[layer_i];

        const cap = ls.slots.len;
        const promo = @min(self.n_miss, cap);
        for (0..promo) |a| {
            const w = self.n_miss - 1 - a;
            const src = &self.work[w];
            var dst: *Slot = undefined;
            if (ls.n_slots < cap) {
                dst = &ls.slots[ls.n_slots];
                ls.n_slots += 1;
            } else {
                dst = &ls.slots[0];
                for (ls.slots[1..]) |*s| {
                    if (s.used < dst.used) dst = s;
                }
            }
            std.mem.swap([]align(slab_align) u8, &dst.slab, &src.slab);
            dst.eid = src.eid;
            src.eid = invalid_eid;
            self.clock += 1;
            dst.used = self.clock;
            // The promoted slab keeps resolving for this expert; pointers are
            // cleared below anyway.
        }

        for (self.active[0..self.n_active]) |eid| ls.resolved[eid] = .{ null, null, null };
        self.n_active = 0;
        self.n_miss = 0;
        self.acquired_layer = null;
        self.mutex.unlock();
    }

    fn findCached(self: *ExpertStore, ls: *LayerState, eid: u32) ?*Slot {
        _ = self;
        for (ls.slots[0..ls.n_slots]) |*slot| {
            if (slot.eid == eid) return slot;
        }
        return null;
    }

    fn findPinned(ls: *LayerState, eid: u32) ?*Slot {
        for (ls.pinned) |*slot| {
            if (slot.eid == eid) return slot;
        }
        return null;
    }

    fn resolveSlot(self: *ExpertStore, ls: *LayerState, eid: u32, slot: *Slot) void {
        _ = self;
        for (0..3) |p| ls.resolved[eid][p] = @ptrCast(slot.slab.ptr + ls.proj_off[p]);
    }

    /// One expert's gate+up+down blocks into `slot.slab` — three `pread`s
    /// (the projections are separate GGUF tensors, so they are not adjacent
    /// on disk; coalescing would need a converter-ordered container).
    fn readExpert(self: *ExpertStore, ls: *LayerState, eid: u32, slot: *Slot) Error!void {
        for (&ls.projs, 0..) |*g, p| {
            const dst = slot.slab[ls.proj_off[p]..][0..g.expert_bytes];
            try self.preadFull(g.part, dst, g.expertFileOffset(eid));
            self.stats.bytes_read += g.expert_bytes;
        }
    }

    fn preadFull(self: *ExpertStore, part: u16, buf: []u8, offset: u64) Error!void {
        var done: usize = 0;
        while (done < buf.len) {
            const n = try preadOnce(self.fds[part], buf[done..], offset + done);
            if (n == 0) return Error.UnexpectedEndOfFile;
            done += n;
        }
    }

    // ---- pilot: router-lookahead prefetch ----------------------------------

    /// Predicted routing for `layer_i`'s NEXT acquire (router lookahead:
    /// apply layer L+1's router to layer L's post-attention state — measured
    /// 87.6-90.5% top-8 recall on the Qwen3 MoEs vs ~41% for "same as last
    /// token" upstream). Marks the prediction for recall scoring and enqueues
    /// readahead for the experts not already pinned or cached; the dedicated
    /// I/O thread issues the actual advice. Call between ops on the forward
    /// thread — never between `acquire` and `release`.
    pub fn pilotHint(self: *ExpertStore, layer_i: usize, eids: []const usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(self.acquired_layer == null);
        if (!self.finalized or layer_i >= self.layers.len or !self.registered[layer_i]) return;
        const ls = &self.layers[layer_i];

        if (self.pilot_thread == null) self.pilotStart() catch return;

        ls.pred_epoch +%= 1;
        if (ls.pred_epoch == ls.pred_scored) ls.pred_epoch +%= 1; // skip the ambiguous wrap value
        for (eids) |e| {
            if (e >= ls.n_expert) continue;
            ls.pred_marks[e] = ls.pred_epoch;
            const eid: u32 = @intCast(e);
            if (findPinned(ls, eid) != null) continue;
            if (self.findCached(ls, eid) != null) continue;
            for (&ls.projs) |*g| self.pilotEnqueue(.{ .offset = g.expertFileOffset(eid), .len = @intCast(@min(g.expert_bytes, std.math.maxInt(u32))), .part = g.part });
        }
    }

    fn pilotStart(self: *ExpertStore) !void {
        if (self.pilot_ring.len == 0) {
            self.pilot_ring = try self.allocator.alloc(PilotRange, pilot_ring_cap);
        }
        self.pilot_thread = try std.Thread.spawn(.{}, pilotWorker, .{self});
    }

    fn pilotEnqueue(self: *ExpertStore, range: PilotRange) void {
        const w = self.pilot_w.load(.monotonic);
        const r = self.pilot_r.load(.acquire);
        if (w -% r >= pilot_ring_cap) return; // full: drop the hint
        self.pilot_ring[w % pilot_ring_cap] = range;
        self.pilot_w.store(w +% 1, .release);
        self.stats.pilot_ranges += 1;
    }

    fn pilotWorker(self: *ExpertStore) void {
        while (!self.pilot_stop.load(.acquire)) {
            const r = self.pilot_r.load(.monotonic);
            const w = self.pilot_w.load(.acquire);
            if (r == w) {
                sleepMicros(200);
                continue;
            }
            const range = self.pilot_ring[r % pilot_ring_cap];
            self.pilot_r.store(r +% 1, .release);
            hintWillNeed(self.fds[range.part], range.offset, range.len);
        }
    }

    // ---- the learning cache: persistent usage histogram --------------------

    const usage_magic = "FUCEXPT1";

    /// Persist the usage histogram to the sidecar (`<gguf>.experts`),
    /// atomically (tmp + rename): counts accumulate across sessions, and at
    /// the next startup auto-pin turns them into a pinned hot tier. Call at
    /// end of generation / turn boundaries; a failure loses only learning.
    pub fn saveUsage(self: *ExpertStore) Error!void {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(self.allocator);
        try bytes.appendSlice(self.allocator, usage_magic);
        try appendInt(u32, self.allocator, &bytes, @intCast(self.layers.len));
        for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
            if (!reg) continue;
            try appendInt(u32, self.allocator, &bytes, @intCast(layer_i));
            try appendInt(u32, self.allocator, &bytes, @intCast(ls.n_expert));
            for (ls.usage) |count| try appendInt(u64, self.allocator, &bytes, count);
        }

        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{self.usage_path});
        defer self.allocator.free(tmp_path);
        {
            const fd = try openWriteTrunc(self.allocator, tmp_path);
            defer closeFd(fd);
            try writeFull(fd, bytes.items);
        }
        try renamePath(self.allocator, tmp_path, self.usage_path);
    }

    /// Merge the sidecar histogram into the in-memory counts (add, so a
    /// session's own routing keeps accumulating on top). Any mismatch —
    /// missing file, other model's geometry, torn write — ignores the file
    /// wholesale. Returns the total merged pair count.
    fn loadUsage(self: *ExpertStore) u64 {
        const bytes = readWholeFile(self.allocator, self.usage_path) orelse return 0;
        defer self.allocator.free(bytes);
        var r = UsageReader{ .bytes = bytes };
        if (!std.mem.eql(u8, r.take(usage_magic.len) orelse return 0, usage_magic)) return 0;
        const n_layers = r.int(u32) orelse return 0;
        if (n_layers != self.layers.len) return 0;

        // Validate the whole record set against the registered geometry
        // before merging anything: an invalid tail must not half-apply.
        var check = r;
        for (self.layers, self.registered, 0..) |*ls, reg, layer_i| {
            if (!reg) continue;
            if ((check.int(u32) orelse return 0) != layer_i) return 0;
            if ((check.int(u32) orelse return 0) != ls.n_expert) return 0;
            if (check.take(ls.n_expert * @sizeOf(u64)) == null) return 0;
        }
        if (check.bytes.len != check.at) return 0;

        var total: u64 = 0;
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            _ = r.int(u32);
            _ = r.int(u32);
            for (ls.usage) |*count| {
                const stored = r.int(u64) orelse unreachable;
                count.* +|= stored;
                total +|= stored;
            }
        }
        return total;
    }

    const UsageReader = struct {
        bytes: []const u8,
        at: usize = 0,

        fn take(self: *UsageReader, n: usize) ?[]const u8 {
            if (self.at + n > self.bytes.len) return null;
            defer self.at += n;
            return self.bytes[self.at..][0..n];
        }

        fn int(self: *UsageReader, comptime T: type) ?T {
            const raw = self.take(@sizeOf(T)) orelse return null;
            return std.mem.readInt(T, raw[0..@sizeOf(T)], .little);
        }
    };

    fn appendInt(comptime T: type, allocator: Allocator, bytes: *std.ArrayList(u8), value: T) Error!void {
        var raw: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &raw, value, .little);
        try bytes.appendSlice(allocator, &raw);
    }

    // ---- live tier adaptation ----------------------------------------------

    /// One adaptation pass over the pinned tier: per layer, replace up to
    /// `max_swaps_per_layer` of the coldest pinned experts with hotter
    /// streamed ones (25% + fixed hysteresis — see `tierPickSwap`), reading
    /// the replacement into the existing pinned slab; then halve the heat so
    /// the signal follows the current workload. Call at safe boundaries
    /// (between generations / chat turns), never inside an acquire. Returns
    /// the number of swaps performed.
    pub fn repinPass(self: *ExpertStore, max_swaps_per_layer: usize) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        std.debug.assert(self.acquired_layer == null);
        if (!self.finalized) return 0;

        var swaps: usize = 0;
        for (self.layers, self.registered) |*ls, reg| {
            if (!reg) continue;
            if (ls.pinned.len > 0) {
                var done: usize = 0;
                while (done < max_swaps_per_layer) : (done += 1) {
                    const pick = tierPickSwap(ls) orelse break;
                    const slot = &ls.pinned[pick.slot];
                    // Invalidate first: a failed read leaves the slot skipped
                    // by lookups instead of resolving to stale bytes.
                    slot.eid = invalid_eid;
                    slot.ensureCapacity(self.allocator, ls.slab_bytes) catch break;
                    self.readExpert(ls, pick.eid, slot) catch break;
                    slot.eid = pick.eid;
                    swaps += 1;
                }
            }
            for (ls.heat) |*h| h.* >>= 1;
        }
        return swaps;
    }
};

/// The streamed counterpart of a resident stacked-expert RHS: same geometry,
/// but expert blocks resolve through the store's acquire-scoped pointer
/// table instead of a slice into one big buffer.
pub const StreamedMoeRhs = struct {
    store: *ExpertStore,
    layer_state: *LayerState,
    layer: usize,
    proj: Proj,
    quant: StreamedQuant,
    k: usize,
    out_dim: usize,
    n_expert: usize,
    blocks_per_column: usize,

    /// Virtual stacked row count, mirroring the resident arms' `n`.
    pub fn rows(self: *const StreamedMoeRhs) usize {
        return self.n_expert * self.out_dim;
    }

    /// Expert `e`'s blocks for this projection. Only valid between the
    /// store's `acquire` (which resolved `e`) and `release`.
    pub inline fn expertBytes(self: *const StreamedMoeRhs, e: usize) [*]const u8 {
        const ptr = self.layer_state.resolved[e][@intFromEnum(self.proj)];
        if (ptr) |p| return p;
        @panic("streamed MoE expert used without acquire");
    }
};

test {
    _ = @import("expert_store_tests.zig");
}
