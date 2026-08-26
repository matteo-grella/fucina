//! The ES `Trainer`: slot registration (facade tensors, registries, ternary
//! genomes), the seed/stream derivations (checkpoint contracts), perturb/
//! restore/materialize, the ES update with reward shaping, the ternary
//! vote-and-threshold update, anchored weight decay, and the step/
//! evaluateMembers drivers. Re-exported by `es.zig`; algorithm and numerics
//! are documented there.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const dtype_mod = @import("../dtype.zig");
const exec_mod = @import("../exec.zig");
const param_registry = @import("../param_registry.zig");
const rng = @import("../rng.zig");
const thread = @import("../thread.zig");
const common = @import("common.zig");
const kernels = @import("kernels.zig");
const slots_mod = @import("slots.zig");
const ternary_mod = @import("ternary.zig");

const Allocator = std.mem.Allocator;
const ExecContext = exec_mod.ExecContext;

const BlockTQ2_0 = common.BlockTQ2_0;
const Config = common.Config;
const EsError = common.EsError;
const Stats = common.Stats;
const seed_domain = common.seed_domain;
const noise_domain = common.noise_domain;
const ternary_domain = common.ternary_domain;

const CacheRegion = slots_mod.CacheRegion;
const Slot = slots_mod.Slot;
const StreamCache = slots_mod.StreamCache;
const makeRelease = slots_mod.makeRelease;
const validateFacadePtr = slots_mod.validateFacadePtr;

const TernarySlot = ternary_mod.TernarySlot;
const UndoEntry = ternary_mod.UndoEntry;
const VoteEntry = ternary_mod.VoteEntry;
const ternary_block_len = ternary_mod.ternary_block_len;
const ternaryFlipAt = ternary_mod.ternaryFlipAt;
const moveCode = ternary_mod.moveCode;
const crumbGet = ternary_mod.crumbGet;
const crumbSet = ternary_mod.crumbSet;
const voteBefore = ternary_mod.voteBefore;

const rewardStats = kernels.rewardStats;
const perturbSlot = kernels.perturbSlot;
const materializeSlot = kernels.materializeSlot;
const updateSlot = kernels.updateSlot;
const anchorSlot = kernels.anchorSlot;

pub const Trainer = struct {
    allocator: Allocator,
    /// Read freely; replace only through `applyResumedConfig` (a resume),
    /// which re-validates it and re-sizes what depends on it. Field
    /// assignment after `init` bypasses both.
    config: Config,
    slots: std.ArrayList(Slot) = .empty,
    /// Ternary genome slots (`addTernaryParam`) — kept apart from the float
    /// slots so the float noise-stream enumeration (slot index) never shifts
    /// when ternary slots register (checkpoint contract).
    ternary_slots: std.ArrayList(TernarySlot) = .empty,
    /// Advances once per `update`. Member seeds derive from it — persist it
    /// (e.g. in trainer_state.json) to resume the exact population stream.
    iteration: u64 = 0,
    /// In-place tripwire: the member currently applied to the shared
    /// parameters, if any.
    active_member: ?usize = null,
    /// `.snapshot` restore slab (lazy; concatenated slot bytes).
    snapshot: ?[]u8 = null,
    /// Slab freshness: a snapshot-mode restore leaves theta == slab, so
    /// per-member re-copies are skipped until an update dirties it.
    snapshot_valid: bool = false,
    /// Per-iteration noise-stream cache (see `Config.cache_streams`).
    stream_cache: ?StreamCache = null,
    /// The AWD anchor theta_0 (concatenated slot bytes, dtype-preserved).
    /// Captured ONCE by `captureAnchor` and never refreshed. On GPU builds
    /// it lives in resident bytes so the device anchor kernel can read it.
    anchor: ?[]u8 = null,
    anchor_resident: bool = false,
    /// Ternary-update scratch, trainer-owned so steady-state updates reuse
    /// capacity instead of allocating: the per-slot vote accumulator
    /// (cleared between slots — vote isolation) and the staged top-K entry
    /// lists of every slot for one update, concatenated in slot order
    /// (`TernarySlot.pending` entries per slot).
    ternary_votes: std.AutoHashMapUnmanaged(usize, f32) = .empty,
    ternary_entries: std.ArrayList(VoteEntry) = .empty,

    const Self = @This();

    pub fn init(allocator: Allocator, config: Config) !Self {
        try validateConfig(config);
        return .{ .allocator = allocator, .config = config };
    }

    /// The `init` config checks, shared with `applyResumedConfig` so a resumed
    /// configuration can never be weaker-checked than a fresh one.
    fn validateConfig(config: Config) !void {
        if (!(config.sigma > 0) or !std.math.isFinite(config.sigma)) return EsError.InvalidConfig;
        if (config.alpha) |alpha| {
            if (!(alpha > 0) or !std.math.isFinite(alpha)) return EsError.InvalidConfig;
        }
        if (config.population < 2) return EsError.InvalidConfig;
        if (config.antithetic and config.population % 2 != 0) return EsError.InvalidConfig;
        // `!(x > 0)`/`!(x <= 1)` reject NaN alongside the range violations.
        if (!(config.ternary_flip_rate > 0) or !(config.ternary_flip_rate <= 1)) return EsError.InvalidConfig;
        if (!(config.ternary_update_fraction > 0) or !(config.ternary_update_fraction <= 1)) return EsError.InvalidConfig;
        if (!(config.ternary_update_decay >= 0) or !std.math.isFinite(config.ternary_update_decay)) return EsError.InvalidConfig;
        if (config.anchor_decay != .none) {
            if (!(config.anchor_lambda > 0) or !std.math.isFinite(config.anchor_lambda)) return EsError.InvalidConfig;
            const alpha = config.alpha orelse (config.sigma / 2);
            // The l2 shrink factor (1 - alpha*lambda) must stay positive.
            if (config.anchor_decay == .l2 and !(alpha * config.anchor_lambda < 1)) return EsError.InvalidConfig;
        }
    }

    /// Replace the configuration after registration: a checkpoint resume,
    /// where the saved sigma/alpha/population/scheme/anchor/seed win over
    /// the CLI. The only sanctioned way to change `config` once `init` has
    /// returned. Runs the whole `init` validation on the new config (an odd
    /// population under `antithetic` is rejected here exactly as at init),
    /// then re-sizes what the old config sized: every ternary slot's undo
    /// log (one entry per flip, at the new `ternary_flip_rate`) and the
    /// per-iteration noise cache, which is dropped outright so no stream
    /// filled under the old seed, scheme, or stream count survives (it
    /// re-fills lazily). Transactional: on any error the live config and the
    /// slots are untouched. `iteration` is restored by the caller as before.
    /// Rejected with `MemberActive` while a member is applied in place.
    pub fn applyResumedConfig(self: *Self, config: Config) !void {
        try validateConfig(config);
        if (self.active_member != null) return EsError.MemberActive;

        // Phase 1 (fallible): the replacement undo logs, allocated beside
        // the live ones. Only slots whose flip count changes get one.
        const n_ternary = self.ternary_slots.items.len;
        const replacements = try self.allocator.alloc(?[]UndoEntry, n_ternary);
        defer self.allocator.free(replacements);
        @memset(replacements, null);
        errdefer for (replacements) |maybe| {
            if (maybe) |undo| self.allocator.free(undo);
        };
        for (self.ternary_slots.items, replacements) |*slot, *replacement| {
            const flips = flipCountFor(config, slot.len);
            if (flips != slot.undo.len) replacement.* = try self.allocator.alloc(UndoEntry, flips);
        }

        // Phase 2 (infallible): commit.
        self.config = config;
        for (self.ternary_slots.items, replacements) |*slot, maybe| {
            const undo = maybe orelse continue;
            self.allocator.free(slot.undo);
            slot.undo = undo;
            slot.undo_len = 0;
        }
        if (self.stream_cache) |*cache| {
            cache.deinit(self.allocator);
            self.stream_cache = null;
        }
    }

    pub fn deinit(self: *Self) void {
        self.ternary_votes.deinit(self.allocator);
        self.ternary_entries.deinit(self.allocator);
        if (self.anchor) |slab| self.freeAnchor(slab);
        if (self.stream_cache) |*cache| cache.deinit(self.allocator);
        if (self.snapshot) |slab| self.allocator.free(slab);
        for (self.slots.items) |*slot| slot.deinit(self.allocator);
        self.slots.deinit(self.allocator);
        for (self.ternary_slots.items) |*slot| slot.deinit(self.allocator);
        self.ternary_slots.deinit(self.allocator);
        self.* = undefined;
    }

    /// The effective learning rate: `config.alpha`, defaulting to sigma/2.
    pub fn alphaValue(self: *const Self) f32 {
        return self.config.alpha orelse (self.config.sigma / 2);
    }

    /// Register one f32/f16/bf16 facade tensor (autograd variable, constant,
    /// or grad-free typed tensor — ES treats them all the same). Retains a
    /// refcounted storage view; `name` (optional) is borrowed.
    pub fn addParam(self: *Self, t: anytype) !void {
        try self.addParamNamed(t, null);
    }

    pub fn addParamNamed(self: *Self, t: anytype, name: ?[]const u8) !void {
        const T = comptime validateFacadePtr(@TypeOf(t));
        if (!t.value.isContiguous()) return EsError.NonContiguousParam;

        const Raw = @TypeOf(t.value);
        const retained = try self.allocator.create(Raw);
        {
            errdefer self.allocator.destroy(retained);
            retained.* = try t.value.cloneView();
        }
        // From here the retained view is owned by the slot value: addSlot
        // releases it through `release` on any failure path.
        try self.addSlot(.{
            .name = name,
            .dtype = T.dtype,
            .bytes = std.mem.sliceAsBytes(retained.data()),
            .n = retained.len(),
            .retained = retained,
            .release = makeRelease(Raw),
        });
    }

    /// Register every entry of a `ParamRegistry` — trainable AND frozen
    /// (ES needs no gradients, so frozen f16/bf16 model weights qualify).
    /// Buffers and names are BORROWED: the registry must outlive the trainer.
    /// Entries whose storage is already registered are SKIPPED, not rejected:
    /// tied weights (e.g. a tied output/embedding matrix collected under two
    /// paths) perturb once, matching torch named_parameters() deduplication.
    /// Returns the number of slots actually added.
    pub fn addRegistry(self: *Self, registry: *const param_registry.ParamRegistry) !usize {
        var added: usize = 0;
        entries: for (0..registry.parameterCount()) |i| {
            const entry = registry.view(i);
            const n = switch (entry.dtype) {
                inline .f32, .f16, .bf16 => |dt| entry.bytes.len / @sizeOf(dtype_mod.Scalar(dt)),
                else => return EsError.UnsupportedDType,
            };
            for (self.slots.items) |*existing| {
                if (existing.bytes.ptr == entry.bytes.ptr) continue :entries;
            }
            try self.addSlot(.{
                .name = entry.name,
                .dtype = entry.dtype,
                .bytes = entry.bytes,
                .n = n,
                .retained = null,
                .release = null,
            });
            added += 1;
        }
        return added;
    }

    /// Register one ternary genome. `blocks` are BORROWED mutable storage
    /// (they must outlive the trainer) holding `len` logical elements as
    /// 2-bit crumbs: `len` must be a positive multiple of 256 with
    /// blocks.len == len/256, and every crumb must be a valid ternary code
    /// (0/1/2). Crumb code 3 — unrepresentable by the encoders, possible in
    /// a corrupt or untrusted GGUF — is rejected here so it can never reach
    /// the flip/vote machinery; `len` is whole blocks, so the scan has no
    /// partial-block tail to exempt. The blocks themselves are the training
    /// state; `d` scales are never touched. Ternary slots do not participate
    /// in snapshots, stream caches, or anchored weight decay (all float-slot
    /// machinery) — their restore path is the per-slot undo log.
    pub fn addTernaryParam(self: *Self, blocks: []BlockTQ2_0, len: usize) !void {
        try self.addTernaryParamNamed(blocks, len, null);
    }

    pub fn addTernaryParamNamed(self: *Self, blocks: []BlockTQ2_0, len: usize, name: ?[]const u8) !void {
        // Same tripwire as addSlot: the active member's undo logs describe
        // the slot set at perturb time.
        if (self.active_member != null) return EsError.MemberActive;
        if (len == 0 or len % ternary_block_len != 0) return EsError.InvalidConfig;
        if (blocks.len != len / ternary_block_len) return EsError.InvalidConfig;
        for (self.ternary_slots.items) |*existing| {
            if (existing.blocks.ptr == blocks.ptr) return EsError.DuplicateParam;
        }
        // Reject crumb code 3 up front: a byte holds one in some lane iff
        // both bits of a pair are set. len % 256 == 0 (checked above) makes
        // every stored crumb logical, so the scan is whole-block.
        for (blocks) |*block| {
            for (block.qs) |byte| {
                if (byte & (byte >> 1) & 0x55 != 0) return EsError.InvalidConfig;
            }
        }
        // The undo log is sized ONCE at the fixed per-member flip count.
        const undo = try self.allocator.alloc(UndoEntry, self.ternaryFlipCount(len));
        errdefer self.allocator.free(undo);
        try self.ternary_slots.append(self.allocator, .{
            .name = name,
            .blocks = blocks,
            .len = len,
            .undo = undo,
        });
    }

    /// Snapshot the CURRENT parameters as the AWD anchor theta_0. Call once
    /// after registration, while the parameters still hold the values the
    /// decay should pull toward (pretrained weights / initial adapters) —
    /// in particular BEFORE loading a training checkpoint on resume, so the
    /// anchor stays the initial model rather than the resumed one. Required
    /// before `update` when `anchor_decay != .none`. AWD is a float-slot
    /// regularizer: a ternary-only trainer captures an EMPTY slab (genomes
    /// restore through undo logs, not anchors) and the decay step in
    /// `update` no-ops over its zero float slots; NoParams only when no
    /// slot of either kind is registered.
    pub fn captureAnchor(self: *Self) !void {
        if (self.active_member != null) return EsError.MemberActive;
        if (self.slots.items.len == 0 and self.ternary_slots.items.len == 0) return EsError.NoParams;
        var total: usize = 0;
        for (self.slots.items) |*slot| total += slot.bytes.len;
        if (self.anchor == null) {
            self.anchor = blk: {
                if (comptime backend_mod.offload.enabled) {
                    // A ternary-only (zero-length) slab needs no device
                    // residency.
                    if (total > 0) {
                        if (backend_mod.offload.allocResidentBytes(total)) |dev| {
                            self.anchor_resident = true;
                            break :blk dev;
                        }
                    }
                }
                break :blk try self.allocator.alloc(u8, total);
            };
        }
        const slab = self.anchor.?;
        var offset: usize = 0;
        for (self.slots.items) |*slot| {
            @memcpy(slab[offset..][0..slot.bytes.len], slot.bytes);
            offset += slot.bytes.len;
        }
    }

    fn addSlot(self: *Self, slot: Slot) !void {
        errdefer if (slot.retained) |retained| slot.release.?(retained, self.allocator);
        // Same tripwire as perturb/update: growing the parameter set while a
        // member is applied in place would make the eventual restore subtract
        // noise the new slot never received (.regenerate) or invalidate the
        // live snapshot slab (.snapshot).
        if (self.active_member != null) return EsError.MemberActive;
        switch (slot.dtype) {
            .f32, .f16, .bf16 => {},
            else => return EsError.UnsupportedDType,
        }
        for (self.slots.items) |*existing| {
            if (existing.bytes.ptr == slot.bytes.ptr) return EsError.DuplicateParam;
        }
        if (self.snapshot) |slab| {
            // The lazy slab is sized to the slots present at first perturb;
            // growing the parameter set afterwards would silently corrupt
            // snapshot restores, so re-size it eagerly here.
            self.allocator.free(slab);
            self.snapshot = null;
            self.snapshot_valid = false;
        }
        if (self.stream_cache) |*cache| {
            // Same sizing argument as the snapshot slab.
            cache.deinit(self.allocator);
            self.stream_cache = null;
        }
        if (self.anchor) |slab| {
            // A stale anchor would silently decay the new slot toward
            // garbage; force a fresh captureAnchor.
            self.freeAnchor(slab);
            self.anchor = null;
        }
        try self.slots.append(self.allocator, slot);
    }

    pub fn paramCount(self: *const Self) usize {
        return self.slots.items.len + self.ternary_slots.items.len;
    }

    pub fn elementCount(self: *const Self) usize {
        var n: usize = 0;
        for (self.slots.items) |*slot| n += slot.n;
        for (self.ternary_slots.items) |*slot| n += slot.len;
        return n;
    }

    /// The seed of population member `member` at the CURRENT iteration — a
    /// pure function of (config.seed, iteration, member). Checkpoint
    /// contract; see the module doc.
    pub fn memberSeed(self: *const Self, member: usize) u64 {
        return rng.at(
            self.config.seed ^ seed_domain,
            self.iteration *% @as(u64, self.config.population) +% @as(u64, member),
        );
    }

    /// The member whose seed drives `member`'s noise: itself, or — under
    /// antithetic sampling — the even member of its (+, -) pair.
    fn noiseMemberIndex(self: *const Self, member: usize) usize {
        return if (self.config.antithetic) member & ~@as(usize, 1) else member;
    }

    /// +1 for members applying their pair's draw, -1 for the mirrored twin.
    fn noiseSign(self: *const Self, member: usize) f32 {
        return if (self.config.antithetic and member % 2 == 1) -1 else 1;
    }

    /// The noise-stream seed of (member, slot) under the configured scheme.
    fn slotStreamSeed(self: *const Self, member_seed: u64, slot_index: usize) u64 {
        return switch (self.config.noise) {
            .iid => rng.at(member_seed ^ noise_domain, slot_index),
            .correlated => member_seed,
        };
    }

    /// The ternary flip seed of member `member` at the CURRENT iteration —
    /// `memberSeed`'s twin in the dedicated ternary domain. Checkpoint
    /// contract; see the module doc.
    pub fn ternaryMemberSeed(self: *const Self, member: usize) u64 {
        return rng.at(
            self.config.seed ^ ternary_domain,
            self.iteration *% @as(u64, self.config.population) +% @as(u64, member),
        );
    }

    /// Ternary slot `slot_index`'s flip stream for a ternary member seed —
    /// always per-slot (the `noise` scheme governs gaussian slots only).
    /// Flip i of the stream reads index = at(seed, 2i) % len and a delta
    /// bit at(seed, 2i+1) & 1 (1 = +1, 0 = -1); the antithetic odd member
    /// negates the deltas of its pair's stream. Checkpoint contract.
    pub fn ternarySlotStreamSeed(member_seed: u64, slot_index: usize) u64 {
        return rng.at(member_seed ^ ternary_domain, slot_index);
    }

    /// Flips per member on a ternary slot of `len` logical elements:
    /// max(1, round(ternary_flip_rate * len)).
    pub fn ternaryFlipCount(self: *const Self, len: usize) usize {
        return flipCountFor(self.config, len);
    }

    fn flipCountFor(config: Config, len: usize) usize {
        const raw = @round(@as(f64, config.ternary_flip_rate) * @as(f64, @floatFromInt(len)));
        return @max(1, @as(usize, @intFromFloat(raw)));
    }

    /// theta += sigma * eps_member, in place over every registered slot
    /// (chunk-parallel). Exactly one member may be active at a time.
    pub fn perturb(self: *Self, ctx: *ExecContext, member: usize) !void {
        if (self.active_member != null) return EsError.MemberActive;
        if (self.slots.items.len == 0 and self.ternary_slots.items.len == 0) return EsError.NoParams;
        if (self.config.restore_mode == .snapshot and !self.snapshot_valid) {
            try self.takeSnapshot();
            self.snapshot_valid = true;
        }
        try self.applySigned(ctx, member, 1.0);
        self.perturbTernary(member);
        self.active_member = member;
    }

    /// Undo `perturb(member)`: regenerate-subtract (`.regenerate`) or memcpy
    /// the snapshot back (`.snapshot`); ternary slots always replay their
    /// undo logs (bitwise, either mode).
    pub fn restore(self: *Self, ctx: *ExecContext, member: usize) !void {
        const active = self.active_member orelse return EsError.MemberNotActive;
        if (active != member) return EsError.MemberNotActive;
        switch (self.config.restore_mode) {
            .regenerate => try self.applySigned(ctx, member, -1.0),
            .snapshot => {
                const slab = self.snapshot.?;
                var offset: usize = 0;
                for (self.slots.items) |*slot| {
                    @memcpy(slot.bytes, slab[offset..][0..slot.bytes.len]);
                    offset += slot.bytes.len;
                }
            },
        }
        self.restoreTernary();
        self.active_member = null;
    }

    /// Apply `member`'s sparse trit flips to every ternary slot, recording
    /// (index, old code) per flip: clamping at the rails is lossy, so exact
    /// restore replays the log in reverse (regenerate-subtract cannot work).
    fn perturbTernary(self: *Self, member: usize) void {
        if (self.ternary_slots.items.len == 0) return;
        const t_seed = self.ternaryMemberSeed(self.noiseMemberIndex(member));
        const mirror = self.config.antithetic and member % 2 == 1;
        for (self.ternary_slots.items, 0..) |*slot, k| {
            const stream_seed = ternarySlotStreamSeed(t_seed, k);
            slot.undo_len = 0;
            for (0..self.ternaryFlipCount(slot.len)) |i| {
                const flip = ternaryFlipAt(stream_seed, @intCast(i), slot.len, mirror);
                const old = crumbGet(slot.blocks, flip.index);
                slot.undo[slot.undo_len] = .{ .index = flip.index, .code = old };
                slot.undo_len += 1;
                crumbSet(slot.blocks, flip.index, moveCode(old, flip.delta));
            }
        }
    }

    /// Replay every ternary slot's undo log in reverse — bitwise restore
    /// (colliding flips on one index unwind through their intermediates).
    fn restoreTernary(self: *Self) void {
        for (self.ternary_slots.items) |*slot| {
            var i = slot.undo_len;
            while (i > 0) {
                i -= 1;
                crumbSet(slot.blocks, slot.undo[i].index, slot.undo[i].code);
            }
            slot.undo_len = 0;
        }
    }

    fn freeAnchor(self: *Self, slab: []u8) void {
        if (comptime backend_mod.offload.enabled) {
            if (self.anchor_resident) {
                backend_mod.offload.freeResidentBytes(slab);
                self.anchor_resident = false;
                return;
            }
        }
        self.allocator.free(slab);
    }

    fn takeSnapshot(self: *Self) !void {
        var total: usize = 0;
        for (self.slots.items) |*slot| total += slot.bytes.len;
        if (self.snapshot == null) self.snapshot = try self.allocator.alloc(u8, total);
        const slab = self.snapshot.?;
        var offset: usize = 0;
        for (self.slots.items) |*slot| {
            @memcpy(slab[offset..][0..slot.bytes.len], slot.bytes);
            offset += slot.bytes.len;
        }
    }

    fn applySigned(self: *Self, ctx: *ExecContext, member: usize, sign: f32) !void {
        const member_seed = self.memberSeed(self.noiseMemberIndex(member));
        const scaled = sign * self.noiseSign(member) * self.config.sigma;
        const stream = self.streamIndex(member);
        const region = try self.cacheRegionForStream(stream);
        for (self.slots.items, 0..) |*slot, k| {
            const stream_seed = self.slotStreamSeed(member_seed, k);
            switch (slot.dtype) {
                inline .f32, .f16, .bf16 => |dt| perturbSlot(
                    dt,
                    ctx,
                    slot.elems(dt),
                    stream_seed,
                    scaled,
                    if (region) |r| .{ .data = r.slotRegion(&self.stream_cache.?, k), .filled = r.filled } else null,
                ),
                else => unreachable, // addSlot rejects other dtypes
            }
        }
        if (region) |r| self.stream_cache.?.filled[r.stream] = true;
    }

    /// The noise-stream index a member draws: itself, or its antithetic
    /// pair (member/2) — matches `update`'s stream enumeration.
    fn streamIndex(self: *const Self, member: usize) usize {
        return if (self.config.antithetic) member / 2 else member;
    }

    fn streamCount(self: *const Self) usize {
        return if (self.config.antithetic) self.config.population / 2 else self.config.population;
    }

    /// The cache region of `stream` for this iteration (null when caching
    /// is off). Allocates the cache lazily on first use.
    fn cacheRegionForStream(self: *Self, stream: usize) !?CacheRegion {
        if (!self.config.cache_streams) return null;
        if (self.stream_cache == null) {
            self.stream_cache = try StreamCache.init(self.allocator, self.slots.items, self.streamCount());
        }
        const cache = &self.stream_cache.?;
        return .{
            .stream = stream,
            .base = stream * cache.total_elems,
            .filled = cache.filled[stream],
        };
    }

    /// Write theta + sigma * eps_member into caller-owned replica buffers
    /// WITHOUT touching the shared parameters: `dst[k]` receives slot k's
    /// perturbed bytes (lengths must match registration order exactly).
    /// Serial on purpose — member-level parallelism comes from calling this
    /// for different members on different threads (`evaluateMembers`), which
    /// is safe while no member is in-place active.
    pub fn materializeMember(self: *const Self, member: usize, dst: []const []u8) !void {
        if (self.active_member != null) return EsError.MemberActive;
        if (dst.len != self.slots.items.len) return EsError.ReplicaShapeMismatch;
        const member_seed = self.memberSeed(self.noiseMemberIndex(member));
        const scaled = self.noiseSign(member) * self.config.sigma;
        for (self.slots.items, dst, 0..) |*slot, dst_bytes, k| {
            if (dst_bytes.len != slot.bytes.len) return EsError.ReplicaShapeMismatch;
            const stream_seed = self.slotStreamSeed(member_seed, k);
            switch (slot.dtype) {
                inline .f32, .f16, .bf16 => |dt| materializeSlot(
                    dt,
                    slot.elemsConst(dt),
                    // Replica buffers must be Scalar(dtype)-aligned (allocate
                    // them as typed slices, not raw bytes).
                    @alignCast(std.mem.bytesAsSlice(dtype_mod.Scalar(dt), dst_bytes)),
                    stream_seed,
                    scaled,
                ),
                else => unreachable,
            }
        }
    }

    /// Ternary twin of `materializeMember`: copy each ternary slot's blocks
    /// into caller-owned replicas (`dst[k]` = ternary slot k in registration
    /// order, block counts must match) and apply `member`'s flips there,
    /// without touching the shared genome. Mixed float+ternary replicas
    /// call both materializers.
    pub fn materializeTernaryMember(self: *const Self, member: usize, dst: []const []BlockTQ2_0) !void {
        if (self.active_member != null) return EsError.MemberActive;
        if (dst.len != self.ternary_slots.items.len) return EsError.ReplicaShapeMismatch;
        const t_seed = self.ternaryMemberSeed(self.noiseMemberIndex(member));
        const mirror = self.config.antithetic and member % 2 == 1;
        for (self.ternary_slots.items, dst, 0..) |*slot, dst_blocks, k| {
            if (dst_blocks.len != slot.blocks.len) return EsError.ReplicaShapeMismatch;
            @memcpy(dst_blocks, slot.blocks);
            const stream_seed = ternarySlotStreamSeed(t_seed, k);
            for (0..self.ternaryFlipCount(slot.len)) |i| {
                const flip = ternaryFlipAt(stream_seed, @intCast(i), slot.len, mirror);
                crumbSet(dst_blocks, flip.index, moveCode(crumbGet(dst_blocks, flip.index), flip.delta));
            }
        }
    }

    /// One ES update from the population's rewards (`rewards[n]` = member
    /// n's reward at the current iteration): normalize, regenerate every
    /// member's noise, apply `theta += (alpha/population) * sum_n C_n *
    /// eps_n` (chunk-parallel, fp32 accumulation), then advance `iteration`.
    /// Mutate-last: every fallible step (validation, scratch allocation, the
    /// ternary vote collection) runs before the first parameter byte
    /// changes, so a failed update is a no-op — theta, the genomes,
    /// `iteration`, and the snapshot slab all stay consistent.
    pub fn update(self: *Self, ctx: *ExecContext, rewards: []const f32) !Stats {
        if (self.active_member != null) return EsError.MemberActive;
        if (self.slots.items.len == 0 and self.ternary_slots.items.len == 0) return EsError.NoParams;
        if (rewards.len != self.config.population) return EsError.RewardCountMismatch;
        // AWD precondition, checked up front (mutate-last); the slab itself
        // is read only after the ES update applies.
        const anchor_slab: ?[]u8 = if (self.config.anchor_decay != .none)
            self.anchor orelse return EsError.AnchorMissing
        else
            null;

        const stats = rewardStats(rewards);
        const coeffs = try self.allocator.alloc(f32, rewards.len);
        defer self.allocator.free(coeffs);
        switch (self.config.reward_norm) {
            .z_score => for (coeffs, rewards) |*c, r| {
                c.* = @floatCast((@as(f64, r) - stats.mean_reward) / (stats.std_reward + 1e-8));
            },
            .centered_ranks => {
                const order = try self.allocator.alloc(usize, rewards.len);
                defer self.allocator.free(order);
                for (order, 0..) |*member, i| member.* = i;
                // Ascending by (reward, member index): the index tiebreak
                // makes the comparator a total order, so the sort result —
                // and therefore the coefficients — are deterministic.
                std.mem.sort(usize, order, rewards, struct {
                    fn lessThan(r: []const f32, a: usize, b: usize) bool {
                        if (r[a] != r[b]) return r[a] < r[b];
                        return a < b;
                    }
                }.lessThan);
                const denom: f32 = @floatFromInt(rewards.len - 1);
                for (order, 0..) |member, rank| {
                    coeffs[member] = @as(f32, @floatFromInt(rank)) / denom - 0.5;
                }
            },
            .none => @memcpy(coeffs, rewards),
        }

        // Antithetic pairs fold to one stream each: (+eps, -eps) with
        // coefficients C_2k and C_2k+1 contribute (C_2k - C_2k+1) * eps_k —
        // the OpenAI-starter pair-difference form — halving update-side
        // noise regeneration. Division stays by the FULL population (the
        // starter's `g /= returns_n2.size`).
        const n_streams = if (self.config.antithetic) rewards.len / 2 else rewards.len;
        const stream_coeffs = try self.allocator.alloc(f32, n_streams);
        defer self.allocator.free(stream_coeffs);
        if (self.config.antithetic) {
            for (stream_coeffs, 0..) |*c, k| c.* = coeffs[2 * k] - coeffs[2 * k + 1];
        } else {
            @memcpy(stream_coeffs, coeffs);
        }

        const member_seeds = try self.allocator.alloc(u64, n_streams);
        defer self.allocator.free(member_seeds);
        for (member_seeds, 0..) |*seed, k| {
            seed.* = self.memberSeed(if (self.config.antithetic) 2 * k else k);
        }

        const stream_seeds = try self.allocator.alloc(u64, n_streams);
        defer self.allocator.free(stream_seeds);

        // Per-stream cache regions (updateSlot iterates members innermost,
        // so regions travel as parallel slices; null when caching is off).
        var cache_regions: ?[]CacheRegion = null;
        defer if (cache_regions) |regions| self.allocator.free(regions);
        if (self.config.cache_streams) {
            const regions = try self.allocator.alloc(CacheRegion, n_streams);
            errdefer self.allocator.free(regions);
            for (regions, 0..) |*region, stream| {
                region.* = (try self.cacheRegionForStream(stream)).?;
            }
            cache_regions = regions;
        }

        // Ternary slots vote on the SAME folded stream coefficients, so
        // mixed trainers keep one reward pipeline. Collection is the LAST
        // fallible step of the update; the staged moves apply infallibly
        // after the float slots.
        try self.collectTernaryVotes(stream_coeffs);

        const scale = self.alphaValue() / @as(f32, @floatFromInt(self.config.population));
        for (self.slots.items, 0..) |*slot, k| {
            for (stream_seeds, member_seeds) |*stream, member_seed| {
                stream.* = self.slotStreamSeed(member_seed, k);
            }
            switch (slot.dtype) {
                inline .f32, .f16, .bf16 => |dt| updateSlot(
                    dt,
                    ctx,
                    slot.elems(dt),
                    stream_seeds,
                    stream_coeffs,
                    scale,
                    if (cache_regions) |regions| .{ .cache = &self.stream_cache.?, .slot_index = k, .regions = regions } else null,
                ),
                else => unreachable,
            }
        }
        if (cache_regions) |regions| {
            for (regions) |region| self.stream_cache.?.filled[region.stream] = true;
        }

        self.applyTernaryVotes();

        // AWD (Eq. 4 / Algorithm 1 line 9 of arXiv:2605.30148): the proximal
        // pull toward the anchor reads the POST-update theta and applies the
        // coupled step alpha * lambda, elementwise over every FLOAT
        // parameter (a no-op over zero float slots — ternary-only trainers).
        if (anchor_slab) |slab| {
            const decay_step = self.alphaValue() * self.config.anchor_lambda;
            var offset: usize = 0;
            for (self.slots.items) |*slot| {
                switch (slot.dtype) {
                    inline .f32, .f16, .bf16 => |dt| anchorSlot(
                        dt,
                        ctx,
                        slot.elems(dt),
                        @alignCast(std.mem.bytesAsSlice(dtype_mod.Scalar(dt), slab[offset..][0..slot.bytes.len])),
                        decay_step,
                        self.config.anchor_decay,
                    ),
                    else => unreachable,
                }
                offset += slot.bytes.len;
            }
        }

        self.iteration += 1;
        // The cache holds THIS iteration's streams; the next iteration draws
        // fresh ones. Theta changed, so a snapshot-mode slab is stale too.
        if (self.stream_cache) |*cache| @memset(cache.filled, false);
        self.snapshot_valid = false;
        return stats;
    }

    /// Fallible half of the ternary vote-and-threshold update (EGGROLL's
    /// single-bin integer recipe on trits), run BEFORE any parameter
    /// mutation: regenerate every stream's flips, accumulate vote[index] +=
    /// coeff * delta (colliding flips accumulate; under the antithetic fold
    /// the odd member's mirrored deltas contribute (C+ - C-) * delta_even,
    /// exactly like the gaussian fold, so only the even member's stream
    /// regenerates), then stage each slot's top-K nonzero votes in
    /// `ternary_entries` (`slot.pending` entries per slot, in application
    /// order). Hashmap iteration order is nondeterministic, so votes are
    /// collected and sorted by (|vote| desc, index asc) — a pinned total
    /// order (checkpoint contract); the f32 vote sums themselves accumulate
    /// in fixed stream-then-flip order. K = min(touched, max(1, round(len *
    /// effective_fraction))) per slot, effective_fraction =
    /// ternary_update_fraction / (1 + ternary_update_decay * iteration).
    /// The vote/entry containers are trainer-owned scratch (capacity
    /// retained across updates), so steady-state updates stop allocating.
    fn collectTernaryVotes(self: *Self, stream_coeffs: []const f32) !void {
        self.ternary_entries.clearRetainingCapacity();
        if (self.ternary_slots.items.len == 0) return;
        const eff = @as(f64, self.config.ternary_update_fraction) /
            (1.0 + @as(f64, self.config.ternary_update_decay) * @as(f64, @floatFromInt(self.iteration)));
        for (self.ternary_slots.items, 0..) |*slot, k| {
            self.ternary_votes.clearRetainingCapacity();
            const flips = self.ternaryFlipCount(slot.len);
            for (stream_coeffs, 0..) |coeff, stream| {
                const member = if (self.config.antithetic) 2 * stream else stream;
                const stream_seed = ternarySlotStreamSeed(self.ternaryMemberSeed(member), k);
                // Flips whose one-bin move clamped to a no-op at perturb
                // time still vote — the EGGROLL integer recipe counts them.
                // A possible future refinement is filtering votes where
                // moveCode(current, delta) == current.
                for (0..flips) |i| {
                    const flip = ternaryFlipAt(stream_seed, @intCast(i), slot.len, false);
                    const gop = try self.ternary_votes.getOrPut(self.allocator, flip.index);
                    if (!gop.found_existing) gop.value_ptr.* = 0;
                    gop.value_ptr.* += coeff * @as(f32, @floatFromInt(flip.delta));
                }
            }
            const base = self.ternary_entries.items.len;
            var it = self.ternary_votes.iterator();
            while (it.next()) |kv| {
                if (kv.value_ptr.* != 0) {
                    try self.ternary_entries.append(self.allocator, .{ .index = kv.key_ptr.*, .vote = kv.value_ptr.* });
                }
            }
            std.mem.sort(VoteEntry, self.ternary_entries.items[base..], {}, voteBefore);
            const cap = @max(1, @as(usize, @intFromFloat(@round(eff * @as(f64, @floatFromInt(slot.len))))));
            const kept = @min(self.ternary_entries.items.len - base, cap);
            self.ternary_entries.shrinkRetainingCapacity(base + kept);
            slot.pending = kept;
        }
    }

    /// Infallible half: apply each slot's staged top-K votes as clamped
    /// one-bin moves toward sign(vote), consuming `ternary_entries` in the
    /// exact order `collectTernaryVotes` staged them (slot order, |vote|
    /// desc / index asc within a slot).
    fn applyTernaryVotes(self: *Self) void {
        var offset: usize = 0;
        for (self.ternary_slots.items) |*slot| {
            for (self.ternary_entries.items[offset..][0..slot.pending]) |entry| {
                const delta: i8 = if (entry.vote > 0) 1 else -1;
                crumbSet(slot.blocks, entry.index, moveCode(crumbGet(slot.blocks, entry.index), delta));
            }
            offset += slot.pending;
            slot.pending = 0;
        }
    }

    /// The sequential in-place driver: for each member, perturb → evaluate →
    /// restore; then one `update`. `evaluator` is duck-typed over
    /// `eval(member: usize) !f32` (capture the model/data/ctx it needs).
    pub fn step(self: *Self, ctx: *ExecContext, evaluator: anytype) !Stats {
        const rewards = try self.allocator.alloc(f32, self.config.population);
        defer self.allocator.free(rewards);
        for (rewards, 0..) |*reward, member| {
            try self.perturb(ctx, member);
            const value = evaluator.eval(member) catch |err| {
                // Leave theta unperturbed on the way out; the tripwire state
                // is consistent for a retry.
                try self.restore(ctx, member);
                return err;
            };
            try self.restore(ctx, member);
            reward.* = value;
        }
        return self.update(ctx, rewards);
    }

    /// Member-parallel evaluation over OS threads: workers pull member
    /// indices from a shared counter and call `evaluator.evalMember(worker,
    /// member) !f32` — `worker` (< `workers`) selects the caller's replica
    /// (populate it via `materializeMember`). `workers` must be >= 1 (it is
    /// clamped to the population and a 64-thread ceiling, so the `worker <
    /// workers` bound always holds for replica tables sized by `workers`).
    /// The evaluator must be thread-safe across distinct workers; the shared
    /// parameters are only READ here (no member may be in-place active). The
    /// first error stops the fan-out and is returned after every worker has
    /// joined.
    pub fn evaluateMembers(self: *const Self, evaluator: anytype, rewards: []f32, workers: usize) !void {
        if (workers == 0) return EsError.InvalidConfig;
        if (self.active_member != null) return EsError.MemberActive;
        if (rewards.len != self.config.population) return EsError.RewardCountMismatch;
        const max_workers = 64; // spawned-thread array bound below
        const worker_count = @min(@min(workers, self.config.population), max_workers);
        if (worker_count == 1) {
            for (rewards, 0..) |*reward, member| reward.* = try evaluator.evalMember(0, member);
            return;
        }

        const Evaluator = @TypeOf(evaluator);
        const Shared = struct {
            evaluator: Evaluator,
            rewards: []f32,
            next: std.atomic.Value(usize) = .init(0),
            failed: std.atomic.Value(bool) = .init(false),
            err_mutex: thread.Mutex = .{},
            err: ?anyerror = null,

            fn run(shared: *@This(), worker: usize) void {
                while (!shared.failed.load(.acquire)) {
                    const member = shared.next.fetchAdd(1, .monotonic);
                    if (member >= shared.rewards.len) return;
                    const value = shared.evaluator.evalMember(worker, member) catch |err| {
                        shared.err_mutex.lock();
                        if (shared.err == null) shared.err = err;
                        shared.err_mutex.unlock();
                        shared.failed.store(true, .release);
                        return;
                    };
                    shared.rewards[member] = value;
                }
            }
        };

        var shared = Shared{ .evaluator = evaluator, .rewards = rewards };
        var threads: [64]std.Thread = undefined;
        var spawned: usize = 0;
        // Workers 1..worker_count on spawned threads; worker 0 runs here.
        while (spawned + 1 < worker_count) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(.{}, Shared.run, .{ &shared, spawned + 1 }) catch break;
        }
        Shared.run(&shared, 0);
        for (threads[0..spawned]) |worker_thread| worker_thread.join();
        if (shared.err) |err| return err;
    }
};
