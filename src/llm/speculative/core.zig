//! Draft-model-free speculative decoding (core).
//!
//! A `DraftSource` proposes a continuation of the committed token stream from
//! any cheap, DETERMINISTIC oracle (n-gram index, prompt lookup, token
//! recycling, ...). The decoder verifies the draft with ONE batched forward
//! (`Model.forwardStepAllLogits`) and commits the longest prefix the target
//! model itself would have produced.
//!
//! Losslessness — one code path for greedy AND sampled: because the drafter is
//! deterministic (a one-hot proposal distribution q), Leviathan rejection
//! sampling degenerates to running the FULL sampling pipeline (penalties,
//! temperature, top-k/top-p/min-p) on the target logits at each verified
//! position, conditioned on the HYPOTHETICAL prefix (committed history + the
//! draft tokens accepted so far):
//!
//!   - accept position i while sampled == draft[i];
//!   - at the first mismatch, the sampled token IS the correction token
//!     (provably distributed as the target distribution);
//!   - if the whole draft is accepted, the (k+1)-th row's sample is a free
//!     bonus token.
//!
//! Greedy is the same path (`temperature <= 0` makes the sampler an argmax, so
//! "sampled == draft" is argmax equality). Token IDS are compared, never
//! probabilities. The KV cache is truncated back to the accepted length.
//!
//! Penalties see the hypothetical prefix naturally: accepted tokens are
//! committed to `history` before the next position is sampled, so each
//! `Sampler.next` call conditions on exactly the tokens its logits row is
//! conditioned on — and rejected drafts never touch `history`, so no rollback
//! is needed.
//!
//! RNG/lossless contract: exactly one RNG draw is consumed per COMMITTED token
//! (positions past the first mismatch are never sampled), the same pattern as
//! a plain run. Stop-awareness is part of this contract: when
//! `Options.stop_token` is set, committing it ends the verify row loop
//! immediately — a plain run stops there too, so sampling any further row
//! would consume draws the plain run never makes and desync a persistent
//! sampler for the rest of the conversation. So given bitwise-identical
//! logits the output token stream is identical to the non-speculative run.
//! Logits, however, are computed in verify batches of m = 1+draft rows
//! instead of m = 1; rows are independent
//! through every kernel until the m-dependent thresholds (fused K-quant FFN at
//! seq >= 12, tiled attention at seq >= 48), beyond which reassociation drift
//! (~1e-6 rel) can flip a near-tied sample. Lossless therefore means: same
//! DISTRIBUTION always; same sample stream whenever the logits match bitwise
//! (which the tests below verify for the small-m regime).

const std = @import("std");
const fucina = @import("fucina");
const kv_cache = @import("../kv_cache.zig");
const sampler_mod = @import("../sampler.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const KvCache = kv_cache.KvCache;
const Sampler = sampler_mod.Sampler;
const Logits = fucina.Tensor(.{ .seq, .vocab });

/// Verification feedback for one verified position (Token-Recycling-style
/// sources build their successor index from these). CANONICAL definition —
/// `recycling.zig` and `sam_index.zig` re-export this type; keep exactly one.
pub const TopKRow = struct {
    /// The committed-stream token whose successor distribution the candidates
    /// describe (the verify input token at this position).
    token: usize,
    /// Top candidate token ids from the target logits at this position, most
    /// probable first, captured BEFORE penalties mutate the row. Borrowed:
    /// valid only for the duration of the `observeTopK` call.
    topk: []const u32,
};

/// Vtable interface for draft proposers — modular and externally injectable;
/// the decoder works with any deterministic source.
pub const DraftSource = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Propose up to `buf.len` continuation tokens given the full committed
        /// token context (the last element is the token just committed).
        /// Returns the number of tokens written into `buf`; 0 = no draft.
        suggest: *const fn (ptr: *anyopaque, context: []const usize, buf: []usize) usize,
        /// Observe newly COMMITTED tokens (for index updates).
        observe: *const fn (ptr: *anyopaque, committed: []const usize) void,
        /// Observe verification logits feedback. Null = the source has no use
        /// for it and the decoder skips computing the top-k entirely.
        observeTopK: ?*const fn (ptr: *anyopaque, positions: []const TopKRow) void = null,
    };

    pub fn suggest(self: DraftSource, context: []const usize, buf: []usize) usize {
        return self.vtable.suggest(self.ptr, context, buf);
    }

    pub fn observe(self: DraftSource, committed: []const usize) void {
        self.vtable.observe(self.ptr, committed);
    }

    pub fn observeTopK(self: DraftSource, positions: []const TopKRow) void {
        if (self.vtable.observeTopK) |f| f(self.ptr, positions);
    }

    pub fn wantsTopK(self: DraftSource) bool {
        return self.vtable.observeTopK != null;
    }
};

pub const Stats = struct {
    /// Total `step` iterations.
    steps: usize = 0,
    /// Iterations that ran a batched verify pass.
    spec_steps: usize = 0,
    /// Iterations that fell back to a plain step (no draft / draft < min_draft).
    fallback_steps: usize = 0,
    /// Iterations that ran plain because speculation was (auto-)disabled.
    disabled_steps: usize = 0,
    /// Draft tokens submitted for verification.
    drafted: usize = 0,
    /// Draft tokens accepted (committed verbatim).
    accepted: usize = 0,
    /// Verify passes that ended in a rejection (correction token committed).
    rejected_steps: usize = 0,
    /// Free bonus tokens (whole draft accepted, k+1-th row sampled).
    bonus: usize = 0,
    /// Total tokens committed.
    committed: usize = 0,

    pub fn tokensPerStep(self: Stats) f64 {
        if (self.steps == 0) return 0;
        return @as(f64, @floatFromInt(self.committed)) / @as(f64, @floatFromInt(self.steps));
    }

    pub fn acceptanceRate(self: Stats) f64 {
        if (self.drafted == 0) return 0;
        return @as(f64, @floatFromInt(self.accepted)) / @as(f64, @floatFromInt(self.drafted));
    }

    pub fn writeSummary(self: Stats, writer: *std.Io.Writer) !void {
        try writer.print(
            "spec: steps={d} (verify {d}, fallback {d}, off {d}) committed={d} ({d:.2} tok/step) drafted={d} accepted={d} ({d:.1}% acc) bonus={d} rejected={d}",
            .{
                self.steps,
                self.spec_steps,
                self.fallback_steps,
                self.disabled_steps,
                self.committed,
                self.tokensPerStep(),
                self.drafted,
                self.accepted,
                self.acceptanceRate() * 100.0,
                self.bonus,
                self.rejected_steps,
            },
        );
    }
};

pub const Options = struct {
    /// Longest draft submitted to one verify pass.
    max_draft: usize = 16,
    /// Drafts shorter than this fall back to a plain single-token step (a
    /// 1-token verify costs a 2-row forward to win at most 1 extra token).
    min_draft: usize = 2,
    /// Master switch; false = every step is a plain step.
    enabled: bool = true,
    /// Candidates per verified position handed to `observeTopK` (only computed
    /// when the source wants the feedback). Must be >= 1 when the source's
    /// `observeTopK` is non-null — `init` rejects 0 there (a top-K-consuming
    /// source with a zero feedback width would silently never learn).
    topk_feedback: usize = 8,
    /// Stream-end marker (e.g. the chat turn's stop id). When a committed
    /// token equals it, the verify row loop BREAKS immediately: no rows past
    /// the stop are sampled (the bonus row included), preserving the
    /// one-RNG-draw-per-plain-committed-token contract across a stop that
    /// arrives as an accepted draft token mid-batch. Null = no stop token.
    stop_token: ?usize = null,

    // ---- cost-aware AUTO-OFF gate (see `CostGate`) ----
    /// Verify steps held in the gate's rolling window. Fallback steps cost
    /// the same as plain steps and never enter the window — only verify
    /// economics can lose time.
    rate_window: usize = 16,
    /// Evaluate the gate as soon as the window holds this many DRAFTED tokens
    /// (and at least 2 verify steps — a single verify is too noisy to act on),
    /// not after a fixed step count: clear losses trip fast.
    min_window_drafted: usize = 8,
    /// Speculation stays enabled while the rolling ESTIMATED SPEEDUP —
    /// committed tokens per plain-step-equivalent of verify cost — is at
    /// least this.
    min_speedup: f32 = 1.0,
    /// Hysteresis: a re-probe re-enables only at min_speedup + probe_margin,
    /// so a marginal regime doesn't flap on/off.
    probe_margin: f32 = 0.10,
    /// Disabled steps before the first re-probe; doubles on every failed
    /// probe (exponential backoff: 128 -> 256 -> 512 -> ...), capped at
    /// `reprobe_max`. A passed probe restarts the schedule at the base.
    reprobe_after: usize = 128,
    reprobe_max: usize = 1024,
    /// Verify steps one re-probe runs before deciding (a SHORT probe, not a
    /// full window).
    probe_steps: usize = 4,
    /// Static verify-cost model: plain-step equivalents of one verify pass at
    /// draft length k (linear interpolation between points, clamped at the
    /// ends). Defaults are the measured Qwen3-0.6B-Q4_K_S economics (M1 Max,
    /// ReleaseFast, `--spec-bench`). Used whenever live timing is unavailable
    /// (decoder `io == null`, or before the first plain-step sample).
    cost_table: []const CostPoint = &default_cost_table,
    /// Cap the draft budget by the rolling acceptance rate:
    /// budget = max(min_draft, 2 + ceil(acceptance * max_draft)) — low-
    /// acceptance phases verify small cheap drafts instead of max_draft
    /// losses.
    adapt_budget: bool = true,
};

/// One static cost-model point: a verify pass over `draft_len` draft tokens
/// costs `cost` plain decode steps.
pub const CostPoint = struct { draft_len: usize, cost: f32 };

/// Measured verify economics on Qwen3-0.6B-Q4_K_S (M1 Max, ReleaseFast,
/// `fucina-zig-qwen3 --spec-bench`): verify-k forward vs single-token step.
pub const default_cost_table = [_]CostPoint{
    .{ .draft_len = 2, .cost = 1.65 },
    .{ .draft_len = 4, .cost = 1.42 },
    .{ .draft_len = 8, .cost = 2.84 },
    .{ .draft_len = 16, .cost = 4.5 },
};

/// Interpolated verify cost (plain-step equivalents) for a draft of length
/// `draft_len`: linear between table points, clamped to the first/last entry
/// outside the table's range.
pub fn tableCost(table: []const CostPoint, draft_len: usize) f32 {
    std.debug.assert(table.len >= 1);
    if (draft_len <= table[0].draft_len) return table[0].cost;
    var i: usize = 1;
    while (i < table.len) : (i += 1) {
        if (draft_len <= table[i].draft_len) {
            const a = table[i - 1];
            const b = table[i];
            const t = @as(f32, @floatFromInt(draft_len - a.draft_len)) /
                @as(f32, @floatFromInt(b.draft_len - a.draft_len));
            return a.cost + t * (b.cost - a.cost);
        }
    }
    return table[table.len - 1].cost;
}

/// Cost-aware speculation gate.
///
/// The old rate gate compared committed tokens/step against a flat threshold,
/// ignoring that a verify pass COSTS more than the plain step it replaces
/// (measured: verify-8 ~ 2.84 plain steps on the 0.6B), so e.g. 1.56 tok/step
/// at k~8 was kept alive as a "win" while actually losing ~45%. This gate
/// estimates the true speedup over a rolling window of verify steps:
///
///     est_speedup = committed_tokens / verify_cost_in_plain_step_equivalents
///
/// Verify cost comes from the static `Options.cost_table` curve, live-rescaled
/// when the decoder has a clock: measured verify/plain forward ratios feed a
/// clamped EWMA multiplier on the table (model-agnostic, tracks thermal drift,
/// and a single noisy measurement cannot flip the gate — see `verifyCost`).
/// Without a clock the table applies as-is.
///
/// Off -> re-probe with exponential backoff and a short probe; re-enable
/// requires `probe_margin` hysteresis. The gate also adapts the draft budget
/// to the window's acceptance rate. Gating decides WHEN speculation runs,
/// never WHAT is committed — losslessness is untouched.
pub const CostGate = struct {
    const State = enum { on, off, probing };

    const Entry = struct {
        committed: usize,
        accepted: usize,
        drafted: usize,
        cost: f32,
    };

    options: Options,
    /// Ring over the last `rate_window` verify steps.
    window: []Entry,
    len: usize = 0,
    idx: usize = 0,
    sum_committed: usize = 0,
    sum_accepted: usize = 0,
    sum_drafted: usize = 0,
    /// f64: the rolling sum sees an unbounded stream of f32 add/subtract
    /// pairs; an f32 accumulator drifts incrementally (the window resets on
    /// every off/probe transition, but an always-on run never resets).
    sum_cost: f64 = 0,

    state: State = .on,
    /// Steps spent disabled since speculation turned off.
    off_steps: usize = 0,
    /// Current wait (disabled steps) before the next re-probe.
    backoff: usize,
    /// Verify steps left in the current probe.
    probe_left: usize = 0,

    /// Live cost model: EWMA of the plain-step forward, ns.
    plain_ns: f64 = 0,
    plain_samples: usize = 0,
    /// Live rescale of the static cost table (1.0 = table as-is), learned
    /// from measured verify/plain ratios — see `verifyCost`.
    cost_scale: f64 = 1.0,

    const ewma_alpha: f64 = 0.25;
    /// Smoothing/clamping of the live table rescale: one sample moves the
    /// scale by at most scale_alpha of its (clamped) deviation, so a single
    /// noisy measurement cannot flip the gate.
    const scale_alpha: f64 = 0.2;
    const scale_min: f64 = 0.25;
    const scale_max: f64 = 4.0;

    fn init(allocator: Allocator, options: Options) !CostGate {
        // Boundary validation, not debug asserts: these options come straight
        // from embedder configuration, and each degenerate value is ReleaseFast
        // UB downstream (zero-length ring -> OOB + modulo-by-zero in
        // recordVerify; empty table -> OOB in tableCost; probe_steps 0 ->
        // probe_left usize wrap). Fail loudly instead.
        if (options.rate_window < 2) return error.RateWindowTooSmall;
        if (options.probe_steps < 1) return error.ProbeStepsZero;
        if (options.cost_table.len < 1) return error.CostTableEmpty;
        if (options.reprobe_after < 1) return error.ReprobeAfterZero;
        const window = try allocator.alloc(Entry, options.rate_window);
        return .{ .options = options, .window = window, .backoff = options.reprobe_after };
    }

    fn deinit(self: *CostGate, allocator: Allocator) void {
        allocator.free(self.window);
        self.* = undefined;
    }

    /// True when this step may speculate; advances the re-probe countdown
    /// while off.
    fn allows(self: *CostGate) bool {
        switch (self.state) {
            .on, .probing => return true,
            .off => {
                self.off_steps += 1;
                if (self.off_steps >= self.backoff) {
                    self.state = .probing;
                    self.probe_left = self.options.probe_steps;
                    self.resetWindow();
                    return true;
                }
                return false;
            },
        }
    }

    fn resetWindow(self: *CostGate) void {
        self.len = 0;
        self.idx = 0;
        self.sum_committed = 0;
        self.sum_accepted = 0;
        self.sum_drafted = 0;
        self.sum_cost = 0;
    }

    /// Feed one plain-step forward time into the live cost model.
    fn notePlainNs(self: *CostGate, ns: u64) void {
        const x: f64 = @floatFromInt(ns);
        self.plain_ns = if (self.plain_samples == 0) x else ewma_alpha * x + (1.0 - ewma_alpha) * self.plain_ns;
        self.plain_samples += 1;
    }

    /// One verify pass's cost in plain-step equivalents. The static table
    /// gives the cost curve's shape; live timing (when available)
    /// continuously RESCALES it: each verify contributes a measured-ratio /
    /// table-cost sample to a clamped EWMA multiplier. One noisy measurement
    /// (scheduler stall, thermal hiccup) moves the estimate by at most
    /// scale_alpha of its clamped deviation — it cannot flip the gate — while
    /// a model whose true economics differ from the table is learned within a
    /// few verifies. A verify is never cheaper than the plain step it
    /// replaces: clamped to >= 1.
    fn verifyCost(self: *CostGate, draft_len: usize, verify_ns: ?u64) f32 {
        const base: f64 = tableCost(self.options.cost_table, draft_len);
        if (verify_ns) |ns| {
            if (self.plain_samples > 0 and self.plain_ns > 0) {
                const measured = @max(@as(f64, @floatFromInt(ns)) / self.plain_ns, 1.0);
                const sample = std.math.clamp(measured / base, scale_min, scale_max);
                self.cost_scale = std.math.clamp(
                    scale_alpha * sample + (1.0 - scale_alpha) * self.cost_scale,
                    scale_min,
                    scale_max,
                );
            }
        }
        return @floatCast(@max(self.cost_scale * base, 1.0));
    }

    /// Record one verify step's outcome and run the gate decision.
    fn recordVerify(self: *CostGate, committed: usize, accepted: usize, drafted: usize, cost: f32) void {
        if (self.len == self.window.len) {
            const old = self.window[self.idx];
            self.sum_committed -= old.committed;
            self.sum_accepted -= old.accepted;
            self.sum_drafted -= old.drafted;
            self.sum_cost -= old.cost;
        } else {
            self.len += 1;
        }
        self.window[self.idx] = .{ .committed = committed, .accepted = accepted, .drafted = drafted, .cost = cost };
        self.idx = (self.idx + 1) % self.window.len;
        self.sum_committed += committed;
        self.sum_accepted += accepted;
        self.sum_drafted += drafted;
        self.sum_cost += cost;

        const evaluable = self.len >= 2 and self.sum_drafted >= self.options.min_window_drafted;
        switch (self.state) {
            .on => if (evaluable and self.estSpeedup() < self.options.min_speedup) {
                // Fresh degradation episode: back off from the base wait.
                self.toOff(self.options.reprobe_after);
            },
            .probing => {
                self.probe_left -= 1;
                if (evaluable and self.estSpeedup() < self.options.min_speedup) {
                    // Clear loss mid-probe: fail early, double the backoff.
                    self.toOff(@min(self.backoff * 2, self.options.reprobe_max));
                } else if (self.probe_left == 0) {
                    if (self.estSpeedup() >= self.options.min_speedup + self.options.probe_margin) {
                        self.state = .on; // probe passed; backoff restarts at base on the next trip
                    } else {
                        self.toOff(@min(self.backoff * 2, self.options.reprobe_max));
                    }
                }
            },
            .off => unreachable, // allows() filtered this step out
        }
    }

    fn toOff(self: *CostGate, next_backoff: usize) void {
        self.state = .off;
        self.off_steps = 0;
        self.backoff = next_backoff;
        self.resetWindow();
    }

    /// Rolling estimated speedup vs plain decoding (committed tokens per
    /// plain-step equivalent). Infinite on an empty window.
    pub fn estSpeedup(self: *const CostGate) f32 {
        if (self.sum_cost <= 0) return std.math.inf(f32);
        return @floatCast(@as(f64, @floatFromInt(self.sum_committed)) / self.sum_cost);
    }

    /// Acceptance-adaptive draft budget: low-acceptance phases shrink to
    /// small, cheap verifies. An empty window grants the full budget.
    fn budgetCap(self: *const CostGate, max_draft: usize, min_draft: usize) usize {
        if (self.sum_drafted == 0) return max_draft;
        const ceil_acc_budget = (self.sum_accepted * max_draft + self.sum_drafted - 1) / self.sum_drafted;
        return @min(max_draft, @max(min_draft, 2 + ceil_acc_budget));
    }
};

/// Streaming sink for committed tokens (stdout decoder, SSE writer, buffer...).
pub const TokenSink = struct {
    ptr: *anyopaque,
    func: *const fn (ptr: *anyopaque, token: usize) anyerror!void,

    pub fn emit(self: TokenSink, token: usize) !void {
        return self.func(self.ptr, token);
    }
};

/// Test/debug instrumentation: called with every logits row the sampler is
/// about to see (pre-penalty), its absolute position (= committed history
/// length at sampling time) and its index within the verify batch (0 for
/// plain steps). The losslessness tests use it to prove replay equivalence.
pub const VerifyRowHook = struct {
    ptr: *anyopaque,
    func: *const fn (ptr: *anyopaque, abs_pos: usize, batch_index: usize, row: []const f32) anyerror!void,
};

/// `Model` is duck-typed: it must expose `forwardStep` and
/// `forwardStepAllLogits` with the qwen3/gemma4 signatures over the shared
/// `KvCache`. (qwen35's recurrent cache cannot rewind — out of scope.)
pub fn SpeculativeDecoder(comptime Model: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        source: DraftSource,
        options: Options,
        stats: Stats = .{},
        /// Verify input scratch: [carried token, draft...] (1 + max_draft).
        verify_buf: []usize,
        /// observeTopK scratch: (1 + max_draft) * topk_feedback ids + rows.
        topk_ids: []u32,
        topk_rows: []TopKRow,
        /// Cost-aware AUTO-OFF gate.
        gate: CostGate,
        /// Clock for the live verify/plain cost model. Null (default) falls
        /// back to the static `Options.cost_table` economics; embedders set
        /// it after init (like `on_verify_row`) to gate on measured ratios.
        io: ?std.Io = null,
        on_verify_row: ?VerifyRowHook = null,

        pub fn init(allocator: Allocator, source: DraftSource, options: Options) !Self {
            std.debug.assert(options.max_draft >= 1);
            // A source that consumes verification feedback with a zero
            // feedback width is a configuration error: it would silently
            // starve the source (empty topk rows). Fail loudly instead.
            if (source.wantsTopK() and options.topk_feedback == 0) return error.TopKFeedbackDisabled;
            const verify_buf = try allocator.alloc(usize, 1 + options.max_draft);
            errdefer allocator.free(verify_buf);
            const topk_ids = try allocator.alloc(u32, (1 + options.max_draft) * options.topk_feedback);
            errdefer allocator.free(topk_ids);
            const topk_rows = try allocator.alloc(TopKRow, 1 + options.max_draft);
            errdefer allocator.free(topk_rows);
            var gate = try CostGate.init(allocator, options);
            errdefer gate.deinit(allocator);
            return .{
                .allocator = allocator,
                .source = source,
                .options = options,
                .verify_buf = verify_buf,
                .topk_ids = topk_ids,
                .topk_rows = topk_rows,
                .gate = gate,
            };
        }

        pub fn deinit(self: *Self) void {
            self.gate.deinit(self.allocator);
            self.allocator.free(self.topk_rows);
            self.allocator.free(self.topk_ids);
            self.allocator.free(self.verify_buf);
            self.* = undefined;
        }

        /// One decode iteration. Invariant (the standard decode-loop shape):
        /// `history` holds every committed token (prompt + generated) and its
        /// LAST element is the token just committed but not yet in `kv`, i.e.
        /// `history.items.len == kv.len + 1`. `history` must be allocated with
        /// `ctx.allocator` (the decoder appends committed tokens to it).
        /// Emits each committed token through `sink`, updates `stats`, feeds
        /// `observe`/`observeTopK`. Returns the number of tokens committed
        /// this iteration (>= 1).
        pub fn step(
            self: *Self,
            ctx: *ExecContext,
            model: *const Model,
            kv: *KvCache,
            sampler: *Sampler,
            history: *std.ArrayList(usize),
            sink: TokenSink,
        ) !usize {
            // Decode-state contract, validated at RUNTIME (not just a debug
            // assert): an empty or kv-desynced history would index out of
            // bounds / corrupt the cache in ReleaseFast (e.g. an empty
            // prompt reaching plainStep's `items[len - 1]`).
            if (history.items.len == 0 or history.items.len != kv.len + 1) {
                return error.InvalidDecodeState;
            }
            std.debug.assert(kv.len + 1 <= kv.capacity);
            self.stats.steps += 1;

            if (!self.gateAllows()) {
                self.stats.disabled_steps += 1;
                const committed = try self.plainStep(ctx, model, kv, sampler, history, sink);
                self.stats.committed += committed;
                return committed;
            }

            // 1 + draft rows must fit in the cache; the gate's acceptance-
            // adaptive budget keeps low-acceptance phases on cheap verifies.
            const room = kv.capacity - kv.len - 1;
            var max_draft = @min(self.options.max_draft, room);
            if (self.options.adapt_budget) {
                max_draft = @min(max_draft, self.gate.budgetCap(self.options.max_draft, self.options.min_draft));
            }
            var draft_len: usize = 0;
            if (max_draft >= self.options.min_draft) {
                // The source is an externally injectable vtable: clamp its
                // claimed length at runtime (a lying return value would walk
                // verify_buf out of bounds in ReleaseFast).
                draft_len = @min(self.source.suggest(history.items, self.verify_buf[1 .. 1 + max_draft]), max_draft);
            }

            const committed = if (draft_len < self.options.min_draft) blk: {
                self.stats.fallback_steps += 1;
                break :blk try self.plainStep(ctx, model, kv, sampler, history, sink);
            } else try self.verifyStep(ctx, model, kv, sampler, history, sink, draft_len);

            self.stats.committed += committed;
            return committed;
        }

        /// True when this step may speculate; advances the re-probe counter
        /// while disabled.
        fn gateAllows(self: *Self) bool {
            if (!self.options.enabled) return false;
            return self.gate.allows();
        }

        /// Monotonic ns for the live cost model; null without a clock.
        fn nowNs(self: *const Self) ?i96 {
            const io = self.io orelse return null;
            return std.Io.Clock.awake.now(io).nanoseconds;
        }

        /// Plain single-token decode step (the non-speculative baseline).
        fn plainStep(
            self: *Self,
            ctx: *ExecContext,
            model: *const Model,
            kv: *KvCache,
            sampler: *Sampler,
            history: *std.ArrayList(usize),
            sink: TokenSink,
        ) !usize {
            const last = history.items[history.items.len - 1];
            // Restore `history.len == kv.len + 1` on EVERY error path: the
            // forward advances kv before its own fallible tail ops, and
            // append/emit can fail after it. history.items.len is read at
            // unwind time; truncate clamps.
            errdefer kv.truncate(history.items.len - 1);
            const t0 = self.nowNs();
            var logits = try model.forwardStep(ctx, kv, &.{last}, kv.len);
            defer logits.deinit();
            if (t0) |start| self.gate.notePlainNs(@intCast(self.nowNs().? - start));
            if (self.on_verify_row) |hook| {
                try hook.func(hook.ptr, history.items.len, 0, try logits.dataConst());
            }
            const next = try sampler.next(ctx, &logits, history.items);
            try history.append(ctx.allocator, next);
            try sink.emit(next);
            self.source.observe(history.items[history.items.len - 1 ..]);
            return 1;
        }

        /// Batched verify pass over `verify_buf[0 .. 1 + draft_len]` (carried
        /// token + draft, already written by `step`).
        fn verifyStep(
            self: *Self,
            ctx: *ExecContext,
            model: *const Model,
            kv: *KvCache,
            sampler: *Sampler,
            history: *std.ArrayList(usize),
            sink: TokenSink,
            draft_len: usize,
        ) !usize {
            self.stats.spec_steps += 1;
            self.stats.drafted += draft_len;

            const pos0 = kv.len;
            const start_len = history.items.len; // committed length at iteration start
            self.verify_buf[0] = history.items[start_len - 1];
            const verify = self.verify_buf[0 .. 1 + draft_len];

            // Restore `history.len == kv.len + 1` on EVERY error path: the
            // forward advances kv (to pos0 + 1 + draft_len) before its own
            // fallible tail ops, and the row loop appends to history as it
            // commits. history.items.len is read at unwind time; truncate
            // clamps. This also drops unverified draft rows from the cache.
            errdefer kv.truncate(history.items.len - 1);

            const t0 = self.nowNs();
            var logits = try model.forwardStepAllLogits(ctx, kv, verify, pos0);
            defer logits.deinit();
            const verify_ns: ?u64 = if (t0) |start| @intCast(self.nowNs().? - start) else null;
            const wants_topk = self.source.wantsTopK();
            var accepted: usize = 0;
            var verified: usize = 0;
            var i: usize = 0;
            while (i <= draft_len) : (i += 1) {
                var row = try logits.narrow(ctx, .seq, i, 1);
                defer row.deinit();
                if (wants_topk) {
                    const k = @min(self.options.topk_feedback, row.dim(.vocab));
                    var top = try row.topK(ctx, .vocab, k, .top);
                    defer top.deinit();
                    const ids = try top.indices.dataConst();
                    const dst = self.topk_ids[i * self.options.topk_feedback ..][0..k];
                    // Indices arrive as f32: exact integer representation only
                    // holds below 2^24 — vocab sizes (~152k max today) are far
                    // below that bound.
                    for (dst, ids[0..k]) |*d, s| d.* = @intFromFloat(s);
                    self.topk_rows[i] = .{ .token = verify[i], .topk = dst };
                }
                if (self.on_verify_row) |hook| {
                    try hook.func(hook.ptr, history.items.len, i, try row.dataConst());
                }
                // The full pipeline, conditioned on the hypothetical prefix
                // (history already holds the accepted draft tokens). Penalties
                // mutate the row in place; each row is sampled exactly once.
                const sampled = try sampler.next(ctx, &row, history.items);
                verified += 1;
                try history.append(ctx.allocator, sampled);
                try sink.emit(sampled);
                const matched = i < draft_len and sampled == verify[i + 1];
                if (matched) {
                    accepted += 1; // sampled == draft: committed verbatim
                } else if (i == draft_len) {
                    self.stats.bonus += 1; // whole draft accepted: free token
                } else {
                    self.stats.rejected_steps += 1; // sampled = correction token
                }
                if (!matched) break;
                // Stop-awareness (RNG/lossless contract): committing the stop
                // token ends the pass NOW. A plain run stops here too, so
                // sampling any further row — the bonus row included — would
                // consume RNG draws the plain run never makes.
                if (self.options.stop_token) |stop| {
                    if (sampled == stop) break;
                }
            }
            self.stats.accepted += accepted;

            // Keep every committed token except the newest one; the newest
            // token, including an accepted mid-batch stop token, enters the
            // cache on the NEXT forward.
            kv.truncate(history.items.len - 1);

            self.source.observe(history.items[start_len..]);
            if (wants_topk) self.source.observeTopK(self.topk_rows[0..verified]);

            const committed = history.items.len - start_len;
            self.gate.recordVerify(committed, accepted, draft_len, self.gate.verifyCost(draft_len, verify_ns));
            return committed;
        }
    };
}

test {
    _ = @import("core_tests.zig");
}

// The cost-gate tests below stay INLINE: they exercise non-pub `CostGate`
// internals (init/allows/recordVerify/budgetCap/verifyCost/notePlainNs/
// resetWindow/State + the cost_scale/plain_ns fields), which a sibling test
// file cannot reach. The behavioral / losslessness tests live in
// speculative_tests.zig.

test "cost gate: trips on losing verifies, exponential backoff, probe hysteresis" {
    const allocator = std.testing.allocator;
    var gate = try CostGate.init(allocator, .{
        .rate_window = 8,
        .min_window_drafted = 4,
        .reprobe_after = 4,
        .reprobe_max = 16,
        .probe_steps = 2,
    });
    defer gate.deinit(allocator);

    // One losing verify is never enough to trip (too noisy).
    try std.testing.expect(gate.allows());
    gate.recordVerify(1, 0, 8, 2.84);
    try std.testing.expectEqual(CostGate.State.on, gate.state);
    // Second losing verify: est = 2/5.68 = 0.35 < 1.0 -> off.
    gate.recordVerify(1, 0, 8, 2.84);
    try std.testing.expectEqual(CostGate.State.off, gate.state);

    // Disabled for reprobe_after steps (the 4th attempt becomes the probe).
    try std.testing.expect(!gate.allows());
    try std.testing.expect(!gate.allows());
    try std.testing.expect(!gate.allows());
    try std.testing.expect(gate.allows());
    try std.testing.expectEqual(CostGate.State.probing, gate.state);

    // Failed probe doubles the backoff: 4 -> 8.
    gate.recordVerify(1, 0, 8, 2.84);
    gate.recordVerify(1, 0, 8, 2.84);
    try std.testing.expectEqual(CostGate.State.off, gate.state);
    try std.testing.expectEqual(@as(usize, 8), gate.backoff);
    for (0..7) |_| try std.testing.expect(!gate.allows());
    try std.testing.expect(gate.allows());

    // Marginal probe (est between min_speedup and +probe_margin) fails the
    // hysteresis: est = 4/3.8 = 1.05 in [1.0, 1.1) -> still off; 8 -> 16.
    gate.recordVerify(2, 1, 2, 1.9);
    gate.recordVerify(2, 1, 2, 1.9);
    try std.testing.expectEqual(CostGate.State.off, gate.state);
    try std.testing.expectEqual(@as(usize, 16), gate.backoff);

    // Backoff caps at reprobe_max.
    for (0..15) |_| try std.testing.expect(!gate.allows());
    try std.testing.expect(gate.allows());
    gate.recordVerify(1, 0, 8, 2.84);
    gate.recordVerify(1, 0, 8, 2.84);
    try std.testing.expectEqual(@as(usize, 16), gate.backoff); // min(32, 16)

    // Winning probe passes the hysteresis and re-enables.
    for (0..15) |_| try std.testing.expect(!gate.allows());
    try std.testing.expect(gate.allows());
    gate.recordVerify(6, 5, 8, 2.84); // est = 12/5.68 = 2.11 >= 1.1
    gate.recordVerify(6, 5, 8, 2.84);
    try std.testing.expectEqual(CostGate.State.on, gate.state);

    // In the ON state the margin does NOT apply: est in [1.0, 1.1) stays on.
    gate.resetWindow();
    gate.recordVerify(2, 1, 4, 1.9);
    gate.recordVerify(2, 1, 4, 1.9); // est = 4/3.8 = 1.05 >= 1.0
    try std.testing.expectEqual(CostGate.State.on, gate.state);
    // A fresh trip from ON restarts the backoff at the base wait (the window
    // is already evaluable, so one clearly losing verify tips the estimate).
    gate.recordVerify(1, 0, 8, 4.5); // est = 5/8.3 = 0.60 < 1.0
    try std.testing.expectEqual(CostGate.State.off, gate.state);
    try std.testing.expectEqual(@as(usize, 4), gate.backoff);
}

test "cost gate: acceptance-adaptive draft budget" {
    const allocator = std.testing.allocator;
    var gate = try CostGate.init(allocator, .{});
    defer gate.deinit(allocator);

    // Empty window: full budget.
    try std.testing.expectEqual(@as(usize, 16), gate.budgetCap(16, 2));
    // 25% rolling acceptance: 2 + ceil(0.25 * 16) = 6.
    gate.recordVerify(3, 2, 8, 2.84);
    gate.recordVerify(2, 0, 0, 1.0); // bonus-only entry, no drafts
    gate.recordVerify(1, 0, 0, 1.0);
    try std.testing.expectEqual(@as(usize, 6), gate.budgetCap(16, 2));
    // Zero acceptance floors at max(min_draft, 2).
    gate.resetWindow();
    gate.recordVerify(1, 0, 8, 2.84);
    try std.testing.expectEqual(@as(usize, 2), gate.budgetCap(16, 2));
    try std.testing.expectEqual(@as(usize, 4), gate.budgetCap(16, 4));
    // Full acceptance: cap clamps to max_draft.
    gate.resetWindow();
    gate.recordVerify(17, 16, 16, 4.5);
    try std.testing.expectEqual(@as(usize, 16), gate.budgetCap(16, 2));
}

test "cost gate: live cost model rescales the static table, robust to outliers" {
    const allocator = std.testing.allocator;
    var gate = try CostGate.init(allocator, .{});
    defer gate.deinit(allocator);

    // No plain-step sample yet: static table, scale untouched.
    try std.testing.expectApproxEqAbs(@as(f32, 2.84), gate.verifyCost(8, 2_500_000), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), gate.cost_scale, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f32, 2.84), gate.verifyCost(8, null), 1e-6);

    // A measurement that matches the table keeps the scale at 1.
    gate.notePlainNs(1_000_000);
    try std.testing.expectApproxEqAbs(@as(f32, 2.84), gate.verifyCost(8, 2_840_000), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), gate.cost_scale, 1e-6);

    // A single 2x outlier moves the cost by scale_alpha (20%), not 100%:
    // sample 2.0 -> scale 0.8*1.0 + 0.2*2.0 = 1.2 -> cost 1.2 * 2.84.
    try std.testing.expectApproxEqAbs(@as(f32, 1.2 * 2.84), gate.verifyCost(8, 5_680_000), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), gate.cost_scale, 1e-6);

    // Sustained 2x measurements converge the scale toward 2 (model truly
    // differs from the table) — and the learned scale also applies when a
    // step has no measurement.
    for (0..40) |_| _ = gate.verifyCost(8, 5_680_000);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), gate.cost_scale, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 * 2.84), gate.verifyCost(8, null), 1e-2);

    // Extreme samples are clamped before entering the EWMA.
    var gate2 = try CostGate.init(allocator, .{});
    defer gate2.deinit(allocator);
    gate2.notePlainNs(1_000_000);
    _ = gate2.verifyCost(8, 1_000_000_000); // 1000x outlier -> sample clamped to 4
    try std.testing.expectApproxEqAbs(@as(f64, 0.8 + 0.2 * 4.0), gate2.cost_scale, 1e-6);

    // A verify can never be cheaper than the plain step it replaces.
    var gate3 = try CostGate.init(allocator, .{});
    defer gate3.deinit(allocator);
    gate3.notePlainNs(1_000_000);
    for (0..60) |_| _ = gate3.verifyCost(2, 100_000); // measured below 1 clamps to 1
    try std.testing.expect(gate3.verifyCost(2, 100_000) >= 1.0);

    // Plain EWMA: 0.25 * new + 0.75 * old.
    gate.notePlainNs(2_000_000);
    try std.testing.expectApproxEqAbs(@as(f64, 1_250_000), gate.plain_ns, 1.0);
}
