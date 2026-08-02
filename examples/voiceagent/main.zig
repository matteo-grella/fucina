//! Native cascade voice agent in a TUI: microphone → Parakeet streaming STT
//! (learned <EOU> endpointing) → in-process Qwen3 chat → Qwen3-TTS →
//! speakers. Every stage is a Fucina port running in this one process — no
//! Python, no server, no browser.
//!
//!   zig build voiceagent -- \
//!       --asr models/parakeet/realtime_eou_120m-v1-f16.gguf \
//!       --chat models/Qwen3-0.6B-Q8_0.gguf \
//!       --tts models/qwen3-tts/qwen-talker-0.6b-customvoice-F32.gguf \
//!       --codec models/qwen3-tts/qwen-tokenizer-12hz-F32.gguf \
//!       [--speaker Aiden] [--lang english] [--system "…"] [--threads N]
//!
//! Half-duplex turn-taking (the reference speech-to-speech local client's
//! discipline): the mic streams while LISTENING; once the EOU model closes
//! the utterance the agent THINKS (streaming the reply text) and SPEAKS
//! (TTS frames decode chunk-by-chunk into the playback ring while the talker
//! keeps generating); the mic ring is drained and re-armed afterwards.
//! The reply text is revealed word-by-word in sync with the voice speaking
//! it (`--eager-text` streams it during THINK instead, ahead of the audio).
//! You can also type. The live transcript shows dim as a suggestion: `→`
//! accepts it into the line, anything else starts clean, and the turn then
//! ends on Enter. Any key interrupts a reply in progress; Ctrl-C quits.
//!
//! Live audio I/O — manually verified (no automated parity; CI has no audio
//! device). The model stages themselves are pinned by the qwen3tts parity
//! suites and the parakeet/qwen3 golden tests.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");
const audio_mod = @import("nam_audio");
const play_mod = @import("omnivoice_play");
const duplex = @import("duplex.zig");
const rail_mod = @import("rail.zig");
const aec_mod = @import("aec.zig");

const qtts = llm.qwen3tts;
const parakeet_loader = llm.parakeet.loader;
const parakeet_frontend = llm.parakeet.frontend;
const parakeet_streaming = llm.parakeet.streaming;
const parakeet_weights = llm.parakeet.weights;
const parakeet_tokenizer = llm.parakeet.tokenizer;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;

const usage =
    \\usage: zig build voiceagent -- --asr <parakeet-eou.gguf> --chat <qwen3.gguf>
    \\         --tts <talker.gguf> --codec <codec.gguf>
    \\         [--speaker NAME] [--lang NAME] [--system PROMPT] [--eager-text]
    \\         [--seed N] [--max-reply N] [--threads N] [--mic-device N] [--list-devices]
    \\         [--sim user.wav [--sim-barge over-reply.wav]]  (headless scripted run)
    \\
;

// --- small arg helpers ------------------------------------------------------

fn flagVal(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |a, i| if (std.mem.eql(u8, a, name) and i + 1 < args.len) return args[i + 1];
    return null;
}
fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a, name)) return true;
    return false;
}
fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

/// Last line of defense against transcribing our own playback: a short
/// utterance whose words mostly fuzzy-match the agent's last reply is echo
/// leakage (reverb tail, imperfect cancellation), not the user.
fn echoGuard(utterance: []const u8, last_reply: []const u8, strict: bool) bool {
    if (last_reply.len == 0) return false;
    var words: usize = 0;
    var matched: usize = 0;
    var it = std.mem.tokenizeAny(u8, utterance, BargeCtx.word_delims);
    while (it.next()) |w| {
        words += 1;
        var rt = std.mem.tokenizeAny(u8, last_reply, BargeCtx.word_delims);
        while (rt.next()) |rw| {
            if (BargeCtx.editLe1(w, rw)) {
                matched += 1;
                break;
            }
        }
    }
    // Inside the echo-tail window (utterance completed right at the turn
    // boundary) mis-transcribed echo words dilute the match — be stricter.
    const threshold: usize = if (strict) 4 else 7;
    return words >= 2 and words <= 12 and matched * 10 >= words * threshold;
}

/// Valid cross-correlation `out[m] = Σ_n sig[m+n]·kern[n]`, as ONE core
/// `conv1d` (PyTorch Conv1d semantics are cross-correlation — no kernel flip).
///
/// The kernel taps are split across `chans` OUTPUT channels rather than left
/// as a single 1×1-channel conv: the backend kernel vectorizes over
/// out-channels (weights are out-channel contiguous), so a one-channel conv
/// wastes the whole vector width. Channel `c` carries taps `[c·T, (c+1)·T)`,
/// so its contribution to lag `m` sits at time `m + c·T` and the split is
/// undone by a strided sum — `out[m] = Σ_c y[m + c·T][c]`.
fn correlate(ctx: *ExecContext, sig: []f32, kern: []f32, out: []f32) !void {
    const chans = 16;
    const taps = kern.len / chans;
    std.debug.assert(taps * chans == kern.len);
    std.debug.assert(sig.len >= out.len + kern.len - 1);

    const wbuf = try ctx.allocator.alloc(f32, taps * chans);
    defer ctx.allocator.free(wbuf);
    // weight[(k*in_per_group + i)*out + o], in_per_group = 1.
    for (0..chans) |c| {
        for (0..taps) |k| wbuf[k * chans + c] = kern[c * taps + k];
    }

    var sig_t = try Tensor(.{ .time, .ch }).fromBorrowedSlice(ctx, .{ sig.len, 1 }, sig);
    defer sig_t.deinit();
    var kern_t = try Tensor(.{ .tap, .ch, .out }).fromBorrowedSlice(ctx, .{ taps, 1, chans }, wbuf);
    defer kern_t.deinit();
    var y = try sig_t.conv1d(ctx, .time, .ch, .tap, .out, &kern_t, 1, 0, 1, 1);
    defer y.deinit();
    const yd = try y.dataConst();

    for (out, 0..) |*o, m| {
        var acc: f32 = 0;
        for (0..chans) |c| acc += yd[(m + c * taps) * chans + c];
        o.* = acc;
    }
}

// --- AEC pump: 48 kHz mic+reference rings → 16 kHz echo-cancelled hops ------

const AecPump = struct {
    engine: *duplex.Engine,
    sess: ?*aec_mod.Session, // null = AEC off (raw mic passthrough)
    ctx: *ExecContext, // owns the delay-scan conv1d; not shared across threads
    dec_mic: duplex.Decimator3,
    dec_ref: duplex.Decimator3,
    mic16: [4096]f32 = undefined,
    ref16: [4096]f32 = undefined,
    fill: usize = 0,
    // barge-in gate state
    floor: f32 = 1e-4,
    hot_hops: usize = 0,
    lag_votes: [76]u8 = @splat(0), // 4 ms bins over 0..300 ms
    votes_total: usize = 0,
    converge_hops: usize = 0, // AEC reconvergence hold after a (re)lock reset
    raw_ema: f32 = 0, // mic level (pre-AEC), for the TUI meter
    res_ema: f32 = 0, // residual level (post-AEC)
    ncc_measured: bool = false, // a correlation scan ran WITH far-end energy
    // Bulk-delay alignment (the piece production stacks pair with every
    // canceller): the acoustic+buffer echo path delays the mic's copy of the
    // playback by tens of ms, and GTCRN has no align block — so estimate the
    // bulk lag by cross-correlation and delay the REFERENCE to match.
    ref_hist: [16384]f32 = @splat(0), // ~1 s @ 16 kHz
    mic_hist: [16384]f32 = @splat(0),
    abs_in: usize = 0, // absolute 16 kHz samples pushed (mic/ref aligned)
    hop_start: usize = 0, // absolute index of the next hop to consume
    next_est: usize = 0,
    delay16: usize = 0,
    locked: bool = false,
    echo_ncc: f32 = 0, // decaying best NCC — "playback leaks into mic" signal
    lock_ms_event: ?usize = null, // set on (re)lock for the TUI to report
    debug: bool = false, // --aec-debug: trace lock/gate decisions to stderr

    /// Seconds of captured audio so far — the pump's own clock, independent
    /// of wall time (and of the sim's real-time pacing).
    fn clock(self: *const AecPump) f64 {
        return @as(f64, @floatFromInt(self.abs_in)) / 16000.0;
    }

    fn init(engine: *duplex.Engine, sess: ?*aec_mod.Session, ctx: *ExecContext) AecPump {
        return .{ .engine = engine, .sess = sess, .ctx = ctx, .dec_mic = duplex.Decimator3.init(), .dec_ref = duplex.Decimator3.init() };
    }

    /// Drain the 48 kHz rings in lockstep, decimate, run the AEC per
    /// 256-sample hop, and call `consume(residual_hop)` for each hop.
    /// Returns the RMS of the loudest processed hop (0 if none).
    fn pump(self: *AecPump, consume: anytype) !f32 {
        var raw_mic: [3072]f32 = undefined;
        var raw_ref: [3072]f32 = undefined;
        var loudest: f32 = 0;
        while (true) {
            const avail = @min(self.engine.mic48.len(), self.engine.ref48.len());
            const want = @min(avail, raw_mic.len);
            if (want < 3) break;
            const gm = self.engine.mic48.pop(raw_mic[0..want]);
            const gr = self.engine.ref48.pop(raw_ref[0..gm]);
            std.debug.assert(gm == gr);
            const nm = self.dec_mic.run(raw_mic[0..gm], self.mic16[self.fill..]);
            const nr = self.dec_ref.run(raw_ref[0..gr], self.ref16[self.fill..]);
            std.debug.assert(nm == nr);
            self.fill += nm;
            const mask = self.ref_hist.len - 1;
            for (0..nm) |k| {
                self.ref_hist[(self.abs_in + k) & mask] = self.ref16[self.fill - nm + k];
                self.mic_hist[(self.abs_in + k) & mask] = self.mic16[self.fill - nm + k];
            }
            self.abs_in += nm;
            if (self.abs_in >= self.next_est) {
                // ~every 128 ms: one core conv1d (~1.6 ms), so ~1.2% of a
                // core. The cadence bounds how fast the vote histogram can
                // reach a lock, and barge-in cannot fire before it does.
                self.next_est = self.abs_in + 2048;
                self.estimateDelay();
            }
            while (self.fill >= aec_mod.hop) {
                var res: [aec_mod.hop]f32 = undefined;
                var raw_acc: f32 = 0;
                for (self.mic16[0..aec_mod.hop]) |v| raw_acc += v * v;
                self.raw_ema = 0.9 * self.raw_ema + 0.1 * @sqrt(raw_acc / aec_mod.hop);
                if (self.sess) |sess| {
                    var ref_al: [aec_mod.hop]f32 = undefined;
                    for (0..aec_mod.hop) |n| {
                        const idx = self.hop_start + n;
                        ref_al[n] = if (idx >= self.delay16) self.ref_hist[(idx - self.delay16) & mask] else 0;
                    }
                    try sess.step(self.mic16[0..aec_mod.hop], &ref_al, &res);
                } else {
                    @memcpy(&res, self.mic16[0..aec_mod.hop]);
                }
                self.hop_start += aec_mod.hop;
                var acc: f32 = 0;
                for (res) |v| acc += v * v;
                const rms = @sqrt(acc / aec_mod.hop);
                self.res_ema = 0.9 * self.res_ema + 0.1 * rms;
                loudest = @max(loudest, rms);
                try consume.hop(&res, rms);
                std.mem.copyForwards(f32, self.mic16[0 .. self.fill - aec_mod.hop], self.mic16[aec_mod.hop..self.fill]);
                std.mem.copyForwards(f32, self.ref16[0 .. self.fill - aec_mod.hop], self.ref16[aec_mod.hop..self.fill]);
                self.fill -= aec_mod.hop;
            }
        }
        return loudest;
    }

    /// Cross-correlate the last 100 ms of mic against the reference over
    /// 0..300 ms of lag. Two consistent confident peaks lock the bulk delay
    /// (resetting the canceller state on change). Silent when either side is
    /// quiet.
    ///
    /// All 4801 lags are ONE core `conv1d`: PyTorch Conv1d semantics are
    /// cross-correlation with no kernel flip, so a `[6400, 1]` reference
    /// against a `[1600, 1, 1]` mic kernel yields `corr[m] = Σ ref[m+n]·mic[n]`
    /// directly, and the lag-l score is `corr[max_lag - l]`. The per-lag
    /// reference energy stays a running scalar update — O(1) per lag, nothing
    /// for a tensor op to win.
    fn estimateDelay(self: *AecPump) void {
        const W = 1600; // 100 ms
        const max_lag = 4800; // 300 ms
        const mask = self.ref_hist.len - 1;
        if (self.abs_in < W + max_lag) return;
        const end = self.abs_in;
        var micw: [W]f32 = undefined;
        var mic_e: f32 = 0;
        for (0..W) |n| {
            const v = self.mic_hist[(end - W + n) & mask];
            micw[n] = v;
            mic_e += v * v;
        }
        var refc: [W + max_lag]f32 = undefined;
        for (0..refc.len) |n| refc[n] = self.ref_hist[(end - refc.len + n) & mask];
        if (mic_e < 1e-4) {
            self.echo_ncc *= 0.5;
            return;
        }
        var corr_buf: [max_lag + 1]f32 = undefined;
        correlate(self.ctx, &refc, &micw, &corr_buf) catch return;
        const corr: []const f32 = &corr_buf;

        var ref_e: f32 = 0;
        for (refc[max_lag..]) |v| ref_e += v * v;
        var best_ncc: f32 = 0;
        var best_lag: usize = 0;
        var lag: usize = 0;
        var e = ref_e;
        while (lag <= max_lag) : (lag += 1) {
            if (lag > 0) {
                const a = refc[max_lag - lag];
                const d = refc[refc.len - lag];
                e += a * a - d * d;
            }
            if (e > 1e-4) {
                const ncc = corr[max_lag - lag] / (@sqrt(mic_e) * @sqrt(e));
                if (ncc > best_ncc) {
                    best_ncc = ncc;
                    best_lag = lag;
                }
            }
        }
        self.ncc_measured = true; // scan ran with mic energy + far activity
        self.echo_ncc = @max(best_ncc, self.echo_ncc * 0.7);
        if (self.debug) std.debug.print("[aec-dbg] t={d:.2}s scan ncc={d:.3} lag={d}ms locked={}\n", .{ self.clock(), best_ncc, best_lag * 1000 / 16000, self.locked });
        if (best_ncc < 0.15) return;
        // Histogram vote over 4 ms bins: a bin reaching 3, counting immediate
        // neighbours, locks the delay. Reverberant rooms smear a single-shot
        // correlation across bins, so a marginal peak (ncc just over 0.15) can
        // land in the wrong one and needs corroborating.
        //
        // Vote weight tracks confidence, and only for DISCOVERY: from no lock
        // at all, an unambiguous peak identifies the echo path outright and
        // counts 3 on its own, because Stage 1 of barge-in is gated on
        // `locked` and cannot fire until it holds. Revising an already-locked
        // delay stays at one vote per scan: during double-talk the peak
        // legitimately wanders (the near-end speech correlates too), and every
        // re-lock resets the canceller and holds the gate ~0.5 s.
        const bin = @min(best_lag / 64, self.lag_votes.len - 1);
        const vote: u8 = if (best_ncc >= 0.5 and !self.locked) 3 else 1;
        self.lag_votes[bin] = self.lag_votes[bin] +| vote;
        self.votes_total += 1;
        if (self.votes_total > 24) {
            for (&self.lag_votes) |*v| v.* -= v.* / 4;
            self.votes_total = 0;
        }
        var mass: usize = self.lag_votes[bin];
        if (bin > 0) mass += self.lag_votes[bin - 1];
        if (bin + 1 < self.lag_votes.len) mass += self.lag_votes[bin + 1];
        if (mass >= 3) {
            const drift = @max(self.delay16, best_lag) - @min(self.delay16, best_lag);
            if (!self.locked or drift > 320) {
                self.delay16 = best_lag;
                self.locked = true;
                if (self.sess) |sx| sx.reset();
                self.lock_ms_event = best_lag * 1000 / 16000;
                self.lag_votes = @splat(0);
                self.votes_total = 0;
                // A freshly reset canceller passes echo while it reconverges:
                // hold the gate for ~0.5 s so the transient cannot fire it.
                self.hot_hops = 0;
                self.converge_hops = 30;
                if (self.debug) std.debug.print("[aec-dbg] t={d:.2}s LOCKED lag={d}ms\n", .{ self.clock(), best_lag * 1000 / 16000 });
            }
        }
    }

    /// Sustained-speech gate on the residual: RMS above an adaptive noise
    /// floor for >= `need` consecutive hops. Never fires on uncancelled
    /// echo: while playback demonstrably leaks into the mic (high NCC) but
    /// the bulk delay is not locked yet, the gate holds.
    fn gate(self: *AecPump, rms: f32, need: usize) bool {
        // Reconvergence hold decays with TIME (every hop) — decrementing
        // only on hot hops sticks forever once the canceller goes quiet,
        // and the zero-feed would silence the STT permanently.
        const converging = self.converge_hops > 0;
        if (converging) self.converge_hops -= 1;
        const hot = rms > @max(4.0 * self.floor, 0.010);
        if (!hot) {
            self.floor = 0.98 * self.floor + 0.02 * rms;
            self.hot_hops = 0;
            return false;
        }
        if (converging) return false;
        self.hot_hops += 1;
        return self.hot_hops >= need;
    }

    fn resetGate(self: *AecPump) void {
        self.hot_hops = 0;
    }
};

// --- incremental STT driver (the parakeet example's, EOU-aware) -------------

const IncrementalStreamer = struct {
    sess: *parakeet_streaming.StreamingSession,
    ctx: *ExecContext,
    file: *const fucina.gguf.File,
    weights: *parakeet_weights.ParakeetWeights,
    arena: std.mem.Allocator,
    feat: parakeet_loader.Featurizer,
    dft_basis: *parakeet_frontend.DftBasis,
    mel_params: parakeet_frontend.MelParams,
    n_mels: usize,
    chunk0: usize,
    chunk_main: usize,
    pre_cache: usize,
    tail_margin: usize,
    samples: std.ArrayList(f32) = .empty,
    buffer_idx: usize = 0,
    first: bool = true,

    fn feed(self: *IncrementalStreamer, new_samples: []const f32, is_final: bool) !void {
        if (new_samples.len > 0) try self.samples.appendSlice(self.arena, new_samples);
        // Bounded window: cap the mel buffer at 12 s (trim to 10 s) so a
        // silent open mic cannot grow per-feed cost without bound. Normal
        // utterances never hit the cap, so per-feature normalization sees
        // the same whole-utterance statistics as the parakeet example —
        // trimming tighter destabilizes the features and the EOU with them.
        const max_samples = 12 * 16000;
        if (self.samples.items.len > max_samples) {
            const hop = self.mel_params.stft.hop;
            var drop_frames = (self.samples.items.len - 10 * 16000) / hop;
            const consumed_margin = if (self.buffer_idx > self.pre_cache) self.buffer_idx - self.pre_cache else 0;
            drop_frames = @min(drop_frames, consumed_margin);
            const drop_samples = drop_frames * hop;
            if (drop_samples > 0) {
                const rest = self.samples.items.len - drop_samples;
                std.mem.copyForwards(f32, self.samples.items[0..rest], self.samples.items[drop_samples..]);
                self.samples.shrinkRetainingCapacity(rest);
                self.buffer_idx -= drop_frames;
            }
        }
        if (self.samples.items.len < self.mel_params.stft.n_fft) {
            if (!is_final) return;
        }
        var mel = try parakeet_frontend.melSpectrogramFastWithBasis(self.arena, self.samples.items, self.mel_params, self.feat.fb, self.feat.window, self.dft_basis);
        defer mel.deinit(self.arena);
        const t = mel.n_frames;
        const stable_t = if (is_final) t else (if (t > self.tail_margin) t - self.tail_margin else 0);

        while (self.buffer_idx < t) {
            const chunk_size = if (self.first) self.chunk0 else self.chunk_main;
            const chunk_hi = @min(self.buffer_idx + chunk_size, t);
            if (chunk_hi <= self.buffer_idx) break;
            if (!is_final and chunk_hi > stable_t) break;
            const lo = if (self.first) self.buffer_idx else (if (self.buffer_idx > self.pre_cache) self.buffer_idx - self.pre_cache else 0);
            const win_frames = chunk_hi - lo;
            const is_last = is_final and (chunk_hi >= t);
            const win = try self.arena.alloc(f32, self.n_mels * win_frames);
            defer self.arena.free(win);
            for (0..self.n_mels) |m| {
                for (0..win_frames) |tt| win[m * win_frames + tt] = mel.feats[m * t + (lo + tt)];
            }
            try self.sess.feedMelChunk(self.ctx, self.file, self.weights, win, self.n_mels, win_frames, is_last);
            self.buffer_idx += chunk_size;
            self.first = false;
        }
    }

    fn reset(self: *IncrementalStreamer) void {
        self.samples.clearRetainingCapacity();
        self.buffer_idx = 0;
        self.first = true;
    }
};

// --- barge context: residual-fed STT across THINK/SPEAK/LISTEN --------------
//
// One continuous consumer of AEC-residual hops for the whole conversation.
// The robustness principle: we KNOW the exact text the agent is speaking, so
// interruption is decided in the TEXT domain — the streaming STT runs on the
// residual while the agent thinks and speaks, and only words that are NOT
// part of the reply being spoken count as the user. Echo mis-transcriptions
// reproduce the reply's own words and are filtered, so even an
// underperforming AEC cannot false-fire the turn. The same STT session then
// carries into LISTEN, so the interrupting utterance continues seamlessly to
// its EOU — nothing the user said is lost and no splice replay is needed.

const BargeCtx = struct {
    pump: *AecPump,
    streamer: *IncrementalStreamer,
    engine: *duplex.Engine,
    pieces: []const []const u8,
    reply_acc: *const std.ArrayList(u8), // text generated so far this turn
    allocator: std.mem.Allocator,

    phase: enum { idle, think, speak } = .idle,
    stage: enum { none, paused } = .none, // pause-then-commit ladder
    pending: bool = false, // COMMIT decided (Enter uses the atomic instead)
    hops_seen: usize = 0,
    pause_hop: usize = 0,
    last_hot: usize = 0, // hop index of the most recent above-floor residual
    far_hot_until: usize = 0, // playback recently live (ref window active)
    last_token_count: usize = 0,
    prev_max_run: usize = 0, // novel-run stability across token updates
    strip_words: usize = 0, // leading echo words to drop from a carried utterance
    batch: [aec_mod.hop * 200]f32 = undefined, // up to ~3.2 s of postponed audio
    fill: usize = 0,

    const flush_hops = aec_mod.hop * 3; // ~48 ms normal batch — snappy partials
    /// Backchannels never confirm a barge (and never count as novel).
    const backchannels = [_][]const u8{ "yeah", "ok", "okay", "right", "sure", "uh", "huh", "hmm", "mhm", "yes", "yep", "aha" };
    /// Command words commit immediately when energy-backed.
    const commands = [_][]const u8{ "stop", "wait", "no", "hold", "pause", "quiet", "enough" };

    const Consumer = struct {
        ctx: *BargeCtx,
        fn hop(c: *const @This(), res: []const f32, rms: f32) !void {
            const b = c.ctx;
            b.hops_seen += 1;
            const sustained = b.pump.gate(rms, 12);
            if (b.pump.hot_hops > 0) b.last_hot = b.hops_seen;
            if (b.engine.outPending() > 0) b.far_hot_until = b.hops_seen + 60; // ~1 s: bulk delay + reverb tail

            // Stage 1 (fast, reversible): sustained residual energy PAUSES
            // playback — killing the echo path within ~200 ms so the STT
            // hears the user cleanly — while generation keeps queueing.
            // Only on EVIDENCE: canceller locked, or a MEASURED no-leak
            // (headphones). An unmeasured 0 is not "no leak" — pausing on
            // uncancelled first-reply echo starves the estimator forever.
            if (b.phase == .speak and b.stage == .none and !b.pending and sustained and
                b.engine.outPending() > 0 and
                (b.pump.locked or (b.pump.ncc_measured and b.pump.echo_ncc <= 0.08)))
            {
                b.engine.setPaused(true);
                b.stage = .paused;
                b.pause_hop = b.hops_seen;
                if (b.pump.debug) std.debug.print("[aec-dbg] t={d:.2}s STAGE1 pause (hot={d})\n", .{ b.pump.clock(), b.pump.hot_hops });
            }
            // Why a sustained-energy hop did NOT pause: the single most
            // useful line when barge-in silently does nothing.
            if (b.pump.debug and b.phase == .speak and b.stage == .none and sustained and b.engine.outPending() > 0 and
                !(b.pump.locked or (b.pump.ncc_measured and b.pump.echo_ncc <= 0.08)))
            {
                std.debug.print("[aec-dbg] t={d:.2}s BLOCKED: speech held but locked={} ncc_measured={} echo_ncc={d:.3}\n", .{ b.pump.clock(), b.pump.locked, b.pump.ncc_measured, b.pump.echo_ncc });
            }
            // No confirming words within ~2 s: resume playback (a false
            // trigger costs a hiccup, not the reply).
            if (b.stage == .paused and !b.pending and b.hops_seen > b.pause_hop + 125) {
                b.engine.setPaused(false);
                b.stage = .none;
                b.pump.resetGate();
                b.prev_max_run = 0;
            }

            if (b.fill == b.batch.len) {
                if (b.phase == .speak and b.stage == .none) {
                    b.fill = 0; // unpaused playback: echo-era audio, not worth a splice-free feed
                } else try b.flush();
            }
            if (b.pump.converge_hops > 0) {
                // A reconverging canceller passes echo nearly unmasked; the
                // STT must not transcribe it. Zeros keep the stream
                // contiguous while silencing the transient.
                @memset(b.batch[b.fill..][0..res.len], 0);
            } else {
                @memcpy(b.batch[b.fill..][0..res.len], res);
            }
            b.fill += res.len;
            if (b.fill < flush_hops) return;
            // While the agent audibly speaks, the STT does not run AT ALL:
            // the talker keeps its whole real-time budget, and playback-era
            // echo can never be transcribed, by construction. The STT hears
            // THINK, LISTEN, and pause windows (echo dead, user clean).
            if (b.phase == .speak and b.stage == .none) return;
            try b.flush();
        }
    };

    /// Drain the rings through the AEC, feeding the STT and (in THINK/SPEAK)
    /// evaluating the barge condition.
    fn scan(self: *BargeCtx) !void {
        _ = try self.pump.pump(&Consumer{ .ctx = self });
    }

    fn flush(self: *BargeCtx) !void {
        if (self.fill == 0) return;
        try self.streamer.feed(self.batch[0..self.fill], false);
        self.fill = 0;
        if (self.phase != .idle) try self.evalNovel();
    }

    fn isIn(word: []const u8, comptime list: []const []const u8) bool {
        inline for (list) |l| if (std.ascii.eqlIgnoreCase(word, l)) return true;
        return false;
    }

    /// Case-insensitive edit distance <= 1 (ASR mangles one char often).
    fn editLe1(a: []const u8, b: []const u8) bool {
        if (a.len == b.len) {
            var diff: usize = 0;
            for (a, b) |x, y| {
                if (std.ascii.toLower(x) != std.ascii.toLower(y)) diff += 1;
            }
            return diff <= 1;
        }
        const long = if (a.len > b.len) a else b;
        const short = if (a.len > b.len) b else a;
        if (long.len != short.len + 1) return false;
        var i: usize = 0;
        var j: usize = 0;
        var skipped = false;
        while (j < short.len) {
            if (std.ascii.toLower(long[i]) == std.ascii.toLower(short[j])) {
                i += 1;
                j += 1;
            } else if (!skipped) {
                skipped = true;
                i += 1;
            } else return false;
        }
        return true;
    }

    const word_delims = " \t\n,.!?;:\"()-";

    /// Does `w` fuzzy-match any whole word of the reply being spoken?
    fn matchesReply(self: *const BargeCtx, w: []const u8) bool {
        if (self.phase != .speak) return false; // think: nothing spoken yet
        var it = std.mem.tokenizeAny(u8, self.reply_acc.items, word_delims);
        while (it.next()) |rw| {
            if (editLe1(w, rw)) return true;
        }
        return false;
    }

    const Novelty = struct { total: usize, max_run: usize, command: bool, run_start: usize };

    /// Classify the current transcript: novel words (not echo of the reply,
    /// not backchannels), their longest consecutive run, command hits.
    fn classify(self: *BargeCtx) !Novelty {
        const toks = self.streamer.sess.tokens.items;
        var out = Novelty{ .total = 0, .max_run = 0, .command = false, .run_start = 0 };
        if (toks.len == 0) return out;
        const text = try parakeet_tokenizer.detokenize(self.allocator, self.pieces, toks);
        defer self.allocator.free(text);
        var run: usize = 0;
        var word_idx: usize = 0;
        var it = std.mem.tokenizeAny(u8, text, word_delims);
        while (it.next()) |w| {
            defer word_idx += 1;
            if (w.len < 2 or isIn(w, &backchannels) or self.matchesReply(w)) {
                run = 0;
                continue;
            }
            if (isIn(w, &commands)) out.command = true;
            out.total += 1;
            run += 1;
            if (run > out.max_run) {
                out.max_run = run;
                out.run_start = word_idx + 1 - run;
            }
        }
        return out;
    }

    /// COMMIT rules — words the agent is not saying, cross-checked with the
    /// acoustics. Never on words alone from a leaking, unaligned canceller.
    fn evalNovel(self: *BargeCtx) !void {
        const toks = self.streamer.sess.tokens.items;
        if (toks.len == self.last_token_count or toks.len == 0) return;
        self.last_token_count = toks.len;
        const nv = try self.classify();
        const stable = nv.max_run >= 2 and self.prev_max_run >= 1;
        self.prev_max_run = nv.max_run;
        // While playback leaks into an unlocked canceller, the transcript IS
        // the echo — only ref-silence or a lock make words trustworthy.
        const far_live = self.hops_seen < self.far_hot_until;
        if (!self.pump.locked and far_live and (!self.pump.ncc_measured or self.pump.echo_ncc > 0.08)) return;
        const energy_recent = self.hops_seen < self.last_hot + 60; // ~1 s
        const commit =
            (nv.command and (self.stage == .paused or energy_recent)) or
            (stable and energy_recent) or
            (nv.total >= 1 and self.pump.locked and self.pump.hot_hops >= 12);
        if (commit) {
            if (self.pump.debug) std.debug.print("[aec-dbg] t={d:.2}s COMMIT (novel={d} run={d} cmd={})\n", .{ self.pump.clock(), nv.total, nv.max_run, nv.command });
            self.pending = true;
            // Echo mis-transcriptions BEFORE the confirmed novel run are the
            // agent's own words — slice them off the carried utterance.
            self.strip_words = nv.run_start;
        }
    }

    /// Words in the current transcript the agent is not saying (tail-carry).
    fn novelCount(self: *BargeCtx) !usize {
        return (try self.classify()).total;
    }

    /// Fresh turn: fresh counters. `keep_session` carries an in-flight
    /// interrupted utterance into LISTEN instead.
    fn beginTurn(self: *BargeCtx, keep_session: bool) void {
        self.pending = false;
        self.phase = .idle;
        self.stage = .none;
        self.engine.setPaused(false);
        self.hops_seen = 0;
        self.pause_hop = 0;
        self.last_hot = 0;
        self.far_hot_until = 0;
        self.prev_max_run = 0;
        self.pump.resetGate();
        if (!keep_session) {
            self.last_token_count = 0;
            self.fill = 0;
            self.strip_words = 0;
        }
    }
};

/// Skip the first `n` words of `text`, returning the remainder slice.
fn skipWords(text: []const u8, n: usize) []const u8 {
    if (n == 0) return text;
    var it = std.mem.tokenizeAny(u8, text, BargeCtx.word_delims);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (it.next() == null) return "";
    }
    return std.mem.trimStart(u8, it.rest(), BargeCtx.word_delims);
}

// --- reply writer: streams chat tokens to the TUI + accumulates -------------

const ReplyWriter = struct {
    interface: std.Io.Writer,
    arena: std.mem.Allocator,
    text: *std.ArrayList(u8), // shared with BargeCtx.reply_acc
    tui: *Tui,
    barge: *BargeCtx,
    rail: *RailCtl,

    fn init(arena: std.mem.Allocator, tui: *Tui, barge: *BargeCtx, rail: *RailCtl, text: *std.ArrayList(u8), buffer: []u8) ReplyWriter {
        return .{
            .interface = .{ .vtable = &.{ .drain = drain }, .buffer = buffer },
            .arena = arena,
            .tui = tui,
            .barge = barge,
            .rail = rail,
            .text = text,
        };
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        _ = splat;
        const self: *ReplyWriter = @alignCast(@fieldParentPtr("interface", w));
        var n: usize = 0;
        const buffered = w.buffered();
        if (buffered.len > 0) {
            self.text.appendSlice(self.arena, buffered) catch return error.WriteFailed;
            if (self.tui.eager_text) self.tui.assistantDelta(buffered);
            w.end = 0;
        }
        for (data) |chunk| {
            self.text.appendSlice(self.arena, chunk) catch return error.WriteFailed;
            if (self.tui.eager_text) self.tui.assistantDelta(chunk);
            n += chunk.len;
        }
        // Keep the rings drained and the residual STT live while thinking:
        // words spoken over the think gap abort the reply before its first
        // frame instead of being lost.
        self.barge.scan() catch {};
        self.rail.draw(self.barge.pump.res_ema);
        return n;
    }
};

// --- minimal ANSI TUI -------------------------------------------------------

/// Resolve `--prompt-color` to an SGR parameter string. Named colours map to
/// their bold variant — the mark is one glyph, and bold is what keeps it
/// legible against a busy scrollback. Anything unrecognized passes through
/// verbatim, so `38;5;208` (256-colour) or `2;37` (dim) work as written.
fn promptSgr(name: ?[]const u8) []const u8 {
    const n = name orelse return "1;37";
    const table = [_]struct { []const u8, []const u8 }{
        .{ "white", "1;37" },  .{ "black", "1;30" }, .{ "red", "1;31" },
        .{ "green", "1;32" },  .{ "yellow", "1;33" }, .{ "blue", "1;34" },
        .{ "magenta", "1;35" }, .{ "cyan", "1;36" },  .{ "gray", "1;90" },
        .{ "grey", "1;90" },
    };
    for (table) |e| if (std.ascii.eqlIgnoreCase(n, e[0])) return e[1];
    return n;
}

// --- typed input: the keyboard as a second path into the same turn ---------
//
// The live transcript is a SUGGESTION, shown dim, and accepting it is
// explicit: right-arrow (or End, or ctrl-E) takes it into the editable line;
// typing anything else starts clean and the suggestion is gone. That is the
// shell-autosuggestion gesture, so it reads without being taught, and it means
// the keyboard can never silently prepend words you did not ask for.
//
// Either way, typing takes the turn off EOU endpointing: it then ends on
// Enter, because an endpointing model firing mid-sentence would submit half a
// line.
//
// The reader thread only moves bytes; decoding and editing happen on the turn
// loop, so all editing state stays single-threaded.

/// Saved terminal state for the SIGINT handler, which cannot take a closure.
var saved_termios: ?std.posix.termios = null;

/// Keystrokes arrive immediately and unechoed, so the prompt line owns what
/// appears on screen. ISIG stays ON: Ctrl-C keeps working through the existing
/// handler instead of needing its own key case.
fn enableRawMode() void {
    const current = std.posix.tcgetattr(0) catch return;
    saved_termios = current;
    var raw = current;
    raw.lflag.ICANON = false;
    raw.lflag.ECHO = false;
    std.posix.tcsetattr(0, .FLUSH, raw) catch {};
}

fn restoreTermios() void {
    if (saved_termios) |t| std.posix.tcsetattr(0, .FLUSH, t) catch {};
}

/// Raw terminal bytes, produced by the reader thread, drained by the turn.
const KeyQueue = struct {
    mutex: std.Io.Mutex = .init,
    bytes: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    fn lock(self: *KeyQueue) void {
        std.Io.Threaded.mutexLock(&self.mutex);
    }
    fn unlock(self: *KeyQueue) void {
        std.Io.Threaded.mutexUnlock(&self.mutex);
    }

    fn push(self: *KeyQueue, data: []const u8) void {
        self.lock();
        defer self.unlock();
        self.bytes.appendSlice(self.allocator, data) catch {};
    }

    fn drain(self: *KeyQueue, out: *std.ArrayList(u8)) void {
        self.lock();
        defer self.unlock();
        out.clearRetainingCapacity();
        out.appendSlice(self.allocator, self.bytes.items) catch {};
        self.bytes.clearRetainingCapacity();
    }
};

const Typed = struct {
    buf: std.ArrayList(u8) = .empty,
    active: bool = false,

    fn reset(self: *Typed) void {
        self.buf.clearRetainingCapacity();
        self.active = false;
    }
};

const KeyAction = enum {
    none,
    edited,
    /// Submit the typed line.
    submit,
    /// Enter while the suggestion is still dim: send the transcript as it
    /// stands, without waiting for the endpointing model. Enter therefore
    /// means SUBMIT in every state, which is what keeps `→` from being a
    /// trap — accepting changes what you can edit, never how you send.
    submit_suggestion,
    interrupt,
};

/// Take the dim suggestion into the line, exactly once per turn.
fn acceptSuggestion(typed: *Typed, allocator: std.mem.Allocator, suggestion: []const u8) bool {
    if (typed.active or suggestion.len == 0) return false;
    typed.active = true;
    typed.buf.clearRetainingCapacity();
    typed.buf.appendSlice(allocator, suggestion) catch {};
    return true;
}

/// Apply drained bytes to the typed buffer. `suggestion` is the live
/// transcript: right-arrow/End/ctrl-E accept it, anything else starts clean.
fn applyKeys(typed: *Typed, allocator: std.mem.Allocator, keys: []const u8, suggestion: []const u8) KeyAction {
    var action: KeyAction = .none;
    var i: usize = 0;
    while (i < keys.len) : (i += 1) {
        switch (keys[i]) {
            '\r', '\n' => {
                if (typed.active) return if (typed.buf.items.len > 0) .submit else .interrupt;
                return if (suggestion.len > 0) .submit_suggestion else .interrupt;
            },
            0x05 => if (acceptSuggestion(typed, allocator, suggestion)) { // ctrl-E
                action = .edited;
            },
            0x7F, 0x08 => { // backspace: one whole codepoint
                while (typed.buf.pop()) |b| {
                    if ((b & 0xC0) != 0x80) break;
                }
                action = .edited;
            },
            0x15 => { // ctrl-U
                typed.buf.clearRetainingCapacity();
                action = .edited;
            },
            0x17 => { // ctrl-W: trailing word
                while (typed.buf.items.len > 0 and typed.buf.items[typed.buf.items.len - 1] == ' ') _ = typed.buf.pop();
                while (typed.buf.items.len > 0 and typed.buf.items[typed.buf.items.len - 1] != ' ') _ = typed.buf.pop();
                action = .edited;
            },
            0x1B => { // CSI/SS3: consume the sequence, then act on its final byte
                var final: u8 = 0;
                i += 1;
                if (i < keys.len and (keys[i] == '[' or keys[i] == 'O')) {
                    i += 1;
                    while (i < keys.len and !(keys[i] >= 0x40 and keys[i] <= 0x7E)) i += 1;
                    if (i < keys.len) final = keys[i];
                }
                // Right arrow and End accept; every other sequence is
                // swallowed rather than inserted as garbage.
                if ((final == 'C' or final == 'F') and acceptSuggestion(typed, allocator, suggestion)) {
                    action = .edited;
                }
            },
            else => |c| {
                if (c < 0x20) continue; // other control bytes
                // Typing without accepting starts CLEAN: the suggestion was
                // not asked for, so it goes away rather than prefixing you.
                if (!typed.active) {
                    typed.active = true;
                    typed.buf.clearRetainingCapacity();
                }
                typed.buf.append(allocator, c) catch {};
                action = .edited;
            },
        }
    }
    return action;
}

const Tui = struct {
    out: *std.Io.Writer,
    arena: std.mem.Allocator,
    /// The user's prompt mark, pre-rendered with its SGR wrapper (plain under
    /// NO_COLOR) so the print path stays a single formatted write.
    prompt_mark: []const u8 = "\x1b[1;37m❯\x1b[0m",
    color: bool = true,
    /// True while the transient prompt line (mark + live partial) is on
    /// screen. Unlike a streaming reply line it is ERASED rather than
    /// newline-terminated when a permanent line has to print, then redrawn —
    /// so a log can never strand an empty prompt in the scrollback.
    prompt_active: bool = false,
    partial_buf: [128]u8 = undefined,
    partial_len: usize = 0,
    /// The prompt is showing TYPED text rather than a live transcript.
    prompt_typed: bool = false,
    /// Turns left to show the key affordance beside a dim suggestion. It
    /// teaches both gestures at once — the point being that Enter sends in
    /// either state, so accepting is never a corner you can get stuck in.
    hint_turns: usize = 2,
    /// Legacy display: stream the reply text during THINK instead of
    /// revealing it in sync with the voice (--eager-text).
    eager_text: bool = false,
    /// True while a \r-updated (unterminated) line is on screen.
    open_line: bool = false,
    /// Terminal rows; the rail is PINNED on the last row, outside the
    /// scroll region (DECSTBM 1..rows-1) — content prints can never collide
    /// with it, even when the region scrolls under an open streaming line.
    rows: usize = 0,
    region_set: bool = false,

    fn banner(self: *Tui) void {
        self.out.print("\x1b[1mfucina voiceagent\x1b[0m — mic → parakeet EOU → qwen3 → qwen3-tts (all native)\n", .{}) catch {};
        self.out.print("speak or type; talking over the reply interrupts it; Ctrl-C quits\n\n", .{}) catch {};
        self.out.flush() catch {};
    }

    fn closeLine(self: *Tui) void {
        if (self.open_line) {
            self.out.print("\n", .{}) catch {};
            self.open_line = false;
        }
    }

    fn termRows() usize {
        var ws: std.posix.winsize = undefined;
        const rc = std.posix.system.ioctl(1, std.posix.T.IOCGWINSZ, @intFromPtr(&ws));
        if (rc == 0 and ws.row > 1) return ws.row;
        return 24;
    }

    /// (Re)pin the rail row: scroll region = all rows but the last.
    fn pinRail(self: *Tui) void {
        const rows = termRows();
        if (self.region_set and rows == self.rows) return;
        self.rows = rows;
        // set region, then park the cursor back inside it (bottom of region)
        self.out.print("\x1b7\x1b[1;{d}r\x1b8", .{rows - 1}) catch {};
        if (!self.region_set) self.out.print("\x1b[{d};1H", .{rows - 1}) catch {};
        self.region_set = true;
        self.out.flush() catch {};
    }

    /// Draw the rail on its pinned row without disturbing the cursor.
    fn pinnedDraw(self: *Tui, rail_line: []const u8) void {
        self.pinRail();
        self.out.print("\x1b7\x1b[{d};1H\x1b[2K{s}\x1b8", .{ self.rows, rail_line }) catch {};
        self.out.flush() catch {};
    }

    /// Restore the full scroll region on exit.
    fn unpinRail(self: *Tui) void {
        if (!self.region_set) return;
        self.region_set = false;
        self.out.print("\x1b[r\x1b[{d};1H\n", .{self.rows}) catch {};
        self.out.flush() catch {};
    }

    /// The user's turn opens: put the prompt mark on screen straight away, so
    /// "your turn" is something you can see rather than infer from an empty
    /// line and a level meter. The live transcript then fills in after it.
    fn promptBegin(self: *Tui) void {
        self.prompt_active = true;
        self.partial_len = 0;
        self.prompt_typed = false;
        self.drawPrompt();
    }

    fn promptEnd(self: *Tui) void {
        self.prompt_active = false;
        self.partial_len = 0;
    }

    fn drawPrompt(self: *Tui) void {
        const p = self.partial_buf[0..self.partial_len];
        // The transcript is a suggestion until accepted, so it renders dim —
        // the same grammar as a shell autosuggestion. Accepted or typed text
        // is plain: it is yours and will not be revised under you.
        if (self.color and !self.prompt_typed) {
            const hint: []const u8 = if (self.hint_turns > 0 and p.len > 0) "   ⏎ send  → edit" else "";
            self.out.print("\r\x1b[2K{s} \x1b[90m{s}\x1b[2m{s}\x1b[0m", .{ self.prompt_mark, p, hint }) catch {};
        } else {
            self.out.print("\r\x1b[2K{s} {s}", .{ self.prompt_mark, p }) catch {};
        }
        self.open_line = true;
        self.out.flush() catch {};
    }

    /// Render the typed buffer (tail-clipped like the transcript, so a long
    /// line never wraps under the pinned rail).
    fn typedUpdate(self: *Tui, text: []const u8) void {
        var tail = if (text.len > 100) text[text.len - 100 ..] else text;
        while (tail.len > 0 and (tail[0] & 0xC0) == 0x80) tail = tail[1..];
        @memcpy(self.partial_buf[0..tail.len], tail);
        self.partial_len = tail.len;
        self.prompt_typed = true;
        self.drawPrompt();
    }

    /// Live partial transcript: rewritten in place after the prompt mark, so
    /// the line the words form IS the line that settles as the final turn.
    fn partialUpdate(self: *Tui, text: []const u8) void {
        var tail = if (text.len > 100) text[text.len - 100 ..] else text;
        while (tail.len > 0 and (tail[0] & 0xC0) == 0x80) tail = tail[1..];
        @memcpy(self.partial_buf[0..tail.len], tail);
        self.partial_len = tail.len;
        self.prompt_typed = false;
        self.drawPrompt();
    }

    /// Rewrite the current (status) line in place.
    fn status(self: *Tui, comptime fmt: []const u8, args: anytype) void {
        self.out.print("\r\x1b[2K" ++ fmt, args) catch {};
        self.open_line = true;
        self.out.flush() catch {};
    }


    /// The user's turn carries a prompt mark; the reply carries none, so the
    /// agent's words are just the text on the screen.
    fn userFinal(self: *Tui, text: []const u8) void {
        self.promptEnd(); // the finalized turn REPLACES the prompt in place
        self.out.print("\r\x1b[2K{s} {s}\n", .{ self.prompt_mark, text }) catch {};
        // Eager text streams straight onto the next line; the synced reveal
        // claims it later, when the voice starts.
        self.open_line = self.eager_text;
        self.out.flush() catch {};
    }

    /// Open the reply line (synced reveal: words follow the voice). Unprefixed
    /// — the line is claimed, nothing is drawn on it yet.
    fn botPrefix(self: *Tui) void {
        self.open_line = true;
        self.out.flush() catch {};
    }

    /// Stream a reply chunk; newlines flattened so the line stays coherent.
    fn assistantDelta(self: *Tui, chunk: []const u8) void {
        for (chunk) |ch| {
            self.out.writeByte(if (ch == '\n') ' ' else ch) catch {};
        }
        self.out.flush() catch {};
    }

    /// Unspoken remainder of a cut reply: kept for the record, dimmed.
    fn assistantDim(self: *Tui, chunk: []const u8) void {
        self.out.print("\x1b[2m", .{}) catch {};
        for (chunk) |ch| {
            self.out.writeByte(if (ch == '\n') ' ' else ch) catch {};
        }
        self.out.print("\x1b[0m", .{}) catch {};
        self.out.flush() catch {};
    }

    /// Append a full line. An open STATUS line is terminated; an open PROMPT
    /// line is erased and redrawn underneath, since it is transient state
    /// rather than content worth keeping.
    fn line(self: *Tui, comptime fmt: []const u8, args: anytype) void {
        if (self.prompt_active) {
            self.out.print("\r\x1b[2K", .{}) catch {};
            self.open_line = false;
        } else self.closeLine();
        self.out.print(fmt ++ "\n", .{} ++ args) catch {};
        if (self.prompt_active) self.drawPrompt() else self.out.flush() catch {};
    }
};

// --- synced reveal: the reply text appears as the voice speaks it -----------
//
// The screen and the voice are one event: nothing of the reply is printed
// during THINK; during SPEAK the text advances on the playback clock
// (samples pushed to the out-ring minus samples still queued = samples
// actually played), so a barge pause holds the text and an interrupt cuts it
// where the voice stopped. The producers flag voiced samples as they push,
// so the map runs over the VOICED window — leading silence delays nothing
// and the trailing pad before EOS stretches nothing: text position =
// (played − voice onset) / voiced span. Words reveal at onset (the word
// being spoken extends to its end — snap UP), matching how speech is heard.
// Until the talker finishes, a fixed speaking-rate estimate paces the first
// words. Timing is estimated (neither engine emits word timestamps) —
// word-granularity snapping keeps the drift invisible. --eager-text
// restores the legacy stream-at-THINK display.

const Reveal = struct {
    tui: *Tui,
    engine: *duplex.Engine,
    eager: bool,
    text: []const u8 = &.{},
    shown: usize = 0,
    active: bool = false,
    total_s: ?f64 = null,
    pushed48: std.atomic.Value(u64) = .init(0),
    lead48: std.atomic.Value(u64) = .init(0), // samples before the first voiced one
    voiced_hi48: std.atomic.Value(u64) = .init(0), // just past the last voiced sample
    all_pushed: std.atomic.Value(bool) = .init(false),

    /// Chars/s until the exact duration is known (qwen3-tts measures ~13.5
    /// on the greedy golden); only a reply's first seconds ride on this.
    const fallback_rate = 14.0;
    /// |sample| above this counts as voice (the sim's radiated-voice gate).
    const voice_thresh = 0.01;

    fn begin(self: *Reveal, text: []const u8) void {
        if (self.eager) return;
        self.text = text;
        self.shown = 0;
        self.total_s = null;
        self.pushed48.store(0, .monotonic);
        self.lead48.store(0, .monotonic);
        self.voiced_hi48.store(0, .monotonic);
        self.all_pushed.store(false, .monotonic);
        self.active = true;
        self.tui.botPrefix();
    }

    /// Producer side: one source sample (= 2 samples at 48 kHz) entered the
    /// out-ring. Single producer at a time; the voiced high-water store is
    /// the RELEASE that publishes lead48 — a relaxed pair would let the
    /// reader see hi != 0 with a stale lead of 0 and reveal ahead of onset.
    fn notePushed(self: *Reveal, voiced: bool) void {
        const base = self.pushed48.fetchAdd(2, .monotonic);
        if (voiced) {
            if (self.voiced_hi48.load(.monotonic) == 0) self.lead48.store(base, .monotonic);
            self.voiced_hi48.store(base + 2, .release);
        }
    }

    fn setTotal(self: *Reveal, secs: f64) void {
        if (self.active) self.total_s = secs;
    }

    /// Generation done and fully pushed: the push counter IS the duration.
    fn setTotalFromPushed(self: *Reveal) void {
        self.setTotal(@as(f64, @floatFromInt(self.pushed48.load(.monotonic))) / 48000.0);
    }

    /// Every sample of the reply is in the ring: the voiced high-water mark
    /// is final, so the map's end anchor becomes exact.
    fn markAllPushed(self: *Reveal) void {
        self.all_pushed.store(true, .monotonic);
    }

    fn playedSeconds(self: *Reveal) f64 {
        // pushed first: pushed only grows, so an older pushed paired with a
        // current pending can only UNDERSTATE progress — whichever thread
        // races between the reads, the reveal trails the audio, never leads.
        const pushed = self.pushed48.load(.monotonic);
        const pending: u64 = self.engine.outPending();
        if (pushed <= pending) return 0;
        return @as(f64, @floatFromInt(pushed - pending)) / 48000.0;
    }

    /// Advance the reveal to the playback position: the word the voice is
    /// in right now is shown whole.
    fn step(self: *Reveal) void {
        if (!self.active) return;
        const hi = self.voiced_hi48.load(.acquire); // pairs with notePushed's release
        if (hi == 0) return; // nothing voiced pushed yet — still silence
        const lead: f64 = @as(f64, @floatFromInt(self.lead48.load(.monotonic))) / 48000.0;
        const pos = self.playedSeconds() - lead;
        if (pos <= 0) return;
        // Speech-span end: exact once all audio is pushed; the frames-based
        // total (minus onset) bridges until then; else the rate estimate.
        const hi_s = @as(f64, @floatFromInt(hi)) / 48000.0;
        const span: ?f64 = if (self.all_pushed.load(.monotonic) and hi_s > lead)
            hi_s - lead
        else if (self.total_s) |ts|
            (if (ts > lead) ts - lead else null)
        else
            null;
        const goal: f64 = if (span) |sp|
            (pos / sp) * @as(f64, @floatFromInt(self.text.len))
        else
            fallback_rate * pos;
        var target: usize = @intFromFloat(@max(goal, 0));
        if (target >= self.text.len) {
            target = self.text.len;
        } else {
            // reveal the CURRENT word to its end (onset, not completion);
            // spaceless scripts cap the extension and cut on a UTF-8
            // boundary instead.
            var i = target;
            while (i < self.text.len and i - target < 32 and !isBreak(self.text[i])) i += 1;
            if (i < self.text.len and !isBreak(self.text[i])) {
                while (target > self.shown and (self.text[target] & 0xC0) == 0x80) target -= 1;
            } else target = i;
        }
        if (target <= self.shown) return;
        self.tui.assistantDelta(self.text[self.shown..target]);
        self.shown = target;
    }

    fn isBreak(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t';
    }

    /// Reply fully played: flush whatever rounding left over.
    fn finishAll(self: *Reveal) void {
        if (!self.active) return;
        self.active = false;
        if (self.shown < self.text.len) self.tui.assistantDelta(self.text[self.shown..]);
        self.shown = self.text.len;
    }

    /// Interrupted: the line keeps what was spoken bright; the unspoken
    /// remainder stays for the record, dimmed.
    fn cutNow(self: *Reveal) void {
        if (!self.active) return;
        self.active = false;
        if (self.shown < self.text.len) self.tui.assistantDim(self.text[self.shown..]);
        self.shown = self.text.len;
    }
};

// --- Signal Rail controller: agent states → rail frames (rail.zig) ----------

const RailCtl = struct {
    tui: *Tui,
    io: std.Io,
    width: usize,
    profile: rail_mod.GlyphProfile,
    color: rail_mod.ColorMode,
    state: rail_mod.State = .idle,
    entry_tick: u64 = 0,
    input_q: u3 = 0,
    output_q: u3 = 0,
    out_level_bits: std.atomic.Value(u32) = .init(0),
    seed: u64 = 0,
    last_tick: u64 = std.math.maxInt(u64),
    // LISTENING → IDLE after ~3 s of sustained silence with no transcript;
    // back within one tick on any input energy (spec IDLE↔LISTENING).
    silence_ticks: u32 = 0,
    hold_listening: bool = false, // a partial transcript pins LISTENING
    progress: ?f32 = null, // ACTING determinate progress
    question_pending: bool = false, // spoken reply asked the user something
    cells: [rail_mod.max_width]rail_mod.Cell = undefined,
    buf: std.ArrayList(u8) = .empty,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, tui: *Tui, io: std.Io, ascii: bool) RailCtl {
        // largest preset fitting the terminal (label 12 + caps 2 + aux 10)
        const cols = termCols();
        const avail = if (cols > 26) cols - 26 else 25;
        const width: usize = if (avail >= 61) 61 else if (avail >= 49) 49 else if (avail >= 37) 37 else 25;
        const no_color = std.c.getenv("NO_COLOR") != null;
        return .{
            .tui = tui,
            .io = io,
            .allocator = allocator,
            .width = width,
            .profile = if (ascii) .ascii else .instrument_square,
            .color = if (no_color) .mono else .truecolor,
        };
    }

    fn termCols() usize {
        var ws: std.posix.winsize = undefined;
        const rc = std.posix.system.ioctl(1, std.posix.T.IOCGWINSZ, @intFromPtr(&ws));
        if (rc == 0 and ws.col > 0) return ws.col;
        return 80;
    }

    fn tick(self: *const RailCtl) u64 {
        return @intCast(@divTrunc(nowNs(self.io), rail_mod.tick_ns));
    }

    fn set(self: *RailCtl, st: rail_mod.State) void {
        if (self.state == st) return;
        self.state = st;
        self.entry_tick = self.tick();
        self.last_tick = std.math.maxInt(u64); // force redraw
        if (st == .thinking) self.seed +%= 1;
    }

    /// Transitional states advance on the clock (captured→thinking,
    /// interrupted/complete→listening).
    fn autoAdvance(self: *RailCtl, t: u64) void {
        const dt = t -% self.entry_tick;
        switch (self.state) {
            .captured => if (dt >= 4) self.set(.thinking),
            .interrupted => if (dt >= 10) self.set(.listening),
            .complete => if (dt >= 14) {
                self.set(if (self.question_pending) .needs_input else .listening);
                self.question_pending = false;
            },
            .needs_input => if (dt >= 180) self.set(.idle), // ~15 s unanswered
            else => {},
        }
    }

    fn setOutputLevel(self: *RailCtl, rms: f32) void {
        self.out_level_bits.store(@bitCast(rms), .monotonic);
    }

    /// Fatal-error exit: play the spec ERROR entry (saturate → blackout →
    /// settled fracture) and leave the fracture as the rail's last image;
    /// the full error name goes to the permanent log.
    fn showError(self: *RailCtl, err: anyerror) void {
        self.set(.err);
        var i: usize = 0;
        while (i < 6) : (i += 1) {
            self.last_tick = std.math.maxInt(u64);
            self.draw(0);
            std.Io.sleep(self.io, .{ .nanoseconds = rail_mod.tick_ns }, .awake) catch {};
        }
        self.tui.line("\x1b[31merror: {s}\x1b[0m", .{@errorName(err)});
        self.last_tick = std.math.maxInt(u64);
        self.draw(0); // re-pin the settled fracture under the log line
    }

    /// Tick-gated draw onto the pinned bottom row.
    fn draw(self: *RailCtl, input_level: f32) void {
        const t = self.tick();
        if (t == self.last_tick) return;
        self.last_tick = t;
        self.autoAdvance(t);
        const q_target = rail_mod.quantize(input_level * 8.0);
        if (self.state == .listening) {
            if (q_target == 0 and !self.hold_listening) {
                self.silence_ticks +|= 1;
                // ~2 s: a transcript-in-progress pins LISTENING, so only
                // true silence counts — and the EOU model closes turns at
                // sub-second pauses, so 2 s of quiet means nobody's talking.
                if (self.silence_ticks >= 24) self.set(.idle);
            } else self.silence_ticks = 0;
        } else if (self.state == .idle or self.state == .needs_input) {
            // wake within one tick on input energy OR the first transcribed
            // token, whichever arrives first
            if (q_target >= 1 or self.hold_listening) self.set(.listening);
        } else self.silence_ticks = 0;
        self.input_q = rail_mod.stepLevel(self.input_q, q_target);
        const out_rms: f32 = @bitCast(self.out_level_bits.load(.monotonic));
        self.output_q = rail_mod.stepLevel(self.output_q, rail_mod.quantize(out_rms * 4.0));

        rail_mod.frame(self.state, .{
            .width = self.width,
            .tick = t,
            .entry_tick = self.entry_tick,
            .input_q = self.input_q,
            .output_q = self.output_q,
            .seed = self.seed,
            .progress = self.progress,
        }, &self.cells);

        var aux_buf: [16]u8 = undefined;
        const aux: []const u8 = switch (self.state) {
            .thinking => std.fmt.bufPrint(&aux_buf, "T+{d:0>4.1}", .{@as(f64, @floatFromInt(t -% self.entry_tick)) / 12.0}) catch "",
            .err => "FATAL",
            .acting => if (self.progress) |pv| std.fmt.bufPrint(&aux_buf, "{d:0>3.0}%", .{pv * 100.0}) catch "--" else "--",
            .needs_input => "INPUT",
            .waiting => "HOLD",
            .complete => "DONE",
            .interrupted => "CUT",
            .listening => "", // level is the display itself
            else => "",
        };

        self.buf.clearRetainingCapacity();
        rail_mod.render(&self.buf, self.allocator, self.state, self.cells[0..self.width], aux, self.profile, self.color) catch return;
        self.tui.pinnedDraw(self.buf.items);
    }
};

// --- TTS streaming sink: frames → chunked decode → playback ring ------------

const SpeakSink = struct {
    ctx: *ExecContext,
    dec: *const qtts.codec.Decoder,
    sess: *qtts.codec.Streaming,
    engine: *duplex.Engine,
    up2: *duplex.Upsampler2,
    barge: *BargeCtx,
    full_duplex: bool,
    allocator: std.mem.Allocator,
    interrupt: *std.atomic.Value(bool),
    abort: ?*const std.atomic.Value(bool), // sim-done: never block on audio
    rail: *RailCtl,
    reveal: *Reveal,
    io: std.Io,
    kq: usize,
    chunk_first: usize, // small first chunk: fast time-to-first-audio
    chunk_main: usize, // large steady chunks: amortize the left-context cost
    left_ctx: usize,
    frames: std.ArrayList(i32) = .empty, // frame-major [t][16]
    emitted: usize = 0, // frames already handed to the decode worker
    first_push_ns: ?i96 = null,
    err: ?anyerror = null,

    // --- decode worker: codec runs OVERLAPPED with the talker ------------
    // The talker alone meets real time (~13 fps vs 12.5); serialized with
    // the chunked codec (25-frame left context => ~3x per-frame overhead at
    // chunk 12) production drops to ~5 fps and the voice breaks up. A
    // dedicated thread (own ExecContext) decodes chunk N while the talker
    // generates N+1, so production returns to talker rate.
    mutex: std.Io.Mutex = .init,
    jobs: std.ArrayList(Job) = .empty,
    worker_busy: bool = false,
    stop_worker: bool = false,
    thread: ?std.Thread = null,

    const Job = struct { kt: []i32, span: usize, ctx_frames: usize };

    fn lock(self: *SpeakSink) void {
        std.Io.Threaded.mutexLock(&self.mutex);
    }
    fn unlock(self: *SpeakSink) void {
        std.Io.Threaded.mutexUnlock(&self.mutex);
    }

    fn stopRequested(self: *SpeakSink) bool {
        if (self.abort) |a| if (a.load(.acquire)) return true;
        return self.interrupt.load(.acquire) or self.barge.pending;
    }

    fn startWorker(self: *SpeakSink) !void {
        self.sess.reset();
        self.thread = try std.Thread.spawn(.{}, workerMain, .{self});
    }

    fn workerMain(self: *SpeakSink) void {
        while (true) {
            self.lock();
            const job: ?Job = if (self.jobs.items.len > 0) self.jobs.orderedRemove(0) else null;
            if (job != null) self.worker_busy = true;
            const stopping = self.stop_worker;
            self.unlock();

            if (job) |j| {
                self.decodeJob(j) catch |e| {
                    if (self.err == null) self.err = e;
                };
                self.allocator.free(j.kt);
                self.lock();
                self.worker_busy = false;
                self.unlock();
            } else if (stopping) {
                return;
            } else {
                std.Io.sleep(self.io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .awake) catch {};
            }
        }
    }

    fn decodeJob(self: *SpeakSink, job: Job) !void {
        const audio = try self.sess.step(self.ctx, job.kt, job.span);
        defer self.ctx.allocator.free(audio);
        const emitted = audio[job.ctx_frames * qtts.codec.hop_length ..];
        var lvl: f32 = 0;
        for (emitted) |x| lvl += x * x;
        self.rail.setOutputLevel(@sqrt(lvl / @as(f32, @floatFromInt(@max(1, emitted.len)))));
        for (audio[job.ctx_frames * qtts.codec.hop_length ..]) |x| {
            var pair: [2]f32 = undefined;
            self.up2.push(x, &pair);
            var off: usize = 0;
            while (off < 2) {
                off += self.engine.out48.push(pair[off..]);
                if (off == 2) break;
                if (self.stopRequested() or self.stopFlag()) return;
                std.Io.sleep(self.io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .awake) catch {};
            }
            self.reveal.notePushed(@abs(x) > Reveal.voice_thresh);
        }
        if (self.first_push_ns == null) self.first_push_ns = nowNs(self.io);
    }

    fn stopFlag(self: *SpeakSink) bool {
        self.lock();
        defer self.unlock();
        return self.stop_worker;
    }

    /// Enqueue frames [emitted, upto) for decoding — streaming: new frames
    /// only, no left-context replay.
    fn enqueue(self: *SpeakSink, upto: usize) !void {
        const ctx_frames: usize = 0;
        const s0 = self.emitted;
        const span = upto - s0;
        const kt = try self.allocator.alloc(i32, self.kq * span);
        errdefer self.allocator.free(kt);
        for (0..span) |t| {
            for (0..self.kq) |k| kt[k * span + t] = self.frames.items[(s0 + t) * self.kq + k];
        }
        self.lock();
        defer self.unlock();
        try self.jobs.append(self.allocator, .{ .kt = kt, .span = span, .ctx_frames = ctx_frames });
        self.emitted = upto;
    }

    /// All queued jobs decoded and pushed?
    fn queueIdle(self: *SpeakSink) bool {
        self.lock();
        defer self.unlock();
        return self.jobs.items.len == 0 and !self.worker_busy;
    }

    /// Stop the worker: discard queued jobs, join the thread. Idempotent —
    /// the error path reaches it via errdefer after the explicit call.
    fn stopWorker(self: *SpeakSink) void {
        if (self.thread == null) return;
        self.lock();
        self.stop_worker = true;
        for (self.jobs.items) |j| self.allocator.free(j.kt);
        self.jobs.clearRetainingCapacity();
        self.unlock();
        if (self.thread) |t| t.join();
        self.thread = null;
        self.jobs.deinit(self.allocator);
    }

    fn onFrame(user: ?*anyopaque, frame: []const i32) bool {
        const self: *SpeakSink = @ptrCast(@alignCast(user.?));
        if (self.stopRequested()) return false;
        // Between TTS frames: drain the rings through the AEC; the residual
        // STT decides interruption in the text domain (BargeCtx).
        if (self.full_duplex) {
            self.barge.scan() catch |e| {
                self.err = e;
                return false;
            };
            if (self.stopRequested()) return false;
        }
        if (self.err != null) return false;
        self.rail.set(if (self.barge.stage == .paused) .waiting else .speaking);
        self.rail.draw(self.barge.pump.res_ema);
        self.reveal.step();
        self.frames.appendSlice(self.allocator, frame) catch |e| {
            self.err = e;
            return false;
        };
        const total = self.frames.items.len / self.kq;
        const chunk = if (self.emitted == 0) self.chunk_first else self.chunk_main;
        if (total - self.emitted >= chunk) {
            self.enqueue(total) catch |e| {
                self.err = e;
                return false;
            };
        }
        return true;
    }

    fn finish(self: *SpeakSink) !void {
        const total = self.frames.items.len / self.kq;
        if (total > self.emitted) try self.enqueue(total);
    }
};

// --- sim driver: scripted "device" for headless end-to-end tests ------------
//
// Replaces the miniaudio duplex stream with a thread that calls the SAME
// Engine.callback on a real-time clock. The synthetic mic is user-speech WAVs
// plus a delayed, attenuated copy of the frames the agent actually played
// (a true echo path — the AEC must cancel it or the barge gate false-fires)
// plus a small noise floor. A second WAV is injected over the reply to
// exercise barge-in. Deterministic; ends when the post-barge reply drains.

const SimDriver = struct {
    engine: *duplex.Engine,
    io: std.Io,
    speech: []const f32, // 48 kHz user utterance, injected at speech_at
    barge: ?[]const f32, // injected once the reply has played ~1.2 s voiced
    done: std.atomic.Value(bool) = .init(false),
    reason: []const u8 = "running",

    t: usize = 0, // global 48 kHz frame counter
    speech_at: usize = 24000, // 0.5 s in
    barge_at: ?usize = null,
    again: bool = false, // inject `barge` as a SECOND TURN (quiet listen), not over the reply
    /// Voiced playback (48 kHz frames) before the barge is injected. The
    /// default 1.2 s is LATE — by then the delay estimator has usually
    /// locked. `--sim-barge-after-ms` drives it down to the case that
    /// actually fails live: talking over the FIRST moments of a reply.
    barge_after_voiced: usize = 48000 * 12 / 10,
    voiced: usize = 0, // voiced played frames, total
    post_gap: bool = false, // interrupt gap seen after barge injection
    voiced_after_barge: usize = 0, // reply-2 voice: counted only post-gap
    silence_run: usize = 0,
    noise: u32 = 0x2545f491,
    delay: [16384]f32 = @splat(0),
    dpos: usize = 0,
    lp: f32 = 0, // one-pole state: speaker+room coloring on the echo
    echo_delay: usize = 5760, // 120 ms default (live-realistic); ≥ period
    echo_step_at: ?usize = null, // +10 ms delay step at this frame (relock)

    const period = 480; // 10 ms
    const echo_gain: f32 = 0.35;
    const noise_amp: f32 = 3.0e-4;

    fn whiteNoise(self: *SimDriver) f32 {
        self.noise = self.noise *% 1664525 +% 1013904223;
        return @as(f32, @floatFromInt(@as(i32, @bitCast(self.noise)))) * 0x1p-31;
    }

    fn run(self: *SimDriver) void {
        var played: [period]f32 = undefined;
        var mic: [period]f32 = undefined;
        const start_ns = nowNs(self.io);
        // Service the callback FOREVER (thread dies with the process): the
        // main thread may be blocked on ring space when the script ends.
        while (true) {
            // mic = noise floor + echo (past playback through a speaker
            // nonlinearity + multi-tap room response + coloring lowpass)
            // + scripted speech
            const taps = [_]struct { d: usize, g: f32 }{
                .{ .d = 0, .g = 1.0 },    .{ .d = 336, .g = 0.55 },
                .{ .d = 624, .g = 0.40 }, .{ .d = 1104, .g = 0.28 },
                .{ .d = 1968, .g = 0.18 },
            };
            if (self.echo_step_at) |sa| if (self.t >= sa) {
                self.echo_delay += 480; // +10 ms mid-run: relock test
                self.echo_step_at = null;
            };
            for (0..period) |i| {
                const gt = self.t + i;
                var m = self.whiteNoise() * noise_amp;
                var e: f32 = 0;
                for (taps) |tp| {
                    const idx = (self.dpos + self.delay.len + i - self.echo_delay - tp.d) & (self.delay.len - 1);
                    e += self.delay[idx] * tp.g;
                }
                self.lp += 0.4 * (e - self.lp);
                m += self.lp * echo_gain;
                if (gt >= self.speech_at and gt - self.speech_at < self.speech.len)
                    m += self.speech[gt - self.speech_at];
                if (self.barge_at) |b0| if (self.barge) |b| {
                    if (gt >= b0 and gt - b0 < b.len) m += b[gt - b0];
                };
                mic[i] = m;
            }
            duplex.Engine.callback(@ptrCast(self.engine), &played, &mic, period);
            // What the "speaker" radiates: mildly tanh-driven playback.
            for (0..period) |i| {
                const x = played[i];
                self.delay[(self.dpos + i) & (self.delay.len - 1)] = std.math.tanh(1.6 * x) / 1.6;
            }
            self.dpos = (self.dpos + period) & (self.delay.len - 1);

            // script state from what actually reached the "speaker"
            var v: usize = 0;
            for (played) |x| {
                if (@abs(x) > 0.01) v += 1;
            }
            if (v > period / 4) {
                self.voiced += period;
                self.silence_run = 0;
                if (self.barge_at != null and self.post_gap) self.voiced_after_barge += period;
            } else {
                self.silence_run += period;
                if (self.barge_at != null and self.silence_run > 24000) self.post_gap = true;
            }

            if (self.barge_at == null and self.barge != null) {
                if (self.again) {
                    // second turn: after the reply played and 2 s of silence
                    if (self.voiced >= 48000 / 2 and self.silence_run > 2 * 48000) self.barge_at = self.t + period;
                } else if (self.voiced >= self.barge_after_voiced) {
                    self.barge_at = self.t + period; // talk over the reply
                }
            }
            const barge_done = if (self.barge) |b|
                (self.barge_at != null and self.t > self.barge_at.? + b.len and self.voiced_after_barge > 48000 / 2)
            else
                (self.t > self.speech_at + self.speech.len and self.voiced > 48000 / 2);
            if (barge_done and self.silence_run > 3 * 48000 and !self.done.load(.acquire)) {
                self.reason = "complete";
                self.done.store(true, .release);
            }
            if (self.t > 120 * 48000 and !self.done.load(.acquire)) {
                self.reason = "TIMEOUT";
                self.done.store(true, .release);
            }

            self.t += period;
            const target = start_ns + @divTrunc(@as(i96, self.t) * std.time.ns_per_s, 48000);
            const now = nowNs(self.io);
            if (target > now) {
                std.Io.sleep(self.io, .{ .nanoseconds = @intCast(target - now) }, .awake) catch {};
            }
        }
    }
};

/// Load a PCM16 mono 24 kHz WAV (our own TTS writer's fixed 44-byte layout)
/// and upsample ×2 to the 48 kHz stream rate.
fn loadWav48(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1 << 28));
    defer allocator.free(bytes);
    if (bytes.len < 44 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE"))
        return error.BadWav;
    const channels = std.mem.readInt(u16, bytes[22..24], .little);
    const rate = std.mem.readInt(u32, bytes[24..28], .little);
    const bits = std.mem.readInt(u16, bytes[34..36], .little);
    if (channels != 1 or rate != 24000 or bits != 16 or !std.mem.eql(u8, bytes[36..40], "data"))
        return error.BadWav;
    const n = std.mem.readInt(u32, bytes[40..44], .little) / 2;
    var up = duplex.Upsampler2.init();
    const out = try allocator.alloc(f32, @as(usize, n) * 2);
    errdefer allocator.free(out);
    for (0..n) |i| {
        const s = std.mem.readInt(i16, bytes[44 + i * 2 ..][0..2], .little);
        var pair: [2]f32 = undefined;
        up.push(@as(f32, @floatFromInt(s)) / 32768.0, &pair);
        out[i * 2] = pair[0];
        out[i * 2 + 1] = pair[1];
    }
    return out;
}

// --- local tools (Hermes format; rail ACTING / NEEDS INPUT) ------------------

const tools_system_block =
    \\
    \\
    \\# Tools
    \\You may call one of these tools when useful. To call one, respond with
    \\ONLY a tool call block, nothing else:
    \\<tool_call>
    \\{"name": "<name>", "arguments": {...}}
    \\</tool_call>
    \\Available tools:
    \\{"name": "set_timer", "description": "Run a countdown timer for N seconds, speak when done", "parameters": {"seconds": "number"}}
    \\{"name": "get_time", "description": "Get the current time of day", "parameters": {}}
;

const ToolCall = struct { name: []const u8, seconds: f64 = 0 };

/// Extract a Hermes `<tool_call>` from a reply, if present.
fn parseToolCall(reply: []const u8) ?ToolCall {
    const open = std.mem.indexOf(u8, reply, "<tool_call>") orelse return null;
    const close = std.mem.indexOfPos(u8, reply, open, "</tool_call>") orelse return null;
    const body = std.mem.trim(u8, reply[open + 11 .. close], " \n\r\t");
    var tc = ToolCall{ .name = "" };
    if (std.mem.indexOf(u8, body, "set_timer") != null) {
        tc.name = "set_timer";
        if (std.mem.indexOf(u8, body, "\"seconds\"")) |sp| {
            var i = sp + 9;
            while (i < body.len and (body[i] == ':' or body[i] == ' ' or body[i] == '"')) i += 1;
            var j = i;
            while (j < body.len and (std.ascii.isDigit(body[j]) or body[j] == '.')) j += 1;
            tc.seconds = std.fmt.parseFloat(f64, body[i..j]) catch 5;
        } else tc.seconds = 5;
        return tc;
    }
    if (std.mem.indexOf(u8, body, "get_time") != null) {
        tc.name = "get_time";
        return tc;
    }
    return null;
}

// --- main -------------------------------------------------------------------

pub fn main(init: std.process.Init) anyerror!void {
    // Long-lived process: real allocator so per-turn frees actually free
    // (an arena would balloon RSS across turns — KVs, mels, transcripts).
    const allocator = std.heap.smp_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [8192]u8 = undefined;
    var out_writer = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_writer.interface;
    defer out.flush() catch {};

    if (flagVal(args, "--threads")) |t| fucina.parallel.setMaxThreads(try std.fmt.parseInt(usize, t, 10));

    var dev = try audio_mod.Audio.init();
    defer dev.deinit();
    if (hasFlag(args, "--list-devices")) {
        var caps: [audio_mod.max_devices]audio_mod.DeviceInfo = undefined;
        const cap_list = try dev.listDevices(.capture, &caps);
        for (cap_list, 0..) |d, i| try out.print("capture {d}: {s}\n", .{ i, d.nameSlice() });
        var pouts: [play_mod.max_devices]play_mod.DeviceInfo = undefined;
        const play_list = try play_mod.listPlaybackDevices(&pouts);
        for (play_list, 0..) |d, i| try out.print("playback {d}: {s}\n", .{ i, d.nameSlice() });
        return;
    }

    const asr_path = flagVal(args, "--asr") orelse return out.print("{s}", .{usage});
    const chat_path = flagVal(args, "--chat") orelse return out.print("{s}", .{usage});
    const tts_path = flagVal(args, "--tts") orelse return out.print("{s}", .{usage});
    const codec_path = flagVal(args, "--codec") orelse "";
    const speaker = flagVal(args, "--speaker") orelse "Aiden";
    const lang = flagVal(args, "--lang") orelse "english";
    // Spoken conversation, not written assistance. "Helpful assistant" primes
    // a model to answer and stop, which over TTS reads as a lecture; framing
    // it as a phone call gets turn-taking instead. Kept short and POSITIVE on
    // purpose — at 1.7B, prohibitions ("no preamble", "don't say sure") are
    // ignored or recited back verbatim, while a concrete persona plus one
    // instruction about what to DO holds.
    const base_system = flagVal(args, "--system") orelse
        "Talk like a friend on the phone: two short spoken sentences of ordinary words, " ++
        "no lists and no markdown. React to what they told you, or ask them something back.";
    const tools_on = !hasFlag(args, "--no-tools");
    const eager_text = hasFlag(args, "--eager-text");
    const system = if (tools_on)
        try std.mem.concat(allocator, u8, &.{ base_system, tools_system_block })
    else
        base_system;
    const seed: i64 = if (flagVal(args, "--seed")) |s| try std.fmt.parseInt(i64, s, 10) else 7;
    const max_reply: usize = if (flagVal(args, "--max-reply")) |s| try std.fmt.parseInt(usize, s, 10) else 160;

    const use_color = std.c.getenv("NO_COLOR") == null;
    const prompt_mark: []const u8 = if (!use_color)
        "❯"
    else
        try std.fmt.allocPrint(allocator, "\x1b[{s}m❯\x1b[0m", .{promptSgr(flagVal(args, "--prompt-color"))});
    var tui = Tui{ .out = out, .arena = allocator, .eager_text = eager_text, .prompt_mark = prompt_mark, .color = use_color };
    tui.banner();
    // Ctrl-C must not leave the terminal with a clamped scroll region.
    const on_sigint = struct {
        fn handle(_: std.c.SIG) callconv(.c) void {
            // tcsetattr is async-signal-safe; call it raw rather than through
            // the Zig wrapper. Cooked mode must come back even on Ctrl-C.
            if (saved_termios) |t| _ = std.c.tcsetattr(0, .FLUSH, &t);
            const restore = "\x1b[0m\x1b[r\x1b[9999;1H\n";
            _ = std.c.write(1, restore, restore.len);
            std.c._exit(130);
        }
    }.handle;
    std.posix.sigaction(std.posix.SIG.INT, &.{
        .handler = .{ .handler = on_sigint },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    }, null);
    var railctl = RailCtl.init(allocator, &tui, io, hasFlag(args, "--rail-ascii"));
    defer railctl.buf.deinit(allocator);
    defer tui.unpinRail();
    // True errors fracture the rail before the process dies (runs before
    // unpinRail in LIFO defer order, so the fracture survives on screen).
    errdefer |e| railctl.showError(e);

    // --- load the four stages -----------------------------------------------
    const t0 = nowNs(io);
    tui.status("[load] asr…", .{});
    var asr_file = try fucina.gguf.File.loadMmap(allocator, io, asr_path);
    defer asr_file.deinit();
    const asr_cfg = try parakeet_loader.Config.fromGguf(&asr_file);
    const asr_sc = (parakeet_loader.StreamingConfig.fromGguf(&asr_file) catch null) orelse {
        tui.line("error: --asr must be a streaming (EOU) parakeet model", .{});
        return;
    };
    const asr_feat = try parakeet_loader.loadFeaturizer(&asr_file, asr_cfg);
    var dft_basis = try parakeet_frontend.DftBasis.init(allocator, asr_cfg.n_fft);
    defer dft_basis.deinit();
    var asr_ctx: ExecContext = undefined;
    asr_ctx.init(allocator);
    defer asr_ctx.deinit();
    var asr_weights = parakeet_weights.ParakeetWeights.init(&asr_ctx, &asr_file);
    defer asr_weights.deinit();
    var sess = try parakeet_streaming.StreamingSession.init(allocator, &asr_file, asr_cfg, asr_sc, &asr_weights, "en");
    defer sess.deinit();
    const pieces = try parakeet_loader.loadPieces(&asr_file, allocator);

    tui.status("[load] chat…", .{});
    var chat_file = try fucina.gguf.File.loadMmap(allocator, io, chat_path);
    defer chat_file.deinit();
    var chat_ctx: ExecContext = undefined;
    chat_ctx.init(allocator);
    defer chat_ctx.deinit();
    const chat_cfg = try llm.qwen3.model.Config.fromGguf(&chat_file);
    var chat_model = try llm.qwen3.model.Model.loadGgufFromFile(&chat_ctx, &chat_file, chat_cfg);
    defer chat_model.deinit();
    var chat_tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &chat_file, .{});
    defer chat_tok.deinit();
    const template = llm.chat.Template.detect(chat_file.getString("tokenizer.chat_template")) orelse {
        tui.line("error: chat model has no recognizable chat template", .{});
        return;
    };
    var convo = try llm.chat.Conversation(llm.qwen3.model.Model, llm.tokenizer).init(&chat_ctx, &chat_model, &chat_tok, template, .{
        .system = system,
        .max_response_tokens = max_reply,
        .think_off = true,
    });
    defer convo.deinit();

    tui.status("[load] tts…", .{});
    var tts_file = try fucina.gguf.File.loadMmap(allocator, io, tts_path);
    defer tts_file.deinit();
    const tts_arch = tts_file.getString("general.architecture") orelse "";
    const use_pocket = std.mem.eql(u8, tts_arch, "pocket-tts");

    var tts_ctx: ExecContext = undefined;
    tts_ctx.init(allocator);
    defer tts_ctx.deinit();

    // Pocket engine (continuous-latent streaming; no separate codec stage).
    var pocket_engine: ?llm.pockettts.pocket.Engine = null;
    defer if (pocket_engine) |*pe| pe.deinit();
    if (use_pocket) {
        pocket_engine = try llm.pockettts.pocket.Engine.init(&tts_ctx, &tts_file, flagVal(args, "--voice") orelse "alba");
    }

    // Qwen3-TTS stages (skipped under pocket).
    var tts_model: qtts.model.Model = undefined;
    var tts_model_loaded = false;
    defer if (tts_model_loaded) tts_model.deinit();
    var tts_tok: llm.tokenizer.Tokenizer = undefined;
    var tts_tok_loaded = false;
    defer if (tts_tok_loaded) tts_tok.deinit();
    var codec_file: fucina.gguf.File = undefined;
    var codec_loaded = false;
    defer if (codec_loaded) codec_file.deinit();
    var codec_ctx: ExecContext = undefined;
    codec_ctx.init(allocator);
    defer codec_ctx.deinit();
    var codec_dec: qtts.codec.Decoder = undefined;
    var codec_dec_loaded = false;
    defer if (codec_dec_loaded) codec_dec.deinit();
    var tts_kvs: qtts.pipeline.Kvs = undefined;
    var tts_kvs_loaded = false;
    defer if (tts_kvs_loaded) tts_kvs.deinit();
    var codec_sess: qtts.codec.Streaming = undefined;
    var codec_sess_loaded = false;
    defer if (codec_sess_loaded) codec_sess.deinit();
    if (!use_pocket) {
        if (codec_path.len == 0) {
            tui.line("error: --codec is required for a Qwen3-TTS talker", .{});
            return;
        }
        tts_model = try qtts.model.Model.load(&tts_ctx, &tts_file);
        tts_model_loaded = true;
        tts_tok = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &tts_file, .{});
        tts_tok_loaded = true;
        tui.status("[load] codec…", .{});
        codec_file = try fucina.gguf.File.loadMmap(allocator, io, codec_path);
        codec_loaded = true;
        codec_dec = try qtts.codec.load(&codec_ctx, &codec_file);
        codec_dec_loaded = true;
        tts_kvs = try qtts.pipeline.Kvs.init(allocator, &tts_model);
        tts_kvs_loaded = true;
        codec_sess = try qtts.codec.Streaming.init(allocator, &codec_dec);
        codec_sess_loaded = true;
    }

    // --- AEC (GTCRN, ported at fixture parity) -------------------------------
    const aec_path = flagVal(args, "--aec") orelse "models/aec/gtcrn_aec.gguf";
    const aec_off = hasFlag(args, "--no-aec");
    var aec_file: ?fucina.gguf.File = null;
    defer if (aec_file) |*f| f.deinit();
    var aec_model: ?aec_mod.Model = null;
    defer if (aec_model) |*m| m.deinit();
    var aec_sess: ?aec_mod.Session = null;
    defer if (aec_sess) |*sx| sx.deinit();
    // Shared by the GTCRN session and the delay scan: both run on the pump's
    // caller, never concurrently with each other.
    var aec_ctx: ExecContext = undefined;
    aec_ctx.init(allocator);
    defer aec_ctx.deinit();
    if (!aec_off) {
        if (fucina.gguf.File.loadMmap(allocator, io, aec_path)) |f| {
            aec_file = f;
            aec_model = try aec_mod.Model.load(allocator, &aec_file.?);
            aec_sess = try aec_mod.Session.init(allocator, &aec_ctx, &aec_model.?);
        } else |_| {
            tui.line("[aec] {s} not found — running HALF-DUPLEX (no barge-in)", .{aec_path});
        }
    }
    const full_duplex = aec_sess != null;

    // --- audio I/O: one 48 kHz duplex stream, callback-clock-aligned ---------
    // (--sim <wav> [--sim-barge <wav>] swaps the device for a scripted driver)
    var engine = duplex.Engine{};
    var sim: ?*SimDriver = null;
    if (flagVal(args, "--sim")) |sim_path| {
        const sd = try allocator.create(SimDriver);
        sd.* = .{
            .engine = &engine,
            .io = io,
            .speech = try loadWav48(allocator, io, sim_path),
            .barge = if (flagVal(args, "--sim-barge")) |bp| try loadWav48(allocator, io, bp) else null,
        };
        if (flagVal(args, "--sim-echo-ms")) |ms| sd.echo_delay = @max(SimDriver.period, (try std.fmt.parseInt(usize, ms, 10)) * 48);
        if (flagVal(args, "--sim-barge-after-ms")) |ms| sd.barge_after_voiced = (try std.fmt.parseInt(usize, ms, 10)) * 48;
        if (hasFlag(args, "--sim-echo-step")) sd.echo_step_at = 15 * 48000;
        if (hasFlag(args, "--sim-again")) sd.again = true;
        const th = try std.Thread.spawn(.{}, SimDriver.run, .{sd});
        th.detach();
        sim = sd;
    } else {
        const mic_index: ?usize = if (flagVal(args, "--mic-device")) |m| try std.fmt.parseInt(usize, m, 10) else null;
        try dev.start(mic_index, null, duplex.stream_rate, duplex.period_frames, duplex.Engine.callback, &engine);
    }

    var pump = AecPump.init(&engine, if (aec_sess) |*sx| sx else null, &aec_ctx);
    pump.debug = hasFlag(args, "--aec-debug");
    var up2 = duplex.Upsampler2.init();
    var reveal = Reveal{ .tui = &tui, .engine = &engine, .eager = eager_text };
    var barge: BargeCtx = undefined; // wired below once the streamer exists

    var reply_acc: std.ArrayList(u8) = .empty;
    defer reply_acc.deinit(allocator);
    var streamer = IncrementalStreamer{
        .sess = &sess,
        .ctx = &asr_ctx,
        .file = &asr_file,
        .weights = &asr_weights,
        .arena = allocator,
        .feat = asr_feat,
        .dft_basis = &dft_basis,
        .mel_params = .{
            .stft = .{ .n_fft = asr_cfg.n_fft, .hop = asr_cfg.hop_length, .win_length = asr_cfg.win_length, .mag_power = asr_cfg.mag_power, .preemph = asr_cfg.preemph },
            .n_mels = asr_cfg.n_mels,
            .log_guard = asr_cfg.log_zero_guard,
            .normalize_per_feature = asr_cfg.normalize == .per_feature,
        },
        .n_mels = asr_cfg.n_mels,
        .chunk0 = @intCast(@max(1, asr_sc.chunk_size[0])),
        .chunk_main = @intCast(@max(1, asr_sc.chunk_size[1])),
        .pre_cache = @intCast(@max(0, asr_sc.pre_encode_cache_size[1])),
        .tail_margin = 8,
    };
    defer streamer.samples.deinit(allocator);

    tui.line("[ready] all stages loaded in {d:.1} s — speak when ready", .{@as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9});
    // Reserve the rail row NOW, while nothing is on the current line. The
    // first pin parks the cursor at the bottom of the new scroll region, so
    // deferring it to the first rail draw would move the cursor out from
    // under the turn-one prompt and strand it.
    tui.pinRail();

    // Keyboard: a detached reader moves bytes onto the queue and raises the
    // interrupt flag. Any keystroke still interrupts a reply exactly as
    // before; the bytes stay queued, so typing OVER the agent cuts it off and
    // the characters you typed open the next turn's line.
    var interrupt = std.atomic.Value(bool).init(false);
    var keys = KeyQueue{ .allocator = allocator };
    defer keys.bytes.deinit(allocator);
    var typed = Typed{};
    defer typed.buf.deinit(allocator);
    var key_scratch: std.ArrayList(u8) = .empty;
    defer key_scratch.deinit(allocator);

    enableRawMode();
    defer restoreTermios();

    const stdin_thread = try std.Thread.spawn(.{}, struct {
        fn run(flag: *std.atomic.Value(bool), q: *KeyQueue) void {
            var buf: [64]u8 = undefined;
            while (true) {
                const n = std.posix.read(0, &buf) catch return;
                if (n == 0) return;
                q.push(buf[0..n]);
                flag.store(true, .release);
            }
        }
    }.run, .{ &interrupt, &keys });
    stdin_thread.detach();

    // --- conversation loop --------------------------------------------------
    barge = .{
        .pump = &pump,
        .streamer = &streamer,
        .engine = &engine,
        .pieces = pieces,
        .reply_acc = &reply_acc,
        .allocator = allocator,
    };
    const resetStt = struct {
        fn go(st: *IncrementalStreamer, b: *BargeCtx, seen: *usize) void {
            st.reset();
            st.sess.tokens.clearRetainingCapacity();
            st.sess.enc.reset();
            st.sess.state.reset();
            st.sess.frames_consumed = 0;
            seen.* = st.sess.eou_events;
            b.last_token_count = 0;
            b.fill = 0;
        }
    }.go;

    var last_token_count: usize = 0;
    var eou_seen: usize = sess.eou_events;
    var reply_buf: [256]u8 = undefined;
    var continue_session = false;
    var interrupts_fired: usize = 0;
    var last_reply: []u8 = &.{};
    defer if (last_reply.len > 0) allocator.free(last_reply);

    while (true) {
        if (railctl.state != .interrupted and railctl.state != .complete) railctl.set(.listening);
        barge.beginTurn(continue_session);
        // A voice barge carries its in-flight utterance (same STT session)
        // into LISTEN — nothing the user said is lost, no splice replay.
        // The mic/ref rings are NEVER discarded: they are pumped in every
        // phase, so mic-vs-reference alignment survives turn boundaries.
        if (!continue_session) {
            resetStt(&streamer, &barge, &eou_seen);
            last_token_count = 0;
        }
        continue_session = false;
        interrupt.store(false, .release);

        // LISTEN until the EOU model closes the utterance: the streamer is
        // fed the AEC RESIDUAL (raw mic when --no-aec) via BargeCtx.
        const t_listen = nowNs(io);
        // The buffer is per-turn; bytes typed over the previous reply are
        // still on the QUEUE and get applied fresh below.
        typed.reset();
        tui.promptBegin();
        const utterance: []u8 = listen: while (true) {
            if (sim) |sd| if (sd.done.load(.acquire)) {
                const tb: f64 = if (sd.barge_at) |b| @as(f64, @floatFromInt(b)) / 48000.0 else -1;
                tui.line("[sim] {s} at t={d:.1} s (barge injected at {d:.1} s), interrupts {d}, underruns {d}", .{
                    sd.reason,
                    @as(f64, @floatFromInt(sd.t)) / 48000.0,
                    tb,
                    interrupts_fired,
                    engine.underruns.load(.monotonic),
                });
                return;
            };
            try barge.scan();
            // A breath/noise EOU with an empty transcript must not
            // truncate the NEXT utterance to its first token: resync.
            if (sess.eou_events > eou_seen and sess.tokens.items.len == 0) {
                eou_seen = sess.eou_events;
            }
            keys.drain(&key_scratch);
            if (key_scratch.items.len > 0) {
                // Seed from whatever the STT has heard so far, so the first
                // keystroke lands at the end of the transcript.
                const heard: []u8 = if (typed.active or sess.tokens.items.len == 0)
                    try allocator.dupe(u8, "")
                else
                    try parakeet_tokenizer.detokenize(allocator, pieces, sess.tokens.items);
                defer allocator.free(heard);
                switch (applyKeys(&typed, allocator, key_scratch.items, heard)) {
                    .submit => break :listen try allocator.dupe(u8, typed.buf.items),
                    // Send the transcript without waiting for the EOU model.
                    .submit_suggestion => break :listen try allocator.dupe(u8, heard),
                    .edited => tui.typedUpdate(typed.buf.items),
                    // Enter with nothing to send, and no reply to interrupt.
                    .interrupt, .none => {},
                }
            }
            if (!typed.active and sess.tokens.items.len != last_token_count) {
                last_token_count = sess.tokens.items.len;
                const text = try parakeet_tokenizer.detokenize(allocator, pieces, sess.tokens.items);
                defer allocator.free(text);
                tui.partialUpdate(text);
            }
            railctl.hold_listening = typed.active or sess.tokens.items.len != 0;
            railctl.draw(pump.res_ema);
            // Typing takes the turn off endpointing: it ends on Enter.
            if (!typed.active and sess.eou_events > eou_seen and sess.tokens.items.len > 0) {
                const text = try parakeet_tokenizer.detokenize(allocator, pieces, sess.tokens.items);
                break :listen text;
            }
            std.Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .awake) catch {};
        };
        railctl.set(.captured);
        // CAPTURED is semantically visible (the input-committed collapse):
        // draw it NOW — the next natural draw waits on chat prefill, which
        // often outlives the 4-tick transition.
        railctl.draw(pump.res_ema);
        defer allocator.free(utterance);
        // Typed text is not a transcript: it has no leading echo words to
        // slice, and it cannot be a self-transcription.
        const utt = if (typed.active) utterance else skipWords(utterance, barge.strip_words);
        barge.strip_words = 0;
        const t_eou = nowNs(io);
        if (utt.len == 0) continue;

        if (pump.lock_ms_event) |ms| {
            pump.lock_ms_event = null;
            tui.line("\x1b[90m[aec] echo-path delay locked: {d} ms\x1b[0m", .{ms});
        }
        const tail_window = t_eou - t_listen < 2 * std.time.ns_per_s;
        if (!typed.active and echoGuard(utt, last_reply, tail_window)) {
            tui.line("\x1b[90m[echo-guard] ignored self-transcription: \"{s}\"\x1b[0m", .{utt});
            continue;
        }

        if (tui.hint_turns > 0) tui.hint_turns -= 1;
        tui.userFinal(utt);
        railctl.last_tick = std.math.maxInt(u64);
        railctl.draw(pump.res_ema); // keep the collapse on screen through the gap

        // THINK: fresh STT session for the reply window (the finished
        // utterance's tokens must not count as interrupting words), then the
        // reply streams onto the reply line while the residual listener
        // watches for the user talking over the think gap.
        resetStt(&streamer, &barge, &eou_seen);
        last_token_count = 0;
        reply_acc.clearRetainingCapacity();
        barge.phase = .think;
        var reply = ReplyWriter.init(allocator, &tui, &barge, &railctl, &reply_acc, &reply_buf);
        _ = try convo.send(utt, &reply.interface);
        try reply.interface.flush();
        tui.closeLine();
        const reply_text = std.mem.trim(u8, reply_acc.items, " \n");
        if (reply_text.len == 0) {
            if (barge.pending) {
                interrupts_fired += 1;
                continue_session = full_duplex;
                railctl.set(.interrupted);
                tui.line("\x1b[90m[turn] interrupted during think\x1b[0m", .{});
            }
            continue;
        }
        // --- tool rounds: Hermes <tool_call> → run locally under ACTING →
        // <tool_response> back → the model's follow-up becomes the reply.
        var spoken = try allocator.dupe(u8, reply_text);
        defer allocator.free(spoken);
        var tool_rounds: usize = 0;
        while (tools_on and tool_rounds < 3) {
            const tc = parseToolCall(spoken) orelse break;
            tool_rounds += 1;
            var result_buf: [128]u8 = undefined;
            var result: []const u8 = "ok";
            if (std.mem.eql(u8, tc.name, "set_timer")) {
                const total_s = std.math.clamp(tc.seconds, 1, 600);
                tui.line("\x1b[90m[tool] set_timer {d:.0} s\x1b[0m", .{total_s});
                railctl.set(.acting);
                const t_start = nowNs(io);
                var cancelled = false;
                while (true) {
                    const el = @as(f64, @floatFromInt(nowNs(io) - t_start)) / 1e9;
                    if (el >= total_s) break;
                    railctl.progress = @floatCast(el / total_s);
                    railctl.draw(pump.res_ema);
                    if (full_duplex) barge.scan() catch {};
                    if (interrupt.load(.acquire) or barge.pending) {
                        cancelled = true;
                        interrupt.store(false, .release);
                        barge.pending = false;
                        break;
                    }
                    std.Io.sleep(io, .{ .nanoseconds = 20 * std.time.ns_per_ms }, .awake) catch {};
                }
                railctl.progress = null;
                result = if (cancelled)
                    std.fmt.bufPrint(&result_buf, "timer cancelled by the user", .{}) catch "cancelled"
                else
                    std.fmt.bufPrint(&result_buf, "timer finished after {d:.0} seconds", .{total_s}) catch "done";
            } else if (std.mem.eql(u8, tc.name, "get_time")) {
                tui.line("\x1b[90m[tool] get_time\x1b[0m", .{});
                railctl.set(.acting);
                var i: usize = 0;
                while (i < 5) : (i += 1) { // brief indeterminate flash
                    railctl.last_tick = std.math.maxInt(u64);
                    railctl.draw(pump.res_ema);
                    std.Io.sleep(io, .{ .nanoseconds = rail_mod.tick_ns }, .awake) catch {};
                }
                // wall-clock via the awake clock epoch offset is unavailable;
                // derive HH:MM (UTC) from the realtime epoch seconds.
                const epoch_s: u64 = @intCast(@divTrunc(std.Io.Clock.real.now(io).nanoseconds, std.time.ns_per_s));
                const day_s = epoch_s % 86400;
                result = std.fmt.bufPrint(&result_buf, "{d:0>2}:{d:0>2} UTC", .{
                    day_s / 3600,
                    (day_s % 3600) / 60,
                }) catch "unknown";
            } else break;
            railctl.set(.thinking);

            var resp_buf: [256]u8 = undefined;
            const tool_msg = std.fmt.bufPrint(&resp_buf, "<tool_response>\n{s}\n</tool_response>", .{result}) catch break;
            reply_acc.clearRetainingCapacity();
            var reply2 = ReplyWriter.init(allocator, &tui, &barge, &railctl, &reply_acc, &reply_buf);
            tui.userFinal("(tool result)");
            _ = try convo.send(tool_msg, &reply2.interface);
            try reply2.interface.flush();
            tui.closeLine();
            allocator.free(spoken);
            spoken = try allocator.dupe(u8, std.mem.trim(u8, reply_acc.items, " \n"));
        }
        // never speak a raw tool-call block
        const reply_text2: []const u8 = if (std.mem.indexOf(u8, spoken, "<tool_call>")) |off|
            std.mem.trim(u8, spoken[0..off], " \n")
        else
            spoken;
        if (reply_text2.len == 0) continue;

        if (last_reply.len > 0) allocator.free(last_reply);
        last_reply = try allocator.dupe(u8, reply_text2);

        // SPEAK: pocket streams 80 ms frames straight to the out-ring;
        // qwen3-tts goes through the pipelined codec worker.
        railctl.set(.speaking);
        // Only Enter pressed FROM HERE interrupts this reply; a fresh onset
        // grace for the auxiliary energy gate.
        interrupt.store(false, .release);
        barge.phase = .speak;
        pump.resetGate();
        engine.armWarmup();
        engine.setSpeaking(true);
        engine.gaps.store(0, .monotonic);
        var frames_spoken: usize = 0;
        var first_push: ?i96 = null;
        reveal.begin(reply_text2);

        if (pocket_engine) |*pe| {
            const PocketCb = struct {
                engine: *duplex.Engine,
                up2: *duplex.Upsampler2,
                barge: *BargeCtx,
                interrupt: *std.atomic.Value(bool),
                abort: ?*const std.atomic.Value(bool),
                full_duplex: bool,
                io: std.Io,
                first_push_ns: *?i96,
                rail: *RailCtl,
                reveal: *Reveal,

                fn stop(c: *const @This()) bool {
                    if (c.abort) |a| if (a.load(.acquire)) return true;
                    return c.interrupt.load(.acquire) or c.barge.pending;
                }

                fn onFrame(c: *@This(), pcm: []const f32) bool {
                    if (c.stop()) return false;
                    if (c.full_duplex) c.barge.scan() catch {};
                    if (c.stop()) return false;
                    var acc: f32 = 0;
                    for (pcm) |x| acc += x * x;
                    c.rail.setOutputLevel(@sqrt(acc / @as(f32, @floatFromInt(pcm.len))));
                    c.rail.set(if (c.barge.stage == .paused) .waiting else .speaking);
                    c.rail.draw(c.barge.pump.res_ema);
                    c.reveal.step();
                    for (pcm) |x| {
                        var pair: [2]f32 = undefined;
                        c.up2.push(x, &pair);
                        var off: usize = 0;
                        while (off < 2) {
                            off += c.engine.out48.push(pair[off..]);
                            if (off == 2) break;
                            if (c.stop()) return false;
                            std.Io.sleep(c.io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .awake) catch {};
                        }
                        c.reveal.notePushed(@abs(x) > Reveal.voice_thresh);
                    }
                    if (c.first_push_ns.* == null) c.first_push_ns.* = nowNs(c.io);
                    return true;
                }
            };
            var pcb = PocketCb{
                .engine = &engine,
                .up2 = &up2,
                .barge = &barge,
                .interrupt = &interrupt,
                .abort = if (sim) |sd| &sd.done else null,
                .full_duplex = full_duplex,
                .io = io,
                .first_push_ns = &first_push,
                .rail = &railctl,
                .reveal = &reveal,
            };
            frames_spoken = try pe.speak(reply_text2, .{
                .temp = pe.temp_default,
                .seed = @bitCast(seed),
            }, &pcb, PocketCb.onFrame);
            reveal.setTotalFromPushed();
            reveal.markAllPushed();
            if (!pcb.stop()) {
                engine.releaseWarmup();
                while (engine.outPending() > 0) {
                    if (engine.outPending() < duplex.period_frames * 4) engine.setSpeaking(false);
                    if (full_duplex) barge.scan() catch {};
                    railctl.set(if (barge.stage == .paused) .waiting else .speaking);
                    railctl.hold_listening = sess.tokens.items.len != 0;
            railctl.draw(pump.res_ema);
                    reveal.step();
                    if (pcb.stop()) break;
                    std.Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .awake) catch {};
                }
            }
        } else {
            var prompt = try qtts.prompt.build(allocator, &tts_model, &tts_tok, .{
                .text = reply_text2,
                .language = lang,
                .speaker = speaker,
            });
            defer prompt.deinit();
            var sink = SpeakSink{
                .ctx = &codec_ctx,
                .dec = &codec_dec,
                .sess = &codec_sess,
                .engine = &engine,
                .up2 = &up2,
                .barge = &barge,
                .full_duplex = full_duplex,
                .allocator = allocator,
                .interrupt = &interrupt,
                .abort = if (sim) |sd| &sd.done else null,
                .rail = &railctl,
                .reveal = &reveal,
                .io = io,
                .kq = tts_model.specials.num_code_groups,
                .chunk_first = 12,
                .chunk_main = 12,
                .left_ctx = 0,
            };
            defer sink.frames.deinit(allocator);
            try sink.startWorker();
            // A generate error must not unwind past a live worker: it would
            // keep decoding into frames/contexts the defers tear down.
            errdefer sink.stopWorker();

            var result = try qtts.pipeline.generate(&tts_ctx, &tts_model, &prompt, .{
                .seed = seed,
                .max_new_tokens = 512,
            }, &tts_kvs, SpeakSink.onFrame, &sink, null);
            defer result.deinit();

            if (!sink.stopRequested() and sink.err == null) {
                sink.finish() catch |e| {
                    sink.err = e;
                };
                // Frame count is final: the reveal switches to exact pacing.
                reveal.setTotal(@as(f64, @floatFromInt(sink.frames.items.len / sink.kq)) *
                    @as(f64, @floatFromInt(qtts.codec.hop_length)) / 24000.0);
                // Let the worker decode the tail while we keep scanning.
                while (!sink.queueIdle() and !sink.stopRequested()) {
                    if (full_duplex) barge.scan() catch {};
                    reveal.step();
                    std.Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .awake) catch {};
                }
                // Final ONLY on the clean exit: a stop leaves the worker
                // mid-job, still raising the voiced high-water mark — a
                // premature "final" maps played audio onto a shrunken span
                // and bright-reveals words the voice never spoke.
                if (!sink.stopRequested()) reveal.markAllPushed();
                engine.releaseWarmup(); // short replies below the cushion play now
                // Drain: wait for the out-ring, then the device tail.
                while (engine.outPending() > 0) {
                    if (engine.outPending() < duplex.period_frames * 4) engine.setSpeaking(false);
                    if (full_duplex) barge.scan() catch {};
                    railctl.set(if (barge.stage == .paused) .waiting else .speaking);
                    railctl.hold_listening = sess.tokens.items.len != 0;
            railctl.draw(pump.res_ema);
                    reveal.step();
                    if (sink.stopRequested()) break;
                    std.Io.sleep(io, .{ .nanoseconds = 10 * std.time.ns_per_ms }, .awake) catch {};
                }
            }
            sink.stopWorker();
            if (sink.err) |e| return @as(anyerror!void, e);
            frames_spoken = result.frames;
            first_push = sink.first_push_ns;
        }
        engine.setSpeaking(false);

        if (interrupt.load(.acquire) or barge.pending) {
            // Cut the reveal BEFORE discarding the queue: the discard makes
            // played jump to pushed, which would reveal unspoken words.
            reveal.cutNow();
            engine.setPaused(false);
            engine.releaseWarmup();
            engine.discard();
            // Wait out the callback's discard (normally one 5 ms period,
            // bounded in case the device stalls): a discard left pending
            // could fire mid-NEXT-reply and drop counted samples, putting
            // that whole turn's reveal ahead of its audio.
            var discard_spins: usize = 0;
            while (engine.discard_requested.load(.acquire) and discard_spins < 100) : (discard_spins += 1) {
                std.Io.sleep(io, .{ .nanoseconds = 2 * std.time.ns_per_ms }, .awake) catch {};
            }
            interrupts_fired += 1;
            // A voice barge continues its utterance in LISTEN; Enter starts
            // the next turn clean.
            continue_session = full_duplex and barge.pending;
            // The rail carries the INTERRUPTED status (retract + hard cut);
            // the log keeps only the permanent per-turn record. The reply
            // line keeps the full generated text — spoken words bright,
            // the unspoken remainder dimmed (all bright under --eager-text).
            railctl.set(.interrupted);
            tui.line("\x1b[90m[turn] interrupted — {d} frames generated, queue discarded\x1b[0m", .{frames_spoken});
        } else {
            reveal.finishAll();
            std.Io.sleep(io, .{ .nanoseconds = 60 * std.time.ns_per_ms }, .awake) catch {};
            const ttfa_s = if (first_push) |t1| @as(f64, @floatFromInt(t1 - t_eou)) / 1e9 else 0.0;
            // The user may have started speaking near the reply tail without
            // reaching the barge threshold — carry those words forward.
            const tail_novel = if (full_duplex) barge.novelCount() catch 0 else 0;
            continue_session = tail_novel >= 1;
            railctl.question_pending = std.mem.endsWith(u8, std.mem.trimEnd(u8, reply_text2, " "), "?");
            railctl.set(.complete);
            tui.line("\x1b[90m[turn] first-audio-queued {d:.1} s, {d} frames, audible gaps {d}\x1b[0m", .{
                ttfa_s,
                frames_spoken,
                engine.gaps.load(.monotonic),
            });
        }
        barge.phase = .idle;
    }
}

test {
    _ = @import("aec.zig");
    _ = @import("rail.zig");
}

test "typed input: the suggestion is accepted explicitly, never implicitly" {
    const a = std.testing.allocator;
    var t = Typed{};
    defer t.buf.deinit(a);

    // Enter means SUBMIT in every state: with the suggestion still dim it
    // sends the transcript, so `→` can never strand you in a state where
    // nothing submits.
    try std.testing.expectEqual(KeyAction.submit_suggestion, applyKeys(&t, a, "\r", "hello world"));
    try std.testing.expect(!t.active);
    // Only with nothing to send at all is it an interrupt.
    try std.testing.expectEqual(KeyAction.interrupt, applyKeys(&t, a, "\r", ""));
    try std.testing.expect(!t.active);

    // Typing WITHOUT accepting starts clean — the suggestion is discarded,
    // never prepended.
    try std.testing.expectEqual(KeyAction.edited, applyKeys(&t, a, "hi", "hello world"));
    try std.testing.expectEqualStrings("hi", t.buf.items);
    try std.testing.expectEqual(KeyAction.submit, applyKeys(&t, a, "\r", ""));

    // Right arrow accepts it verbatim, and later keys append to it.
    var t2 = Typed{};
    defer t2.buf.deinit(a);
    try std.testing.expectEqual(KeyAction.edited, applyKeys(&t2, a, "\x1b[C", "hello world"));
    try std.testing.expect(t2.active);
    try std.testing.expectEqualStrings("hello world", t2.buf.items);
    _ = applyKeys(&t2, a, "!", "hello world and more");
    try std.testing.expectEqualStrings("hello world!", t2.buf.items);

    // End and ctrl-E are the same gesture; accepting twice is a no-op.
    var t3 = Typed{};
    defer t3.buf.deinit(a);
    _ = applyKeys(&t3, a, "\x05", "from ctrl-e");
    try std.testing.expectEqualStrings("from ctrl-e", t3.buf.items);
    _ = applyKeys(&t3, a, "\x1b[C", "should not re-adopt");
    try std.testing.expectEqualStrings("from ctrl-e", t3.buf.items);

    // Accepting an empty suggestion does nothing at all.
    var t4 = Typed{};
    defer t4.buf.deinit(a);
    try std.testing.expectEqual(KeyAction.none, applyKeys(&t4, a, "\x1b[C", ""));
    try std.testing.expect(!t4.active);
}

test "typed input: editing keys" {
    const a = std.testing.allocator;
    var t = Typed{};
    defer t.buf.deinit(a);

    _ = applyKeys(&t, a, "ab caf\u{00e9}", ""); // multibyte tail
    try std.testing.expectEqualStrings("ab caf\u{00e9}", t.buf.items);

    // Backspace removes a whole codepoint, not one byte.
    _ = applyKeys(&t, a, "\x7F", "");
    try std.testing.expectEqualStrings("ab caf", t.buf.items);

    // ctrl-W drops the trailing word; ctrl-U clears the line.
    _ = applyKeys(&t, a, "\x17", "");
    try std.testing.expectEqualStrings("ab ", t.buf.items);
    _ = applyKeys(&t, a, "\x15", "");
    try std.testing.expectEqualStrings("", t.buf.items);

    // Arrow keys are swallowed whole rather than inserted as garbage.
    _ = applyKeys(&t, a, "x\x1b[Dy", "");
    try std.testing.expectEqualStrings("xy", t.buf.items);
}

// Pins the delay scan's core `conv1d` correlation against a direct scalar
// reference: the same values (SIMD reassociation aside) and, what the lock
// actually consumes, the same argmax lag.
test "aec delay scan: core conv1d correlation matches the scalar reference" {
    const W = 1600;
    const max_lag = 4800;
    const plant = 1234; // ref carries mic delayed by this many samples

    var ctx: ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();

    var prng = std.Random.DefaultPrng.init(0xA1CE);
    const random = prng.random();
    var micw: [W]f32 = undefined;
    for (&micw) |*v| v.* = random.floatNorm(f32);
    var refc: [W + max_lag]f32 = undefined;
    for (&refc) |*v| v.* = 0.05 * random.floatNorm(f32);
    // The lag-l window starts at refc[max_lag - l], so plant the echo there.
    for (0..W) |n| refc[max_lag - plant + n] += micw[n];

    var want: [max_lag + 1]f32 = undefined;
    for (0..max_lag + 1) |lag| {
        var c: f32 = 0;
        const seg = refc[max_lag - lag ..][0..W];
        for (micw, seg) |x, y| c += x * y;
        want[lag] = c;
    }

    var corr_buf: [max_lag + 1]f32 = undefined;
    try correlate(&ctx, &refc, &micw, &corr_buf);
    const corr: []const f32 = &corr_buf;

    var worst: f32 = 0;
    var best: f32 = -std.math.inf(f32);
    var best_lag: usize = 0;
    for (0..max_lag + 1) |lag| {
        const got = corr[max_lag - lag];
        worst = @max(worst, @abs(got - want[lag]));
        if (got > best) {
            best = got;
            best_lag = lag;
        }
    }
    try std.testing.expectEqual(@as(usize, plant), best_lag);
    if (worst > 1e-2) {
        std.debug.print("[aec-delay] conv1d vs scalar max|diff| {d:.6} (peak {d:.1})\n", .{ worst, best });
        return error.TestUnexpectedResult;
    }
}
