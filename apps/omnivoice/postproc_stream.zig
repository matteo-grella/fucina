//! Streaming TTS waveform post-processing for the OmniVoice example.
//!
//! Port of `refs/omnivoice.cpp/src/audio-postproc-stream.h`: three stateful
//! stages chained as a pipeline, crossfader -> silence remover -> fade and
//! pad. Each stage exposes `push(samples, emit)` and `flush(emit)`; `emit`
//! forwards downstream (the reference's `bool(const float*, int)` abort
//! channel maps onto Zig error propagation from the sink).
//!
//! Order matches `_post_process_audio` in omnivoice.py (and the buffered
//! `pipeline.postFilter`):
//!   1. cross_fade_chunks (concat with fade out + silence + fade in)
//!   2. remove_silence (mid silence drop, edge trim)
//!   3. volume scale (ref_rms branch only; voice design skips here — the
//!      global peak is unknowable while streaming, so peak/0.5 never runs
//!      and voice-design output sits ~6-12 dB below the buffered path)
//!   4. fade_and_pad (fade in + fade out + leading/trailing silence pad)
//!
//! Equivalence to the buffered functions (per the reference's own analysis,
//! verified in postproc_stream_tests.zig):
//! - `Crossfader` is sample-exact vs `crossFadeChunks` when every chunk is
//!   at least fade_n samples (0.1 s) long — always true on OmniVoice chunks.
//! - `SilenceRemover` is sample-exact vs `removeSilence` on chunks where mid
//!   silent groups are scoped within the min_sil_n look-ahead horizon.
//! - `FadePad` is sample-exact vs `fadeAndPad` for inputs of at least
//!   2*fade_n samples; between fade_n and 2*fade_n the streaming version
//!   fades in over fade_n and out over the remainder while the batch one
//!   fades both ends over len/2 (documented reference divergence).

const std = @import("std");

const postproc = @import("postproc.zig");

const Allocator = std.mem.Allocator;

/// Downstream sample sink: type-erased context + function pointer. Stages
/// never call it with an empty slice. An error from the sink aborts the
/// whole pipeline (the reference's `emit(...) == false` path).
pub const Emit = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, samples: []const f32) anyerror!void,

    pub fn call(self: Emit, samples: []const f32) anyerror!void {
        return self.func(self.ctx, samples);
    }
};

// ---------------------------------------------------------------------------
// Stage 1: streaming cross fade (reference `crossfader_stream`)
// ---------------------------------------------------------------------------

/// Holds the last fade_n samples of the most recent emission as `pending`.
/// On the next push, applies fade out to the tail of pending, emits pending,
/// emits silence_n zeros, applies fade in to the head of the new chunk,
/// emits the new chunk minus its own tail. The retained tail becomes the new
/// pending. Last chunk: flush emits pending verbatim (no trailing fade out),
/// matching the Python behaviour of leaving the final chunk's tail intact.
pub const Crossfader = struct {
    allocator: Allocator,
    fade_n: usize,
    silence_n: usize,
    first_chunk: bool,
    pending: std.ArrayList(f32),

    pub fn init(allocator: Allocator, sr: i32, silence_dur: f64) Crossfader {
        const total_n: i32 = @intFromFloat(silence_dur * @as(f64, @floatFromInt(sr)));
        const fade_n = @divTrunc(total_n, 3);
        return .{
            .allocator = allocator,
            .fade_n = @intCast(@max(fade_n, 0)),
            .silence_n = @intCast(@max(fade_n, 0)),
            .first_chunk = true,
            .pending = .empty,
        };
    }

    pub fn deinit(self: *Crossfader) void {
        self.pending.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn push(self: *Crossfader, samples: []const f32, emit: Emit) !void {
        if (samples.len == 0) {
            return;
        }

        if (self.first_chunk) {
            // Emit body, retain last fade_n as pending.
            const emit_n = samples.len -| self.fade_n;
            if (emit_n > 0) {
                try emit.call(samples[0..emit_n]);
            }
            self.pending.clearRetainingCapacity();
            try self.pending.appendSlice(self.allocator, samples[emit_n..]);
            self.first_chunk = false;
            return;
        }

        // Fade out tail of pending.
        const fout_n = @min(self.fade_n, self.pending.items.len);
        if (fout_n > 0) {
            const denom: f32 = @floatFromInt(@max(fout_n - 1, 1));
            const tail = self.pending.items[self.pending.items.len - fout_n ..];
            for (tail, 0..) |*s, j| {
                const w = 1.0 - @as(f32, @floatFromInt(j)) / denom;
                s.* *= w;
            }
        }

        // Emit pending (now fade out applied at its tail).
        if (self.pending.items.len > 0) {
            try emit.call(self.pending.items);
        }

        // Emit silence gap.
        if (self.silence_n > 0) {
            const silence = try self.allocator.alloc(f32, self.silence_n);
            defer self.allocator.free(silence);
            @memset(silence, 0.0);
            try emit.call(silence);
        }

        // Apply fade in to the first fade_n of the new chunk in a copy.
        const chunk_copy = try self.allocator.dupe(f32, samples);
        defer self.allocator.free(chunk_copy);
        const fin_n = @min(self.fade_n, chunk_copy.len);
        if (fin_n > 0) {
            const denom: f32 = @floatFromInt(@max(fin_n - 1, 1));
            for (chunk_copy[0..fin_n], 0..) |*s, j| {
                const w = @as(f32, @floatFromInt(j)) / denom;
                s.* *= w;
            }
        }

        // Emit body of the new chunk, retain last fade_n as new pending.
        const emit_n = chunk_copy.len -| self.fade_n;
        if (emit_n > 0) {
            try emit.call(chunk_copy[0..emit_n]);
        }
        self.pending.clearRetainingCapacity();
        try self.pending.appendSlice(self.allocator, chunk_copy[emit_n..]);
    }

    pub fn flush(self: *Crossfader, emit: Emit) !void {
        if (self.pending.items.len > 0) {
            try emit.call(self.pending.items);
            self.pending.clearRetainingCapacity();
        }
    }
};

// ---------------------------------------------------------------------------
// Stage 2: streaming silence remover (reference `silence_remover_stream`)
// ---------------------------------------------------------------------------

/// Accumulates pushed samples (float and int16 in lockstep, quantized with
/// the exact pydub recipe) and advances an emit cursor as scan windows
/// close. Trail trim runs at flush by reversing the un-emitted suffix and
/// reusing `postproc.detectLeadingSilence`.
///
/// Latency: up to min_sil_n samples (500 ms at 24 kHz, mid_sil=500). Once a
/// silent group closes (next non-silent seek_step), the prefix up to its
/// determined drop boundary emits in one shot.
pub const SilenceRemover = struct {
    allocator: Allocator,
    sr: i32,
    trail_sil_ms: i32,
    thresh_lin: f64,

    seek_step: usize,
    min_sil_n: usize,
    keep_n: usize,
    chunk_n: i32,
    lead_keep: usize,

    buf_f: std.ArrayList(f32),
    buf_s: std.ArrayList(i16),
    emit_pos: usize,
    lead_done: bool,
    scan_pos: usize,
    /// Contiguous run of silent window starts (the reference keeps a vector
    /// but only reads front()/back(): first/last are sufficient state).
    silent_grp: ?struct { first: usize, last: usize },

    pub fn init(
        allocator: Allocator,
        sr: i32,
        mid_sil_ms: i32,
        lead_sil_ms: i32,
        trail_sil_ms: i32,
        thresh_db: f64,
    ) SilenceRemover {
        return .{
            .allocator = allocator,
            .sr = sr,
            .trail_sil_ms = trail_sil_ms,
            .thresh_lin = 32768.0 * std.math.pow(f64, 10.0, thresh_db / 20.0),
            .seek_step = @intCast(@divTrunc(sr, 100)),
            .min_sil_n = @intCast(@divTrunc(sr * mid_sil_ms, 1000)),
            .keep_n = @intCast(@divTrunc(sr * mid_sil_ms, 1000)),
            .chunk_n = @divTrunc(sr, 100),
            .lead_keep = @intCast(@divTrunc(sr * lead_sil_ms, 1000)),
            .buf_f = .empty,
            .buf_s = .empty,
            .emit_pos = 0,
            .lead_done = false,
            .scan_pos = 0,
            .silent_grp = null,
        };
    }

    pub fn deinit(self: *SilenceRemover) void {
        self.buf_f.deinit(self.allocator);
        self.buf_s.deinit(self.allocator);
        self.* = undefined;
    }

    /// Appends samples to both buffers (reference `append`), quantizing with
    /// the same recipe as `postproc.f32ToS16`.
    fn append(self: *SilenceRemover, samples: []const f32) !void {
        try self.buf_f.appendSlice(self.allocator, samples);
        try self.buf_s.ensureUnusedCapacity(self.allocator, samples.len);
        for (samples) |x| {
            self.buf_s.appendAssumeCapacity(postproc.f32ToS16Sample(x));
        }
    }

    pub fn push(self: *SilenceRemover, samples: []const f32, emit: Emit) !void {
        if (samples.len == 0) {
            return;
        }
        try self.append(samples);

        // Lead trim phase. Scan chunks of chunk_n until the first non-silent
        // is found; until then no emission happens.
        if (!self.lead_done) {
            const trim: usize = @intCast(postproc.detectLeadingSilence(self.buf_s.items, self.thresh_lin, self.chunk_n));
            if (trim >= self.buf_s.items.len) {
                return;
            }
            const p = trim -| self.lead_keep;
            self.emit_pos = p;
            self.lead_done = true;
            self.scan_pos = self.emit_pos;
        }

        // Mid scan: advance scan_pos by seek_step while we have a full
        // min_sil_n window ahead. Track contiguous silent runs.
        while (self.scan_pos + self.min_sil_n <= self.buf_s.items.len) {
            const r = postproc.sliceRmsS16(self.buf_s.items, self.scan_pos, self.min_sil_n);
            const is_silent = r <= self.thresh_lin;
            if (is_silent) {
                if (self.silent_grp) |*grp| {
                    grp.last = self.scan_pos;
                } else {
                    self.silent_grp = .{ .first = self.scan_pos, .last = self.scan_pos };
                }
            } else if (self.silent_grp != null) {
                try self.closeSilentGroup(emit);
            }
            self.scan_pos += self.seek_step;
        }

        // Emit safe prefix: when no silent group is pending, every sample up
        // to scan_pos is decided. With a pending group, hold everything until
        // the group closes (its drop boundary depends on the group's end).
        if (self.silent_grp == null and self.scan_pos > self.emit_pos) {
            try emit.call(self.buf_f.items[self.emit_pos..self.scan_pos]);
            self.emit_pos = self.scan_pos;
        }
    }

    /// Closes the active silent group [s, e] following the pydub pairwise
    /// midpoint dedup rule: if the gap is at least 2*keep_n, drop the middle
    /// and keep keep_n on each side; otherwise keep all samples but split the
    /// overlap at (s + e) / 2 to avoid duplicating samples in the concat.
    fn closeSilentGroup(self: *SilenceRemover, emit: Emit) !void {
        const grp = self.silent_grp.?;
        const s = grp.first;
        const e = grp.last + self.min_sil_n;
        if (e - s >= 2 * self.keep_n) {
            const emit_end = s + self.keep_n;
            if (emit_end > self.emit_pos) {
                try emit.call(self.buf_f.items[self.emit_pos..emit_end]);
            }
            self.emit_pos = e - self.keep_n;
        } else {
            const mid = (s + e) / 2;
            if (mid > self.emit_pos) {
                try emit.call(self.buf_f.items[self.emit_pos..mid]);
            }
            self.emit_pos = mid;
        }
        self.silent_grp = null;
    }

    /// Trail trim: reverse the un-emitted suffix and run the same leading
    /// silence detector as `removeSilence` does. Trailing silence beyond
    /// trail_sil_ms gets dropped, matching the buffered path verbatim.
    pub fn flush(self: *SilenceRemover, emit: Emit) !void {
        // A still-open silent group at flush is the trailing silence: the
        // pydub split_on_silence keeps margin keep_n past the last non
        // silent, then the trail trim reduces that margin to trail_sil_ms.
        var end_emit: usize = self.buf_s.items.len;
        if (self.silent_grp) |grp| {
            end_emit = @min(grp.first + self.keep_n, self.buf_s.items.len);
            self.silent_grp = null;
        }

        if (self.emit_pos >= end_emit) {
            return;
        }

        // Build a reversed int16 view of the un-emitted suffix and let the
        // existing detector find the trailing silence length in samples.
        const rev = try self.allocator.dupe(i16, self.buf_s.items[self.emit_pos..end_emit]);
        defer self.allocator.free(rev);
        std.mem.reverse(i16, rev);

        const trim_back = postproc.detectLeadingSilence(rev, self.thresh_lin, self.chunk_n);
        const trail_keep = @divTrunc(self.sr * self.trail_sil_ms, 1000);
        const drop_trail: usize = @intCast(@max(0, trim_back - trail_keep));

        var emit_n = end_emit - self.emit_pos;
        if (drop_trail >= emit_n) {
            self.emit_pos = end_emit;
            return;
        }
        emit_n -= drop_trail;
        try emit.call(self.buf_f.items[self.emit_pos..][0..emit_n]);
        self.emit_pos = end_emit;
    }
};

// ---------------------------------------------------------------------------
// Stage 3: streaming fade and pad (reference `fade_pad_stream`)
// ---------------------------------------------------------------------------

/// Emits pad_n zeros at start, holds the first fade_n samples to apply fade
/// in, then streams the body while keeping the last fade_n samples as a tail
/// buffer for the closing fade out. Flush applies fade out, emits the tail,
/// then emits pad_n trailing zeros.
pub const FadePad = struct {
    allocator: Allocator,
    fade_n: usize,
    pad_n: usize,

    started: bool,
    head_faded_in: bool,
    head_buf: std.ArrayList(f32),
    tail_buf: std.ArrayList(f32),

    pub fn init(allocator: Allocator, sr: i32, fade_dur: f64, pad_dur: f64) FadePad {
        const fade_n: i32 = @intFromFloat(fade_dur * @as(f64, @floatFromInt(sr)));
        const pad_n: i32 = @intFromFloat(pad_dur * @as(f64, @floatFromInt(sr)));
        return .{
            .allocator = allocator,
            .fade_n = @intCast(@max(fade_n, 0)),
            .pad_n = @intCast(@max(pad_n, 0)),
            .started = false,
            .head_faded_in = false,
            .head_buf = .empty,
            .tail_buf = .empty,
        };
    }

    pub fn deinit(self: *FadePad) void {
        self.head_buf.deinit(self.allocator);
        self.tail_buf.deinit(self.allocator);
        self.* = undefined;
    }

    fn emitZeros(self: *FadePad, n: usize, emit: Emit) !void {
        if (n == 0) {
            return;
        }
        const z = try self.allocator.alloc(f32, n);
        defer self.allocator.free(z);
        @memset(z, 0.0);
        try emit.call(z);
    }

    pub fn push(self: *FadePad, samples: []const f32, emit: Emit) !void {
        if (samples.len == 0) {
            return;
        }

        // Leading pad: emit pad_n zeros once, on the first non-empty push.
        if (!self.started) {
            try self.emitZeros(self.pad_n, emit);
            self.started = true;
        }

        var remaining = samples;

        // Phase 1: collect first fade_n samples, apply fade in, emit.
        if (!self.head_faded_in) {
            const need = self.fade_n - self.head_buf.items.len;
            const take = @min(need, remaining.len);
            try self.head_buf.appendSlice(self.allocator, remaining[0..take]);
            remaining = remaining[take..];

            if (self.head_buf.items.len == self.fade_n) {
                const denom: f32 = @floatFromInt(@max(self.fade_n -| 1, 1));
                for (self.head_buf.items, 0..) |*s, j| {
                    s.* *= @as(f32, @floatFromInt(j)) / denom;
                }
                if (self.fade_n > 0) {
                    try emit.call(self.head_buf.items);
                }
                self.head_buf.clearRetainingCapacity();
                self.head_faded_in = true;
            } else {
                return;
            }
        }

        if (remaining.len == 0) {
            return;
        }

        // Phase 2: append to tail_buf, emit all but the last fade_n.
        try self.tail_buf.appendSlice(self.allocator, remaining);
        if (self.tail_buf.items.len > self.fade_n) {
            const emit_n = self.tail_buf.items.len - self.fade_n;
            try emit.call(self.tail_buf.items[0..emit_n]);
            self.tail_buf.replaceRangeAssumeCapacity(0, emit_n, &.{});
        }
    }

    pub fn flush(self: *FadePad, emit: Emit) !void {
        if (!self.started) {
            return;
        }

        if (!self.head_faded_in) {
            // Total audio shorter than fade_n. Apply degraded fade with
            // k = total / 2, matching fade_and_pad_audio for short inputs.
            const total = self.head_buf.items.len;
            const k = @min(self.fade_n, total / 2);
            if (k > 0) {
                const denom: f32 = @floatFromInt(@max(k - 1, 1));
                for (self.head_buf.items[0..k], 0..) |*s, j| {
                    s.* *= @as(f32, @floatFromInt(j)) / denom;
                }
                const tail = self.head_buf.items[total - k ..];
                for (tail, 0..) |*s, j| {
                    s.* *= 1.0 - @as(f32, @floatFromInt(j)) / denom;
                }
            }
            if (total > 0) {
                try emit.call(self.head_buf.items);
            }
            self.head_buf.clearRetainingCapacity();
        } else {
            // Tail buffer holds the last fade_n samples. Apply fade out and
            // emit. For inputs of at least 2*fade_n samples (always true on
            // OmniVoice outputs), this matches fade_and_pad_audio exactly.
            const k = self.tail_buf.items.len;
            if (k > 0) {
                const denom: f32 = @floatFromInt(@max(k - 1, 1));
                for (self.tail_buf.items, 0..) |*s, j| {
                    s.* *= 1.0 - @as(f32, @floatFromInt(j)) / denom;
                }
                try emit.call(self.tail_buf.items);
            }
            self.tail_buf.clearRetainingCapacity();
        }

        // Trailing pad.
        try self.emitZeros(self.pad_n, emit);
    }
};

// ---------------------------------------------------------------------------
// The OmniVoice streaming chain (reference tts_synthesize_long_stream_internal
// stage wiring: cf -> silence remove -> volume scale -> fade and pad -> sink)
// ---------------------------------------------------------------------------

/// The fixed streaming post-proc chain with the reference constants:
/// cross fade 0.3 s, silence remove (mid=500, lead=100, trail=100, -50 dBFS),
/// volume scale (resolved up front by the caller), fade+pad 0.1/0.1 s.
/// `pushChunk` feeds one fully synthesized chunk; `finish` drains the stages
/// in pipeline order (cf.flush -> sr.flush -> fp.flush).
pub const Pipeline = struct {
    allocator: Allocator,
    cf: Crossfader,
    sr_stage: SilenceRemover,
    fp: FadePad,
    volume_scale: f32,
    sink: Emit,

    pub fn init(allocator: Allocator, sr: i32, volume_scale: f32, sink: Emit) Pipeline {
        return .{
            .allocator = allocator,
            .cf = Crossfader.init(allocator, sr, 0.3),
            .sr_stage = SilenceRemover.init(allocator, sr, 500, 100, 100, -50.0),
            .fp = FadePad.init(allocator, sr, 0.1, 0.1),
            .volume_scale = volume_scale,
            .sink = sink,
        };
    }

    pub fn deinit(self: *Pipeline) void {
        self.cf.deinit();
        self.sr_stage.deinit();
        self.fp.deinit();
        self.* = undefined;
    }

    fn emitPostCf(ctx: *anyopaque, samples: []const f32) anyerror!void {
        const self: *Pipeline = @ptrCast(@alignCast(ctx));
        try self.sr_stage.push(samples, .{ .ctx = self, .func = emitPostSilence });
    }

    fn emitPostSilence(ctx: *anyopaque, samples: []const f32) anyerror!void {
        const self: *Pipeline = @ptrCast(@alignCast(ctx));
        // The scale = 1 fast path skips the copy (exact compare, like the
        // reference).
        if (self.volume_scale == 1.0) {
            return self.fp.push(samples, self.sink);
        }
        const scaled = try self.allocator.dupe(f32, samples);
        defer self.allocator.free(scaled);
        for (scaled) |*s| {
            s.* *= self.volume_scale;
        }
        try self.fp.push(scaled, self.sink);
    }

    /// Pushes one decoded chunk through the full chain.
    pub fn pushChunk(self: *Pipeline, samples: []const f32) !void {
        try self.cf.push(samples, .{ .ctx = self, .func = emitPostCf });
    }

    /// Drains the stages in pipeline order.
    pub fn finish(self: *Pipeline) !void {
        try self.cf.flush(.{ .ctx = self, .func = emitPostCf });
        try self.sr_stage.flush(.{ .ctx = self, .func = emitPostSilence });
        try self.fp.flush(self.sink);
    }
};

test {
    _ = @import("postproc_stream_tests.zig");
}
