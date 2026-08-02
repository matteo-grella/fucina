//! Full-duplex audio engine for the voice agent: ONE 48 kHz mono stream
//! (nam miniaudio duplex). The realtime callback plays the TTS out-ring and
//! captures the microphone — and pushes THE EXACT frames it just wrote to the
//! output (silence included) into a reference ring paired with the mic ring
//! on the same callback clock. That sample-aligned far-end reference is what
//! the AEC consumes; the alignment every production stack gets from WebRTC's
//! plumbing falls out of the single-callback design here.
//!
//! Rate plumbing: TTS pushes 24 kHz (×2 halfband upsample to the stream
//! rate); the AEC/STT chain runs at 16 kHz (÷3 polyphase decimation, the
//! SAME filter on mic and reference so they stay phase-matched).

const std = @import("std");

pub const stream_rate = 48000;
pub const period_frames = 256;

/// SPSC f32 ring; single producer + single consumer, power-of-two capacity.
pub fn Ring(comptime cap_pow2: usize) type {
    return struct {
        const Self = @This();
        pub const cap = cap_pow2;
        buf: [cap]f32 = undefined,
        head: std.atomic.Value(usize) = .init(0), // producer
        tail: std.atomic.Value(usize) = .init(0), // consumer

        pub fn len(self: *const Self) usize {
            return self.head.load(.acquire) -% self.tail.load(.acquire);
        }

        pub fn push(self: *Self, samples: []const f32) usize {
            var h = self.head.load(.monotonic);
            const t = self.tail.load(.acquire);
            var n: usize = 0;
            while (n < samples.len and h -% t < cap) {
                self.buf[h & (cap - 1)] = samples[n];
                h +%= 1;
                n += 1;
            }
            self.head.store(h, .release);
            return n;
        }

        pub fn pop(self: *Self, out: []f32) usize {
            const h = self.head.load(.acquire);
            var t = self.tail.load(.monotonic);
            var n: usize = 0;
            while (t != h and n < out.len) {
                out[n] = self.buf[t & (cap - 1)];
                t +%= 1;
                n += 1;
            }
            // release: the producer's tail-acquire must order our buf reads
            // before it reuses the freed slots (ARM load->store reordering).
            self.tail.store(t, .release);
            return n;
        }

        pub fn free(self: *const Self) usize {
            return cap - self.len();
        }

        /// Consumer-side drop of everything queued.
        pub fn discardAll(self: *Self) void {
            self.tail.store(self.head.load(.acquire), .release);
        }
    };
}

/// ÷3 FIR decimator, 45-tap windowed-sinc lowpass (cutoff ≈ 6.6 kHz),
/// identical for mic and reference so the pair stays phase-matched.
pub const Decimator3 = struct {
    pub const taps = 45;
    coef: [taps]f32,
    hist: [taps]f32 = @splat(0),
    pos: usize = 0,
    phase: usize = 0,

    pub fn init() Decimator3 {
        var d = Decimator3{ .coef = undefined };
        const fc = 0.825 / 3.0; // normalized cutoff with transition margin
        for (0..taps) |i| {
            const n = @as(f32, @floatFromInt(i)) - @as(f32, taps - 1) / 2.0;
            const sinc = if (n == 0) 2.0 * fc else @sin(2.0 * std.math.pi * fc * n) / (std.math.pi * n);
            const w = 0.54 - 0.46 * @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, taps - 1));
            d.coef[i] = sinc * w;
        }
        return d;
    }

    /// Consume 48 kHz samples, emit 16 kHz samples into `out`. Returns the
    /// number of output samples written (≤ out.len; in.len/3 rounded by
    /// carried phase).
    pub fn run(self: *Decimator3, in: []const f32, out: []f32) usize {
        var n: usize = 0;
        for (in) |x| {
            self.hist[self.pos] = x;
            self.pos = (self.pos + 1) % taps;
            self.phase += 1;
            if (self.phase == 3) {
                self.phase = 0;
                var acc: f32 = 0;
                for (0..taps) |k| {
                    acc += self.coef[k] * self.hist[(self.pos + taps - 1 - k) % taps];
                }
                if (n < out.len) {
                    out[n] = acc;
                    n += 1;
                }
            }
        }
        return n;
    }

    pub fn reset(self: *Decimator3) void {
        self.hist = @splat(0);
        self.phase = 0;
        self.pos = 0;
    }
};

/// ×2 halfband interpolator (31-tap) for the 24 kHz → 48 kHz TTS path.
pub const Upsampler2 = struct {
    pub const taps = 31;
    coef: [taps]f32,
    hist: [taps]f32 = @splat(0),
    pos: usize = 0,

    pub fn init() Upsampler2 {
        var u = Upsampler2{ .coef = undefined };
        const fc = 0.225; // of the 48 kHz output rate
        for (0..taps) |i| {
            const n = @as(f32, @floatFromInt(i)) - @as(f32, taps - 1) / 2.0;
            const sinc = if (n == 0) 2.0 * fc else @sin(2.0 * std.math.pi * fc * n) / (std.math.pi * n);
            const w = 0.54 - 0.46 * @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, taps - 1));
            u.coef[i] = 2.0 * sinc * w;
        }
        return u;
    }

    /// One 24 kHz sample in → two 48 kHz samples out.
    pub fn push(self: *Upsampler2, x: f32, out: *[2]f32) void {
        // zero-stuffed convolution: even output uses the stuffed zero grid
        self.hist[self.pos] = x;
        const p = self.pos;
        self.pos = (self.pos + 1) % taps;
        var a: f32 = 0;
        var b: f32 = 0;
        // output samples correspond to filter phases 0 and 1 over the
        // zero-stuffed history (input on even indices).
        var k: usize = 0;
        while (k < taps) : (k += 2) {
            const h = self.hist[(p + taps - k / 2) % taps];
            a += self.coef[k] * h;
        }
        k = 1;
        while (k < taps) : (k += 2) {
            const h = self.hist[(p + taps - (k - 1) / 2) % taps];
            b += self.coef[k] * h;
        }
        out[0] = a;
        out[1] = b;
    }
};

pub const Engine = struct {
    mic48: Ring(1 << 19) = .{}, // ~10.9 s: rings are never discarded, only pumped
    ref48: Ring(1 << 19) = .{},
    out48: Ring(1 << 19) = .{}, // ~10.9 s of queued speech
    discard_requested: std.atomic.Value(bool) = .init(false),
    paused: std.atomic.Value(bool) = .init(false),
    /// Playback holds until the out-ring reaches this many queued samples
    /// (one-shot latch, re-armed per reply): a cushion against a producer
    /// that momentarily falls behind real time.
    warmup: std.atomic.Value(bool) = .init(false),
    speaking: std.atomic.Value(bool) = .init(false),
    was_fed: bool = false, // callback-local: ring was non-empty
    gaps: std.atomic.Value(usize) = .init(0), // audible starvation events
    underruns: std.atomic.Value(usize) = .init(0),

    pub const warmup_samples = 2 * stream_rate; // 2 s cushion

    /// Realtime duplex callback (miniaudio thread): play out-ring → output,
    /// capture input → mic ring, and mirror the played frames → ref ring.
    /// While paused, silence plays and the out-ring is held. mic and ref are
    /// pushed as a PAIR or dropped as a pair — sample alignment between them
    /// is an invariant maintained here, at the sole producer.
    pub fn callback(user: ?*anyopaque, output: ?[*]f32, input: ?[*]const f32, frame_count: c_uint) callconv(.c) void {
        const self: *Engine = @ptrCast(@alignCast(user.?));
        const n: usize = frame_count;
        if (self.discard_requested.load(.acquire)) {
            self.out48.discardAll();
            self.discard_requested.store(false, .release);
        }
        const paused = self.paused.load(.acquire);
        var off: usize = 0;
        while (off < n) {
            var outbuf: [4096]f32 = undefined;
            const sl = @min(n - off, outbuf.len);
            const chunk = outbuf[0..sl];
            if (paused) {
                @memset(chunk, 0);
            } else if (self.warmup.load(.acquire) and self.out48.len() < warmup_samples) {
                @memset(chunk, 0); // hold: cushion not built yet
            } else {
                self.warmup.store(false, .release); // latch open for this reply
                const got = self.out48.pop(chunk);
                if (got < sl) {
                    @memset(chunk[got..], 0);
                    if (!(got == 0 and self.out48.len() == 0)) _ = self.underruns.fetchAdd(1, .monotonic);
                }
                // Honest starvation metric: non-empty -> empty transition
                // while a reply is audibly in flight = one audible gap.
                if (self.speaking.load(.monotonic)) {
                    if (got > 0) {
                        self.was_fed = true;
                    } else if (self.was_fed) {
                        self.was_fed = false;
                        _ = self.gaps.fetchAdd(1, .monotonic);
                        // Starved mid-reply: hold until the cushion rebuilds
                        // — one clean pause beats machine-gun stutter. The
                        // producer's finish() force-releases for the tail.
                        self.warmup.store(true, .release);
                    }
                }
            }
            if (output) |o| @memcpy(o[off..][0..sl], chunk);
            // The reference is EXACTLY what went to the device this callback.
            if (input) |i| {
                if (self.mic48.free() >= sl and self.ref48.free() >= sl) {
                    _ = self.ref48.push(chunk);
                    _ = self.mic48.push(i[off..][0..sl]);
                } // else: drop the PAIR together — alignment over completeness
            } else if (self.ref48.free() >= sl) {
                _ = self.ref48.push(chunk);
            }
            off += sl;
        }
    }

    /// Ask the callback to drop all queued speech (barge-in).
    pub fn discard(self: *Engine) void {
        self.discard_requested.store(true, .release);
    }

    /// Hold/resume playback without losing the queue (pause-then-commit).
    pub fn setPaused(self: *Engine, p: bool) void {
        self.paused.store(p, .release);
    }

    /// Arm the warm-up latch for a new reply (playback holds until the
    /// cushion is built or the latch is released).
    pub fn armWarmup(self: *Engine) void {
        self.warmup.store(true, .release);
    }

    /// Release the latch (generation finished / short reply / barge).
    pub fn releaseWarmup(self: *Engine) void {
        self.warmup.store(false, .release);
    }

    pub fn setSpeaking(self: *Engine, v: bool) void {
        self.speaking.store(v, .monotonic);
    }

    pub fn outPending(self: *Engine) usize {
        return self.out48.len();
    }
};
