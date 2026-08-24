//! GTCRN-AEC — acoustic echo cancellation for the voice agent. A 1:1 Zig port
//! of LocalVQE's scalar reference (refs/LocalVQE/ggml/gtcrn/gtcrn.cpp, itself
//! validated <1e-4 vs PyTorch): the 48,965-param AEC-aware GTCRN — ERB
//! sub-band + SFE + grouped-conv encoder + 2× grouped dual-path RNN +
//! grouped-deconv decoder + complex-ratio mask on the mic spectrum, with its
//! own 512/256 windowed-DFT STFT (matrices shipped in the GGUF; BatchNorms
//! and deconvs pre-folded by the exporter — one conv code path).
//!
//! `Session` is the deployment shape: `step` consumes ONE 16 kHz hop
//! (256 samples) of mic + far-end reference and emits 256 echo-cancelled
//! samples, carrying all recurrent state (depthwise-conv time history, TRA
//! and inter-GRU hiddens) across frames. The whole-clip entry used by the
//! parity tests is a fresh session iterated per frame, so the fixture gates
//! pin the exact streaming path. The model is 49 K params at 16 ms hops.
//!
//! Core ops carry the ERB analysis/synthesis matmuls (`dot`) and the (F, C)
//! LayerNorms (`layerNormAffineRows`); the rest is hand-rolled, because at
//! this frame size the core dispatch thresholds all sit above these shapes:
//! `shouldUseBlas` wants m,n,k >= 16 and threading wants m >= 32 with
//! m*n*k >= 1 M (src/parallel.zig), while the frames here are [16, 129] and
//! [16, 33]. Below those thresholds a tensor op buys a vector kernel over a
//! scalar loop and nothing more, which the small matmuls (the 8.4 K-MAC
//! DPGRNN FCs) do not recover against the wrapper cost. The convs would
//! further require `Frame` in channel-last order — core conv2d/conv1d are
//! channel-last, and a per-call permute pays a full `prepareContiguous`
//! materialize in every consuming op, so that is a global layout decision
//! rather than a local substitution.
//!
//! `zig build test-voiceagent -Doptimize=ReleaseFast` prints us/frame against
//! the 16 ms hop budget, so the balance above is re-checkable.
//!
//! Inputs follow the reference convention: `e` = the microphone (echoic)
//! signal the mask is applied to, `y` = the far-end (loudspeaker) reference.

const std = @import("std");
const fucina = @import("fucina");

const gguf = fucina.gguf;
const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const Tensor = fucina.Tensor;

pub const fft_size = 512;
pub const hop = 256;
pub const n_bins = 257;
pub const sample_rate = 16000;

pub const Error = error{
    MissingTensor,
    NotF32,
    BadShape,
};

/// A weight view: numpy-order dims over the mmap'd f32 payload.
const W = struct {
    dims: [4]usize,
    nd: usize,
    data: []const f32,

    fn dim(self: *const W, i: usize) usize {
        return self.dims[i];
    }
};

pub const Model = struct {
    allocator: Allocator,
    weights: std.StringHashMap(W),

    pub fn load(allocator: Allocator, file: *const gguf.File) !Model {
        var weights = std.StringHashMap(W).init(allocator);
        errdefer weights.deinit();
        for (file.tensors) |*t| {
            if (t.ggml_type != .f32) return Error.NotF32;
            if (!std.mem.isAligned(@intFromPtr(t.data.ptr), @alignOf(f32))) return Error.BadShape;
            const data = @as([*]const f32, @ptrCast(@alignCast(t.data.ptr)))[0 .. t.data.len / 4];
            // GGUF ne is fastest-first; record numpy order (reversed).
            var dims = [4]usize{ 1, 1, 1, 1 };
            for (0..t.n_dims) |i| dims[i] = t.dims[t.n_dims - 1 - i];
            try weights.put(t.name, .{ .dims = dims, .nd = t.n_dims, .data = data });
        }
        return .{ .allocator = allocator, .weights = weights };
    }

    pub fn deinit(self: *Model) void {
        self.weights.deinit();
        self.* = undefined;
    }

    fn get(self: *const Model, name: []const u8) !*const W {
        return self.weights.getPtr(name) orelse Error.MissingTensor;
    }

    fn getf(self: *const Model, comptime fmt: []const u8, args: anytype, buf: []u8) !*const W {
        return self.get(std.fmt.bufPrint(buf, fmt, args) catch unreachable);
    }
};

// ── frame tensors: (C, F) per time step, row-major ──────────────────────────

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// One time frame of activations: (C, F) row-major.
const Frame = struct {
    c: usize,
    f: usize,
    d: []f32,

    fn at(self: *const Frame, c: usize, f: usize) f32 {
        return self.d[c * self.f + f];
    }
    fn atP(self: *Frame, c: usize, f: usize) *f32 {
        return &self.d[c * self.f + f];
    }
};

/// Bump-style frame arena: all frames of one step live in a reusable buffer.
const FrameArena = struct {
    buf: []f32,
    used: usize = 0,

    fn alloc(self: *FrameArena, c: usize, f: usize) Frame {
        const n = c * f;
        std.debug.assert(self.used + n <= self.buf.len);
        const d = self.buf[self.used..][0..n];
        self.used += n;
        @memset(d, 0);
        return .{ .c = c, .f = f, .d = d };
    }

    fn reset(self: *FrameArena) void {
        self.used = 0;
    }
};

// ── per-block streaming state ───────────────────────────────────────────────

/// GT block state: depthwise-conv input history (2·dil frames of the hidden
/// (C_h, F) activation, oldest first) + the TRA GRU hidden.
const GtState = struct {
    dil: usize,
    hist_frames: usize, // 2*dil
    ch: usize, // hidden channels
    fdim: usize,
    hist: []f32, // [hist_frames][ch][fdim], ring
    hist_pos: usize = 0,
    tra_h: []f32, // TRA GRU hidden (2*C_block)

    fn init(allocator: Allocator, dil: usize, ch: usize, fdim: usize, tra_h: usize) !GtState {
        const hist = try allocator.alloc(f32, 2 * dil * ch * fdim);
        @memset(hist, 0);
        errdefer allocator.free(hist);
        const th = try allocator.alloc(f32, tra_h);
        @memset(th, 0);
        return .{ .dil = dil, .hist_frames = 2 * dil, .ch = ch, .fdim = fdim, .hist = hist, .tra_h = th };
    }

    fn deinit(self: *GtState, allocator: Allocator) void {
        allocator.free(self.hist);
        allocator.free(self.tra_h);
        self.* = undefined;
    }

    fn histFrame(self: *const GtState, back: usize) []const f32 {
        // back = 1..hist_frames (frames before current), oldest kept in ring
        const idx = (self.hist_pos + self.hist_frames - back) % self.hist_frames;
        return self.hist[idx * self.ch * self.fdim ..][0 .. self.ch * self.fdim];
    }

    fn push(self: *GtState, frame: []const f32) void {
        @memcpy(self.hist[self.hist_pos * self.ch * self.fdim ..][0 .. self.ch * self.fdim], frame);
        self.hist_pos = (self.hist_pos + 1) % self.hist_frames;
    }
};

/// DPGRNN state: inter GRU hiddens per frequency bin, for rnn1 and rnn2.
const DpState = struct {
    h1: []f32, // [33][H1]
    h2: []f32, // [33][H2]
    h1_w: usize,
    h2_w: usize,

    fn init(allocator: Allocator, fdim: usize, h1: usize, h2: usize) !DpState {
        const a = try allocator.alloc(f32, fdim * h1);
        @memset(a, 0);
        errdefer allocator.free(a);
        const b = try allocator.alloc(f32, fdim * h2);
        @memset(b, 0);
        return .{ .h1 = a, .h2 = b, .h1_w = h1, .h2_w = h2 };
    }

    fn deinit(self: *DpState, allocator: Allocator) void {
        allocator.free(self.h1);
        allocator.free(self.h2);
        self.* = undefined;
    }
};

// ── GRU cell (PyTorch layout, gate order [r,z,n]) ───────────────────────────

fn gruStep(x: []const f32, h: []f32, wih: *const W, whh: *const W, bih: *const W, bhh: *const W, gi: []f32, gh: []f32) void {
    const hn = whh.dim(1);
    const in = wih.dim(1);
    for (0..3 * hn) |g| {
        var a: f32 = bih.data[g];
        const wr = wih.data[g * in ..][0..in];
        for (x, wr) |xv, wv| a += xv * wv;
        gi[g] = a;
        var c: f32 = bhh.data[g];
        const hr = whh.data[g * hn ..][0..hn];
        for (h, hr) |hv, wv| c += hv * wv;
        gh[g] = c;
    }
    for (0..hn) |k| {
        const r = sigmoid(gi[k] + gh[k]);
        const z = sigmoid(gi[hn + k] + gh[hn + k]);
        const n = std.math.tanh(gi[2 * hn + k] + r * gh[2 * hn + k]);
        h[k] = (1.0 - z) * n + z * h[k];
    }
}

/// Full GRU over a sequence with fresh zero hidden (used for the per-frame
/// bidirectional intra RNNs over the frequency axis).
fn gruSeq(x: []const f32, seq: usize, in: usize, wih: *const W, whh: *const W, bih: *const W, bhh: *const W, reverse: bool, out: []f32, h: []f32, gi: []f32, gh: []f32) void {
    const hn = whh.dim(1);
    @memset(h[0..hn], 0);
    for (0..seq) |s| {
        const ti = if (reverse) seq - 1 - s else s;
        gruStep(x[ti * in ..][0..in], h[0..hn], wih, whh, bih, bhh, gi, gh);
        @memcpy(out[ti * hn ..][0..hn], h[0..hn]);
    }
}

// ── session ─────────────────────────────────────────────────────────────────

/// Debug stage capture for the parity tests: per-step (C,F) copies keyed by
/// the fixture stage names.
pub const Cap = struct {
    allocator: Allocator,
    map: std.StringHashMap([]f32),

    pub fn init(allocator: Allocator) Cap {
        return .{ .allocator = allocator, .map = std.StringHashMap([]f32).init(allocator) };
    }

    pub fn deinit(self: *Cap) void {
        var it = self.map.valueIterator();
        while (it.next()) |v| self.allocator.free(v.*);
        self.map.deinit();
        self.* = undefined;
    }

    fn put(self: *Cap, name: []const u8, d: []const f32) void {
        const gop = self.map.getOrPut(name) catch return;
        if (gop.found_existing) {
            if (gop.value_ptr.len == d.len) {
                @memcpy(gop.value_ptr.*, d);
                return;
            }
            self.allocator.free(gop.value_ptr.*);
        }
        gop.value_ptr.* = self.allocator.dupe(f32, d) catch return;
    }
};

pub const Session = struct {
    allocator: Allocator,
    ctx: *ExecContext,
    model: *const Model,
    cap: ?*Cap = null,

    // streaming conv/GRU state per block
    enc_gt: [3]GtState, // enc2 (dil 1), enc3 (dil 2), enc4 (dil 5)
    dec_gt: [3]GtState, // dec0 (dil 5), dec1 (dil 2), dec2 (dil 1)
    dp: [2]DpState,

    arena: FrameArena,
    scratch: []f32, // GRU/linear scratch

    // audio-level streaming: 512-sample analysis buffers + OLA tail
    mic_buf: [fft_size]f32 = @splat(0),
    ref_buf: [fft_size]f32 = @splat(0),
    ola: [fft_size]f32 = @splat(0),
    ola_env: [fft_size]f32 = @splat(0),

    pub fn init(allocator: Allocator, ctx: *ExecContext, model: *const Model) !Session {
        // Block channel geometry from the checkpoint: gt blocks operate on
        // C=16 (chunked to 8+8, hidden 16 after pc1 of SFE(8)=24ch → 16).
        var buf: [96]u8 = undefined;
        var enc_gt: [3]GtState = undefined;
        var dec_gt: [3]GtState = undefined;
        const dils = [3]usize{ 1, 2, 5 };
        inline for (0..3) |i| {
            const pc1 = try model.getf("encoder.en_convs.{d}.pc1.w", .{i + 2}, &buf);
            const hidden = pc1.dim(0);
            const tra_fc = try model.getf("encoder.en_convs.{d}.tra.fc.w", .{i + 2}, &buf);
            const tra_h = tra_fc.dim(1); // 2C
            enc_gt[i] = try GtState.init(allocator, dils[i], hidden, 33, tra_h);
        }
        inline for (0..3) |i| {
            const pc1 = try model.getf("decoder.de_convs.{d}.pc1.w", .{i}, &buf);
            const hidden = pc1.dim(0);
            const tra_fc = try model.getf("decoder.de_convs.{d}.tra.fc.w", .{i}, &buf);
            const tra_h = tra_fc.dim(1);
            dec_gt[i] = try GtState.init(allocator, dils[2 - i], hidden, 33, tra_h);
        }
        var dp: [2]DpState = undefined;
        inline for (0..2) |i| {
            const whh1 = try model.getf("dpgrnn{d}.inter_rnn.rnn1.weight_hh_l0", .{i + 1}, &buf);
            const whh2 = try model.getf("dpgrnn{d}.inter_rnn.rnn2.weight_hh_l0", .{i + 1}, &buf);
            dp[i] = try DpState.init(allocator, 33, whh1.dim(1), whh2.dim(1));
        }

        const arena_buf = try allocator.alloc(f32, 256 * 1024);
        errdefer allocator.free(arena_buf);
        const scratch = try allocator.alloc(f32, 64 * 1024);

        return .{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .enc_gt = enc_gt,
            .dec_gt = dec_gt,
            .dp = dp,
            .arena = .{ .buf = arena_buf },
            .scratch = scratch,
        };
    }

    pub fn deinit(self: *Session) void {
        for (&self.enc_gt) |*s| s.deinit(self.allocator);
        for (&self.dec_gt) |*s| s.deinit(self.allocator);
        for (&self.dp) |*s| s.deinit(self.allocator);
        self.allocator.free(self.arena.buf);
        self.allocator.free(self.scratch);
        self.* = undefined;
    }

    pub fn reset(self: *Session) void {
        for (&self.enc_gt) |*s| {
            @memset(s.hist, 0);
            @memset(s.tra_h, 0);
            s.hist_pos = 0;
        }
        for (&self.dec_gt) |*s| {
            @memset(s.hist, 0);
            @memset(s.tra_h, 0);
            s.hist_pos = 0;
        }
        for (&self.dp) |*s| {
            @memset(s.h1, 0);
            @memset(s.h2, 0);
        }
        self.mic_buf = @splat(0);
        self.ref_buf = @splat(0);
        self.ola = @splat(0);
        self.ola_env = @splat(0);
    }

    // ── spectral step: one frame of (257×2) mic + ref → masked mic frame ──

    /// `spec_e`/`spec_y`: interleaved (re,im) per bin, 514 floats each.
    /// Writes the masked mic spectrum into `out` (514 floats).
    pub fn stepSpec(self: *Session, spec_e: []const f32, spec_y: []const f32, out: []f32) !void {
        const m = self.model;
        self.arena.reset();

        // feat(e) || feat(y) → (18, 129)
        var ft = self.arena.alloc(18, 129);
        try self.featInto(spec_e, 0, &ft);
        try self.featInto(spec_y, 9, &ft);
        if (self.cap) |cp| cp.put("feat", ft.d);

        // encoder
        const en0 = try self.convBlock(&ft, "encoder.en_convs.0", 1, 2, false, false);
        if (self.cap) |cp| cp.put("enc0", en0.d);
        const en1 = try self.convBlock(&en0, "encoder.en_convs.1", 2, 2, false, false);
        if (self.cap) |cp| cp.put("enc1", en1.d);
        const en2 = try self.gtBlock(&en1, "encoder.en_convs.2", &self.enc_gt[0]);
        if (self.cap) |cp| cp.put("enc2", en2.d);
        const en3 = try self.gtBlock(&en2, "encoder.en_convs.3", &self.enc_gt[1]);
        if (self.cap) |cp| cp.put("enc3", en3.d);
        const en4 = try self.gtBlock(&en3, "encoder.en_convs.4", &self.enc_gt[2]);
        if (self.cap) |cp| cp.put("enc4", en4.d);

        // dual-path RNNs
        const d1 = try self.dpgrnn(&en4, "dpgrnn1", &self.dp[0]);
        if (self.cap) |cp| cp.put("dpgrnn1", d1.d);
        const d2 = try self.dpgrnn(&d1, "dpgrnn2", &self.dp[1]);
        if (self.cap) |cp| cp.put("dpgrnn2", d2.d);

        // decoder with skip adds
        var x = addFrames(self.arena.alloc(d2.c, d2.f), &d2, &en4);
        x = try self.gtBlock(&x, "decoder.de_convs.0", &self.dec_gt[0]);
        if (self.cap) |cp| cp.put("dec0", x.d);
        x = addFrames(self.arena.alloc(x.c, x.f), &x, &en3);
        x = try self.gtBlock(&x, "decoder.de_convs.1", &self.dec_gt[1]);
        if (self.cap) |cp| cp.put("dec1", x.d);
        x = addFrames(self.arena.alloc(x.c, x.f), &x, &en2);
        x = try self.gtBlock(&x, "decoder.de_convs.2", &self.dec_gt[2]);
        if (self.cap) |cp| cp.put("dec2", x.d);
        x = addFrames(self.arena.alloc(x.c, x.f), &x, &en1);
        x = try self.convBlock(&x, "decoder.de_convs.3", 2, 2, true, false);
        if (self.cap) |cp| cp.put("dec3", x.d);
        x = addFrames(self.arena.alloc(x.c, x.f), &x, &en0);
        x = try self.convBlock(&x, "decoder.de_convs.4", 1, 2, true, true);
        if (self.cap) |cp| cp.put("dec4", x.d);

        // ERB synthesis → complex ratio mask applied to spec_e
        const bs = try m.get("erb.bs");
        var mask = self.arena.alloc(2, n_bins);
        for (0..2) |c| {
            for (0..65) |f| mask.atP(c, f).* = x.at(c, f);
        }
        {
            const bands = self.scratch[0 .. 2 * 64];
            for (0..2) |c| {
                for (0..64) |i| bands[c * 64 + i] = x.at(c, 65 + i);
            }
            var b_t = try Tensor(.{ .c, .band }).fromBorrowedSlice(self.ctx, .{ 2, 64 }, bands);
            defer b_t.deinit();
            var bs_t = try Tensor(.{ .binhi, .band }).fromBorrowedConstSlice(self.ctx, .{ 192, 64 }, bs.data[0 .. 192 * 64]);
            defer bs_t.deinit();
            var out_t = try b_t.dot(self.ctx, &bs_t, .band);
            defer out_t.deinit();
            const od = try out_t.dataConst();
            for (0..2) |c| {
                for (0..192) |j| mask.atP(c, 65 + j).* = od[c * 192 + j];
            }
        }
        if (self.cap) |cp| cp.put("mask", mask.d);
        for (0..n_bins) |f| {
            const er = spec_e[f * 2];
            const ei = spec_e[f * 2 + 1];
            const mr = mask.at(0, f);
            const mi = mask.at(1, f);
            out[f * 2] = er * mr - ei * mi;
            out[f * 2 + 1] = ei * mr + er * mi;
        }
    }

    /// AUDIO step: 256 fresh 16 kHz samples of mic + reference in, 256
    /// echo-cancelled samples out (OLA with the shipped synthesis window;
    /// one hop of algorithmic latency).
    pub fn step(self: *Session, mic: *const [hop]f32, ref: *const [hop]f32, out: *[hop]f32) !void {
        const m = self.model;
        // slide analysis buffers
        std.mem.copyForwards(f32, self.mic_buf[0 .. fft_size - hop], self.mic_buf[hop..]);
        @memcpy(self.mic_buf[fft_size - hop ..], mic);
        std.mem.copyForwards(f32, self.ref_buf[0 .. fft_size - hop], self.ref_buf[hop..]);
        @memcpy(self.ref_buf[fft_size - hop ..], ref);

        var se: [n_bins * 2]f32 = undefined;
        var sy: [n_bins * 2]f32 = undefined;
        var so: [n_bins * 2]f32 = undefined;
        stftFrame(m, &self.mic_buf, &se) catch |e| return e;
        stftFrame(m, &self.ref_buf, &sy) catch |e| return e;
        try self.stepSpec(&se, &sy, &so);

        var ft: [fft_size]f32 = undefined;
        try istftFrame(m, &so, &ft);
        const win2 = (try m.get("stft.win2")).data;
        // OLA: shift tails, add this frame, emit the completed hop.
        std.mem.copyForwards(f32, self.ola[0 .. fft_size - hop], self.ola[hop..]);
        @memset(self.ola[fft_size - hop ..], 0);
        std.mem.copyForwards(f32, self.ola_env[0 .. fft_size - hop], self.ola_env[hop..]);
        @memset(self.ola_env[fft_size - hop ..], 0);
        for (0..fft_size) |i| {
            self.ola[i] += ft[i];
            self.ola_env[i] += win2[i];
        }
        for (0..hop) |i| {
            const w = self.ola_env[i];
            out[i] = if (w > 1e-11) self.ola[i] / w else 0;
        }
    }

    // ── building blocks (per-frame) ─────────────────────────────────────────

    /// feat: one spectrum frame → 9 channels of (129) into `dst` rows
    /// [c0..c0+9): [mag,re,im] ERB-banded then 3-tap SFE.
    fn featInto(self: *Session, spec: []const f32, c0: usize, dst: *Frame) !void {
        const m = self.model;
        const bm = try m.get("erb.bm");
        var f3 = self.arena.alloc(3, n_bins);
        for (0..n_bins) |f| {
            const re = spec[f * 2];
            const im = spec[f * 2 + 1];
            f3.atP(1, f).* = re;
            f3.atP(2, f).* = im;
            f3.atP(0, f).* = @sqrt(re * re + im * im + 1e-12);
        }
        var banded = self.arena.alloc(3, 129);
        for (0..3) |c| {
            for (0..65) |f| banded.atP(c, f).* = f3.at(c, f);
        }
        // ERB analysis: the 192 high bins of each of the 3 feature planes
        // through the [64, 192] band matrix. Core `dot` — `bm` is already
        // row-major [out, in], no repack; the gather is only because the
        // per-plane runs are strided by n_bins inside the frame.
        {
            const hi = self.scratch[0 .. 3 * 192];
            for (0..3) |c| {
                for (0..192) |i| hi[c * 192 + i] = f3.at(c, 65 + i);
            }
            var hi_t = try Tensor(.{ .c, .binhi }).fromBorrowedSlice(self.ctx, .{ 3, 192 }, hi);
            defer hi_t.deinit();
            var bm_t = try Tensor(.{ .band, .binhi }).fromBorrowedConstSlice(self.ctx, .{ 64, 192 }, bm.data[0 .. 64 * 192]);
            defer bm_t.deinit();
            var out_t = try hi_t.dot(self.ctx, &bm_t, .binhi);
            defer out_t.deinit();
            const od = try out_t.dataConst();
            for (0..3) |c| {
                for (0..64) |j| banded.atP(c, 65 + j).* = od[c * 64 + j];
            }
        }
        // SFE ×3
        for (0..3) |c| {
            for (0..129) |f| {
                const fm = if (f >= 1) banded.at(c, f - 1) else 0;
                const f0 = banded.at(c, f);
                const fp = if (f + 1 < 129) banded.at(c, f + 1) else 0;
                dst.atP(c0 + c * 3 + 0, f).* = fm;
                dst.atP(c0 + c * 3 + 1, f).* = f0;
                dst.atP(c0 + c * 3 + 2, f).* = fp;
            }
        }
    }

    /// LayerNorm over the WHOLE (F, C) buffer — the checkpoint ships
    /// `intra_ln.w/b` as [33, 16] = 528, i.e. torch `LayerNorm((F, C))`, so
    /// this is one 528-wide row, not per-row normalization.
    fn layerNormWhole(self: *Session, x: []f32, w: *const W, b: *const W, eps: f32) !void {
        var y = try self.ctx.layerNormRows(x, 1, x.len, eps, .{ .weight = w.data[0..x.len], .bias = b.data[0..x.len] });
        defer y.deinit();
        @memcpy(x, y.dataConst());
    }

    /// Frequency-only conv2d for one frame (KT must be 1 for stateless blocks).
    fn conv2dF(self: *Session, x: *const Frame, w: *const W, b: ?*const W, groups: usize, stride_f: usize, pad_f: usize) Frame {
        const oc = w.dim(0);
        const ing = w.dim(1);
        const kf = w.dim(3);
        const fout = (x.f + 2 * pad_f - (kf - 1) - 1) / stride_f + 1;
        var y = self.arena.alloc(oc, fout);
        const opg = oc / groups;
        const ipg = x.c / groups;
        for (0..groups) |g| {
            for (0..opg) |o| {
                const co = g * opg + o;
                const bias: f32 = if (b) |bb| bb.data[co] else 0;
                for (0..fout) |f| {
                    var s = bias;
                    for (0..ing) |ci| {
                        const cin = g * ipg + ci;
                        for (0..kf) |k| {
                            const fi = @as(i64, @intCast(f * stride_f + k)) - @as(i64, @intCast(pad_f));
                            if (fi < 0 or fi >= x.f) continue;
                            s += w.data[((co * ing + ci) * w.dim(2) + 0) * kf + k] * x.at(cin, @intCast(fi));
                        }
                    }
                    y.atP(co, f).* = s;
                }
            }
        }
        return y;
    }

    fn prelu(x: *Frame, slope: *const W) void {
        const scalar = slope.data.len == 1;
        for (0..x.c) |c| {
            const a = if (scalar) slope.data[0] else slope.data[c];
            for (0..x.f) |f| {
                const p = x.atP(c, f);
                if (p.* < 0) p.* *= a;
            }
        }
    }

    fn convBlock(self: *Session, x: *const Frame, comptime p: []const u8, groups: usize, stride_f: usize, deconv: bool, is_last: bool) !Frame {
        const m = self.model;
        var buf: [96]u8 = undefined;
        const w = try m.getf(p ++ ".w", .{}, &buf);
        const b = try m.getf(p ++ ".b", .{}, &buf);
        var y: Frame = undefined;
        if (!deconv) {
            y = self.conv2dF(x, w, b, groups, stride_f, 2);
        } else {
            // zero-insert upsample then stride-1 conv
            const fup = (x.f - 1) * stride_f + 1;
            var xu = self.arena.alloc(x.c, fup);
            for (0..x.c) |c| {
                for (0..x.f) |f| xu.atP(c, f * stride_f).* = x.at(c, f);
            }
            y = self.conv2dF(&xu, w, b, groups, 1, 2);
        }
        if (is_last) {
            for (y.d) |*v| v.* = std.math.tanh(v.*);
        } else {
            prelu(&y, try m.getf(p ++ ".prelu", .{}, &buf));
        }
        return y;
    }

    /// GT conv block with streaming depthwise time history + TRA state.
    fn gtBlock(self: *Session, x: *const Frame, comptime p: []const u8, st: *GtState) !Frame {
        const m = self.model;
        var buf: [96]u8 = undefined;
        const half = x.c / 2;

        // x1 = first half, SFE ×3 (frame-local)
        var s = self.arena.alloc(half * 3, x.f);
        for (0..half) |c| {
            for (0..x.f) |f| {
                const fm = if (f >= 1) x.at(c, f - 1) else 0;
                const f0 = x.at(c, f);
                const fp = if (f + 1 < x.f) x.at(c, f + 1) else 0;
                s.atP(c * 3 + 0, f).* = fm;
                s.atP(c * 3 + 1, f).* = f0;
                s.atP(c * 3 + 2, f).* = fp;
            }
        }
        var h = self.conv2dF(&s, try m.getf(p ++ ".pc1.w", .{}, &buf), try m.getf(p ++ ".pc1.b", .{}, &buf), 1, 1, 0);
        prelu(&h, try m.getf(p ++ ".pc1.prelu", .{}, &buf));

        // depthwise causal conv: KT=3 time taps at dilation dil, pad_f=1.
        const dw = try m.getf(p ++ ".dw.w", .{}, &buf);
        const dwb = try m.getf(p ++ ".dw.b", .{}, &buf);
        const kt = dw.dim(2);
        const kf = dw.dim(3);
        std.debug.assert(kt == 3 and h.c == dw.dim(0));
        var hd = self.arena.alloc(h.c, h.f);
        for (0..h.c) |c| {
            for (0..h.f) |f| {
                var acc: f32 = dwb.data[c];
                for (0..kt) |ktap| {
                    // time index: current frame is kt-1; earlier taps read history
                    const back = (kt - 1 - ktap) * st.dil;
                    const src: []const f32 = if (back == 0) h.d else st.histFrame(back);
                    for (0..kf) |k| {
                        const fi = @as(i64, @intCast(f + k)) - 1;
                        if (fi < 0 or fi >= h.f) continue;
                        acc += dw.data[((c * 1 + 0) * kt + ktap) * kf + k] * src[c * h.f + @as(usize, @intCast(fi))];
                    }
                }
                hd.atP(c, f).* = acc;
            }
        }
        st.push(h.d);
        prelu(&hd, try m.getf(p ++ ".dw.prelu", .{}, &buf));

        var h2 = self.conv2dF(&hd, try m.getf(p ++ ".pc2.w", .{}, &buf), try m.getf(p ++ ".pc2.b", .{}, &buf), 1, 1, 0);

        // TRA: gate by sigmoid(fc(gru(mean_f h2²))) with carried hidden.
        {
            const c = h2.c;
            var e = self.scratch[0..c];
            for (0..c) |ci| {
                var acc: f32 = 0;
                for (0..h2.f) |f| {
                    const v = h2.at(ci, f);
                    acc += v * v;
                }
                e[ci] = acc / @as(f32, @floatFromInt(h2.f));
            }
            const gi = self.scratch[c .. c + 3 * st.tra_h.len];
            const gh = self.scratch[c + 3 * st.tra_h.len .. c + 6 * st.tra_h.len];
            gruStep(e, st.tra_h, try m.getf(p ++ ".tra.gru.weight_ih_l0", .{}, &buf), try m.getf(p ++ ".tra.gru.weight_hh_l0", .{}, &buf), try m.getf(p ++ ".tra.gru.bias_ih_l0", .{}, &buf), try m.getf(p ++ ".tra.gru.bias_hh_l0", .{}, &buf), gi, gh);
            const fcw = try m.getf(p ++ ".tra.fc.w", .{}, &buf);
            const fcb = try m.getf(p ++ ".tra.fc.b", .{}, &buf);
            for (0..c) |o| {
                var acc: f32 = fcb.data[o];
                const wr = fcw.data[o * st.tra_h.len ..][0..st.tra_h.len];
                for (st.tra_h, wr) |hv, wv| acc += hv * wv;
                const g = sigmoid(acc);
                for (0..h2.f) |f| h2.atP(o, f).* *= g;
            }
        }

        // channel shuffle with the bypass half
        var y = self.arena.alloc(x.c, x.f);
        for (0..half) |c| {
            for (0..x.f) |f| {
                y.atP(2 * c + 0, f).* = h2.at(c, f);
                y.atP(2 * c + 1, f).* = x.at(half + c, f);
            }
        }
        return y;
    }

    /// DPGRNN for one frame: bidirectional intra GRU over freq (fresh hidden)
    /// + unidirectional inter GRU over time (carried per-bin hidden).
    fn dpgrnn(self: *Session, x: *const Frame, comptime d: []const u8, st: *DpState) !Frame {
        const m = self.model;
        var buf: [96]u8 = undefined;
        const c = x.c; // 16
        const fd = x.f; // 33
        const half = c / 2;

        // xp: (F, C) for this frame
        const xp = self.scratch[0 .. fd * c];
        for (0..fd) |f| {
            for (0..c) |ci| xp[f * c + ci] = x.at(ci, f);
        }

        // intra: grouped bidirectional GRU over the freq sequence.
        // Grouped output widths: each rnn contributes 2*H (bidir), and for
        // the GTCRN configs the total equals C.
        const intra = self.scratch[fd * c .. 2 * fd * c];
        {
            var off: usize = 2 * fd * c;
            const sub = self.scratch[off .. off + fd * half];
            off += fd * half;
            var out_off: usize = 0;
            var in_off: usize = 0;
            inline for (.{ "rnn1", "rnn2" }) |rname| {
                const wih = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".weight_ih_l0", .{}, &buf);
                const whh = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".weight_hh_l0", .{}, &buf);
                const bih = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".bias_ih_l0", .{}, &buf);
                const bhh = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".bias_hh_l0", .{}, &buf);
                const wihr = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".weight_ih_l0_reverse", .{}, &buf);
                const whhr = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".weight_hh_l0_reverse", .{}, &buf);
                const bihr = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".bias_ih_l0_reverse", .{}, &buf);
                const bhhr = try m.getf(d ++ ".intra_rnn." ++ rname ++ ".bias_hh_l0_reverse", .{}, &buf);
                const hn = whh.dim(1);
                for (0..fd) |f| {
                    for (0..half) |i| sub[f * half + i] = xp[f * c + in_off + i];
                }
                const yf = self.scratch[off .. off + fd * hn];
                const yr = self.scratch[off + fd * hn .. off + 2 * fd * hn];
                const h = self.scratch[off + 2 * fd * hn .. off + 2 * fd * hn + hn];
                const gi = self.scratch[off + 2 * fd * hn + hn .. off + 2 * fd * hn + hn + 3 * hn];
                const gh = self.scratch[off + 2 * fd * hn + hn + 3 * hn .. off + 2 * fd * hn + hn + 6 * hn];
                gruSeq(sub, fd, half, wih, whh, bih, bhh, false, yf, h, gi, gh);
                gruSeq(sub, fd, half, wihr, whhr, bihr, bhhr, true, yr, h, gi, gh);
                for (0..fd) |f| {
                    for (0..hn) |k| {
                        intra[f * c + out_off + k] = yf[f * hn + k];
                        intra[f * c + out_off + hn + k] = yr[f * hn + k];
                    }
                }
                out_off += 2 * hn;
                in_off += half;
            }
        }
        // intra FC (2C→C) + LayerNorm over (F,C) + residual
        const intra_fc = self.scratch[2 * fd * c .. 3 * fd * c];
        {
            const w = try m.getf(d ++ ".intra_fc.w", .{}, &buf);
            const b = try m.getf(d ++ ".intra_fc.b", .{}, &buf);
            const in = w.dim(1);
            for (0..fd) |f| {
                for (0..c) |o| {
                    var s: f32 = b.data[o];
                    const wr = w.data[o * in ..][0..in];
                    for (0..in) |i| s += wr[i] * intra[f * in + i];
                    intra_fc[f * c + o] = s;
                }
            }
            const lw = try m.getf(d ++ ".intra_ln.w", .{}, &buf);
            const lb = try m.getf(d ++ ".intra_ln.b", .{}, &buf);
            try self.layerNormWhole(intra_fc, lw, lb, 1e-8);
            for (0..fd * c) |i| intra_fc[i] += xp[i];
        }

        // inter: per-bin unidirectional grouped GRU over time (carried hidden)
        const inter = self.scratch[3 * fd * c .. 4 * fd * c];
        {
            const off: usize = 4 * fd * c;
            var out_off: usize = 0;
            var in_off: usize = 0;
            inline for (.{ "rnn1", "rnn2" }, 0..) |rname, ri| {
                const wih = try m.getf(d ++ ".inter_rnn." ++ rname ++ ".weight_ih_l0", .{}, &buf);
                const whh = try m.getf(d ++ ".inter_rnn." ++ rname ++ ".weight_hh_l0", .{}, &buf);
                const bih = try m.getf(d ++ ".inter_rnn." ++ rname ++ ".bias_ih_l0", .{}, &buf);
                const bhh = try m.getf(d ++ ".inter_rnn." ++ rname ++ ".bias_hh_l0", .{}, &buf);
                const hn = whh.dim(1);
                const hstore = if (ri == 0) st.h1 else st.h2;
                const gi = self.scratch[off .. off + 3 * hn];
                const gh = self.scratch[off + 3 * hn .. off + 6 * hn];
                const xin = self.scratch[off + 6 * hn .. off + 6 * hn + half];
                for (0..fd) |f| {
                    for (0..half) |i| xin[i] = intra_fc[f * c + in_off + i];
                    const h = hstore[f * hn ..][0..hn];
                    gruStep(xin, h, wih, whh, bih, bhh, gi, gh);
                    for (0..hn) |k| inter[f * c + out_off + k] = h[k];
                }
                out_off += hn;
                in_off += half;
            }
        }
        // inter FC + LN + residual
        var out = self.arena.alloc(c, fd);
        {
            const w = try m.getf(d ++ ".inter_fc.w", .{}, &buf);
            const b = try m.getf(d ++ ".inter_fc.b", .{}, &buf);
            const in = w.dim(1);
            const tmp = self.scratch[4 * fd * c .. 5 * fd * c];
            for (0..fd) |f| {
                for (0..c) |o| {
                    var s: f32 = b.data[o];
                    const wr = w.data[o * in ..][0..in];
                    for (0..in) |i| s += wr[i] * inter[f * in + i];
                    tmp[f * c + o] = s;
                }
            }
            const lw = try m.getf(d ++ ".inter_ln.w", .{}, &buf);
            const lb = try m.getf(d ++ ".inter_ln.b", .{}, &buf);
            try self.layerNormWhole(tmp, lw, lb, 1e-8);
            for (0..fd) |f| {
                for (0..c) |ci| out.atP(ci, f).* = tmp[f * c + ci] + intra_fc[f * c + ci];
            }
        }
        return out;
    }
};

fn addFrames(dst: Frame, a: *const Frame, b: *const Frame) Frame {
    for (dst.d, a.d, b.d) |*o, av, bv| o.* = av + bv;
    return dst;
}

// ── STFT/ISTFT via the shipped windowed-DFT matrices ────────────────────────

pub fn stftFrame(m: *const Model, fr: *const [fft_size]f32, out: *[n_bins * 2]f32) !void {
    const wcos = (try m.get("stft.wcos")).data;
    const wsin = (try m.get("stft.wsin")).data;
    for (0..n_bins) |f| {
        const wc = wcos[f * fft_size ..][0..fft_size];
        const ws = wsin[f * fft_size ..][0..fft_size];
        var re: f32 = 0;
        var im: f32 = 0;
        for (fr, wc, ws) |x, c, s| {
            re += c * x;
            im += s * x;
        }
        out[f * 2] = re;
        out[f * 2 + 1] = im;
    }
}

pub fn istftFrame(m: *const Model, spec: *const [n_bins * 2]f32, out: *[fft_size]f32) !void {
    const icos = (try m.get("stft.icos")).data;
    const isin = (try m.get("stft.isin")).data;
    for (0..fft_size) |n| {
        var v: f32 = 0;
        const cr = icos[n * n_bins ..][0..n_bins];
        const sr = isin[n * n_bins ..][0..n_bins];
        for (0..n_bins) |f| {
            v += cr[f] * spec[f * 2] + sr[f] * spec[f * 2 + 1];
        }
        out[n] = v;
    }
}

test {
    _ = @import("aec_tests.zig");
}
