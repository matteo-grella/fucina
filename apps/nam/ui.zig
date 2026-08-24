//! Terminal helpers for the live loop. Two layers:
//!   - RawTerminal: single-key reads (ECHO/ICANON/ISIG/IXON off, VMIN=VTIME=0)
//!     so ^C/^D/^S arrive as bytes, plus a signal handler that always leaves
//!     the terminal sane on SIGINT/SIGTERM/SIGHUP (the "broken terminal after
//!     a kill" failure mode).
//!   - Dashboard: a pinned multi-row status block at the bottom of the screen
//!     (in/out level meters, colored flags, keybindings) drawn in place via a
//!     DECSTBM scroll region, synchronized output (DEC mode 2026), and a
//!     whole-frame diff so a steady signal emits nothing. Log lines still
//!     scroll above it. Falls back to a single rewritten status line when
//!     stdout is not a tty (pipes / CI).

const std = @import("std");

// ---------------------------------------------------------------------------
// ANSI / DEC escapes
// ---------------------------------------------------------------------------
const hide_cursor = "\x1b[?25l";
const show_cursor = "\x1b[?25h";
const reset_region = "\x1b[r"; // DECSTBM: full-screen scroll region
const save_cursor = "\x1b7"; // DECSC
const restore_cursor = "\x1b8"; // DECRC
const sync_begin = "\x1b[?2026h"; // synchronized output on
const sync_end = "\x1b[?2026l"; // synchronized output off (no-op on terminals that ignore it)

const reset = "\x1b[0m";
const dim = "\x1b[2m";
const green = "\x1b[32m";
const yellow = "\x1b[33m";
const red = "\x1b[31m";
const cyan = "\x1b[36m";
const byp_on = "\x1b[1;30;43m"; // bold black on yellow
const clip_on = "\x1b[1;97;41m"; // bold bright-white on red
const mute_on = "\x1b[1;30;46m"; // bold black on cyan

const default_height: usize = 5; // rule + in + out + status + keys

/// Pinned keybinding hints, most-essential first (so the tail is what drops on
/// a narrow terminal). `?` prints the full reference via live.printKeys().
const key_hints = [_][]const u8{
    "q=quit",  "space=byp", "m=mute",  "[ ]=prof", "1-9=slot", ",/.=trim",
    "+/-=gain", "g=gate",   "</>=thr", "t=tuner",  "n=norm",   "a=auto",
    "i/o=dev", "c=clr",     "?=help",
};

// ---------------------------------------------------------------------------
// Raw terminal + signal-safe restore
// ---------------------------------------------------------------------------

extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;

/// Current terminal size in character cells, or null when stdout is not a tty.
fn termSize() ?struct { rows: usize, cols: usize } {
    var ws: std.posix.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
    if (ioctl(1, @intCast(std.posix.T.IOCGWINSZ), &ws) != 0) return null;
    if (ws.row == 0 or ws.col == 0) return null;
    return .{ .rows = ws.row, .cols = ws.col };
}

const SavedTerminal = struct { fd: std.posix.fd_t, termios: std.posix.termios };
/// Set while raw mode is active; read by the signal handler (async context),
/// so it stays a plain optional that is written once on enable and cleared on
/// restore.
var g_saved: ?SavedTerminal = null;

fn signalRestore(sig: std.posix.SIG) callconv(.c) void {
    if (g_saved) |s| std.posix.tcsetattr(s.fd, .NOW, s.termios) catch {};
    // Raw libc write() is async-signal-safe; the buffered writer is not. The
    // leading SGR reset clears any color/background left mid-frame.
    const msg = reset ++ reset_region ++ show_cursor ++ "\r\n";
    _ = std.c.write(1, msg, msg.len);
    // Restore the default disposition and re-raise so the exit status reflects
    // the signal instead of swallowing it.
    var dfl = std.posix.Sigaction{
        .handler = .{ .handler = std.posix.SIG.DFL },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(sig, &dfl, null);
    std.posix.raise(sig) catch {};
}

fn setSignals(handler: ?std.posix.Sigaction.handler_fn) void {
    var act = std.posix.Sigaction{
        .handler = .{ .handler = handler },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(.INT, &act, null);
    std.posix.sigaction(.TERM, &act, null);
    std.posix.sigaction(.HUP, &act, null);
    // Crash paths too (Zig panic -> abort() -> ABRT; memory/illegal faults), so
    // a crash also leaves the terminal sane. signalRestore re-raises to DFL,
    // preserving the core dump / panic trace.
    std.posix.sigaction(.ABRT, &act, null);
    std.posix.sigaction(.SEGV, &act, null);
    std.posix.sigaction(.BUS, &act, null);
    std.posix.sigaction(.ILL, &act, null);
    std.posix.sigaction(.FPE, &act, null);
}

pub const RawTerminal = struct {
    fd: std.posix.fd_t,
    saved: std.posix.termios,
    active: bool,
    /// Set when a non-raw (non-tty) stdin reaches EOF — the caller should quit
    /// rather than spin (a closed stdin can never deliver a quit key).
    eof: bool = false,

    pub fn enable() !RawTerminal {
        const fd: std.posix.fd_t = 0; // stdin
        // Non-tty stdin (piped input, CI, redirected): skip raw mode, still poll
        // reads (an EOF there signals quit via poll()). Check isatty first so a
        // non-tty doesn't trigger std's tcgetattr "unexpected errno" trace dump;
        // the catch stays as a belt-and-suspenders fallback.
        if (std.c.isatty(fd) == 0) return .{ .fd = fd, .saved = undefined, .active = false };
        const saved = std.posix.tcgetattr(fd) catch {
            return .{ .fd = fd, .saved = undefined, .active = false };
        };
        var raw = saved;
        raw.lflag.ECHO = false;
        raw.lflag.ICANON = false;
        // ISIG off => ^C/^D/^Z reach poll() as bytes instead of signals, so the
        // normal quit path (and its defers) runs. IXON off => ^S/^Q never freeze
        // the live display.
        raw.lflag.ISIG = false;
        raw.iflag.IXON = false;
        raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
        raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
        try std.posix.tcsetattr(fd, .NOW, raw);
        g_saved = .{ .fd = fd, .termios = saved };
        setSignals(&signalRestore);
        return .{ .fd = fd, .saved = saved, .active = true };
    }

    pub fn restore(self: *RawTerminal) void {
        if (!self.active) return;
        // Restore the device FIRST, then drop the handlers: at every instant
        // either a handler is still installed (it will restore + re-raise) or
        // the terminal is already cooked — no window where a signal kills us raw.
        std.posix.tcsetattr(self.fd, .NOW, self.saved) catch {};
        g_saved = null;
        setSignals(std.posix.SIG.DFL);
        self.active = false;
    }

    /// Non-blocking single-key read; null when no key is pending.
    pub fn poll(self: *RawTerminal) ?u8 {
        var buf: [1]u8 = undefined;
        const n = std.posix.read(self.fd, &buf) catch return null;
        if (n == 0) {
            // Raw tty (VMIN=0): 0 = no key pending (normal, keep polling).
            // Non-tty stdin: 0 = EOF (closed pipe / /dev/null) — flag it so the
            // caller exits instead of spinning with no way to receive a quit key.
            if (!self.active) self.eof = true;
            return null;
        }
        return buf[0];
    }
};

// ---------------------------------------------------------------------------
// Snapshot the render loop hands to the renderer (decoupled from the atomics).
// ---------------------------------------------------------------------------
pub const View = struct {
    chain_index: usize,
    chain_count: usize,
    chain_name: []const u8,
    /// Number of stages in the active chain (1 = a bare profile).
    stage_count: usize = 1,
    /// Displayed (peak-held, decaying) input/output levels in dBFS.
    in_db: f32,
    out_db: f32,
    in_trim_db: f32,
    out_gain_db: f32,
    bypass: bool,
    muted: bool = false,
    normalize: bool,
    gate_on: bool,
    gate_db: f32,
    clipped: bool,
    oversize: bool,
    silent: bool,
    /// Transient echo of the last MIDI event; empty when none/stale.
    midi_label: []const u8,
    /// Tuner display: while on, the tuner row replaces the keybinding row
    /// (block height never changes; `?` still prints the full key list).
    tuner_on: bool = false,
    tuner: TunerView = .{},
};

/// Snapshot of the tuner for one rendered frame (built by the render loop
/// from the analysis thread's published atomics).
pub const TunerView = struct {
    pub const Mode = enum { listening, mono, poly };
    pub const StringTune = struct { active: bool = false, cents: f32 = 0 };

    mode: Mode = .listening,
    /// Note label ("E2", "F#3"); mono mode only.
    note: []const u8 = "",
    cents: f32 = 0,
    hz: f32 = 0,
    strings: [6]StringTune = @splat(.{}),
};

// ---------------------------------------------------------------------------
// Row composition (pure, width-aware, unit-tested). Escapes contribute bytes
// but zero display columns; multibyte box glyphs are one column each.
// ---------------------------------------------------------------------------

fn bufAppend(buf: []u8, len: *usize, s: []const u8) void {
    const n = @min(s.len, buf.len - len.*);
    @memcpy(buf[len.*..][0..n], s[0..n]);
    len.* += n;
}

const RowBuf = struct {
    data: []u8,
    len: usize = 0,
    vis: usize = 0,
    max_vis: usize,

    /// Bytes with no display width (SGR/escape codes).
    fn raw(self: *RowBuf, s: []const u8) void {
        bufAppend(self.data, &self.len, s);
    }
    /// `s` occupies `w` display columns; dropped whole if it would overflow.
    fn vstr(self: *RowBuf, s: []const u8, w: usize) void {
        if (self.vis + w > self.max_vis) return;
        bufAppend(self.data, &self.len, s);
        self.vis += w;
    }
    /// ASCII text: display width == byte length.
    fn ascii(self: *RowBuf, s: []const u8) void {
        self.vstr(s, s.len);
    }
    fn rep(self: *RowBuf, glyph: []const u8, n: usize) void {
        var i: usize = 0;
        while (i < n) : (i += 1) self.vstr(glyph, 1);
    }
    fn pf(self: *RowBuf, comptime fmt: []const u8, args: anytype) void {
        var tmp: [128]u8 = undefined;
        const s = std.fmt.bufPrint(&tmp, fmt, args) catch return;
        self.vstr(s, s.len); // formatted output here is ASCII
    }
};

fn dbInt(db: f32) i64 {
    return @intFromFloat(@round(std.math.clamp(db, -99, 99)));
}

/// Append up to `w` display columns of `name`, truncating on UTF-8 codepoint
/// boundaries (profile names are filenames and can be multibyte) and padding to
/// `w`. One column per codepoint, matching the box-glyph convention — so the
/// trailing flags keep their fixed columns instead of shifting on accented names.
fn appendName(rb: *RowBuf, name: []const u8, w: usize) void {
    var cols: usize = 0;
    var idx: usize = 0;
    while (idx < name.len and cols < w) {
        const sl: usize = std.unicode.utf8ByteSequenceLength(name[idx]) catch 1;
        if (idx + sl > name.len) break; // truncated/invalid tail: stop cleanly
        rb.vstr(name[idx..][0..sl], 1);
        idx += sl;
        cols += 1;
    }
    while (cols < w) : (cols += 1) rb.vstr(" ", 1);
}

fn appendSigned(rb: *RowBuf, val: f32) void {
    if (val >= 0) rb.ascii("+");
    rb.pf("{d:.1}", .{val});
}

fn meterRow(out: []u8, start: usize, label: []const u8, db: f32, knob_name: []const u8, knob_db: f32, width: usize) usize {
    var rb = RowBuf{ .data = out[start..], .max_vis = width };
    rb.ascii(label);
    var i = label.len;
    while (i < 4) : (i += 1) rb.ascii(" ");

    // Reserve room for the bracket pair, the dB readout, and the knob value.
    const overhead: usize = 30;
    const bar_i = std.math.clamp(@as(i64, @intCast(width)) - @as(i64, overhead), 8, 120);
    const bc: usize = @intCast(bar_i);

    rb.raw(dim);
    rb.vstr("▕", 1);
    rb.raw(reset);

    // -60 dBFS empty .. 0 dBFS full; color by zone so the eye reads "pushing
    // into yellow/red" peripherally without parsing the number.
    const min_db: f32 = -60;
    const frac = std.math.clamp((db - min_db) / (0 - min_db), 0, 1);
    const fill: usize = @intFromFloat(@round(frac * @as(f32, @floatFromInt(bc))));
    const g = bc * 4 / 5; // ~ -12 dBFS
    const y = bc * 19 / 20; // ~ -3 dBFS
    const gf = @min(fill, g);
    if (gf > 0) {
        rb.raw(green);
        rb.rep("█", gf);
    }
    if (fill > g) {
        const yf = @min(fill, y);
        rb.raw(yellow);
        rb.rep("█", yf - g);
    }
    if (fill > y) {
        rb.raw(red);
        rb.rep("█", fill - y);
    }
    rb.raw(dim);
    rb.rep("░", bc - fill);
    rb.raw(reset);
    rb.raw(dim);
    rb.vstr("▏", 1);
    rb.raw(reset);

    rb.pf(" {d:>4} dB", .{dbInt(db)});
    rb.pf("  {s} ", .{knob_name});
    appendSigned(&rb, knob_db);
    rb.raw(reset);
    return start + rb.len;
}

/// Builds the whole block into `out` as rows joined by '\n' (the '\n's are
/// markers for the caller's per-row positioning, never emitted to the terminal).
/// Every row ends in a reset so a row's color never bleeds into the next line's
/// erase. Pure: identical Views produce byte-identical output (the diff basis).
fn composeFrame(v: View, cols: usize, out: []u8) []const u8 {
    const width = @min(cols, 160);
    var len: usize = 0;

    // Row 0: dim rule — a stable visual anchor separating block from log.
    {
        var rb = RowBuf{ .data = out[len..], .max_vis = width };
        rb.raw(dim);
        rb.rep("─", width);
        rb.raw(reset);
        len += rb.len;
    }
    out[len] = '\n';
    len += 1;

    // Rows 1-2: meters.
    len = meterRow(out, len, "in", v.in_db, "trim", v.in_trim_db, width);
    out[len] = '\n';
    len += 1;
    len = meterRow(out, len, "out", v.out_db, "gain", v.out_gain_db, width);
    out[len] = '\n';
    len += 1;

    // Row 3: profile + fixed-position flags (color = state, position never
    // shifts, so the flags are catchable peripherally) + transient notes.
    {
        var rb = RowBuf{ .data = out[len..], .max_vis = width };
        rb.pf("[{d}/{d}] ", .{ v.chain_index + 1, v.chain_count });
        appendName(&rb, v.chain_name, 22);
        if (v.stage_count > 1) rb.pf(" x{d}", .{v.stage_count});
        rb.ascii(" ");
        flag(&rb, v.bypass, byp_on, "BYPASS", "bypass");
        rb.ascii(" ");
        flag(&rb, v.muted, mute_on, "MUTE", "mute");
        rb.ascii(" ");
        flag(&rb, v.normalize, green, "NORM", "norm");
        rb.ascii(" ");
        if (v.gate_on) {
            rb.raw(cyan);
            rb.pf("GATE{d:>4}", .{dbInt(v.gate_db)});
            rb.raw(reset);
        } else {
            rb.raw(dim);
            rb.ascii("gate    ");
            rb.raw(reset);
        }
        rb.ascii(" ");
        flag(&rb, v.clipped, clip_on, "CLIP", "clip");
        if (v.midi_label.len > 0) {
            rb.raw(cyan);
            rb.pf("  [MIDI {s}]", .{v.midi_label});
            rb.raw(reset);
        }
        if (v.silent) {
            rb.raw(yellow);
            rb.ascii("  NO INPUT? press i");
            rb.raw(reset);
        }
        if (v.oversize) {
            rb.raw(red);
            rb.ascii("  oversize!");
            rb.raw(reset);
        }
        rb.raw(reset);
        len += rb.len;
    }
    out[len] = '\n';
    len += 1;

    // Row 4: the tuner while it is on, otherwise keybindings (ASCII),
    // appended token-by-token so a narrow terminal shows as many whole
    // bindings as fit (never a mid-token cut); `q=quit` is first so it
    // always survives, and `?` prints the full list.
    if (v.tuner_on) {
        len = tunerRow(out, len, v.tuner, width);
    } else {
        var rb = RowBuf{ .data = out[len..], .max_vis = width };
        rb.raw(dim);
        for (key_hints, 0..) |k, ki| {
            const sep: usize = if (ki == 0) 0 else 2;
            if (rb.vis + sep + k.len > rb.max_vis) break;
            if (ki != 0) rb.ascii("  ");
            rb.ascii(k);
        }
        rb.raw(reset);
        len += rb.len;
    }
    return out[0..len];
}

/// Color for a cents deviation: in tune (green) within ±3, close (yellow)
/// within ±10, off (red) beyond.
fn centsColor(cents: f32) []const u8 {
    const a = @abs(cents);
    return if (a <= 3.0) green else if (a <= 10.0) yellow else red;
}

/// Standard-tuning string labels for the poly readout (low to high).
const string_labels = [6][]const u8{ "E", "A", "D", "G", "B", "e" };

/// The tuner row. Mono: a ±50-cent needle (2.5 cents/cell) + note, cents to
/// one decimal, and the measured frequency. Poly: per-string cents readout.
fn tunerRow(out: []u8, start: usize, t: TunerView, width: usize) usize {
    var rb = RowBuf{ .data = out[start..], .max_vis = width };
    rb.raw(cyan);
    rb.ascii("tun ");
    rb.raw(reset);
    switch (t.mode) {
        .listening => {
            rb.raw(dim);
            rb.ascii("(listening - pluck a string)   t=off");
            rb.raw(reset);
        },
        .mono => {
            const cells = 41;
            const center = cells / 2;
            const clamped = std.math.clamp(t.cents, -50.0, 50.0);
            const offset: i64 = @intFromFloat(@round(clamped / 50.0 * center));
            const pos: usize = @intCast(center + offset);
            const color = centsColor(t.cents);
            rb.raw(dim);
            rb.vstr("▕", 1);
            var i: usize = 0;
            while (i < cells) : (i += 1) {
                if (i == pos) {
                    rb.raw(reset);
                    rb.raw(color);
                    rb.vstr("●", 1);
                    rb.raw(reset);
                    rb.raw(dim);
                } else if (i == center) {
                    rb.raw(reset);
                    rb.vstr("│", 1);
                    rb.raw(dim);
                } else {
                    rb.vstr("─", 1);
                }
            }
            rb.vstr("▏", 1);
            rb.raw(reset);
            rb.raw(color);
            rb.pf(" {s:<4}", .{t.note});
            appendSigned(&rb, t.cents);
            rb.vstr("¢", 1);
            rb.raw(reset);
            rb.pf("  {d:>7.2} Hz", .{t.hz});
            rb.raw(dim);
            rb.ascii("  t=off");
            rb.raw(reset);
        },
        .poly => {
            for (t.strings, 0..) |s, si| {
                if (si != 0) rb.ascii("  ");
                if (s.active) {
                    rb.raw(centsColor(s.cents));
                    rb.pf("{s}", .{string_labels[si]});
                    if (s.cents >= 0) {
                        rb.pf("+{d:.0}", .{s.cents});
                    } else {
                        rb.pf("{d:.0}", .{s.cents});
                    }
                    rb.raw(reset);
                } else {
                    rb.raw(dim);
                    rb.pf("{s}", .{string_labels[si]});
                    rb.vstr("··", 2);
                    rb.raw(reset);
                }
            }
            rb.raw(dim);
            rb.ascii("   strum  t=off");
            rb.raw(reset);
        },
    }
    return start + rb.len;
}

fn flag(rb: *RowBuf, on: bool, on_color: []const u8, on_text: []const u8, off_text: []const u8) void {
    if (on) {
        rb.raw(on_color);
        rb.ascii(on_text);
    } else {
        rb.raw(dim);
        rb.ascii(off_text);
    }
    rb.raw(reset);
}

// ---------------------------------------------------------------------------
// Dashboard: the pinned block + scroll region.
// ---------------------------------------------------------------------------
pub const Dashboard = struct {
    io: std.Io,
    active: bool = false,
    rows: usize = 0,
    cols: usize = 0,
    height: usize = default_height,
    /// Whether the block + scroll region + hidden cursor are currently
    /// installed (false while dormant on a too-small window).
    reserved: bool = false,
    last_len: usize = 0,
    frame_buf: [4096]u8 = undefined,
    last_frame: [4096]u8 = undefined,
    emit_buf: [6144]u8 = undefined,

    /// Reserves the bottom `height` rows, sets the scroll region above them,
    /// and hides the cursor. Inactive (no-op render) when stdout is not a tty
    /// or the window is too small — the caller then uses statusLineView().
    pub fn init(io: std.Io) Dashboard {
        var d: Dashboard = .{ .io = io };
        const sz = termSize() orelse return d;
        if (sz.rows < d.height + 2 or sz.cols < 24) return d;
        d.rows = sz.rows;
        d.cols = sz.cols;
        d.active = true;
        d.reserve(); // hide cursor, reserve the bottom rows, install the region
        return d;
    }

    pub fn deinit(self: *Dashboard) void {
        if (!self.active) return;
        var b: [24]u8 = undefined;
        self.writeRaw(reset); // defense-in-depth: never leave a stray color/bg
        self.writeRaw(std.fmt.bufPrint(&b, "\x1b[{d};1H", .{self.rows}) catch "");
        self.writeRaw(reset_region);
        self.writeRaw(show_cursor);
        self.writeRaw("\r\n");
        self.active = false;
    }

    fn writeRaw(self: *Dashboard, s: []const u8) void {
        std.Io.File.stdout().writeStreamingAll(self.io, s) catch {};
    }

    /// Hide the cursor, scroll up to reserve the bottom `height` rows, and
    /// install the scroll region above them. Forces a full redraw next frame.
    fn reserve(self: *Dashboard) void {
        self.writeRaw(hide_cursor);
        self.writeRaw("\n" ** default_height);
        self.applyRegion();
        self.reserved = true;
        self.last_len = 0;
    }

    /// Drop the region and show the cursor — the terminal is left sane so log
    /// output scrolls normally while the block is dormant (too-small window).
    fn release(self: *Dashboard) void {
        self.writeRaw(reset_region);
        self.writeRaw(show_cursor);
        self.reserved = false;
    }

    /// Scroll region = rows [1, rows-height]; park the cursor on its last line
    /// so subsequent log writes scroll there, leaving the block untouched.
    fn applyRegion(self: *Dashboard) void {
        var b: [48]u8 = undefined;
        const top = self.rows - self.height;
        self.writeRaw(std.fmt.bufPrint(&b, "\x1b[1;{d}r\x1b[{d};1H", .{ top, top }) catch return);
    }

    pub fn render(self: *Dashboard, v: View) void {
        if (!self.active) return;
        const sz = termSize() orelse return; // can't size: skip, don't corrupt
        const big_enough = sz.rows >= self.height + 2 and sz.cols >= 24;
        if (!big_enough) {
            // No room for the block: drop the region + show the cursor so the
            // terminal stays sane (logs scroll normally) until it grows back.
            if (self.reserved) self.release();
            self.rows = sz.rows;
            self.cols = sz.cols;
            return;
        }
        if (!self.reserved or sz.rows != self.rows or sz.cols != self.cols) {
            // Coming back from dormant, or a resize: clear the screen so a
            // grown window doesn't orphan a ghost copy of the old block, then
            // re-install the region at the new size. (The first reserve happens
            // in init(), which keeps the startup log — this path is resize-only.)
            self.rows = sz.rows;
            self.cols = sz.cols;
            self.writeRaw("\x1b[2J\x1b[H");
            self.reserve();
        }

        const frame = composeFrame(v, self.cols, &self.frame_buf);
        if (self.last_len == frame.len and std.mem.eql(u8, frame, self.last_frame[0..self.last_len])) return;

        const top = self.rows - self.height + 1;
        var w: usize = 0;
        bufAppend(&self.emit_buf, &w, sync_begin);
        bufAppend(&self.emit_buf, &w, save_cursor);
        bufAppend(&self.emit_buf, &w, reset); // reset SGR so each \x1b[2K erases with default colors
        var it = std.mem.splitScalar(u8, frame, '\n');
        var i: usize = 0;
        while (it.next()) |line| : (i += 1) {
            if (i >= self.height) break;
            var hb: [24]u8 = undefined;
            const head = std.fmt.bufPrint(&hb, "\x1b[{d};1H\x1b[2K", .{top + i}) catch continue;
            bufAppend(&self.emit_buf, &w, head);
            bufAppend(&self.emit_buf, &w, line);
        }
        bufAppend(&self.emit_buf, &w, restore_cursor);
        bufAppend(&self.emit_buf, &w, sync_end);
        self.writeRaw(self.emit_buf[0..w]);

        @memcpy(self.last_frame[0..frame.len], frame);
        self.last_len = frame.len;
    }
};

// ---------------------------------------------------------------------------
// Single-line fallback (non-tty) + plain scrolling log.
// ---------------------------------------------------------------------------

/// Single rewritten status line — the fallback when the block is inactive.
pub fn statusLineView(io: std.Io, v: View) void {
    var gbuf: [24]u8 = undefined;
    const gate_label = if (v.gate_on)
        std.fmt.bufPrint(&gbuf, "  GATE {d:.0}", .{v.gate_db}) catch ""
    else
        "";
    var mbuf: [40]u8 = undefined;
    const midi = if (v.midi_label.len > 0)
        std.fmt.bufPrint(&mbuf, "  [MIDI {s}]", .{v.midi_label}) catch ""
    else
        "";
    var sbuf: [16]u8 = undefined;
    const stages_label = if (v.stage_count > 1)
        std.fmt.bufPrint(&sbuf, " x{d}", .{v.stage_count}) catch ""
    else
        "";
    var tbuf: [48]u8 = undefined;
    const tuner_label = if (!v.tuner_on)
        ""
    else switch (v.tuner.mode) {
        .listening => "  [tuner ...]",
        .mono => std.fmt.bufPrint(&tbuf, "  [tuner {s} {d:.1}c]", .{ v.tuner.note, v.tuner.cents }) catch "",
        .poly => "  [tuner strum]",
    };
    statusLine(io, "[{d}/{d}] {s}{s}{s}{s}  in {d:>6.1} dB (trim {d:>5.1})  out {d:>6.1} dB (gain {d:>5.1}{s}){s}{s}{s}{s}{s}{s}", .{
        v.chain_index + 1,
        v.chain_count,
        v.chain_name,
        stages_label,
        if (v.bypass) "  [BYPASS]" else "",
        if (v.muted) "  [MUTE]" else "",
        v.in_db,
        v.in_trim_db,
        v.out_db,
        v.out_gain_db,
        if (v.normalize) " NORM" else "",
        gate_label,
        tuner_label,
        midi,
        if (v.clipped) "  CLIP!" else "",
        if (v.oversize) "  (oversize blocks!)" else "",
        if (v.silent) "  NO INPUT? press i to cycle capture device (also check mic permission)" else "",
    });
}

/// Rewrites the current terminal line (CR + clear-to-end).
pub fn statusLine(io: std.Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "\r\x1b[2K" ++ fmt, args) catch return;
    std.Io.File.stdout().writeStreamingAll(io, text) catch {};
}

pub fn plainLine(io: std.Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt ++ "\n", args) catch return;
    std.Io.File.stdout().writeStreamingAll(io, text) catch {};
}

/// Linear amplitude -> dBFS string for meters.
pub fn dbfs(amplitude: f32) f32 {
    if (amplitude <= 1e-7) return -140.0;
    return 20.0 * std.math.log10(amplitude);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    _ = @import("ui_tests.zig");
}

const test_view = View{
    .chain_index = 1,
    .chain_count = 5,
    .chain_name = "Fender 65",
    .stage_count = 1,
    .in_db = -6,
    .out_db = -30,
    .in_trim_db = 0,
    .out_gain_db = 3,
    .bypass = false,
    .normalize = true,
    .gate_on = true,
    .gate_db = -65,
    .clipped = true,
    .oversize = false,
    .silent = false,
    .midi_label = "",
};

test "composeFrame: meters fill with level, flags rendered, deterministic" {
    var buf: [4096]u8 = undefined;
    const f1 = composeFrame(test_view, 100, &buf);

    // Flags present; filled cells present.
    try std.testing.expect(std.mem.indexOf(u8, f1, "NORM") != null);
    try std.testing.expect(std.mem.indexOf(u8, f1, "CLIP") != null);
    try std.testing.expect(std.mem.indexOf(u8, f1, "GATE") != null);
    try std.testing.expect(std.mem.indexOf(u8, f1, "q=quit") != null);
    try std.testing.expect(std.mem.count(u8, f1, "█") > 0);

    // Higher level => strictly more filled cells.
    var buf2: [4096]u8 = undefined;
    var low = test_view;
    low.in_db = -60;
    low.out_db = -60;
    const f2 = composeFrame(low, 100, &buf2);
    try std.testing.expect(std.mem.count(u8, f1, "█") > std.mem.count(u8, f2, "█"));

    // Exactly `default_height` rows.
    try std.testing.expectEqual(default_height - 1, std.mem.count(u8, f1, "\n"));

    // Pure: identical View => identical bytes (the diff-skip contract).
    var buf3: [4096]u8 = undefined;
    const f3 = composeFrame(test_view, 100, &buf3);
    try std.testing.expectEqualStrings(f1, f3);
}

test "composeFrame: multibyte profile name stays valid UTF-8 (no mid-codepoint cut)" {
    var buf: [4096]u8 = undefined;
    var v = test_view;
    // 1 ASCII + accented chars: a byte-index truncation at column 22 would split
    // a 2-byte codepoint, leaving an invalid partial sequence in the frame.
    v.chain_name = "x" ++ ("é" ** 30);
    const f = composeFrame(v, 100, &buf);
    try std.testing.expect(std.unicode.utf8ValidateSlice(f));
    // Flags still render after the (truncated) name.
    try std.testing.expect(std.mem.indexOf(u8, f, "CLIP") != null);
}

test "composeFrame: tuner row replaces key hints while on" {
    var buf: [4096]u8 = undefined;
    var v = test_view;
    v.tuner_on = true;
    v.tuner = .{ .mode = .mono, .note = "E2", .cents = -3.2, .hz = 82.25 };
    const f = composeFrame(v, 120, &buf);
    try std.testing.expect(std.mem.indexOf(u8, f, "q=quit") == null);
    try std.testing.expect(std.mem.indexOf(u8, f, "E2") != null);
    try std.testing.expect(std.mem.indexOf(u8, f, "●") != null);
    try std.testing.expect(std.mem.indexOf(u8, f, "¢") != null);
    try std.testing.expect(std.mem.indexOf(u8, f, "82.25 Hz") != null);
    try std.testing.expectEqual(default_height - 1, std.mem.count(u8, f, "\n"));
    try std.testing.expect(std.unicode.utf8ValidateSlice(f));

    // Poly readout: active strings show signed cents, inactive are dimmed.
    v.tuner = .{ .mode = .poly };
    v.tuner.strings[0] = .{ .active = true, .cents = -15 };
    v.tuner.strings[4] = .{ .active = true, .cents = 20 };
    const p = composeFrame(v, 120, &buf);
    try std.testing.expect(std.mem.indexOf(u8, p, "E-15") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "B+20") != null);
    try std.testing.expect(std.mem.indexOf(u8, p, "strum") != null);

    // Off again: key hints return.
    v.tuner_on = false;
    const k = composeFrame(v, 120, &buf);
    try std.testing.expect(std.mem.indexOf(u8, k, "q=quit") != null);
    try std.testing.expect(std.mem.indexOf(u8, k, "t=tuner") != null);
}

test "composeFrame: clip flag toggles the colored chip" {
    var on_buf: [4096]u8 = undefined;
    var off_buf: [4096]u8 = undefined;
    const on = composeFrame(test_view, 100, &on_buf);
    var no_clip = test_view;
    no_clip.clipped = false;
    const off = composeFrame(no_clip, 100, &off_buf);
    try std.testing.expect(std.mem.indexOf(u8, on, clip_on ++ "CLIP") != null);
    try std.testing.expect(std.mem.indexOf(u8, off, clip_on ++ "CLIP") == null);
    try std.testing.expect(std.mem.indexOf(u8, off, "clip") != null); // dim off-state
}
