//! Signal Rail — the voiceagent's status rail (Signal Rail 1.0 spec, scaled
//! to this agent's states). A one-row instrument display: state label +
//! `[` logical rail `]` + auxiliary value. Not a spinner, not a waveform —
//! a state-display system with spatial semantics:
//!
//!   INPUT (28%) | PROCESSING (44%) | OUTPUT (28%)
//!
//! listening expands in the input zone; captured input collapses; thinking
//! reads left→right across a deterministic field; speaking emits packets
//! rightward in the output zone; the pause-then-commit hold freezes behind a
//! boundary; interruption cuts movement with a hard stop; a completed reply
//! sweeps once. Deterministic: frames are a pure function of (state, state
//! entry tick, tick, width, quantized levels, seed) — no wall clock, no
//! randomness. 12 Hz base tick (83.33 ms).
//!
//! The logical cell model is renderer-independent (no ANSI inside);
//! `render` maps cells through a glyph profile (instrument square / safe
//! block / ASCII) and a color mode (truecolor obsidian_instrument palette /
//! monochrome dim-normal-bold). Snapshot tests pin the ASCII renders at 25
//! cells. Deviations from the full spec are listed in the voiceagent README.

const std = @import("std");

pub const tick_ns: u64 = 83_333_333; // 12 Hz base tick

pub const State = enum {
    idle,
    listening,
    captured,
    thinking,
    speaking,
    waiting, // pause-then-commit hold (spec WAITING)
    acting, // tool execution (determinate via Ctx.progress, else packet)
    needs_input, // control returned to the user, response expected
    complete,
    interrupted,
    err,
};

pub fn label(s: State) []const u8 {
    return switch (s) {
        .idle => "IDLE",
        .listening => "LISTENING",
        .captured => "CAPTURED",
        .thinking => "THINKING",
        .speaking => "SPEAKING",
        .waiting => "WAITING",
        .acting => "ACTING",
        .needs_input => "NEEDS INPUT",
        .complete => "COMPLETE",
        .interrupted => "INTERRUPTED",
        .err => "ERROR",
    };
}

pub const GlyphRole = enum { track, low, medium, high, peak, head_right, head_left, boundary, hard_boundary, fracture };
pub const ColorRole = enum { track, neutral, neutral_hot, input, input_hot, output, output_hot, success, success_hot, interrupted, err, err_hot };
pub const Emphasis = enum { dim, normal, bright };

pub const Cell = struct {
    glyph: GlyphRole = .track,
    color: ColorRole = .track,
    emphasis: Emphasis = .dim,
};

pub const max_width = 61;

pub const Ctx = struct {
    width: usize, // logical cells (odd preferred; 25..61)
    tick: u64, // global monotonic tick
    entry_tick: u64, // tick at state entry
    input_q: u3 = 0, // quantized 0..4
    output_q: u3 = 0,
    seed: u64 = 0, // deterministic (turn counter)
    progress: ?f32 = null, // ACTING determinate progress (0..1)
    motion: enum { normal, off } = .normal,
};

pub const Zones = struct {
    input: usize,
    processing: usize,
    output: usize,

    pub fn of(width: usize) Zones {
        const in = (width * 28 + 50) / 100;
        const proc = (width * 44 + 50) / 100;
        return .{ .input = in, .processing = proc, .output = width - in - proc };
    }
};

/// Quantize a 0..1 level into 0..4 (spec §14 bands).
pub fn quantize(level: f32) u3 {
    const v = std.math.clamp(if (std.math.isNan(level)) 0 else level, 0, 1);
    if (v < 0.10) return 0;
    if (v < 0.30) return 1;
    if (v < 0.50) return 2;
    if (v < 0.75) return 3;
    return 4;
}

/// Stepped meter response: rise ≤2, fall ≤1 per tick.
pub fn stepLevel(prev: u3, target: u3) u3 {
    if (target > prev) return @min(target, prev + 2);
    if (target < prev) return prev - 1;
    return prev;
}

fn hash3(seed: u64, a: u64, b: u64) u64 {
    var h = seed ^ (a *% 0x9E3779B97F4A7C15) ^ (b *% 0xBF58476D1CE4E5B9);
    h ^= h >> 30;
    h *%= 0xBF58476D1CE4E5B9;
    h ^= h >> 27;
    h *%= 0x94D049BB133111EB;
    return h ^ (h >> 31);
}

fn track(cells: []Cell) void {
    for (cells) |*c| c.* = .{};
}

/// Generate one logical frame. Pure and deterministic.
pub fn frame(state: State, ctx: Ctx, cells: []Cell) void {
    const w = ctx.width;
    const z = Zones.of(w);
    const t = ctx.tick -% ctx.entry_tick;
    track(cells[0..w]);

    switch (state) {
        .idle => {
            // one stable marker at the processing-zone center; no animation
            const c = z.input + z.processing / 2;
            cells[c] = .{ .glyph = .medium, .color = .neutral, .emphasis = .normal };
        },
        .listening => {
            // quantized amplitude expands from the input-zone origin
            const origin = z.input / 2;
            const q: usize = ctx.input_q;
            if (q == 0) {
                cells[origin] = .{ .glyph = .medium, .color = .input, .emphasis = .normal };
            } else {
                const radius = @min(z.input / 2, q + 1);
                var d: usize = 0;
                while (d <= radius) : (d += 1) {
                    const lvl = std.math.clamp(@as(i32, @intCast(q)) + 1 - @as(i32, @intCast(d)), 1, 4);
                    const g: GlyphRole = switch (lvl) {
                        1 => .low,
                        2 => .medium,
                        3 => .high,
                        else => .peak,
                    };
                    const col: ColorRole = if (lvl >= 4) .input_hot else .input;
                    const em: Emphasis = if (lvl >= 4) .bright else .normal;
                    if (origin + d < z.input) cells[origin + d] = .{ .glyph = g, .color = col, .emphasis = em };
                    if (d != 0 and origin >= d) cells[origin - d] = .{ .glyph = g, .color = col, .emphasis = em };
                }
            }
        },
        .captured => {
            // 4-tick collapse toward the origin, ending input-hot
            const origin = z.input / 2;
            const phase = @min(t, 3);
            const spread: usize = switch (phase) {
                0 => 3,
                1 => 2,
                else => 1,
            };
            var d: usize = 0;
            while (d < spread) : (d += 1) {
                const em: Emphasis = if (phase >= 3) .bright else .normal;
                const col: ColorRole = if (phase >= 3) .input_hot else .input;
                if (origin + d < z.input) cells[origin + d] = .{ .glyph = .peak, .color = col, .emphasis = em };
                if (d != 0 and origin >= d) cells[origin - d] = .{ .glyph = .peak, .color = col, .emphasis = em };
            }
        },
        .thinking => {
            // deterministic sparse field + read head, one cell per 2 ticks,
            // never bouncing; new field each pass
            const p0 = z.input;
            const plen = z.processing;
            const steps_per_pass = plen + 1; // +1 = one dim tick between passes
            const pos_in_pass = (t / 2) % steps_per_pass;
            const pass = (t / 2) / steps_per_pass;
            for (0..plen) |i| {
                const h = hash3(ctx.seed, pass, i) % 100;
                if (h < 25) {
                    cells[p0 + i] = .{ .glyph = .low, .color = .neutral, .emphasis = .dim };
                } else if (h < 40) {
                    cells[p0 + i] = .{ .glyph = .medium, .color = .neutral, .emphasis = .normal };
                }
            }
            if (ctx.motion == .off) {
                cells[p0 + plen / 2] = .{ .glyph = .head_right, .color = .neutral_hot, .emphasis = .bright };
            } else if (pos_in_pass < plen) {
                const head = p0 + pos_in_pass;
                cells[head] = .{ .glyph = .head_right, .color = .neutral_hot, .emphasis = .bright };
                if (pos_in_pass >= 1) cells[head - 1] = .{ .glyph = .medium, .color = .neutral, .emphasis = .normal };
                if (pos_in_pass >= 2) cells[head - 2] = .{ .glyph = .low, .color = .neutral, .emphasis = .dim };
            } // else: the one-tick inter-pass dim (field only)
        },
        .speaking => {
            // packets spawn at the output boundary and travel right one
            // cell/tick; spawn period follows the quantized output level
            const o0 = z.input + z.processing;
            const olen = z.output;
            const period: u64 = switch (ctx.output_q) {
                0 => 12,
                1 => 8,
                2 => 6,
                3 => 4,
                else => 3,
            };
            if (ctx.motion == .off) {
                cells[o0 + olen / 2] = .{ .glyph = .medium, .color = .output, .emphasis = .normal };
                cells[o0 + olen / 2 + 1] = .{ .glyph = .head_right, .color = .output_hot, .emphasis = .bright };
            } else {
                // up to 3 live packets, newest last; length follows level
                const plen_pkt: usize = @min(3, @as(usize, ctx.output_q) + 1);
                var spawn_i: u64 = 0;
                var live: usize = 0;
                while (live < 3) : (spawn_i += 1) {
                    if (spawn_i * period > t) break;
                    const age = t - spawn_i * period;
                    if (age >= olen + plen_pkt) continue; // left the rail
                    live += 1;
                    const head_pos = age; // cells traveled
                    var k: usize = 0;
                    while (k < plen_pkt + 1) : (k += 1) {
                        if (head_pos < k) break;
                        const cell_off = head_pos - k;
                        if (cell_off >= olen) continue;
                        const idx = o0 + cell_off;
                        const new: Cell = if (k == 0)
                            .{ .glyph = .head_right, .color = .output_hot, .emphasis = .bright }
                        else if (k == 1)
                            .{ .glyph = .medium, .color = .output, .emphasis = .normal }
                        else
                            .{ .glyph = .low, .color = .output, .emphasis = .dim };
                        // overlap keeps the highest intensity
                        if (@intFromEnum(cells[idx].glyph) < @intFromEnum(new.glyph) or cells[idx].glyph == .track)
                            cells[idx] = new;
                    }
                }
            }
        },
        .waiting => {
            // pause-then-commit: frozen dim residue, pulsing boundary at the
            // processing/output boundary (where emission stopped)
            const b = z.input + z.processing;
            for (0..b) |i| {
                if (hash3(ctx.seed, 7, i) % 100 < 30)
                    cells[i] = .{ .glyph = .low, .color = .neutral, .emphasis = .dim };
            }
            const bright = (t % 11) < 4; // ~300ms bright / ~600ms normal
            cells[b] = .{ .glyph = .boundary, .color = .input, .emphasis = if (bright and ctx.motion == .normal) .bright else .normal };
        },
        .acting => {
            if (ctx.progress) |p| {
                // determinate: committed cells + head + track (spec §21.1)
                const pc = std.math.clamp(p, 0, 1);
                const committed = @min(w - 1, @as(usize, @intFromFloat(pc * @as(f32, @floatFromInt(w)))));
                for (0..committed) |i| cells[i] = .{ .glyph = .medium, .color = .neutral, .emphasis = .normal };
                cells[committed] = .{ .glyph = .head_right, .color = .neutral_hot, .emphasis = .bright };
            } else {
                // indeterminate: bounded work packet through processing +
                // output, two-tick pause, restart — never bouncing (§21.2)
                const span = z.processing + z.output;
                const cycle = span + 2;
                const pos = t % cycle;
                if (pos < span and ctx.motion == .normal) {
                    const head = z.input + pos;
                    cells[head] = .{ .glyph = .head_right, .color = .neutral_hot, .emphasis = .bright };
                    if (pos >= 1) cells[head - 1] = .{ .glyph = .peak, .color = .neutral, .emphasis = .normal };
                    if (pos >= 2) cells[head - 2] = .{ .glyph = .medium, .color = .neutral, .emphasis = .normal };
                    if (pos >= 3) cells[head - 3] = .{ .glyph = .low, .color = .neutral, .emphasis = .dim };
                } else if (ctx.motion == .off) {
                    const b = z.input + z.processing;
                    cells[b - 1] = .{ .glyph = .peak, .color = .neutral, .emphasis = .normal };
                    cells[b] = .{ .glyph = .head_right, .color = .neutral_hot, .emphasis = .bright };
                }
            }
        },
        .needs_input => {
            // control handed back: processing + input markers alternate
            // (A 4 ticks / B 4 ticks / pause 7 ticks — spec §23 rhythm)
            const in_mark = z.input / 2;
            const proc_mark = z.input + z.processing / 2;
            const phase = t % 15;
            const a_hot = phase < 4;
            const b_hot = phase >= 4 and phase < 8;
            cells[proc_mark] = .{
                .glyph = if (a_hot) .medium else .low,
                .color = .input,
                .emphasis = if (a_hot and ctx.motion == .normal) .bright else .normal,
            };
            cells[in_mark] = .{
                .glyph = if (b_hot or ctx.motion == .off) .medium else .low,
                .color = .input_hot,
                .emphasis = if (b_hot and ctx.motion == .normal) .bright else .normal,
            };
        },
        .complete => {
            // one rightward sweep, then a settled medium rail
            const speed = (w + 6) / 7; // cells per tick
            const head = (t + 1) * speed;
            if (head < w and ctx.motion == .normal) {
                for (0..head) |i| cells[i] = .{ .glyph = .medium, .color = .success, .emphasis = .normal };
                cells[head] = .{ .glyph = .head_right, .color = .success_hot, .emphasis = .bright };
            } else {
                for (0..w) |i| cells[i] = .{ .glyph = .medium, .color = .success, .emphasis = .normal };
            }
        },
        .interrupted => {
            // stop, retract to the input/processing boundary, hard cut
            const b = z.input;
            const retract: usize = @min(t, 3);
            const spill = (3 - retract) * 2;
            for (0..b) |i| cells[i] = .{ .glyph = .medium, .color = .interrupted, .emphasis = .normal };
            var i: usize = 0;
            while (i < spill and b + 1 + i < w) : (i += 1) {
                cells[b + 1 + i] = .{ .glyph = .low, .color = .interrupted, .emphasis = .dim };
            }
            cells[b] = .{ .glyph = .hard_boundary, .color = .err, .emphasis = .bright };
        },
        .err => {
            // saturate (2 ticks) → blackout (1) → deterministic fracture
            if (t < 2 and ctx.motion == .normal) {
                for (cells[0..w]) |*c| c.* = .{ .glyph = .peak, .color = .err_hot, .emphasis = .bright };
            } else if (t == 2 and ctx.motion == .normal) {
                for (cells[0..w]) |*c| c.* = .{ .glyph = .fracture, .color = .track, .emphasis = .dim };
            } else {
                var i: usize = 0;
                while (i < w) {
                    const run = 2 + hash3(ctx.seed, 13, i) % 4;
                    const gap = 1 + hash3(ctx.seed, 17, i) % 2;
                    var k: usize = 0;
                    while (k < run and i < w) : (k += 1) {
                        cells[i] = .{ .glyph = .peak, .color = .err, .emphasis = .normal };
                        i += 1;
                    }
                    k = 0;
                    while (k < gap and i < w) : (k += 1) {
                        cells[i] = .{ .glyph = .fracture, .color = .track, .emphasis = .dim };
                        i += 1;
                    }
                }
            }
        },
    }
}

// --- rendering ---------------------------------------------------------------

pub const GlyphProfile = enum { instrument_square, safe_block, ascii };

fn glyphStr(p: GlyphProfile, g: GlyphRole) []const u8 {
    return switch (p) {
        .instrument_square => switch (g) {
            .track => "─",
            .low => "▪",
            .medium => "■",
            .high => "█",
            .peak => "█",
            .head_right => "▐",
            .head_left => "▌",
            .boundary => "│",
            .hard_boundary => "┃",
            .fracture => " ",
        },
        .safe_block => switch (g) {
            .track => "─",
            .low => "▂",
            .medium => "▄",
            .high => "▆",
            .peak => "█",
            .head_right => "▐",
            .head_left => "▌",
            .boundary => "│",
            .hard_boundary => "┃",
            .fracture => " ",
        },
        .ascii => switch (g) {
            .track => "-",
            .low => ".",
            .medium => "=",
            .high => "#",
            .peak => "#",
            .head_right => ">",
            .head_left => "<",
            .boundary => "|",
            .hard_boundary => "!",
            .fracture => " ",
        },
    };
}

/// obsidian_instrument truecolor palette (spec §11.1).
fn rgb(role: ColorRole) [3]u8 {
    return switch (role) {
        .track => .{ 0x30, 0x36, 0x3B },
        .neutral => .{ 0xC9, 0xC3, 0xB4 },
        .neutral_hot => .{ 0xF0, 0xEB, 0xDD },
        .input => .{ 0xD4, 0x9A, 0x32 },
        .input_hot => .{ 0xFF, 0xC7, 0x5A },
        .output => .{ 0x8E, 0xAF, 0x8B },
        .output_hot => .{ 0xC6, 0xE1, 0xBB },
        .success => .{ 0x69, 0xB5, 0x78 },
        .success_hot => .{ 0xA8, 0xDB, 0xA9 },
        .interrupted => .{ 0x8E, 0x45, 0x42 },
        .err => .{ 0xB7, 0x40, 0x3D },
        .err_hot => .{ 0xEF, 0x65, 0x5D },
    };
}

fn labelColor(s: State) ColorRole {
    return switch (s) {
        .idle, .thinking, .acting => .neutral,
        .listening, .captured, .waiting, .needs_input => .input,
        .speaking => .output,
        .complete => .success,
        .interrupted => .interrupted,
        .err => .err_hot,
    };
}

pub const ColorMode = enum { truecolor, mono };

/// Render label + caps + cells + aux into `out` (ANSI). Caller supplies the
/// cells from `frame`.
pub fn render(
    out: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    state: State,
    cells: []const Cell,
    aux: []const u8,
    profile: GlyphProfile,
    color: ColorMode,
) !void {
    const lbl = label(state);
    if (color == .truecolor) {
        const c = rgb(labelColor(state));
        try out.print(allocator, "\x1b[38;2;{d};{d};{d}m", .{ c[0], c[1], c[2] });
    }
    try out.appendSlice(allocator, lbl);
    var pad: usize = if (lbl.len < 12) 12 - lbl.len else 1;
    while (pad > 0) : (pad -= 1) try out.append(allocator, ' ');
    if (color == .truecolor) {
        try out.appendSlice(allocator, "\x1b[38;2;74;81;87m["); // frame color caps
    } else {
        try out.appendSlice(allocator, "\x1b[0m[");
    }
    var prev_sgr: u64 = std.math.maxInt(u64);
    for (cells) |cell| {
        // compact SGR switching: only emit when style changes
        const key: u64 = (@as(u64, @intFromEnum(cell.color)) << 8) | @intFromEnum(cell.emphasis);
        if (key != prev_sgr) {
            prev_sgr = key;
            if (color == .truecolor) {
                const c = rgb(cell.color);
                const em: []const u8 = switch (cell.emphasis) {
                    .dim => "\x1b[2m",
                    .normal => "",
                    .bright => "\x1b[1m",
                };
                try out.print(allocator, "\x1b[0m{s}\x1b[38;2;{d};{d};{d}m", .{ em, c[0], c[1], c[2] });
            } else {
                const em: []const u8 = switch (cell.emphasis) {
                    .dim => "\x1b[0;2m",
                    .normal => "\x1b[0m",
                    .bright => "\x1b[0;1m",
                };
                try out.appendSlice(allocator, em);
            }
        }
        try out.appendSlice(allocator, glyphStr(profile, cell.glyph));
    }
    if (color == .truecolor) {
        try out.appendSlice(allocator, "\x1b[0m\x1b[38;2;74;81;87m]");
    } else {
        try out.appendSlice(allocator, "\x1b[0m]");
    }
    if (aux.len > 0) try out.print(allocator, "  \x1b[2m{s}\x1b[0m", .{aux}) else try out.appendSlice(allocator, "\x1b[0m");
}

/// ASCII-only render of the logical cells (for tests and NO_COLOR pipes).
pub fn renderAsciiCells(cells: []const Cell, out: []u8) []const u8 {
    for (cells, 0..) |c, i| out[i] = glyphStr(.ascii, c.glyph)[0];
    return out[0..cells.len];
}

// --- tests (deterministic, 25 cells, ASCII snapshots) ------------------------

fn snap(state: State, ctx: Ctx) [25]u8 {
    var cells: [25]Cell = undefined;
    var c = ctx;
    c.width = 25;
    frame(state, c, &cells);
    var buf: [25]u8 = undefined;
    _ = renderAsciiCells(&cells, &buf);
    return buf;
}

test "zones sum and split for all preset widths" {
    for ([_]usize{ 25, 37, 49, 61 }) |w| {
        const z = Zones.of(w);
        try std.testing.expectEqual(w, z.input + z.processing + z.output);
        try std.testing.expect(z.input >= 6 and z.output >= 6);
    }
}

test "idle: stable center marker, no motion" {
    const a = snap(.idle, .{ .width = 25, .tick = 0, .entry_tick = 0 });
    const b = snap(.idle, .{ .width = 25, .tick = 100, .entry_tick = 0 });
    try std.testing.expectEqualSlices(u8, &a, &b);
    try std.testing.expectEqualSlices(u8, "------------=------------", &a);
}

test "listening: quantized expansion stays in the input zone" {
    const z = Zones.of(25);
    inline for ([_]u3{ 0, 1, 2, 3, 4 }) |q| {
        const s = snap(.listening, .{ .width = 25, .tick = 5, .entry_tick = 0, .input_q = q });
        for (s[z.input..]) |ch| try std.testing.expectEqual(@as(u8, '-'), ch);
    }
    const s4 = snap(.listening, .{ .width = 25, .tick = 5, .entry_tick = 0, .input_q = 4 });
    try std.testing.expectEqualSlices(u8, "=#####=", s4[0..7]); // symmetric peak around origin 3
}

test "thinking: head advances rightward, never left; deterministic fields" {
    const z = Zones.of(25);
    var last_head: usize = 0;
    var t: u64 = 0;
    while (t < 2 * z.processing) : (t += 2) {
        const s = snap(.thinking, .{ .width = 25, .tick = t, .entry_tick = 0, .seed = 9 });
        const head = std.mem.indexOfScalar(u8, &s, '>') orelse continue;
        try std.testing.expect(head >= z.input and head < z.input + z.processing);
        if (t > 0) try std.testing.expect(head >= last_head);
        last_head = head;
    }
    // same seed+tick = identical frame
    const a = snap(.thinking, .{ .width = 25, .tick = 8, .entry_tick = 0, .seed = 9 });
    const b = snap(.thinking, .{ .width = 25, .tick = 8, .entry_tick = 0, .seed = 9 });
    try std.testing.expectEqualSlices(u8, &a, &b);
}

test "speaking: packets only in the output zone, moving right" {
    const z = Zones.of(25);
    var t: u64 = 0;
    while (t < 30) : (t += 1) {
        const s = snap(.speaking, .{ .width = 25, .tick = t, .entry_tick = 0, .output_q = 3 });
        for (s[0 .. z.input + z.processing]) |ch| try std.testing.expectEqual(@as(u8, '-'), ch);
    }
}

test "waiting: frozen with a single boundary; interrupted: hard cut" {
    const s = snap(.waiting, .{ .width = 25, .tick = 2, .entry_tick = 0, .seed = 3 });
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, &s, "|"));
    const i = snap(.interrupted, .{ .width = 25, .tick = 10, .entry_tick = 0 });
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, &i, "!"));
    // t=0: nothing retracted yet — full 6-cell spill trail past the cut
    const cut0 = snap(.interrupted, .{ .width = 25, .tick = 0, .entry_tick = 0 });
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, &cut0, "!"));
    try std.testing.expectEqual(@as(usize, 6), std.mem.count(u8, &cut0, "."));
    // monochrome-distinct from error (fractures = spaces)
    const e = snap(.err, .{ .width = 25, .tick = 10, .entry_tick = 0, .seed = 5 });
    try std.testing.expect(std.mem.count(u8, &e, " ") >= 2);
}

test "acting: determinate maps progress; indeterminate never bounces" {
    const z = Zones.of(25);
    const half = snap(.acting, .{ .width = 25, .tick = 3, .entry_tick = 0, .progress = 0.5 });
    const committed = std.mem.count(u8, &half, "=");
    try std.testing.expect(committed >= 11 and committed <= 13);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, &half, ">"));
    var last: usize = 0;
    var t: u64 = 0;
    var wrapped = false;
    while (t < 40) : (t += 1) {
        const s2 = snap(.acting, .{ .width = 25, .tick = t, .entry_tick = 0 });
        const head = std.mem.indexOfScalar(u8, &s2, '>') orelse {
            wrapped = true;
            continue;
        };
        try std.testing.expect(head >= z.input);
        if (!wrapped and t > 0 and last != 0) try std.testing.expect(head >= last);
        if (wrapped) wrapped = false;
        last = head;
    }
}

test "needs-input: two markers, input side dominant in phase B" {
    const a = snap(.needs_input, .{ .width = 25, .tick = 1, .entry_tick = 0 });
    const b = snap(.needs_input, .{ .width = 25, .tick = 5, .entry_tick = 0 });
    try std.testing.expect(!std.mem.eql(u8, &a, &b)); // phases differ
    // exactly two non-track cells in every phase
    for ([_][25]u8{ a, b }) |s2| {
        var marks: usize = 0;
        for (s2) |ch| {
            if (ch != '-') marks += 1;
        }
        try std.testing.expectEqual(@as(usize, 2), marks);
    }
}

test "quantizer bands and stepped meter response" {
    try std.testing.expectEqual(@as(u3, 0), quantize(0.05));
    try std.testing.expectEqual(@as(u3, 2), quantize(0.4));
    try std.testing.expectEqual(@as(u3, 4), quantize(0.9));
    try std.testing.expectEqual(@as(u3, 2), stepLevel(0, 4)); // rise capped at +2
    try std.testing.expectEqual(@as(u3, 3), stepLevel(4, 0)); // fall capped at -1
}
