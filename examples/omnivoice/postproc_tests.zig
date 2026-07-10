//! Tests for postproc.zig against goldens from the C++ reference
//! (refs/omnivoice.cpp/src/audio-postproc.h). Goldens were produced by a
//! standalone harness (postproc_golden.cpp) that #includes the reference
//! header, feeds it LCG-generated 24 kHz signals with silent gaps at known
//! ranges, and prints results as u32/u64 hex bit patterns (compiled with
//! clang++ -std=c++17 -O2 -ffp-contract=off).
//!
//! Every float assert is bit-exact except the thresh_lin pow probe: Apple
//! libm pow(10, -2.5) and Zig std.math.pow differ by 2 ulps, so that one
//! test asserts a <= 2 ulp distance. The removeSilence goldens stay
//! bit-exact regardless because no slice RMS of the test signals falls
//! within 2 ulps of the threshold (slice RMS = sqrt(int/int); the nearest
//! values straddle the threshold by orders of magnitude more).

const std = @import("std");
const postproc = @import("postproc.zig");

// ---------------------------------------------------------------------------
// Deterministic signal generator, replicated bit for bit in the C++ harness:
// state = state*1664525 + 1013904223 (u32 wrap), top 24 bits -> [-1, 1) in
// exact f64 arithmetic, scaled by amp in f64, then one rounding to f32.

const Lcg = struct {
    state: u32,

    fn next(self: *Lcg, amp: f64) f32 {
        self.state = self.state *% 1664525 +% 1013904223;
        const u: u32 = self.state >> 8;
        return @floatCast((@as(f64, @floatFromInt(u)) * (1.0 / 8388608.0) - 1.0) * amp);
    }
};

const Seg = struct { n: usize, amp: f64 };

fn makeSignal(allocator: std.mem.Allocator, seed: u32, segs: []const Seg) ![]f32 {
    var total: usize = 0;
    for (segs) |s| total += s.n;

    const out = try allocator.alloc(f32, total);
    var lcg = Lcg{ .state = seed };
    var i: usize = 0;
    for (segs) |s| {
        for (0..s.n) |_| {
            out[i] = lcg.next(s.amp);
            i += 1;
        }
    }
    return out;
}

/// Main 24 kHz signal S (72000 samples = 3 s): 200 ms zeros | 800 ms noise
/// 0.5 | 800 ms quiet noise 0.004 (silent at -50 dBFS: s16 RMS ~75 < ~103.6)
/// | 800 ms noise 0.5 | 400 ms zeros.
const sig_segs = [_]Seg{
    .{ .n = 4800, .amp = 0.0 },
    .{ .n = 19200, .amp = 0.5 },
    .{ .n = 19200, .amp = 0.004 },
    .{ .n = 19200, .amp = 0.5 },
    .{ .n = 9600, .amp = 0.0 },
};

/// Quiet signal Q for the auto-gain branch (whole-signal RMS ~0.0082 < 0.1):
/// 300 ms zeros | 800 ms noise 0.02 | 500 ms zeros.
const quiet_segs = [_]Seg{
    .{ .n = 7200, .amp = 0.0 },
    .{ .n = 19200, .amp = 0.02 },
    .{ .n = 12000, .amp = 0.0 },
};

/// Loud signal L (RMS ~0.29 > 0.1 -> no gain).
const loud_segs = [_]Seg{.{ .n = 4800, .amp = 0.5 }};

fn makeChunks(allocator: std.mem.Allocator) ![3][]f32 {
    var lcg = Lcg{ .state = 0xDEADBEEF };
    const c0 = try allocator.alloc(f32, 1000);
    errdefer allocator.free(c0);
    for (c0) |*s| s.* = lcg.next(0.5);
    const c1 = try allocator.alloc(f32, 700);
    errdefer allocator.free(c1);
    for (c1) |*s| s.* = lcg.next(0.3);
    const c2 = try allocator.alloc(f32, 50);
    errdefer allocator.free(c2);
    for (c2) |*s| s.* = lcg.next(0.8);
    return .{ c0, c1, c2 };
}

// ---------------------------------------------------------------------------
// Golden checking helpers.

const Stats = struct {
    len: usize,
    sum: u64,
    ssq: u64,
    first: []const u32,
    last: []const u32,
};

/// Bit-exact check of length, sequential f64 sum, sequential f64 sum of
/// squares, and the first/last sample bit patterns (accumulation order
/// matches the harness exactly).
fn checkStats(g: Stats, v: []const f32) !void {
    try std.testing.expectEqual(g.len, v.len);

    var sum: f64 = 0.0;
    var ssq: f64 = 0.0;
    for (v) |x| {
        sum += @as(f64, x);
        ssq += @as(f64, x) * @as(f64, x);
    }
    try std.testing.expectEqual(g.sum, @as(u64, @bitCast(sum)));
    try std.testing.expectEqual(g.ssq, @as(u64, @bitCast(ssq)));

    for (g.first, v[0..g.first.len]) |e, x| {
        try std.testing.expectEqual(e, @as(u32, @bitCast(x)));
    }
    for (g.last, v[v.len - g.last.len ..]) |e, x| {
        try std.testing.expectEqual(e, @as(u32, @bitCast(x)));
    }
}

fn expectRanges(expected: []const postproc.Range, got: []const postproc.Range) !void {
    try std.testing.expectEqual(expected.len, got.len);
    for (expected, got) |e, g| {
        try std.testing.expectEqual(e.start, g.start);
        try std.testing.expectEqual(e.end, g.end);
    }
}

// ---------------------------------------------------------------------------
// Goldens (printed by postproc_golden.cpp).

const g_sig = Stats{ .len = 72000, .sum = 0xC0178B12E513D1F0, .ssq = 0x40A939D29B2E0702, .first = &.{ 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000 } };
const g_thresh_lin_bits: u64 = 0x4059E7C6E43390B7;
const g_s16_at_4800 = [_]i16{ 9716, 2166, 2937, -1164, -15813, 3060, 1462, 829, -15072, -12427, -10869, 1541, -12989, -10380, -16162, 3523 };
const g_roundtrip = Stats{ .len = 72000, .sum = 0xC0178CD000000000, .ssq = 0x40A9393B8B399000, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_rms_noise_bits: u64 = 0x40C2789320430339;
const g_rms_quiet_bits: u64 = 0x4052C9119D00351B;
const g_rms_overhang_bits: u64 = 0x0000000000000000;
const g_sil_500 = [_]postproc.Range{.{ .start = 24000, .end = 43200 }};
const g_sil_200 = [_]postproc.Range{ .{ .start = 0, .end = 4800 }, .{ .start = 24000, .end = 43200 }, .{ .start = 62400, .end = 72000 } };
const g_nonsil_500 = [_]postproc.Range{ .{ .start = 0, .end = 24000 }, .{ .start = 43200, .end = 72000 } };
const g_nonsil_200 = [_]postproc.Range{ .{ .start = 4800, .end = 24000 }, .{ .start = 43200, .end = 62400 } };
const g_lead: i32 = 4800;
const g_h1_sil = [_]postproc.Range{ .{ .start = 0, .end = 27 }, .{ .start = 35, .end = 100 } };
const g_h1_nonsil = [_]postproc.Range{.{ .start = 27, .end = 35 }};
const g_h2_sil = [_]postproc.Range{.{ .start = 0, .end = 50 }};
const g_h2_nonsil = [_]postproc.Range{};
const g_h3_sil = [_]postproc.Range{.{ .start = 0, .end = 40 }};
const g_h3_nonsil = [_]postproc.Range{.{ .start = 40, .end = 50 }};
const g_h4_sil = [_]postproc.Range{};
const g_h4_nonsil = [_]postproc.Range{.{ .start = 0, .end = 10 }};
const g_rs500 = Stats{ .len = 62400, .sum = 0xC0178CD000000000, .ssq = 0x40A9393B8B399000, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_rs200 = Stats{ .len = 55200, .sum = 0xC018C73800000000, .ssq = 0x40A93921AADBD800, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_rs_nomid = Stats{ .len = 62400, .sum = 0xC0178CD000000000, .ssq = 0x40A9393B8B399000, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_rs_allzero_len: usize = 100;
const g_c0 = Stats{ .len = 1000, .sum = 0x40354B38A9000000, .ssq = 0x40543BBB2B46E436, .first = &.{ 0xBDAAA108, 0x3EEBD230, 0xBEECED24, 0xBE8EB9BE, 0xBDE46638, 0x3EB0744E, 0xBE900B82, 0x3EF84BF8, 0x3EA9265E, 0xBE4765A4, 0xBDCA5B90, 0xBE65D134, 0xBEF58C58, 0xBE6C0A48, 0x3C14E380, 0xBEF3242C }, .last = &.{ 0x3ECE64EC, 0x3E4F4268, 0x3CE25180, 0xBEA3C80C, 0xBEA3D768, 0xBEABA226, 0xBE4A7C6C, 0xBEB14894, 0x3E0492D4, 0x3E9DAC76, 0x3EE2B2E6, 0xBDCA42C8, 0x3EF56A8C, 0xBEE639F0, 0xBEB7F22C, 0x3E82E5E0 } };
const g_xfade = Stats{ .len = 3350, .sum = 0x402693D1C377C000, .ssq = 0x40447C2B3FB4E1B1, .first = &.{ 0xBDAAA108, 0x3EEBD230, 0xBEECED24, 0xBE8EB9BE, 0xBDE46638, 0x3EB0744E, 0xBE900B82, 0x3EF84BF8, 0x3EA9265E, 0xBE4765A4, 0xBDCA5B90, 0xBE65D134, 0xBEF58C58, 0xBE6C0A48, 0x3C14E380, 0xBEF3242C }, .last = &.{ 0x3E45EAF5, 0xBDBFD077, 0x3E92E0F6, 0x3F08BE36, 0x3C8B6C8C, 0xBEF945A6, 0x3F19038C, 0xBF0FE66C, 0xBF0C7DD9, 0xBF09AA7D, 0xBE54BB85, 0x3D0AEB39, 0x3DCC8C92, 0x3D8490BC, 0x3F17AC07, 0xBD6D6C1A } };
const g_xfade_tiny = Stats{ .len = 1750, .sum = 0x4034277C45DB4000, .ssq = 0x405BE50A0CBA361E, .first = &.{ 0xBDAAA108, 0x3EEBD230, 0xBEECED24, 0xBE8EB9BE, 0xBDE46638, 0x3EB0744E, 0xBE900B82, 0x3EF84BF8, 0x3EA9265E, 0xBE4765A4, 0xBDCA5B90, 0xBE65D134, 0xBEF58C58, 0xBE6C0A48, 0x3C14E380, 0xBEF3242C }, .last = &.{ 0x3E8E9E03, 0xBE064520, 0x3EC7EB16, 0x3F351793, 0x3CB3C89A, 0xBF1C9810, 0x3F3B7125, 0xBF2BFA68, 0xBF23E828, 0xBF1CE00B, 0xBE6CE81A, 0x3D174466, 0x3DD9E3A6, 0x3D8A34DA, 0x3F1AD4F2, 0xBD6D6C1A } };
const g_norm = Stats{ .len = 1000, .sum = 0x403555F58EE99000, .ssq = 0x405450284D73A006, .first = &.{ 0xBDAAF713, 0x3EEC491B, 0xBEED649E, 0xBE8F01B7, 0xBDE4D965, 0x3EB0CD49, 0xBE905425, 0x3EF8C92E, 0x3EA97BAA, 0xBE47CA31, 0xBDCAC19B, 0xBE664518, 0xBEF6082B, 0xBE6C814F, 0x3C152E95, 0xBEF39EC8 }, .last = &.{ 0x3ECECD00, 0x3E4FAAEC, 0x3CE2C3A0, 0xBEA41AA3, 0xBEA42A07, 0xBEABF8B3, 0xBE4AE288, 0xBEB1A1FA, 0x3E04D5AE, 0x3E9DFBF9, 0x3EE32537, 0xBDCAA8C7, 0x3EF5E64E, 0xBEE6AE09, 0xBEB84EEE, 0x3E8327E2 } };
const g_fap_c0 = Stats{ .len = 5800, .sum = 0x4028D2B67BADA000, .ssq = 0x403AC772CF137AD2, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_fap_c1 = Stats{ .len = 1660, .sum = 0xBFFBEB49F6BA0000, .ssq = 0x402672F1ECD471E2, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_fap_10 = Stats{ .len = 10, .sum = 0x3FE1D93360000000, .ssq = 0x3FE2708ED85F63BC, .first = &.{ 0x00000000, 0x3E30D393, 0x3EC4C4F3, 0xBDAE757C, 0xBED5DFF0, 0x3E4E67C6, 0xBE221608, 0x3EC2EC20, 0x3DA33326, 0x00000000 }, .last = &.{ 0x00000000, 0x3E30D393, 0x3EC4C4F3, 0xBDAE757C, 0xBED5DFF0, 0x3E4E67C6, 0xBE221608, 0x3EC2EC20, 0x3DA33326, 0x00000000 } };
const g_fap_3 = Stats{ .len = 7, .sum = 0x3FF7598860000000, .ssq = 0x3FF1161C498A5120, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x3F30D393, 0x3F44C4F3, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x3F30D393, 0x3F44C4F3, 0x00000000, 0x00000000 } };
const g_prep_q_rms_bits: u32 = 0x3C059739;
const g_prep_q_nogain = Stats{ .len = 38400, .sum = 0xC02E2E8AD7F1C000, .ssq = 0x4077FFFFFD850443, .first = &.{ 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000 }, .last = &.{ 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000 } };
const g_prep_qt_rms_bits: u32 = 0x3C059739;
const g_prep_q_trim = Stats{ .len = 26400, .sum = 0xC02E2CB000000000, .ssq = 0x4077FED96DD40000, .first = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 }, .last = &.{ 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000 } };
const g_prep_l_rms_bits: u32 = 0x3E93F393;
const g_prep_l = Stats{ .len = 4800, .sum = 0xC0488F25DF800000, .ssq = 0x40790CFA7FD7C26D, .first = &.{ 0x3D997290, 0xBEA57F66, 0x3E803E60, 0xBEB181F6, 0x3EC40602, 0xBED6BA1E, 0x3EAA17BC, 0xBBE2D780, 0x3EA633E4, 0x3EFD44CC, 0x3EBA513E, 0xBE90DBD2, 0x3D3D8980, 0x3D8CEC70, 0xBE0783E4, 0xBEBC4A4A }, .last = &.{ 0x3D7ECED0, 0xBEBA665C, 0xBEAB5FE6, 0x3DB98EE8, 0xBD7D4E40, 0x3EB849D2, 0x3EDF0B24, 0x3ECD1428, 0x3EED37F8, 0x3CFDF000, 0xBEAC35CC, 0xBE90E016, 0xBDD85FF8, 0x3E5A39F0, 0xBD4C0C70, 0x3ED1A608 } };

// ---------------------------------------------------------------------------

test "LCG main signal matches the C++ harness bit for bit" {
    const allocator = std.testing.allocator;
    const sig = try makeSignal(allocator, 0x12345678, &sig_segs);
    defer allocator.free(sig);
    try checkStats(g_sig, sig);
}

test "thresh_lin: Zig pow within 2 ulps of the libm golden" {
    // The reference computes 32768*pow(10, -50/20) with Apple libm
    // (0x4059E7C6E43390B7); Zig's std.math.pow lands 2 ulps below
    // (0x...B5). Both are ~103.615292775; the delta only matters if a slice
    // RMS falls inside those 2 ulps, which none of the golden signals does
    // (proven by the bit-exact removeSilence goldens below).
    const zig_thresh = 32768.0 * std.math.pow(f64, 10.0, -50.0 / 20.0);
    const zb: u64 = @bitCast(zig_thresh);
    const gb: u64 = g_thresh_lin_bits;
    try std.testing.expect(@max(zb, gb) - @min(zb, gb) <= 2);
}

test "f32ToS16: golden samples + clip and truncation semantics" {
    const allocator = std.testing.allocator;
    const sig = try makeSignal(allocator, 0x12345678, &sig_segs);
    defer allocator.free(sig);

    const s16 = try postproc.f32ToS16(allocator, sig);
    defer allocator.free(s16);
    try std.testing.expectEqualSlices(i16, &g_s16_at_4800, s16[4800..4816]);

    // Clip [-32768, 32767] and truncation toward zero (exact by hand):
    // 0.123*32768 = 4030.464 -> 4030; -0.123 -> -4030 (not -4031).
    const hand = try postproc.f32ToS16(allocator, &.{ 1.0, -1.0, 0.5, -0.5, 0.123, -0.123, 1e-9 });
    defer allocator.free(hand);
    try std.testing.expectEqualSlices(i16, &.{ 32767, -32768, 16384, -16384, 4030, -4030, 0 }, hand);
}

test "s16ToF32: round-trip golden + exact values" {
    const allocator = std.testing.allocator;
    const sig = try makeSignal(allocator, 0x12345678, &sig_segs);
    defer allocator.free(sig);

    const s16 = try postproc.f32ToS16(allocator, sig);
    defer allocator.free(s16);
    const back = try postproc.s16ToF32(allocator, s16);
    defer allocator.free(back);
    try checkStats(g_roundtrip, back);

    const hand = try postproc.s16ToF32(allocator, &.{ -32768, 16384, 1, 0 });
    defer allocator.free(hand);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.5, 0.000030517578125, 0.0 }, hand);
}

test "sliceRmsS16: goldens + clamped and empty slices" {
    const allocator = std.testing.allocator;
    const sig = try makeSignal(allocator, 0x12345678, &sig_segs);
    defer allocator.free(sig);
    const s16 = try postproc.f32ToS16(allocator, sig);
    defer allocator.free(s16);

    try std.testing.expectEqual(g_rms_noise_bits, @as(u64, @bitCast(postproc.sliceRmsS16(s16, 4800, 12000))));
    try std.testing.expectEqual(g_rms_quiet_bits, @as(u64, @bitCast(postproc.sliceRmsS16(s16, 24000, 12000))));
    // Overhanging slice shrinks to the end (all zeros there -> 0.0).
    try std.testing.expectEqual(g_rms_overhang_bits, @as(u64, @bitCast(postproc.sliceRmsS16(s16, 71000, 5000))));
    // Start at/past the end -> empty -> 0.0.
    try std.testing.expectEqual(@as(f64, 0.0), postproc.sliceRmsS16(s16, 72000, 100));
    try std.testing.expectEqual(@as(f64, 0.0), postproc.sliceRmsS16(s16, 80000, 100));
    try std.testing.expectEqual(@as(f64, 0.0), postproc.sliceRmsS16(s16, 100, 0));
}

test "detectSilence / detectNonsilent goldens on the main signal" {
    const allocator = std.testing.allocator;
    const sig = try makeSignal(allocator, 0x12345678, &sig_segs);
    defer allocator.free(sig);
    const s16 = try postproc.f32ToS16(allocator, sig);
    defer allocator.free(s16);

    const thresh: f64 = @bitCast(g_thresh_lin_bits);

    const sil500 = try postproc.detectSilence(allocator, s16, 12000, thresh, 240);
    defer allocator.free(sil500);
    try expectRanges(&g_sil_500, sil500);

    const sil200 = try postproc.detectSilence(allocator, s16, 4800, thresh, 240);
    defer allocator.free(sil200);
    try expectRanges(&g_sil_200, sil200);

    const nonsil500 = try postproc.detectNonsilent(allocator, s16, 12000, thresh, 240);
    defer allocator.free(nonsil500);
    try expectRanges(&g_nonsil_500, nonsil500);

    const nonsil200 = try postproc.detectNonsilent(allocator, s16, 4800, thresh, 240);
    defer allocator.free(nonsil200);
    try expectRanges(&g_nonsil_200, nonsil200);
}

test "detect merge-logic and edge cases (handcrafted goldens)" {
    const allocator = std.testing.allocator;

    // h1: blip at [30,33) splits [0,27] off, but the 77 -> 80 step (last
    // slice appended because 80 % 7 != 0) is neither continuous nor
    // gapped (80 <= 77+20), so [35,100] stays merged.
    var h1 = [_]i16{0} ** 100;
    h1[30] = 20000;
    h1[31] = 20000;
    h1[32] = 20000;
    const h1_sil = try postproc.detectSilence(allocator, &h1, 20, 100.0, 7);
    defer allocator.free(h1_sil);
    try expectRanges(&g_h1_sil, h1_sil);
    const h1_nonsil = try postproc.detectNonsilent(allocator, &h1, 20, 100.0, 7);
    defer allocator.free(h1_nonsil);
    try expectRanges(&g_h1_nonsil, h1_nonsil);

    // h2: fully silent -> silence covers [0, seg_len] -> nonsilent empty.
    const h2 = [_]i16{0} ** 50;
    const h2_sil = try postproc.detectSilence(allocator, &h2, 20, 100.0, 5);
    defer allocator.free(h2_sil);
    try expectRanges(&g_h2_sil, h2_sil);
    const h2_nonsil = try postproc.detectNonsilent(allocator, &h2, 20, 100.0, 5);
    defer allocator.free(h2_nonsil);
    try expectRanges(&g_h2_nonsil, h2_nonsil);

    // h3: silence at the exact front -> degenerate leading (0,0) nonsilent
    // range is stripped.
    var h3 = [_]i16{0} ** 50;
    for (h3[40..50]) |*s| s.* = 12000;
    const h3_sil = try postproc.detectSilence(allocator, &h3, 20, 100.0, 5);
    defer allocator.free(h3_sil);
    try expectRanges(&g_h3_sil, h3_sil);
    const h3_nonsil = try postproc.detectNonsilent(allocator, &h3, 20, 100.0, 5);
    defer allocator.free(h3_nonsil);
    try expectRanges(&g_h3_nonsil, h3_nonsil);

    // h4: shorter than min_silence_len -> no silence, whole segment nonsilent.
    const h4 = [_]i16{0} ** 10;
    const h4_sil = try postproc.detectSilence(allocator, &h4, 20, 100.0, 5);
    defer allocator.free(h4_sil);
    try expectRanges(&g_h4_sil, h4_sil);
    const h4_nonsil = try postproc.detectNonsilent(allocator, &h4, 20, 100.0, 5);
    defer allocator.free(h4_nonsil);
    try expectRanges(&g_h4_nonsil, h4_nonsil);
}

test "detectLeadingSilence: golden + break/clamp semantics" {
    const allocator = std.testing.allocator;
    const sig = try makeSignal(allocator, 0x12345678, &sig_segs);
    defer allocator.free(sig);
    const s16 = try postproc.f32ToS16(allocator, sig);
    defer allocator.free(s16);

    const thresh: f64 = @bitCast(g_thresh_lin_bits);
    try std.testing.expectEqual(g_lead, postproc.detectLeadingSilence(s16, thresh, 240));

    // Breaks at the first loud chunk.
    var h3 = [_]i16{0} ** 50;
    for (h3[40..50]) |*s| s.* = 12000;
    try std.testing.expectEqual(@as(i32, 40), postproc.detectLeadingSilence(&h3, 100.0, 5));

    // All silent: trim steps past the end (49 -> 56) and clamps to len.
    const h2 = [_]i16{0} ** 50;
    try std.testing.expectEqual(@as(i32, 50), postproc.detectLeadingSilence(&h2, 100.0, 7));

    // Empty input -> 0.
    try std.testing.expectEqual(@as(i32, 0), postproc.detectLeadingSilence(&.{}, 100.0, 7));
}

test "removeSilence goldens" {
    const allocator = std.testing.allocator;

    // mid=500ms: the 800 ms gap < 2*keep -> midpoint dedup keeps it whole;
    // only the edges trim (lead keeps 100 ms of 200, trail keeps 100 of 400).
    {
        var a = try makeSignal(allocator, 0x12345678, &sig_segs);
        defer allocator.free(a);
        try postproc.removeSilence(allocator, &a, 24000, 500, 100, 100, -50.0);
        try checkStats(g_rs500, a);
    }

    // mid=200ms: gap shortened 800 -> 400 ms, edges trimmed.
    {
        var a = try makeSignal(allocator, 0x12345678, &sig_segs);
        defer allocator.free(a);
        try postproc.removeSilence(allocator, &a, 24000, 200, 100, 200, -50.0);
        try checkStats(g_rs200, a);
    }

    // mid=0: mid removal skipped entirely, edge trims only.
    {
        var a = try makeSignal(allocator, 0x12345678, &sig_segs);
        defer allocator.free(a);
        try postproc.removeSilence(allocator, &a, 24000, 0, 100, 100, -50.0);
        try checkStats(g_rs_nomid, a);
    }

    // 100 all-zero samples: lead/trail keeps (2400 samples) exceed the
    // buffer, so nothing trims and the length is preserved.
    {
        var a = try allocator.alloc(f32, 100);
        defer allocator.free(a);
        @memset(a, 0.0);
        try postproc.removeSilence(allocator, &a, 24000, 500, 100, 100, -50.0);
        try std.testing.expectEqual(g_rs_allzero_len, a.len);
        for (a) |x| try std.testing.expectEqual(@as(f32, 0.0), x);
    }
}

test "removeSilence: long fully-silent input empties; empty input is a no-op" {
    const allocator = std.testing.allocator;

    // 1 s of zeros with mid=500ms: detect_silence covers [0, seg_len], so
    // detect_nonsilent is empty and the concat drops everything.
    {
        var a = try allocator.alloc(f32, 24000);
        defer allocator.free(a);
        @memset(a, 0.0);
        try postproc.removeSilence(allocator, &a, 24000, 500, 100, 100, -50.0);
        try std.testing.expectEqual(@as(usize, 0), a.len);
    }

    {
        var a = try allocator.alloc(f32, 0);
        defer allocator.free(a);
        try postproc.removeSilence(allocator, &a, 24000, 500, 100, 100, -50.0);
        try std.testing.expectEqual(@as(usize, 0), a.len);
    }
}

test "crossFadeChunks: goldens + degenerate cases" {
    const allocator = std.testing.allocator;
    const chunks = try makeChunks(allocator);
    defer for (chunks) |c| allocator.free(c);
    try checkStats(g_c0, chunks[0]);

    const views = [_][]const f32{ chunks[0], chunks[1], chunks[2] };

    // silence_dur=0.1 -> total_n 2400, fade_n = silence_n = 800; the 50-long
    // third chunk exercises fin_n < fade_n.
    {
        const merged = try postproc.crossFadeChunks(allocator, &views, 24000, 0.1);
        defer allocator.free(merged);
        try checkStats(g_xfade, merged);
    }

    // silence_dur=0.0001 -> total_n = (int)2.4 = 2, fade_n = 0: pure concat.
    {
        const merged = try postproc.crossFadeChunks(allocator, &views, 24000, 0.0001);
        defer allocator.free(merged);
        try checkStats(g_xfade_tiny, merged);
    }

    // No chunks -> empty; one chunk -> verbatim copy.
    {
        const merged = try postproc.crossFadeChunks(allocator, &.{}, 24000, 0.1);
        defer allocator.free(merged);
        try std.testing.expectEqual(@as(usize, 0), merged.len);
    }
    {
        const merged = try postproc.crossFadeChunks(allocator, views[0..1], 24000, 0.1);
        defer allocator.free(merged);
        try std.testing.expectEqualSlices(f32, chunks[0], merged);
        try std.testing.expect(merged.ptr != chunks[0].ptr);
    }
}

test "peakNormalizeHalf: golden + no-op branches" {
    const allocator = std.testing.allocator;
    const chunks = try makeChunks(allocator);
    defer for (chunks) |c| allocator.free(c);

    const a = try allocator.dupe(f32, chunks[0]);
    defer allocator.free(a);
    postproc.peakNormalizeHalf(a);
    try checkStats(g_norm, a);

    // Peak <= 1e-6 leaves the buffer untouched.
    var tiny = [_]f32{ 1e-7, -2e-7, 0.0 };
    postproc.peakNormalizeHalf(&tiny);
    try std.testing.expectEqualSlices(f32, &.{ 1e-7, -2e-7, 0.0 }, &tiny);

    var empty = [_]f32{};
    postproc.peakNormalizeHalf(&empty);
}

test "fadeAndPad: goldens + no-op branches" {
    const allocator = std.testing.allocator;
    const chunks = try makeChunks(allocator);
    defer for (chunks) |c| allocator.free(c);

    // len 1000 with fade_n 1200 -> k = 500; pad_n 2400 each side.
    {
        var a = try allocator.dupe(f32, chunks[0]);
        defer allocator.free(a);
        try postproc.fadeAndPad(allocator, &a, 24000, 0.05, 0.1);
        try checkStats(g_fap_c0, a);
    }

    // fade_n 240, pad_n 480.
    {
        var a = try allocator.dupe(f32, chunks[1]);
        defer allocator.free(a);
        try postproc.fadeAndPad(allocator, &a, 24000, 0.01, 0.02);
        try checkStats(g_fap_c1, a);
    }

    // len 10 -> k = 5, denom 4; no pad.
    {
        var a = try allocator.dupe(f32, chunks[2][0..10]);
        defer allocator.free(a);
        try postproc.fadeAndPad(allocator, &a, 24000, 0.05, 0.0);
        try checkStats(g_fap_10, a);
    }

    // len 3 -> k = 1, denom max(0,1)=1 (fade-in zeroes a[0], fade-out
    // leaves a[2]); pad_n = (int)2.4 = 2.
    {
        var a = try allocator.dupe(f32, chunks[2][0..3]);
        defer allocator.free(a);
        try postproc.fadeAndPad(allocator, &a, 24000, 0.05, 0.0001);
        try checkStats(g_fap_3, a);
    }

    // fade_dur 0 and pad_dur 0: unchanged.
    {
        var a = try allocator.dupe(f32, chunks[2]);
        defer allocator.free(a);
        try postproc.fadeAndPad(allocator, &a, 24000, 0.0, 0.0);
        try std.testing.expectEqualSlices(f32, chunks[2], a);
    }

    // Empty input: early return, no pad applied.
    {
        var a = try allocator.alloc(f32, 0);
        defer allocator.free(a);
        try postproc.fadeAndPad(allocator, &a, 24000, 0.05, 0.1);
        try std.testing.expectEqual(@as(usize, 0), a.len);
    }
}

test "refPreprocessAudio: goldens + empty returns -1" {
    const allocator = std.testing.allocator;

    // Quiet signal, no trim: auto-gain fires unconditionally.
    {
        var a = try makeSignal(allocator, 0xCAFEBABE, &quiet_segs);
        defer allocator.free(a);
        const rms = try postproc.refPreprocessAudio(allocator, &a, 24000, false);
        try std.testing.expectEqual(g_prep_q_rms_bits, @as(u32, @bitCast(rms)));
        try checkStats(g_prep_q_nogain, a);
    }

    // Quiet signal with trim: gain, then remove_silence(200,100,200,-50).
    {
        var a = try makeSignal(allocator, 0xCAFEBABE, &quiet_segs);
        defer allocator.free(a);
        const rms = try postproc.refPreprocessAudio(allocator, &a, 24000, true);
        try std.testing.expectEqual(g_prep_qt_rms_bits, @as(u32, @bitCast(rms)));
        try checkStats(g_prep_q_trim, a);
    }

    // Loud signal (RMS > 0.1): no gain, buffer untouched.
    {
        var a = try makeSignal(allocator, 0x0BADF00D, &loud_segs);
        defer allocator.free(a);
        const rms = try postproc.refPreprocessAudio(allocator, &a, 24000, false);
        try std.testing.expectEqual(g_prep_l_rms_bits, @as(u32, @bitCast(rms)));
        try checkStats(g_prep_l, a);
    }

    // Empty buffer: -1, untouched.
    {
        var a = try allocator.alloc(f32, 0);
        defer allocator.free(a);
        const rms = try postproc.refPreprocessAudio(allocator, &a, 24000, true);
        try std.testing.expectEqual(@as(f32, -1.0), rms);
        try std.testing.expectEqual(@as(usize, 0), a.len);
    }
}
