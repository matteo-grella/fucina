//! The sanctioned environment readers: `flag`, `flagValue`, `positiveUsize`,
//! `nonNegativeUsize` and `stringIs`, each with the libc arm
//! (`std.c.getenv`) and the libc-free Linux arm that scans
//! `/proc/self/environ` (the kernel's copy of the initial environment,
//! readable without libc and without allocating). The tuning-table loader
//! and the substrate gates read every `FUCINA_*` variable through here; a
//! bare `std.c.getenv` does not compile into the static Linux binaries. A
//! std-only leaf. Layer stack: docs/ARCHITECTURE.md.
//!
//! The scanner tests stay inline: they exercise file-private symbols.
const std = @import("std");
const builtin = @import("builtin");

/// Non-negative-usize environment knob, or null when unset/invalid. Unlike
/// `positiveUsize`, `0` is a VALID value: the tuning-table leaves that
/// read through this arm (`FUCINA_SPIN_BUDGET`, the GPU work floors and
/// percentages) give zero a meaning of its own (park immediately, always
/// offload, occupancy-blind). Same per-target arms as `positiveUsize`.
pub fn nonNegativeUsize(comptime name: [:0]const u8) ?usize {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return null;
        return parseNonNegativeUsize(std.mem.sliceTo(value, 0));
    } else if (builtin.os.tag == .linux) {
        return readProcSelfEnviron(name, parseNonNegativeUsize);
    } else {
        return null;
    }
}

/// Like `parsePositiveUsize` but `0` parses as a valid value; empty,
/// non-numeric, or sign-prefixed-negative input is still null (no override).
/// The explicit '-' check matters because `parseInt(usize, "-0")` succeeds —
/// without it "-0" would be a live park-immediately override while every
/// sibling knob treats it as unset.
fn parseNonNegativeUsize(s: []const u8) ?usize {
    if (s.len == 0 or s[0] == '-') return null;
    return std.fmt.parseInt(usize, s, 10) catch null;
}

/// Positive-usize environment knob, or null when unset/invalid/zero. Zig 0.16
/// has no libc-free std getenv (the environment block is only handed to `main`
/// via `std.process.Init`), so the read is per-target:
///  - libc builds (macOS always links libSystem): `std.c.getenv`;
///  - static Linux builds (the fully static ReleaseFast server binary, where
///    the libc arm compiles out and FUCINA_MAX_THREADS used to be a silent
///    no-op): scan /proc/self/environ — the kernel's copy of the initial
///    environment, readable without libc and without allocating;
///  - any other libc-free target (Windows/wasi/freestanding): no override.
pub fn positiveUsize(comptime name: [:0]const u8) ?usize {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return null;
        return parsePositiveUsize(std.mem.sliceTo(value, 0));
    } else if (builtin.os.tag == .linux) {
        return readProcSelfEnviron(name, parsePositiveUsize);
    } else {
        return null;
    }
}

/// Boolean environment flag with the getenv-family truthiness contract
/// (set with a first character other than '0'), same per-target arms as
/// `positiveUsize` — the ONLY sanctioned way to read a flag from code
/// that must also compile into libc-free Linux binaries (bare
/// `std.c.getenv` is a compile error there).
pub fn flag(comptime name: [:0]const u8) bool {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return false;
        return parseFlag(std.mem.sliceTo(value, 0)) == 1;
    } else if (builtin.os.tag == .linux) {
        return (readProcSelfEnviron(name, parseFlag) orelse 0) == 1;
    } else {
        return false;
    }
}

fn parseFlag(s: []const u8) ?usize {
    return if (s.len > 0 and s[0] != '0') 1 else 0;
}

/// Tri-state boolean environment flag: null when unset or set empty (no
/// override), false when the value's first character is `'0'`, true
/// otherwise. The tuning table's boolean leaves read through this, so
/// setting a gate's variable to `0` forces the route off while an unset
/// variable keeps the measured default. Same per-target arms as `flag`.
pub fn flagValue(comptime name: [:0]const u8) ?bool {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return null;
        const parsed = parseFlagValue(std.mem.sliceTo(value, 0)) orelse return null;
        return parsed == 1;
    } else if (builtin.os.tag == .linux) {
        const parsed = readProcSelfEnviron(name, parseFlagValue) orelse return null;
        return parsed == 1;
    } else {
        return null;
    }
}

fn parseFlagValue(s: []const u8) ?usize {
    if (s.len == 0) return null;
    return if (s[0] != '0') 1 else 0;
}

/// True when the variable is set to exactly `expected` (the contract of the
/// string-valued knobs that select a named mode, e.g.
/// `FUCINA_GPU_KERNELS=src`). Same per-target arms as `flag`.
pub fn stringIs(comptime name: [:0]const u8, comptime expected: []const u8) bool {
    if (builtin.link_libc) {
        const value = std.c.getenv(name) orelse return false;
        return std.mem.eql(u8, std.mem.sliceTo(value, 0), expected);
    } else if (builtin.os.tag == .linux) {
        const match = struct {
            fn parse(s: []const u8) ?usize {
                return if (std.mem.eql(u8, s, expected)) 1 else 0;
            }
        };
        return (readProcSelfEnviron(name, match.parse) orelse 0) == 1;
    } else {
        return false;
    }
}

/// Env-value parse contract (unchanged from the original libc-only arm):
/// base-10 usize; 0 or anything unparsable means "no override".
fn parsePositiveUsize(s: []const u8) ?usize {
    const n = std.fmt.parseInt(usize, s, 10) catch 0;
    return if (n >= 1) n else null;
}

/// Linux-only, libc-free env lookup via /proc/self/environ (NUL-separated
/// `KEY=VALUE` records; a trailing NUL is usual but the final record is
/// handled either way). Fixed stack buffers, no allocation. Any I/O failure
/// degrades to "no override".
fn readProcSelfEnviron(comptime name: [:0]const u8, comptime parse: fn ([]const u8) ?usize) ?usize {
    const posix = std.posix;
    const fd = posix.openatZ(
        posix.AT.FDCWD,
        "/proc/self/environ",
        .{ .ACCMODE = .RDONLY, .CLOEXEC = true },
        0,
    ) catch return null;
    // std.posix has no close in 0.16; the raw syscall is fine on this
    // Linux-only path.
    defer _ = std.os.linux.close(fd);

    var scan: EnvironScan(name, parse) = .init;
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = posix.read(fd, &buf) catch return null;
        if (n == 0) break;
        if (scan.feed(buf[0..n])) |decision| return decision;
    }
    return scan.finish();
}

/// Incremental scanner for a NUL-separated `KEY=VALUE` environment block (the
/// /proc/self/environ format), matching getenv semantics: the FIRST record
/// whose key is `name` decides, and an invalid/zero value decides "no
/// override" (later duplicates are not consulted). Platform-independent so the
/// parsing logic is unit-tested on every target, not just Linux.
fn EnvironScan(comptime name: [:0]const u8, comptime parse: fn ([]const u8) ?usize) type {
    return struct {
        const Self = @This();
        const key = name ++ "=";

        // A valid usize value is at most 20 decimal digits, so 64 bytes covers
        // every record that could ever produce an override. Longer records are
        // tracked as overflowed: a matching-but-overlong record still
        // concludes the scan with "no override" (its value cannot parse; first
        // match wins, like getenv).
        entry: [64]u8,
        entry_len: usize,
        overflowed: bool,

        const init: Self = .{ .entry = undefined, .entry_len = 0, .overflowed = false };

        /// Feed the next chunk. Outer null = keep scanning; otherwise the scan
        /// has concluded and the inner `?usize` is the override decision
        /// (null = no override). Stop feeding once concluded.
        fn feed(self: *Self, chunk: []const u8) ??usize {
            for (chunk) |byte| {
                if (byte != 0) {
                    if (self.entry_len < self.entry.len) {
                        self.entry[self.entry_len] = byte;
                        self.entry_len += 1;
                    } else {
                        self.overflowed = true;
                    }
                    continue;
                }
                if (self.concludeEntry()) |decision| return decision;
            }
            return null;
        }

        /// Handle a final record with no trailing NUL. Returns the override,
        /// if any.
        fn finish(self: *Self) ?usize {
            if (self.entry_len == 0 and !self.overflowed) return null;
            return self.concludeEntry() orelse null;
        }

        fn concludeEntry(self: *Self) ??usize {
            defer {
                self.entry_len = 0;
                self.overflowed = false;
            }
            const e = self.entry[0..self.entry_len];
            if (!std.mem.startsWith(u8, e, key)) return null;
            if (self.overflowed) return @as(?usize, null);
            return parse(e[key.len..]);
        }
    };
}

test "parsePositiveUsize: usize base-10, 0/invalid mean no override" {
    try std.testing.expectEqual(@as(?usize, 6), parsePositiveUsize("6"));
    try std.testing.expectEqual(@as(?usize, 8), parsePositiveUsize("0008"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("0"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize(""));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("abc"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("6abc"));
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("-4"));
    // 21 digits overflows usize -> parse error -> no override.
    try std.testing.expectEqual(@as(?usize, null), parsePositiveUsize("111111111111111111111"));
}

test "parseNonNegativeUsize: like parsePositiveUsize but 0 is a valid value" {
    try std.testing.expectEqual(@as(?usize, 0), parseNonNegativeUsize("0"));
    try std.testing.expectEqual(@as(?usize, 512), parseNonNegativeUsize("512"));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize(""));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("abc"));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("-4"));
    // parseInt(usize, "-0") would succeed; the sign check keeps "-0" unset
    // like every sibling knob instead of a live park-immediately override.
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("-0"));
    try std.testing.expectEqual(@as(?usize, null), parseNonNegativeUsize("111111111111111111111"));
}

// The scanner is comptime-keyed (FUCINA_MAX_THREADS / FUCINA_SPIN_BUDGET share
// it); the tests instantiate it with the original key.
const MaxThreadsScan = EnvironScan("FUCINA_MAX_THREADS", parsePositiveUsize);

test "EnvironScan: finds the key among other records" {
    var scan: MaxThreadsScan = .init;
    const block = "PATH=/usr/bin\x00FUCINA_MAX_THREADS=6\x00HOME=/root\x00";
    try std.testing.expectEqual(@as(??usize, @as(?usize, 6)), scan.feed(block));
}

test "EnvironScan: absent key scans to the end with no cap" {
    var scan: MaxThreadsScan = .init;
    try std.testing.expectEqual(@as(??usize, null), scan.feed("PATH=/usr/bin\x00HOME=/root\x00"));
    try std.testing.expectEqual(@as(?usize, null), scan.finish());
}

test "EnvironScan: record split across arbitrary chunk boundaries" {
    const block = "AA=1\x00FUCINA_MAX_THREADS=12\x00BB=2\x00";
    // Byte-at-a-time is the worst-case chunking.
    var scan: MaxThreadsScan = .init;
    var decided: ??usize = null;
    for (block) |byte| {
        if (scan.feed(&[_]u8{byte})) |decision| {
            decided = decision;
            break;
        }
    }
    try std.testing.expectEqual(@as(??usize, @as(?usize, 12)), decided);
}

test "EnvironScan: invalid or zero value concludes with no cap" {
    var scan: MaxThreadsScan = .init;
    try std.testing.expectEqual(
        @as(??usize, @as(?usize, null)),
        scan.feed("FUCINA_MAX_THREADS=abc\x00"),
    );
    scan = .init;
    try std.testing.expectEqual(
        @as(??usize, @as(?usize, null)),
        scan.feed("FUCINA_MAX_THREADS=0\x00FUCINA_MAX_THREADS=5\x00"),
    );
}

test "EnvironScan: final record without trailing NUL" {
    var scan: MaxThreadsScan = .init;
    try std.testing.expectEqual(@as(??usize, null), scan.feed("A=1\x00FUCINA_MAX_THREADS=4"));
    try std.testing.expectEqual(@as(?usize, 4), scan.finish());
}

test "EnvironScan: overlong records" {
    // Overlong record with a matching key concludes with no cap (first match
    // wins; its value cannot be a valid usize).
    var scan: MaxThreadsScan = .init;
    const overlong_match = "FUCINA_MAX_THREADS=" ++ "1" ** 100 ++ "\x00FUCINA_MAX_THREADS=5\x00";
    try std.testing.expectEqual(@as(??usize, @as(?usize, null)), scan.feed(overlong_match));
    // Overlong record with a non-matching key is skipped; the scan continues.
    scan = .init;
    const overlong_other = "JUNK=" ++ "x" ** 200 ++ "\x00FUCINA_MAX_THREADS=6\x00";
    try std.testing.expectEqual(@as(??usize, @as(?usize, 6)), scan.feed(overlong_other));
}

test "EnvironScan: near-miss keys never match" {
    var scan: MaxThreadsScan = .init;
    const block = "FUCINA_MAX_THREADS_X=9\x00XFUCINA_MAX_THREADS=9\x00FUCINA_MAX_THREAD=9\x00\x00";
    try std.testing.expectEqual(@as(??usize, null), scan.feed(block));
    try std.testing.expectEqual(@as(?usize, null), scan.finish());
}

test "readProcSelfEnviron: live smoke on Linux" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    // The test process env is not under our control; assert the read is safe
    // and any produced override is valid.
    if (readProcSelfEnviron("FUCINA_MAX_THREADS", parsePositiveUsize)) |cap| try std.testing.expect(cap >= 1);
}
