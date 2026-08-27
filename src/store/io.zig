//! Platform I/O shims for the streamed expert tier: positional reads and
//! writes, readahead advice, uncached-read and QoS toggles, and whole-file
//! helpers -- all without an `std.Io` handle (the MoE ops run with
//! `io == null` in production decode). Linux goes straight to the syscall
//! layer (no libc requirement); everything else uses libc, which is always
//! linked on the Apple targets. Also home to the store error set: most of
//! its members name I/O failures, and every sibling can import this leaf.
const std = @import("std");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;

pub const Error = error{
    LayerAlreadyRegistered,
    LayerNotRegistered,
    InvalidExpertGeometry,
    StoreNotFinalized,
    ExpertFileOpenFailed,
    ExpertFileReadFailed,
    UsageFileWriteFailed,
    UnexpectedEndOfFile,
    L2NoStripeableLayers,
} || Allocator.Error;

pub const fd_t = std.posix.fd_t;

pub fn openReadOnly(allocator: Allocator, path: []const u8) Error!fd_t {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.openat(linux.AT.FDCWD, path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(rc) != .SUCCESS) return Error.ExpertFileOpenFailed;
            return @intCast(rc);
        },
        else => {
            const rc = std.c.open(path_z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
            if (rc < 0) return Error.ExpertFileOpenFailed;
            return rc;
        },
    }
}

pub fn closeFd(fd: fd_t) void {
    switch (builtin.os.tag) {
        .linux => _ = std.os.linux.close(fd),
        else => _ = std.c.close(fd),
    }
}

/// Monotonic nanoseconds for the read-time stat (the exec ops' clock rides
/// on `std.Io`, which the MoE decode path legitimately runs without).
pub fn monotonicNanos() ?u64 {
    switch (builtin.os.tag) {
        .linux => {
            var ts: std.os.linux.timespec = undefined;
            if (std.os.linux.clock_gettime(.MONOTONIC, &ts) != 0) return null;
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
        else => {
            var ts: std.c.timespec = undefined;
            if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return null;
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        },
    }
}

/// Short sleep for the pilot thread's empty-ring wait (no std.Thread.sleep
/// in this std; the syscall detour is fine at a 200 µs cadence).
pub fn sleepMicros(us: u64) void {
    switch (builtin.os.tag) {
        .linux => {
            var req = std.os.linux.timespec{ .sec = @intCast(us / 1_000_000), .nsec = @intCast((us % 1_000_000) * 1000) };
            _ = std.os.linux.nanosleep(&req, null);
        },
        else => {
            const c = struct {
                extern "c" fn nanosleep(req: *const std.c.timespec, rem: ?*std.c.timespec) c_int;
            };
            var req = std.c.timespec{ .sec = @intCast(us / 1_000_000), .nsec = @intCast((us % 1_000_000) * 1000) };
            _ = c.nanosleep(&req, null);
        },
    }
}

pub fn preadOnce(fd: fd_t, buf: []u8, offset: u64) Error!usize {
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.pread(fd, buf.ptr, buf.len, @intCast(offset));
            if (linux.errno(rc) != .SUCCESS) return Error.ExpertFileReadFailed;
            return rc;
        },
        else => {
            const rc = std.c.pread(fd, buf.ptr, buf.len, @intCast(offset));
            if (rc < 0) return Error.ExpertFileReadFailed;
            return @intCast(rc);
        },
    }
}

/// Full positional write (the L2-tier builder's `pwrite` mirror of
/// `preadOnce`): loops until every byte lands or the write errors.
pub fn pwriteFullFd(fd: fd_t, buf: []const u8, offset: u64) Error!void {
    var done: usize = 0;
    while (done < buf.len) {
        switch (builtin.os.tag) {
            .linux => {
                const linux = std.os.linux;
                const rc = linux.pwrite(fd, buf.ptr + done, buf.len - done, @intCast(offset + done));
                if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
                done += rc;
            },
            else => {
                const rc = std.c.pwrite(fd, buf.ptr + done, buf.len - done, @intCast(offset + done));
                if (rc < 0) return Error.UsageFileWriteFailed;
                done += @intCast(rc);
            },
        }
    }
}

/// Ask the OS not to page-cache reads on `fd` (macOS `F_NOCACHE`; fcntl.h
/// value 48 — the `F_RDADVISE` shim above uses the same raw-constant
/// pattern). Best effort: a failure just leaves normal caching on.
pub fn setUncached(fd: fd_t) void {
    if (builtin.os.tag == .macos) {
        _ = std.c.fcntl(fd, 48, @as(c_int, 1));
    }
}

pub fn openWriteTrunc(allocator: Allocator, path: []const u8) Error!fd_t {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.openat(linux.AT.FDCWD, path_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, 0o644);
            if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
            return @intCast(rc);
        },
        else => {
            const rc = std.c.open(path_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o644));
            if (rc < 0) return Error.UsageFileWriteFailed;
            return rc;
        },
    }
}

pub fn writeFull(fd: fd_t, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        switch (builtin.os.tag) {
            .linux => {
                const linux = std.os.linux;
                const rc = linux.write(fd, bytes.ptr + done, bytes.len - done);
                if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
                done += rc;
            },
            else => {
                const rc = std.c.write(fd, bytes.ptr + done, bytes.len - done);
                if (rc < 0) return Error.UsageFileWriteFailed;
                done += @intCast(rc);
            },
        }
    }
}

pub fn renamePath(allocator: Allocator, old: []const u8, new: []const u8) Error!void {
    const old_z = try allocator.dupeZ(u8, old);
    defer allocator.free(old_z);
    const new_z = try allocator.dupeZ(u8, new);
    defer allocator.free(new_z);
    switch (builtin.os.tag) {
        .linux => {
            const linux = std.os.linux;
            const rc = linux.renameat(linux.AT.FDCWD, old_z, linux.AT.FDCWD, new_z);
            if (linux.errno(rc) != .SUCCESS) return Error.UsageFileWriteFailed;
        },
        else => {
            if (std.c.rename(old_z, new_z) != 0) return Error.UsageFileWriteFailed;
        },
    }
}

/// Whole small file into an allocated buffer, or null on any failure. Capped
/// (histograms are tens of KB) so a bogus path can't balloon memory.
pub fn readWholeFile(allocator: Allocator, path: []const u8) ?[]u8 {
    const max_bytes = 16 << 20;
    const fd = openReadOnly(allocator, path) catch return null;
    defer closeFd(fd);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(allocator);
    var chunk: [65536]u8 = undefined;
    while (bytes.items.len <= max_bytes) {
        const n = preadOnce(fd, &chunk, bytes.items.len) catch return null;
        if (n == 0) return bytes.toOwnedSlice(allocator) catch null;
        bytes.appendSlice(allocator, chunk[0..n]) catch return null;
    }
    return null;
}

/// Best-effort async readahead hint for a file range: tells the kernel to
/// start reading in the background so the synchronous `pread` that follows
/// (or follows several other hinted reads) finds the page cache warm. A
/// failed hint is not an error — it only costs the overlap.
pub fn hintWillNeed(fd: fd_t, offset: u64, len: usize) void {
    switch (builtin.os.tag) {
        .linux => {
            _ = std.os.linux.fadvise(fd, @intCast(offset), @intCast(len), std.os.linux.POSIX_FADV.WILLNEED);
        },
        .macos, .ios => {
            // struct radvisory + F_RDADVISE (fcntl.h): macOS's readahead advice.
            const radvisory = extern struct { ra_offset: i64, ra_count: c_int };
            var adv = radvisory{
                .ra_offset = @intCast(offset),
                .ra_count = @intCast(@min(len, std.math.maxInt(c_int))),
            };
            _ = std.c.fcntl(fd, 44, @intFromPtr(&adv));
        },
        else => {},
    }
}

pub fn preadFullFd(fd: fd_t, buf: []u8, offset: u64) Error!void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = try preadOnce(fd, buf[done..], offset + done);
        if (n == 0) return Error.UnexpectedEndOfFile;
        done += n;
    }
}

/// Demote the calling thread below the compute team (macOS QoS
/// UTILITY): the store's threads live blocked in pread, and when they
/// wake they must not preempt the worker team's fork-joins on the
/// P-cores — miss activity scheduled onto the compute cores inflates
/// every phase that shares them. Self-set once per thread (QoS classes
/// cannot be assigned externally); no-op off macOS. The FORWARD thread
/// also drains batches — it never routes through here, so its QoS is
/// untouched.
threadlocal var io_qos_set: bool = false;
pub fn setIoThreadQos() void {
    if (builtin.os.tag != .macos) return;
    if (io_qos_set) return;
    io_qos_set = true;
    const c = struct {
        extern "c" fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) c_int;
    };
    _ = c.pthread_set_qos_class_self_np(0x11, 0); // QOS_CLASS_UTILITY
}
