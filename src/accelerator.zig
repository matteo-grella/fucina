//! Backend-neutral lifetime primitives for eagerly submitted accelerator work.
//!
//! A `Work` is not a graph node: the operation has already been encoded and
//! submitted when the value is created.  It is only a completion/lifetime
//! token attached to the output storage.  CPU access waits for host visibility;
//! a later accelerator op may retain the token and consume its device result
//! directly.  `Resource` is the analogous storage-lifetime cache entry (for
//! example a Metal page wrapper) and contains no computation.

const std = @import("std");

pub const Provider = enum(u8) {
    metal,
    cuda,
};

pub const WorkVTable = struct {
    /// Complete the submitted work.  `copy_to_host` is false when storage is
    /// being discarded without a CPU reader.  Returns false on a deferred
    /// device error.
    finish: *const fn (ctx: *anyopaque, copy_to_host: bool) bool,
    /// Base device address of the output, when another op can consume it
    /// without a host round trip.  The address stays valid while a Work ref is
    /// held, including after `finish(true)`.
    device_ptr: ?*const fn (ctx: *anyopaque) ?usize = null,
    destroy: *const fn (ctx: *anyopaque) void,
};

const WorkState = enum(u8) {
    submitted,
    completing,
    host_ready,
    discarded,
    failed,
};

pub const Work = struct {
    provider: Provider,
    ctx: *anyopaque,
    vtable: *const WorkVTable,
    refs: std.atomic.Value(u32) = .init(1),
    state: std.atomic.Value(WorkState) = .init(.submitted),

    pub fn init(provider: Provider, ctx: *anyopaque, vtable: *const WorkVTable) Work {
        return .{ .provider = provider, .ctx = ctx, .vtable = vtable };
    }

    pub fn retain(self: *Work) void {
        const old = self.refs.fetchAdd(1, .monotonic);
        std.debug.assert(old > 0);
    }

    pub fn release(self: *Work) void {
        const old = self.refs.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        if (old != 1) return;
        // A producer may become unreachable without ever being read.  Its
        // device resources still cannot be recycled until the submitted work
        // has stopped touching them.
        _ = self.complete(false);
        self.vtable.destroy(self.ctx);
    }

    /// Make the host output visible.  Device execution failures are fatal at
    /// this boundary: the original eager operation has already returned and
    /// cannot safely replay through the CPU without retaining its whole call
    /// frame (which would be a graph in disguise).
    pub fn ensureHost(self: *Work) void {
        if (!self.complete(true)) @panic("asynchronous accelerator operation failed");
    }

    pub fn discard(self: *Work) void {
        _ = self.complete(false);
    }

    pub fn devicePtr(self: *Work, provider: Provider) ?usize {
        if (self.provider != provider) return null;
        return switch (self.state.load(.acquire)) {
            .submitted, .completing, .host_ready => if (self.vtable.device_ptr) |f| f(self.ctx) else null,
            .discarded, .failed => null,
        };
    }

    fn complete(self: *Work, copy_to_host: bool) bool {
        while (true) {
            switch (self.state.load(.acquire)) {
                .host_ready => return true,
                .discarded => {
                    // A buffer cannot be read after its final release.  Seeing
                    // this on a host-read path would mean the external tensor
                    // lifetime contract was violated.
                    if (copy_to_host) @panic("accelerator result was discarded before host access");
                    return true;
                },
                .failed => return false,
                .completing => std.atomic.spinLoopHint(),
                .submitted => {
                    if (self.state.cmpxchgStrong(.submitted, .completing, .acq_rel, .acquire) != null) continue;
                    const ok = self.vtable.finish(self.ctx, copy_to_host);
                    self.state.store(if (!ok) .failed else if (copy_to_host) .host_ready else .discarded, .release);
                    return ok;
                },
            }
        }
    }
};

pub const ResourceVTable = struct {
    destroy: *const fn (ctx: *anyopaque) void,
};

/// Provider-owned cache object tied to one storage allocation's lifetime.
/// Unlike `Work`, this contains no pending computation.
pub const Resource = struct {
    provider: Provider,
    ctx: *anyopaque,
    vtable: *const ResourceVTable,

    pub fn destroy(self: *Resource) void {
        self.vtable.destroy(self.ctx);
    }
};

test "work completes once and separates host-read from discard" {
    const Probe = struct {
        work: Work,
        calls: usize = 0,
        copied: bool = false,

        const vtable: WorkVTable = .{
            .finish = finish,
            .device_ptr = devicePtr,
            .destroy = destroy,
        };

        fn finish(ctx: *anyopaque, copy_to_host: bool) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
            self.copied = copy_to_host;
            return true;
        }
        fn devicePtr(_: *anyopaque) ?usize {
            return 1234;
        }
        fn destroy(_: *anyopaque) void {}
    };

    var probe: Probe = undefined;
    probe = .{ .work = Work.init(.metal, &probe, &Probe.vtable) };
    probe.work.retain();
    try std.testing.expectEqual(@as(?usize, 1234), probe.work.devicePtr(.metal));
    probe.work.ensureHost();
    probe.work.ensureHost();
    try std.testing.expectEqual(@as(usize, 1), probe.calls);
    try std.testing.expect(probe.copied);
    probe.work.release();
    probe.work.release();
}
