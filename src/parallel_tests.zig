//! Behavioral tests for `parallel.zig`: the cached worker-count logic
//! (`cpuThreadCount` clamping and the `setMaxThreads` override precedence).
//! The FUCINA_MAX_THREADS env arm itself is process-global state fixed before
//! main() and cannot be exercised here without racing the rest of the suite;
//! its parsing/scanning logic is unit-tested inline in `parallel.zig` (file-
//! private symbols) and the end-to-end behavior is verified on a static Linux
//! ReleaseFast binary (the configuration where it used to be a silent no-op).
const std = @import("std");
const parallel = @import("parallel.zig");

test "cpuThreadCount caps at the max_threads argument and is at least 1" {
    const unbounded = parallel.cpuThreadCount(std.math.maxInt(usize));
    try std.testing.expect(unbounded >= 1);
    try std.testing.expectEqual(@as(usize, 1), parallel.cpuThreadCount(1));
    const capped = parallel.cpuThreadCount(parallel.vector_max_threads);
    try std.testing.expect(capped >= 1);
    try std.testing.expect(capped <= parallel.vector_max_threads);
    try std.testing.expect(capped <= unbounded);
}

test "setMaxThreads override wins over detection and later lookups" {
    // Snapshot the process-global cache (this also forces first-call
    // detection, so the env var cannot be consulted mid-test) and restore it
    // on exit so the rest of the suite keeps its worker count.
    const prev = parallel.cpuThreadCount(std.math.maxInt(usize));
    defer parallel.setMaxThreads(prev);

    parallel.setMaxThreads(3);
    try std.testing.expectEqual(@as(usize, 3), parallel.cpuThreadCount(std.math.maxInt(usize)));
    // The max_threads argument still caps the override.
    try std.testing.expectEqual(@as(usize, 2), parallel.cpuThreadCount(2));
    try std.testing.expectEqual(@as(usize, 3), parallel.cpuThreadCount(8));

    // n == 0 is documented as ignored.
    parallel.setMaxThreads(0);
    try std.testing.expectEqual(@as(usize, 3), parallel.cpuThreadCount(std.math.maxInt(usize)));
}
