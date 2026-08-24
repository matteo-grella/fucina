//! Behavioral tests for the grammar-aware draft source (`constrained.zig`):
//! forced spans preempt the inner source, free-choice drafts are truncated at
//! the first grammar-invalid token with the inner pending accounting
//! following, and observe/topk/truncate delegation stays correct on both
//! draft origins.

const std = @import("std");
const core = @import("core.zig");
const constrained = @import("constrained.zig");
const logit_processor = @import("../logit_processor.zig");

const DraftSource = core.DraftSource;
const TopKRow = core.TopKRow;
const LogitProcessor = logit_processor.LogitProcessor;
const ConstrainedSource = constrained.ConstrainedSource;

/// Structural-hook test double: `forced` is the span forcedTokens returns
/// (empty = free choice); validPrefixLen accepts leading tokens < `limit`.
const FakeGrammar = struct {
    forced: []const usize = &.{},
    limit: usize = std.math.maxInt(usize),

    fn processor(self: *FakeGrammar) LogitProcessor {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = LogitProcessor.VTable{
        .process = process,
        .commit = commit,
        .forcedTokens = forcedTokens,
        .validPrefixLen = validPrefixLen,
    };

    fn process(ptr: *anyopaque, logits: []f32, history: []const usize) anyerror!void {
        _ = ptr;
        _ = logits;
        _ = history;
    }
    fn commit(ptr: *anyopaque, token: usize) anyerror!void {
        _ = ptr;
        _ = token;
    }
    fn forcedTokens(ptr: *anyopaque, buf: []usize) usize {
        const self: *FakeGrammar = @ptrCast(@alignCast(ptr));
        const n = @min(self.forced.len, buf.len);
        @memcpy(buf[0..n], self.forced[0..n]);
        return n;
    }
    fn validPrefixLen(ptr: *anyopaque, tokens: []const usize) usize {
        const self: *FakeGrammar = @ptrCast(@alignCast(ptr));
        for (tokens, 0..) |t, i| {
            if (t >= self.limit) return i;
        }
        return tokens.len;
    }
};

/// Inner-source test double: always drafts `draft`, records observe /
/// truncatePending traffic.
const FakeInner = struct {
    draft: []const usize = &.{},
    suggests: usize = 0,
    observed: usize = 0,
    truncated_to: ?usize = null,
    wants_topk: bool = false,
    topk_calls: usize = 0,

    fn source(self: *FakeInner) DraftSource {
        return .{ .ptr = self, .vtable = if (self.wants_topk) &vtable_topk else &vtable };
    }

    const vtable = DraftSource.VTable{
        .suggest = suggest,
        .observe = observe,
        .truncatePending = truncatePending,
    };
    const vtable_topk = DraftSource.VTable{
        .suggest = suggest,
        .observe = observe,
        .observeTopK = observeTopK,
        .truncatePending = truncatePending,
    };

    fn suggest(ptr: *anyopaque, context: []const usize, buf: []usize) usize {
        _ = context;
        const self: *FakeInner = @ptrCast(@alignCast(ptr));
        self.suggests += 1;
        const n = @min(self.draft.len, buf.len);
        @memcpy(buf[0..n], self.draft[0..n]);
        return n;
    }
    fn observe(ptr: *anyopaque, committed: []const usize) void {
        const self: *FakeInner = @ptrCast(@alignCast(ptr));
        self.observed += committed.len;
    }
    fn observeTopK(ptr: *anyopaque, positions: []const TopKRow) void {
        const self: *FakeInner = @ptrCast(@alignCast(ptr));
        self.topk_calls += positions.len;
    }
    fn truncatePending(ptr: *anyopaque, new_len: usize) void {
        const self: *FakeInner = @ptrCast(@alignCast(ptr));
        self.truncated_to = new_len;
    }
};

test "forced span preempts the inner source and carries no inner accounting" {
    var grammar = FakeGrammar{ .forced = &.{ 7, 8, 9 } };
    var inner = FakeInner{ .draft = &.{ 1, 2, 3, 4 } };
    var cs = ConstrainedSource.init(grammar.processor(), inner.source());
    const src = cs.source();

    var buf: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 3), src.suggest(&.{5}, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 7, 8, 9 }, buf[0..3]);
    try std.testing.expectEqual(@as(usize, 0), inner.suggests); // never consulted

    // A wrapper truncating the FORCED draft must not touch inner pending.
    src.truncatePending(1);
    try std.testing.expectEqual(@as(?usize, null), inner.truncated_to);
}

test "free choice: inner draft truncated at the first invalid token, pending follows" {
    var grammar = FakeGrammar{ .limit = 3 }; // tokens >= 3 are grammar-invalid
    var inner = FakeInner{ .draft = &.{ 1, 2, 3, 4 } };
    var cs = ConstrainedSource.init(grammar.processor(), inner.source());
    const src = cs.source();

    var buf: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 2), src.suggest(&.{5}, &buf));
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, buf[0..2]);
    try std.testing.expectEqual(@as(?usize, 2), inner.truncated_to); // accounting shrank with the draft

    // A wrapper truncating further forwards to the inner source.
    src.truncatePending(1);
    try std.testing.expectEqual(@as(?usize, 1), inner.truncated_to);
}

test "fully valid inner drafts pass through untruncated" {
    var grammar = FakeGrammar{ .limit = 100 };
    var inner = FakeInner{ .draft = &.{ 1, 2, 3 } };
    var cs = ConstrainedSource.init(grammar.processor(), inner.source());
    const src = cs.source();

    var buf: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 3), src.suggest(&.{5}, &buf));
    try std.testing.expectEqual(@as(?usize, null), inner.truncated_to);
}

test "observe and top-k delegate to the inner source; top-k appetite mirrors it" {
    var grammar = FakeGrammar{};
    var plain_inner = FakeInner{};
    var cs_plain = ConstrainedSource.init(grammar.processor(), plain_inner.source());
    try std.testing.expect(!cs_plain.source().wantsTopK());

    var topk_inner = FakeInner{ .wants_topk = true };
    var cs = ConstrainedSource.init(grammar.processor(), topk_inner.source());
    const src = cs.source();
    try std.testing.expect(src.wantsTopK());

    src.observe(&.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(usize, 3), topk_inner.observed);
    src.observeTopK(&.{.{ .token = 1, .topk = &.{} }});
    try std.testing.expectEqual(@as(usize, 1), topk_inner.topk_calls);
}
