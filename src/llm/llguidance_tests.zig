//! Behavioral + ABI tests for llguidance constrained decoding
//! (`llguidance.zig`). Every test skips unless the build has
//! `-Dllguidance=true` (the stub `Constraint` cannot be exercised beyond its
//! init error); the enabled leg doubles as the ABI round-trip check for the
//! hand-written extern declarations.

const std = @import("std");
const fucina = @import("fucina");
const llg = @import("llguidance.zig");
const bpe = @import("tokenizer.zig");
const spm = @import("spm_tokenizer.zig");
const sampler_mod = @import("sampler.zig");

const ExecContext = fucina.ExecContext;
const Logits = fucina.Tensor(.{ .seq, .vocab });

/// Run `process` on a fresh all-zero row of `n` logits and assert exactly
/// `allowed` survive (everything else must be -inf).
fn expectAllowed(proc: llg.LogitProcessor, n: usize, allowed: []const usize) !void {
    var buf: [16]f32 = @splat(0);
    try proc.process(buf[0..n], &.{});
    outer: for (buf[0..n], 0..) |l, id| {
        for (allowed) |a| if (a == id) {
            try std.testing.expect(l == 0);
            continue :outer;
        };
        try std.testing.expect(l == -std.math.inf(f32));
    }
}

test "disabled build: Constraint.init fails with LlguidanceNotEnabled" {
    if (llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{ "a", "b" };
    var tok = try bpe.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 1 });
    defer tok.deinit();
    try std.testing.expectError(
        llg.Error.LlguidanceNotEnabled,
        llg.Constraint.init(alloc, &tok, .{ .regex = "a" }, .{}),
    );
}

test "version string links and names the vendored engine" {
    if (!llg.enabled) return error.SkipZigTest;
    try std.testing.expect(std.mem.startsWith(u8, llg.version(), "llguidance@"));
}

test "regex constraint: mask walk, grammar-complete stop forcing, reset" {
    if (!llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;

    const vocab = [_][]const u8{ "a", "b", "c", "<|end|>" };
    var tok = try bpe.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 3 });
    defer tok.deinit();

    var constraint = try llg.Constraint.init(alloc, &tok, .{ .regex = "ab+c" }, .{});
    defer constraint.deinit();
    const proc = constraint.processor();

    try expectAllowed(proc, 4, &.{0}); // ^a
    try proc.commit(0);
    try expectAllowed(proc, 4, &.{1}); // b+ needs at least one b
    try proc.commit(1);
    try expectAllowed(proc, 4, &.{ 1, 2 }); // more b, or the closing c
    try proc.commit(2);
    try std.testing.expect(constraint.isAccepting());
    try expectAllowed(proc, 4, &.{3}); // grammar complete: only the stop token
    try proc.commit(3);
    try std.testing.expect(constraint.isStopped());
    try expectAllowed(proc, 4, &.{3}); // terminal state keeps forcing it
    try proc.commit(3); // post-stop commits are ignored, not errors

    try proc.reset(); // re-arm for a fresh reply
    try std.testing.expect(!constraint.isStopped());
    try expectAllowed(proc, 4, &.{0});
}

test "special tokens never match literal grammar text" {
    if (!llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;

    // ".+" matches any text; the control tokens' literal spellings would
    // match it too, but the 0xFF special marker keeps them out of the trie.
    const vocab = [_][]const u8{ "x", "<|im_end|>", "<|other|>" };
    var tok = try bpe.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 1 });
    defer tok.deinit();

    var constraint = try llg.Constraint.init(alloc, &tok, .{ .regex = "x+" }, .{ .eos_token = 1 });
    defer constraint.deinit();
    const proc = constraint.processor();

    try expectAllowed(proc, 3, &.{0}); // not the specials, ".+"-shaped or not
    try proc.commit(0);
    // "x" is accepting: x continues the match, the eos control token closes it.
    try expectAllowed(proc, 3, &.{ 0, 1 });
}

test "json schema constraint over a byte-level vocab" {
    if (!llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;

    const vocab = [_][]const u8{ "{", "}", "\"", "a", ":", "1", ",", "<|end|>" };
    var tok = try bpe.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 7 });
    defer tok.deinit();

    var constraint = try llg.Constraint.init(alloc, &tok, .{ .json_schema =
        \\{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}
    }, .{});
    defer constraint.deinit();
    const proc = constraint.processor();

    // First token must open the object; drive the only legal byte path and
    // assert key content stays forced.
    var row: [8]f32 = @splat(0);
    try proc.process(&row, &.{});
    try std.testing.expect(row[0] == 0); // '{'
    for ([_]usize{ 1, 3, 4, 5, 6, 7 }) |id| try std.testing.expect(row[id] == -std.math.inf(f32));

    for ([_]usize{ 0, 2, 3, 2, 4, 5, 1 }) |t| try proc.commit(t); // {"a":1}
    try std.testing.expect(constraint.isAccepting());
    try expectAllowed(proc, 8, &.{7});
}

test "invalid grammar fails init loudly" {
    if (!llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const vocab = [_][]const u8{ "a", "<|end|>" };
    var tok = try bpe.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 1 });
    defer tok.deinit();
    try std.testing.expectError(
        llg.Error.InvalidGrammar,
        llg.Constraint.init(alloc, &tok, .{ .regex = "a(" }, .{ .log_level = 0 }),
    );
    try std.testing.expectError(
        llg.Error.InvalidGrammar,
        llg.Constraint.init(alloc, &tok, .{ .json_schema = "not json" }, .{ .log_level = 0 }),
    );
}

test "SPM bridge: attrs drive special marking and byte-fallback tokens" {
    if (!llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;

    const vocab = [_][]const u8{ "<unk>", "<s>", "</s>", "▁a", "b", "c", "<0x78>" };
    const scores = [_]f32{ 0, 0, 0, -1, -1, -1, -10 };
    const attrs = [_]spm.Attr{ .unknown, .control, .control, .normal, .normal, .normal, .byte };
    var tok = try spm.Tokenizer.initFromSlices(alloc, &vocab, &scores, &attrs, .{
        .add_bos = false,
        .add_space_prefix = false,
    });
    defer tok.deinit();

    // "▁a" decodes to " a"; "<0x78>" is the raw byte 'x'.
    var constraint = try llg.Constraint.init(alloc, &tok, .{ .regex = " axb?" }, .{ .eos_token = 2 });
    defer constraint.deinit();
    const proc = constraint.processor();

    try expectAllowed(proc, 7, &.{3}); // only "▁a" starts " a..."
    try proc.commit(3);
    try expectAllowed(proc, 7, &.{6}); // the 'x' byte-fallback token
    try proc.commit(6);
    try expectAllowed(proc, 7, &.{ 2, 4 }); // optional b, or close via </s>
}

test "sampler integration: greedy decode is steered through the grammar" {
    if (!llg.enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    var ctx: ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const vocab = [_][]const u8{ "a", "b", "c", "<|end|>" };
    var tok = try bpe.Tokenizer.initFromParts(alloc, &vocab, &.{}, .{ .eos = 3 });
    defer tok.deinit();

    var constraint = try llg.Constraint.init(alloc, &tok, .{ .regex = "ab+c" }, .{});
    defer constraint.deinit();

    var s = sampler_mod.Sampler.init(.{}); // greedy
    s.processor = constraint.processor();

    // Unconstrained argmax would emit "c" forever; the mask forces "abc"+eos.
    const raw = [_]f32{ 0.1, 0.5, 3.0, 1.0 };
    var got: [4]usize = undefined;
    for (&got) |*slot| {
        var logits = try Logits.fromSlice(&ctx, .{ 1, 4 }, &raw);
        defer logits.deinit();
        slot.* = try s.next(&ctx, &logits, &.{});
    }
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, &got);
    try std.testing.expect(constraint.isStopped());

    // Seeded temperature sampling obeys the same mask.
    try constraint.reset();
    var st = sampler_mod.Sampler.init(.{ .temperature = 1.2, .top_k = 4, .seed = 7 });
    st.processor = constraint.processor();
    var first = try Logits.fromSlice(&ctx, .{ 1, 4 }, &raw);
    defer first.deinit();
    try std.testing.expectEqual(@as(usize, 0), try st.next(&ctx, &first, &.{}));
}
