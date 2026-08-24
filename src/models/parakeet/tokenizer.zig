//! Parakeet SentencePiece-BPE detokenizer (decode-only). Matches
//! parakeet.cpp `src/tokenizer.cpp::detokenize`: concat `pieces[id]`, replace the
//! meta-space marker ▁ (U+2581 = E2 96 81) with a regular space, strip one
//! leading space (SentencePiece `decode_ids` behavior). The piece table comes
//! from `parakeet.tokenizer.pieces` (see `parakeet_loader.loadPieces`).
const std = @import("std");

/// Decode token ids to text. Caller owns the result.
pub fn detokenize(allocator: std.mem.Allocator, pieces: []const []const u8, ids: []const i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    for (ids) |id| {
        if (id < 0) continue;
        const ui: usize = @intCast(id);
        if (ui >= pieces.len) continue;
        const p = pieces[ui];
        var i: usize = 0;
        while (i < p.len) {
            if (i + 3 <= p.len and p[i] == 0xE2 and p[i + 1] == 0x96 and p[i + 2] == 0x81) {
                try out.append(allocator, ' ');
                i += 3;
            } else {
                try out.append(allocator, p[i]);
                i += 1;
            }
        }
    }
    // Strip a single leading space.
    if (out.items.len > 0 and out.items[0] == ' ') {
        std.mem.copyForwards(u8, out.items[0 .. out.items.len - 1], out.items[1..]);
        out.shrinkRetainingCapacity(out.items.len - 1);
    }
    return out.toOwnedSlice(allocator);
}

test {
    _ = @import("tokenizer_tests.zig");
}
