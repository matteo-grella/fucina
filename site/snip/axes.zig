// axis names live in the type
const M = Tensor(.{ .batch, .d });
comptime {
    // resolve an axis by name, at compile time
    std.debug.assert(M.axis(.d) == 1);
    std.debug.assert(M.hasTag(.batch));
    std.debug.assert(!M.hasTag(.channel));
}
// broadcasting & contraction align by NAME,
// never by position — compiled to stride math
// with zero runtime tagging cost.
