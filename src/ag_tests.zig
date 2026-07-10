//! Tests for the autograd facade (`ag.zig`): asserts the legacy raw autograd
//! surface (Function/Node/Engine) stays removed from the core scheduler.
const std = @import("std");
const core = @import("ag/core.zig");

test "legacy raw autograd surface stays removed" {
    inline for (.{ "Fun" ++ "ction", "No" ++ "de", "Eng" ++ "ine" }) |decl| {
        try std.testing.expect(!@hasDecl(core, decl));
    }
}
