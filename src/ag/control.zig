//! Public gradient-recording controls for the eager autograd facade.
const std = @import("std");

threadlocal var no_grad_depth: usize = 0;

pub const NoGradScope = struct {
    active: bool = true,

    pub fn close(self: *NoGradScope) void {
        if (!self.active) return;
        std.debug.assert(no_grad_depth > 0);
        no_grad_depth -= 1;
        self.active = false;
    }
};

pub fn noGrad() NoGradScope {
    no_grad_depth += 1;
    return .{};
}

pub fn isGradEnabled() bool {
    return no_grad_depth == 0;
}

test "noGrad scope disables and restores gradient recording" {
    try std.testing.expect(isGradEnabled());
    var outer = noGrad();
    defer outer.close();
    try std.testing.expect(!isGradEnabled());
    {
        var inner = noGrad();
        defer inner.close();
        try std.testing.expect(!isGradEnabled());
    }
    try std.testing.expect(!isGradEnabled());
    outer.close();
    try std.testing.expect(isGradEnabled());
}
