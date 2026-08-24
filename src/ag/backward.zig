//! The VJP registry root. The records live in `backward/<domain>.zig` (the
//! split mirrors `src/exec/`'s taxonomy) and every facade mixin imports
//! its own domain file directly (`float/norm.zig` -> `backward/norm.zig`);
//! this root only forwards the domain tests so one `_ = backward` in
//! ../ag.zig reaches all of them.

test {
    _ = @import("backward/common.zig");
    _ = @import("backward/elementwise.zig");
    _ = @import("backward/matmul.zig");
    _ = @import("backward/reduce.zig");
    _ = @import("backward/stats.zig");
    _ = @import("backward/shape.zig");
    _ = @import("backward/gather_scatter.zig");
    _ = @import("backward/topk.zig");
    _ = @import("backward/softmax.zig");
    _ = @import("backward/norm.zig");
    _ = @import("backward/rope.zig");
    _ = @import("backward/attention.zig");
    _ = @import("backward/loss.zig");
    _ = @import("backward/conv.zig");
    _ = @import("backward/pool.zig");
}
