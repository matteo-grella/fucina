//! The public autograd band: the tagged `Tensor(spec)` facade (eager
//! forward over `tag_ops.zig`/`ExecContext`, tape-recorded backward), graph
//! control (`noGrad`, `checkpoint`, `customVjp`, `gradcheck`), and the VJP
//! registry (`ag/backward/`). This is the surface `fucina.Tensor` exports.
//! Layer stack: docs/ARCHITECTURE.md.
const std = @import("std");
const backward = @import("ag/backward.zig");
const checkpoint_mod = @import("ag/checkpoint.zig");
const control_mod = @import("ag/control.zig");
const custom_mod = @import("ag/custom.zig");
const core = @import("ag/core.zig");
const elemental_mod = @import("ag/elemental.zig");
const gradcheck_mod = @import("ag/gradcheck.zig");
const tags = @import("tags.zig");
const tensor = @import("ag/tensor.zig");

pub const Tensor = tensor.Tensor;
/// The backward engine's error domain (`MissingOutputGradient`,
/// `MissingBackwardGradient`, `BackwardAlreadyRun`).
pub const AgError = core.AgError;
pub const PackedRhs = tensor.PackedRhs;
pub const SliceRange = tensor.SliceRange;
pub const einsumMany = tensor.einsumMany;
pub const checkpoint = checkpoint_mod.checkpoint;
pub const checkpointWithContext = checkpoint_mod.checkpointWithContext;
pub const noGrad = control_mod.noGrad;
pub const isGradEnabled = control_mod.isGradEnabled;
pub const NoGradScope = control_mod.NoGradScope;
pub const customVjp = custom_mod.customVjp;
pub const gradcheck = gradcheck_mod.gradcheck;
pub const GradcheckOptions = gradcheck_mod.Options;
pub const GradcheckResult = gradcheck_mod.Result;

test {
    _ = backward;
    _ = checkpoint_mod;
    _ = control_mod;
    _ = custom_mod;
    _ = core;
    _ = elemental_mod;
    _ = gradcheck_mod;
    _ = tags;
    _ = tensor;
    _ = @import("ag_tests.zig");
}
