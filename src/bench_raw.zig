//! Raw-tensor vocabulary for the microbenchmarks (`bench/*.zig`): re-exports
//! the internal RawTensor and the backend/exec plumbing raw kernels bench
//! against. Lives in `src/` because it is its own module root importing
//! `src/*.zig` siblings by path, and a Zig module cannot import above its
//! root's directory. Never part of the public `fucina` module.
const ag = @import("ag.zig");
const backend = @import("backend.zig");
const dtype = @import("dtype.zig");
const exec = @import("exec.zig");
const tensor = @import("tensor.zig");

pub const Tensor = ag.Tensor;
pub const noGrad = ag.noGrad;
pub const einsumMany = ag.einsumMany;
pub const RawTensor = tensor.Tensor;
pub const RawTensorOf = tensor.TensorOf;
pub const ExecContext = exec.ExecContext;
pub const RopeMode = exec.RopeMode;
pub const RopeTable = exec.RopeTable;
pub const simd = @import("fucina.zig").simd;
pub const active_backend_kind = backend.active_kind;
pub const BlockQ8_0 = dtype.BlockQ8_0;
pub const q8_0_block_size = dtype.q8_0_block_size;
pub const optim = @import("optim.zig");
