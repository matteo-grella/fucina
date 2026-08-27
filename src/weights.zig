//! Model weight I/O and the fused linear/MoE weight containers: GGUF-backed
//! dense and quantized weights (`LinearWeight` — five behaviour containers
//! over ~30 GGUF formats — PTQTP plane weights, `LookupWeight`), MoE
//! expert-stack loading (resident, mmap-borrowed, or disk-streamed through
//! `ExpertStore`), the fused forward
//! helpers the LLM families share (`moeSwiGluFfnSeq`, `linearSeq*`), GGUF
//! host-side vector/matrix readers, and PTQTP decoration. Public as
//! `fucina.weights`.
//!
//! Home modules are imported directly: `fucina.zig` re-exports this file,
//! so the facade cannot be imported from here — the production import
//! graph is cycle-checked (`zig build arch-check`).
//!
//! WHAT BELONGS HERE, and what does not. Shared model code in this tree has
//! three homes, and the subject of the code decides which:
//!
//!   * here (`fucina.weights`, core band) — the subject is a WEIGHT
//!     CONTAINER: how to build one from GGUF bytes, and how to multiply by
//!     it. `linearSeq*` and `moe*FfnSeq` are forward compute but they are
//!     not stray: `LinearWeight.linearSeq` is a union dispatch INTO the
//!     per-format arms and the arms take the container types back, so the
//!     container and its multiply are one mutually-dependent unit.
//!   * `models/model_common.zig` (models band) — the subject is a GGUF FILE's
//!     layout: which tensor names a family's layer trio has, how an
//!     embed/head/norm set is read. Naming conventions, not numerics.
//!   * `models/host_ops.zig` (models band) — the subject is raw f32 HOST SLICES,
//!     for the host-reference ports that run below the Tensor facade.
//!
//! A helper that fits none of these wants a new home with a stated subject,
//! not a fourth un-ruled one.
//!
//! Layout: this file is the facade; the bodies live in `weights/`
//! (`common` substrate, `host` GGUF host readers, `gpu` weight residency,
//! `ptqtp` trit-plane containers, `dense` dense/packed containers and
//! loaders, `stack` quantized byte stacks, `borrowed` zero-copy linears,
//! `moe` expert stacks and streaming with the `moe_stream` options leaf,
//! `linear` the LinearWeight containers and the LookupWeight table,
//! `fuse` fusion and decoration).

const common = @import("weights/common.zig");
const host = @import("weights/host.zig");
const gpu = @import("weights/gpu.zig");
const ptqtp_w = @import("weights/ptqtp.zig");
const dense = @import("weights/dense.zig");
const stack = @import("weights/stack.zig");
const borrowed = @import("weights/borrowed.zig");
const moe = @import("weights/moe.zig");
const linear = @import("weights/linear.zig");
const fuse = @import("weights/fuse.zig");

pub const Error = common.Error;
pub const QuantWeight = common.QuantWeight;

pub const WeightF32 = dense.WeightF32;
pub const WeightF16 = dense.WeightF16;
pub const WeightBf16 = dense.WeightBf16;
pub const WeightQ4_K = dense.WeightQ4_K;
pub const WeightQ5_K = dense.WeightQ5_K;
pub const WeightQ6_K = dense.WeightQ6_K;
pub const WeightQ8_0 = dense.WeightQ8_0;
pub const setDecodeCompact = dense.setDecodeCompact;
pub const linearSeq = dense.linearSeq;

pub const ResidentByteRegistry = gpu.ResidentByteRegistry;

pub const WeightPtqtp = ptqtp_w.WeightPtqtp;
pub const WeightPtqtpFx4 = ptqtp_w.WeightPtqtpFx4;

pub const QuantByteStackPart = stack.QuantByteStackPart;
pub const QuantByteStackOptions = stack.QuantByteStackOptions;
pub const QuantByteStack = stack.QuantByteStack;
pub const makeQuantByteStack = stack.makeQuantByteStack;

pub const LinearWeight = linear.LinearWeight;
pub const DenseWeight = linear.DenseWeight;
pub const PackedWeight = linear.PackedWeight;
pub const ColdQuantWeight = linear.ColdQuantWeight;
pub const LookupWeight = linear.LookupWeight;

pub const loadMoeRhs = moe.loadMoeRhs;
pub const loadMoeRhsPtqtp = moe.loadMoeRhsPtqtp;
pub const MoeStreamOptions = moe.MoeStreamOptions;
pub const createExpertStore = moe.createExpertStore;
pub const StreamedMoeFfnRhs = moe.StreamedMoeFfnRhs;
pub const loadMoeRhsStreamed = moe.loadMoeRhsStreamed;
pub const registerStreamedMoeLayer = moe.registerStreamedMoeLayer;
pub const streamedProjSpec = moe.streamedProjSpec;
pub const streamedProjSpecPtqtp = moe.streamedProjSpecPtqtp;
pub const moeSwiGluFfnSeq = moe.moeSwiGluFfnSeq;
pub const moeGatedFfnSeq = moe.moeGatedFfnSeq;

pub const layerName = host.layerName;
pub const hostVector = host.hostVector;
pub const hostVectorInfo = host.hostVectorInfo;
pub const hostMatrix = host.hostMatrix;
pub const loadVector = host.loadVector;
pub const fillF32 = host.fillF32;

pub const BorrowedQuantLinearOptions = borrowed.BorrowedQuantLinearOptions;
pub const linearSeqBorrowedF16 = borrowed.linearSeqBorrowedF16;
pub const packGroupedQ8_0Rhs = borrowed.packGroupedQ8_0Rhs;
pub const groupedQ8_0GemvFusedInto = borrowed.groupedQ8_0GemvFusedInto;
pub const linearSeqBorrowedQuantized = borrowed.linearSeqBorrowedQuantized;

pub const fuseLinear = fuse.fuseLinear;
pub const PtqtpReport = fuse.PtqtpReport;
pub const decoratePtqtpInto = fuse.decoratePtqtpInto;

test {
    _ = common;
    _ = host;
    _ = gpu;
    _ = ptqtp_w;
    _ = dense;
    _ = stack;
    _ = borrowed;
    _ = moe;
    _ = linear;
    _ = fuse;
    _ = @import("weights/moe_stream.zig");
    _ = @import("weights_tests.zig");
}
