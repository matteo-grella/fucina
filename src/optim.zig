//! Gradient-descent optimizers over the public autograd Tensor facade.
//!
//! Optimizers, each a faithful port of its reference implementation:
//!
//! - `Adam` — PyTorch `torch.optim.Adam` single-tensor path (coupled L2
//!   weight decay: `g += weight_decay * p` before moment updates).
//! - `AdamW` — PyTorch `torch.optim.AdamW` single-tensor path (decoupled decay
//!   applied to the parameter BEFORE the Adam step; `denom = sqrt(v)/sqrt(1-b2^t) + eps`).
//! - `Muon` — Keller Jordan's reference (github.com/KellerJordan/Muon): lerp-form
//!   momentum, Newton-Schulz-5 orthogonalization with the transpose trick, and a
//!   built-in AdamW fallback for non-matrix params (biases, norms) and params the
//!   caller routes there explicitly (embeddings, output heads). Both update-scale
//!   conventions are available: Keller's spectral `sqrt(max(1, rows/cols))` and
//!   Moonlight's RMS-matching `0.2*sqrt(max(rows, cols))` (arXiv 2502.16982).
//! - `Apollo` — the official `apollo_torch` optimizer (arXiv 2412.05270): random
//!   low-rank projection of the gradient, AdamW moments in the compressed space,
//!   channel- or tensor-wise gradient scaling, the Fira norm-growth limiter, and
//!   a scaled-SGD update. APOLLO-Mini is `ApolloConfig.mini()`. Its fallback path
//!   reproduces the reference's legacy-HF AdamW (eps OUTSIDE the bias correction,
//!   decay AFTER the step) — deliberately different from `AdamW` above.
//!
//! Ownership: optimizers hold refcounted views of parameter storage plus raw
//! `*GradState` pointers, so parameter tensors may move by value but must
//! OUTLIVE the optimizer (they own the GradState). Each variable must be
//! registered with exactly one optimizer: duplicates are rejected WITHIN one
//! instance (`error.DuplicateParam`); registering the same tensor with two
//! different instances (e.g. two OptimizerSet groups) is not detectable and
//! silently double-steps — cross-instance uniqueness is the caller's
//! responsibility. Optimizer state (moments, momentum, projections) is owned
//! by the optimizer and freed by `deinit`. Newton-Schulz / projection
//! transients come from the ExecContext BufferPool.
//!
//! Checkpointing: parameter values have two formats. `saveTensors`/
//! `loadTensors` (FZT1) are positional and f32-only: the loading program must
//! list the same tensors in the same order (shapes are validated).
//! `saveStateDict`/`loadStateDict` are safetensors-backed, NAMED, and
//! dtype-aware (f32/f16/bf16, raw byte passthrough): load matches stream entries to the
//! provided list by name, so entry order is free; strict mode (the default)
//! requires an exact one-to-one match, non-strict skips unknown stream
//! entries. Each optimizer's `saveState`/`loadState` serializes moments, step
//! counts, and the structural config fields, validating them on load. When
//! every state buffer is f32 (the default) the v3 frames are written
//! byte-identically (FZAD/FZA3/FZM3/FZP3/FZS3; FZO3 for OptimizerSet), so
//! pre-bf16 builds keep reading new f32 checkpoints. When any state buffer is
//! non-f32 the v4 frames are written instead (FZD4/FZA4/FZM4/FZS4 — Adam,
//! AdamW, Muon, SGD; Apollo state stays f32, always FZP3): identical layout
//! except each state buffer is prefixed by one u8 `StateDType` tag. Loaders
//! accept both versions but require the stored dtype to match the configured
//! one EXACTLY (v3 implies f32 everywhere) — a cross-dtype load errors with
//! `CheckpointDtypeMismatch` rather than converting, because an implicit
//! f32<->bf16 conversion would silently break the bit-exact-resume contract
//! below. Optimizer slots are matched BY
//! NAME — explicit via `addParamNamed`, otherwise the auto-name "param<i>"
//! from the slot's index within its slot list — so named params may be
//! re-registered in any order within their list (Muon/Apollo route params to
//! their fallback by rank, independent of order); unnamed params must keep
//! their relative registration order to reproduce their auto-names. A failed
//! load leaves the target partially restored — treat load errors as fatal for
//! that model/optimizer instance. Resumption replays the optimizer
//! bit-exactly; end-to-end bit-exact training resume additionally requires
//! the surrounding forward/backward replay to be deterministic (true for
//! single-contribution gradients; gradients accumulated from 3+ async
//! dot-backward branches are summed in completion order). APOLLO projections
//! are not stored: P is a deterministic, repo-owned function of
//! (seed, step / update_proj_gap) and is regenerated on the next step.
//!
//! Layout: this file is the facade; the bodies live in `optim/`
//! (`common` substrate, `frame` checkpoint helpers, `moment_pair`
//! Adam/AdamW, `muon`, `apollo`, `sgd`, `schedule`, `set`).

const common = @import("optim/common.zig");
const frame = @import("optim/frame.zig");
const moment_pair = @import("optim/moment_pair.zig");
const muon = @import("optim/muon.zig");
const apollo = @import("optim/apollo.zig");
const sgd = @import("optim/sgd.zig");
const schedule = @import("optim/schedule.zig");
const set = @import("optim/set.zig");
const state_dict = @import("state_dict.zig");

pub const OptimError = common.OptimError;
pub const StateDType = common.StateDType;
pub const Param = common.Param;
pub const sumSquares = common.sumSquares;

pub const AdamWConfig = moment_pair.AdamWConfig;
pub const AdamW = moment_pair.AdamW;
pub const AdamConfig = moment_pair.AdamConfig;
pub const Adam = moment_pair.Adam;

pub const MuonScale = muon.MuonScale;
pub const MuonConfig = muon.MuonConfig;
pub const Muon = muon.Muon;
pub const newtonSchulz5 = muon.newtonSchulz5;

pub const ApolloScaleType = apollo.ApolloScaleType;
pub const ApolloConfig = apollo.ApolloConfig;
pub const Apollo = apollo.Apollo;

pub const SgdConfig = sgd.SgdConfig;
pub const SGD = sgd.SGD;

pub const LrSchedule = schedule.LrSchedule;
pub const warmupCosineFactor = schedule.warmupCosineFactor;

pub const AnyOptimizer = set.AnyOptimizer;
pub const anyOptimizer = set.anyOptimizer;
pub const OptimizerSet = set.OptimizerSet;

pub const saveTensors = frame.saveTensors;
pub const loadTensors = frame.loadTensors;

pub const NamedTensor = state_dict.NamedTensor;
pub const NamedTensorMut = state_dict.NamedTensorMut;
pub const LoadOptions = state_dict.LoadOptions;
pub const saveStateDict = state_dict.saveStateDict;
pub const loadStateDict = state_dict.loadStateDict;

test {
    _ = common;
    _ = frame;
    _ = moment_pair;
    _ = muon;
    _ = apollo;
    _ = sgd;
    _ = schedule;
    _ = set;
    _ = @import("optim_tests.zig");
}
