//! The MoE band: the `MoeRhs` expert-stack container, the single-token
//! decode and batched-prefill expert FFN engines, the phase-chain scheduling
//! every family shares (`chain`), and the decode-scratch views family
//! engines carve. Above exec (a consumer of the runtime), below the weight
//! containers and the model families.
const expert_ffn = @import("moe/expert_ffn.zig");

pub const chain = @import("moe/chain.zig");
/// A MoE expert stack's RHS: resident (borrowed or owned) or streamed through an `ExpertStore`.
pub const MoeRhs = expert_ffn.MoeRhs;
pub const MoePtqtpRhs = expert_ffn.MoePtqtpRhs;
/// Per-phase timing counters for the batched MoE paths.
pub const MoeBatchProfile = expert_ffn.MoeBatchProfile;
/// Single-token routed expert FFN over K-quant expert stacks.
pub const expertFfn = expert_ffn.expertFfn;
/// Batched-prefill routed expert FFN: tokens grouped by expert.
pub const expertFfnBatch = expert_ffn.expertFfnBatch;
pub const lockDecodeScratch = expert_ffn.lockDecodeScratch;
pub const unlockDecodeScratch = expert_ffn.unlockDecodeScratch;
pub const DecodeScratchView = expert_ffn.DecodeScratchView;
pub const DecodeChainScratchView = expert_ffn.DecodeChainScratchView;
pub const carveDecodeScratch = expert_ffn.carveDecodeScratch;
pub const carveDecodeChainScratch = expert_ffn.carveDecodeChainScratch;

test {
    _ = @import("moe/expert_ffn_tests.zig");
}
