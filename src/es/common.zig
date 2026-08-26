//! Shared substrate of the ES module (es/*.zig): the error set, the config/
//! stats types, the noise/restore/decay/shaping enums, the domain
//! separators, the kernel thresholds, and the ternary storage-block alias.
//! Leaf: every other es/ file imports this. The public types are
//! re-exported by `es.zig`.

const dtype_mod = @import("../dtype.zig");

/// Ternary genome storage unit (GGUF tq2_0, the backend quantized-matmul
/// block): 256 2-bit crumbs storing w+1 in {0, 1, 2} plus one fp16 scale
/// `d` that ES never touches (register scales as float slots to train them).
pub const BlockTQ2_0 = dtype_mod.BlockTQ2_0;

pub const EsError = error{
    InvalidConfig,
    AnchorMissing,
    UnsupportedDType,
    NonContiguousParam,
    DuplicateParam,
    NoParams,
    RewardCountMismatch,
    MemberActive,
    MemberNotActive,
    ReplicaShapeMismatch,
};

/// How each member's noise streams relate across parameter slots.
pub const NoiseScheme = enum {
    /// Every (member, slot) pair gets an independent stream — the clean
    /// formulation (the reference archive's "iid" variant).
    iid,
    /// Every slot of a member reuses the SAME stream, so same-length slots
    /// receive identical noise and shorter slots share a prefix — the
    /// reference library's default scheme (an acknowledged artifact of
    /// reseeding one generator per tensor; kept for reference-faithful runs).
    correlated,
};

/// How `restore` undoes a perturbation.
pub const RestoreMode = enum {
    /// Regenerate the noise and subtract (the reference's zero-memory
    /// scheme). Exact up to `(x + t) - t` rounding drift per element.
    regenerate,
    /// Snapshot the parameter bytes on `perturb` and memcpy them back:
    /// bitwise restore, at the cost of one extra copy of the parameters.
    snapshot,
};

/// Anchored weight decay (AWD, arXiv:2605.30148, the ES-at-scale group's
/// anti-forgetting regularizer): a proximal step applied AFTER each ES
/// update, pulling theta toward a fixed ANCHOR (`captureAnchor` — typically
/// the pretrained weights at fine-tuning start; meaningless from scratch,
/// where the anchor would be random init):
///
///     l2: theta <- theta_0 + (1 - alpha*lambda) * (theta - theta_0)
///     l1: theta <- theta_0 + sign(d) * max(|d| - alpha*lambda, 0),
///         d = theta - theta_0   (proximal soft-threshold; induces sparsity)
///
/// The decay step is the COUPLED alpha*lambda (the reference's fixed-alpha
/// form); l2 requires alpha*lambda < 1. Reference defaults: l2 lambda = 10,
/// l1 lambda = 0.01 (with alpha = 5e-4). Counteracts the accumulated
/// random-walk drift in reward-irrelevant directions — the cheap substitute
/// for a large population.
pub const AnchorDecay = enum { none, l1, l2 };

/// Reward shaping applied inside `update`.
pub const RewardNorm = enum {
    /// (r - mean) / (std + 1e-8), stats in f64, ddof = 0 — the reference's
    /// only implemented shaping. Affine, so it PRESERVES outlier magnitude:
    /// one catastrophic member can dominate the update (the classic failure
    /// mode with unbounded rewards such as raw -CE). Self-stops exactly when
    /// every reward ties (all coefficients become 0).
    z_score,
    /// Centered-rank fitness shaping (Salimans et al. 2017, the OpenAI-ES
    /// default; the ES-at-scale reference deliberately drops it): members
    /// are ranked ascending by reward and mapped to the fixed grid
    /// rank/(N-1) - 0.5 in [-0.5, 0.5]. Invariant to any monotone reward
    /// transform and immune to outliers (the worst member is exactly -0.5
    /// no matter how bad). Ties break by MEMBER INDEX, ascending — a pinned,
    /// deterministic order (numpy argsort tie order is implementation-
    /// defined; this mapping is a checkpoint contract). Note: unlike
    /// z_score, all-equal rewards still produce the full utility spread
    /// (a mean-zero phantom step), so pair long runs with eval-based
    /// stopping.
    centered_ranks,
    /// Use raw rewards as coefficients (tests / externally-shaped rewards).
    none,
};

pub const Config = struct {
    /// Perturbation scale. Reference default.
    sigma: f32 = 0.001,
    /// Learning rate with 1/sigma folded in; null = sigma/2 (the reference's
    /// auto default).
    alpha: ?f32 = null,
    /// Population size N. Reference default.
    population: usize = 30,
    /// Mirrored (antithetic) sampling — Salimans et al. 2017's variance
    /// reduction, deliberately stripped by the ES-at-scale reference (kept
    /// opt-in here for the same reason as `centered_ranks`): members pair up
    /// as (+eps, -eps) sharing one noise draw — member 2k draws the stream,
    /// member 2k+1 applies its negation — and the update folds each pair to
    /// (C_2k - C_2k+1) * eps_k, halving update-side noise regeneration.
    /// Requires an even population. Part of the noise checkpoint contract
    /// (persist it alongside seed/population/noise scheme).
    antithetic: bool = false,
    noise: NoiseScheme = .iid,
    restore_mode: RestoreMode = .regenerate,
    /// Cache each iteration's noise streams in RAM: every stream is
    /// regenerated ONCE per iteration (on first use, straight into the
    /// cache) and replayed from memory afterwards — bitwise identical to
    /// regeneration. Memory: n_streams * elementCount * 4 bytes, n_streams
    /// = population (population/2 with antithetic). Pays off only when
    /// replays outnumber fills by enough to beat the cache's allocation and
    /// memory traffic (large populations, small parameter sets); at large
    /// parameter counts the cache can be a net loss — benchmark before
    /// enabling. Replica materialization does not consult the cache.
    cache_streams: bool = false,
    /// AWD penalty (see `AnchorDecay`); `.none` = the reference algorithm.
    anchor_decay: AnchorDecay = .none,
    /// AWD lambda; the applied per-iteration step is alpha * lambda.
    anchor_lambda: f32 = 0,
    reward_norm: RewardNorm = .z_score,
    /// Ternary slots only (`addTernaryParam`): the fraction of a slot's
    /// logical elements each member flips per perturbation — per-member
    /// flips = max(1, round(ternary_flip_rate * len)). Must be in (0, 1].
    /// Part of the flip-stream checkpoint contract like `antithetic`
    /// (re-pass it identically on resume alongside seed/population).
    ternary_flip_rate: f32 = 0.001,
    /// Ternary update budget: the top-K vote cap per update is
    /// K = max(1, round(effective_fraction * len)). Must be in (0, 1].
    /// Checkpoint contract (re-pass it identically on resume).
    ternary_update_fraction: f32 = 0.005,
    /// Decay schedule on the ternary update budget: effective_fraction =
    /// ternary_update_fraction / (1 + ternary_update_decay * iteration) —
    /// EGGROLL's 1/(1+ct) cadence; 0 = constant budget. Must be finite and
    /// >= 0. Checkpoint contract (re-pass it identically on resume).
    ternary_update_decay: f32 = 0.0,
    /// Base seed for the member-seed and noise-stream derivations.
    seed: u64 = 42,
};

/// Per-update reward statistics (pre-normalization, over the population).
pub const Stats = struct {
    mean_reward: f64,
    std_reward: f64,
    min_reward: f32,
    max_reward: f32,
};

/// Domain separators so member-seed and noise-stream derivations never
/// collide with each other or with other rng.at consumers of the same base
/// seed (same pattern as the trainers' dropout domain).
pub const seed_domain: u64 = 0x65735f7365656473; // "es_seeds"
pub const noise_domain: u64 = 0x65735f6e6f697365; // "es_noise"
/// Ternary flip streams live in their own domain: a member's flip stream
/// and its gaussian noise streams derive from the same (config.seed,
/// iteration, member) but can never collide, which is what keeps float
/// slots bitwise unchanged when ternary slots join a trainer.
pub const ternary_domain: u64 = 0x65735f7472697473; // "es_trits"

/// Elementwise noise kernels chunk across the worker pool above this many
/// elements. Noise regeneration is ~an order of magnitude costlier per
/// element than an optimizer step (Box-Muller: two mixes + log/sqrt/cos), so
/// the threshold sits below optim.zig's 1<<17. Every parallel loop here is an
/// element-independent map — results are bitwise identical to the serial
/// path for any thread count.
pub const perturb_min_len: usize = 1 << 14;
/// The update loop does `population` regenerations per element; its
/// threshold is correspondingly lower.
pub const update_min_len: usize = 1 << 11;

/// Stack chunk (f32 elements) for the noise/accumulator scratch of the
/// kernels (kernels.zig) — small enough for worker stacks, large enough to
/// amortize the per-chunk pair alignment.
pub const chunk_len: usize = 2048;
