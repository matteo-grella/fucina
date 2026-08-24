//! The family-independent half of a LoRA trainer: which projections carry
//! adapters, the per-layer adapter set built from that selection, its
//! A/B tensor tuple, the dropout seed stream, and the loss knobs.
//!
//! Everything here is decided by the target selection alone — no model
//! config, no forward pass. A family trainer instantiates `AdapterSet`,
//! supplies its own per-layer projection dimensions, and keeps its forward,
//! its rope tables, and its loss tail. Consumers: `qwen3/train.zig`,
//! `gemma/train.zig`.
//!
//! The seven target names are the attention/FFN projection vocabulary every
//! decoder family in this tree shares; a family whose projections do not map
//! onto them wants its own set, not an eighth field here.

const std = @import("std");
const fucina = @import("fucina");

const ExecContext = fucina.ExecContext;
const lora = fucina.lora;
const rng = fucina.rng;

/// Which frozen projections receive a trainable LoRA adapter.
pub const Targets = struct {
    q: bool = true,
    k: bool = false,
    v: bool = true,
    o: bool = false,
    gate: bool = false,
    up: bool = false,
    down: bool = false,
};

pub const n_targets = 7;
pub const target_names = [n_targets][]const u8{ "q", "k", "v", "o", "gate", "up", "down" };

/// Masked label sentinel for a trainer's `loss`: positions whose label equals
/// this value contribute zero loss and zero gradient
/// (`CrossEntropyOptions.ignore_index`).
pub const ignore_index: usize = std.math.maxInt(usize);

/// Loss knobs for gradient accumulation (defaults reproduce the plain
/// `loss` exactly). Backward WITHOUT `zeroGrad` ADDS into each param's
/// persisted gradient, so N micro-batch lossExt+backward passes followed by
/// ONE clip/step/zeroGrad implement an N-sequence batch. Normalize on the
/// loss side: `.mean` + `loss_scale = 1.0/N` is mean-of-means (the true
/// batch mean only for equal supervised-token counts); `.sum` +
/// `loss_scale = 1.0/total_valid` (valid = labels != `ignore_index` across
/// the window) is the exact token-weighted mean. See docs/TRAINING.md §4
/// "Gradient accumulation".
pub const LossOptions = struct {
    /// CE reduction over the non-ignored positions.
    reduction: enum { mean, sum } = .mean,
    /// Multiplies the returned loss (and thus the gradients) via the
    /// differentiable `scale` op when != 1.
    loss_scale: f32 = 1,
};

/// Per-layer projection shapes, in `target_names` order: `[in, out]` for
/// each of the seven targets. A family builds one of these per layer (the
/// entries for disabled targets are never read), which is what lets
/// per-layer geometries — Gemma's alternating head dims — share this code
/// with fixed-geometry families.
pub const TargetDims = [n_targets][2]usize;

/// The adapter machinery for one target selection. `dropout_domain` is the
/// family's constant for the dropout seed stream: it keeps two families
/// trained from the same `seed` on independent noise.
pub fn AdapterSet(comptime targets: Targets, comptime dropout_domain: u64) type {
    return struct {
        pub fn enabled(comptime t: usize) bool {
            return @field(targets, target_names[t]);
        }

        /// The tagged adapter type for target `t`. The tag pairs are the
        /// projection's own axes, so an adapter composes with the frozen
        /// weight without a reshape.
        pub fn TargetAdapter(comptime t: usize) type {
            return switch (t) {
                0 => lora.Adapter(.embed, .q),
                1 => lora.Adapter(.embed, .k),
                2 => lora.Adapter(.embed, .v),
                3 => lora.Adapter(.attn, .embed),
                4, 5 => lora.Adapter(.embed, .ffn),
                6 => lora.Adapter(.ffn, .embed),
                else => unreachable,
            };
        }

        /// Position of target `t`'s A tensor in `AbTuple` (B is the next
        /// slot). Disabled targets occupy no slot.
        pub fn abIndex(comptime t: usize) usize {
            comptime {
                var j: usize = 0;
                for (0..t) |i| {
                    if (enabled(i)) j += 2;
                }
                return j;
            }
        }

        pub const n_enabled = blk: {
            var n: usize = 0;
            for (0..n_targets) |t| {
                if (enabled(t)) n += 1;
            }
            break :blk n;
        };

        /// One layer's adapters; a disabled target's field is `void`, so the
        /// selection costs nothing at runtime.
        pub const LayerAdapters = struct {
            q: if (targets.q) TargetAdapter(0) else void,
            k: if (targets.k) TargetAdapter(1) else void,
            v: if (targets.v) TargetAdapter(2) else void,
            o: if (targets.o) TargetAdapter(3) else void,
            gate: if (targets.gate) TargetAdapter(4) else void,
            up: if (targets.up) TargetAdapter(5) else void,
            down: if (targets.down) TargetAdapter(6) else void,
        };

        pub const ab_ptr_types = blk: {
            var types: [2 * n_enabled]type = undefined;
            var j: usize = 0;
            for (0..n_targets) |t| {
                if (enabled(t)) {
                    types[j] = *const TargetAdapter(t).ATensor;
                    types[j + 1] = *const TargetAdapter(t).BTensor;
                    j += 2;
                }
            }
            break :blk types;
        };

        /// Flat (A, B) pointer tuple over the enabled adapters — the shape
        /// the checkpointed layer blocks take their adapter inputs in.
        pub const AbTuple = std.meta.Tuple(&ab_ptr_types);

        /// Build one layer's enabled adapters at the shapes `dims` gives.
        /// Each adapter's init draws from its own substream, so a target set
        /// change does not perturb the others' initialization.
        pub fn initLayerAdapters(
            ctx: *ExecContext,
            ads: *LayerAdapters,
            dims: TargetDims,
            config: lora.Config,
            seed: u64,
            layer_i: usize,
        ) !void {
            var built: usize = 0;
            errdefer deinitLayerAdaptersPartial(ads, built);
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    @field(ads.*, target_names[t]) = try TargetAdapter(t).init(
                        ctx,
                        dims[t][0],
                        dims[t][1],
                        config,
                        rng.at(seed, layer_i * n_targets + t),
                    );
                    built += 1;
                }
            }
        }

        /// Deinit the first `built` enabled adapters (full teardown at
        /// `built == n_enabled`); the partial form serves init error paths.
        pub fn deinitLayerAdaptersPartial(ads: *LayerAdapters, built: usize) void {
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    const ordinal = comptime abIndex(t) / 2;
                    if (ordinal < built) @field(ads.*, target_names[t]).deinit();
                }
            }
        }

        pub fn abTuple(ads: *const LayerAdapters) AbTuple {
            var abs: AbTuple = undefined;
            comptime var j: usize = 0;
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    abs[j] = &@field(ads.*, target_names[t]).a;
                    abs[j + 1] = &@field(ads.*, target_names[t]).b;
                    j += 2;
                }
            }
            return abs;
        }

        /// Per-target dropout seeds for one layer at one step. `null` step
        /// (eval) yields all-null: no dropout stream is drawn, so eval never
        /// advances the noise.
        pub fn layerSeeds(seed: u64, n_layers: usize, step: ?u64, layer_i: usize) [n_targets]?u64 {
            var seeds: [n_targets]?u64 = [1]?u64{null} ** n_targets;
            const s = step orelse return seeds;
            const base = (s * @as(u64, n_layers) + layer_i) * n_targets;
            for (&seeds, 0..) |*sd, t| sd.* = rng.at(seed ^ dropout_domain, base + t);
            return seeds;
        }
    };
}

test {
    _ = @import("lora_trainer_tests.zig");
}
