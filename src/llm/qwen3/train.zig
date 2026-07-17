//! LoRA fine-tuning over a frozen Qwen3 `Model` (dense only).
//!
//! `Trainer(targets)` mirrors the inference forward op-for-op — same norms,
//! fused q/k-norm+RoPE, grouped causal attention, SwiGLU — but runs every
//! frozen projection through the DIFFERENTIABLE frozen-RHS `dot` (gradients
//! flow to the f32 activations only; weight memory stays quantized/f16) and
//! adds trainable LoRA deltas on the projections selected by `targets`. The
//! base model is never written: the only parameters are the adapters' A/B.
//!
//! Contracts (see docs/TRAINING.md):
//! - `loss` MUST run inside an open exec scope (`ctx.openExecScope()`); the
//!   defer-deinit forward below relies on scope adoption to keep the graph
//!   alive until `backward()`. It returns a scope-owned borrow.
//! - The trainer caches one RoPE table per distinct sequence length and never
//!   frees them before `deinit` (training positions are always 0..seq-1, so a
//!   table for a given length is immutable and reusable): with
//!   `checkpoint_layers` the backward recompute re-reads the table through
//!   the checkpoint `extra`, and the cache keeps that pointer valid across
//!   ANY interleaving of `loss`/`evalLastLogits`/`backward`.
//! - Dropout is deterministic: per (step, layer, projection) seeds derived
//!   via `rng.at` from the base seed, replayed bitwise by the checkpoint
//!   recompute (seeds travel by value in `extra`). Eval passes null seeds.
//! - MoE configs (`num_experts > 0`) are rejected with `Error.MoeUnsupported`.

const std = @import("std");
const fucina = @import("fucina");
const qwen3 = @import("model.zig");
const cartridge_mod = @import("../cartridge.zig");
const engram_mod = @import("../engram.zig");
const weights = @import("../weights.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const LinearWeight = weights.LinearWeight;
const ParamRegistry = fucina.ParamRegistry;
const Tag = @TypeOf(.tag);
const lora = fucina.lora;
const optim = fucina.optim;
const rng = fucina.rng;

pub const Error = error{
    MoeUnsupported,
    ExecScopeRequired,
    InvalidSequenceLength,
    LabelLengthMismatch,
    InvalidLayerRange,
    InvalidInjection,
    InvalidCartridge,
    InvalidCapture,
    InvalidPacking,
    InvalidEngram,
    /// Cartridge / capture / packed forwards are plain-path only: the
    /// cartridge K/V variables, capture sink, and transient packed rope
    /// tables are not checkpoint inputs, so a recompute would silently
    /// detach, re-fill, or dangle them.
    CartridgeCheckpointUnsupported,
};

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

/// Masked label sentinel for `loss`: positions whose label equals this value
/// contribute zero loss and zero gradient (`CrossEntropyOptions.ignore_index`).
pub const ignore_index: usize = std.math.maxInt(usize);

/// Test seam: the per-block layer type of `qwen3.Model` (not exported by
/// qwen3.zig), reachable through the `layers` field for synthetic-model
/// construction in tests.
pub const ModelLayer = std.meta.Child(@FieldType(qwen3.Model, "layers"));

/// The trainer's residual-stream tensor type ([seq, embed] f32) — the
/// currency of `Trainer.forwardHidden` and `Injection.row`.
pub const Hidden = fucina.Tensor(.{ .seq, .embed });

/// A single-row embedding override for `ForwardOptions.inject`: `row`
/// (a [1, embed] tensor) replaces the token embedding at sequence position
/// `pos` before the first layer runs. The substitution is the DIFFERENTIABLE
/// `setSlice`, so when `row` is a variable, backward routes a gradient into
/// it even though the surrounding embedding rows are a frozen constant.
pub const Injection = struct {
    pos: usize,
    row: *const Hidden,
};

/// Shared capture payload (see `cartridge.KvCapture`): per-layer host
/// copies of the post-q/k-norm, post-RoPE keys and the values of one
/// forward pass. Fill via `ForwardOptions.capture`; `Trainer.captureKv`
/// wraps the flow.
pub const KvCapture = cartridge_mod.KvCapture;

/// Options for `Trainer.forwardHidden` / `Trainer.evalLastLogitsExt`: run
/// layers [start_layer, start_layer + layer_count) over the token embedding
/// (`layer_count == null` runs through the last layer), optionally with a
/// single-row embedding injection applied before the first selected layer.
///
/// `cartridge` prepends a trained KV prefix to every layer's attention
/// (tokens shift to RoPE positions p..p+seq-1 and attend the whole prefix;
/// gradients flow into the cartridge's trainable rows through the frozen
/// stack). `capture` copies each layer's freshly computed token K/V rows
/// out of the forward — the cartridge initialization seam. Both are
/// plain-path only (`CartridgeCheckpointUnsupported` under
/// `checkpoint_layers`).
/// `packed_segments` runs several independent sequences through ONE forward as
/// contiguous segments of the packed row (`packed_segments[i]` = length of
/// i; lengths must sum to `tokens.len`). Every seq-parallel op (the
/// projections, norms, logits — the dominant GEMMs) batches over the packed
/// rows for free; RoPE positions restart per segment, and attention runs
/// per segment over zero-copy narrows so no token ever sees another
/// segment (with a cartridge, each segment sees the shared prefix plus its
/// own causal rows — bit-for-bit the reference block mask). Gradients flow
/// through the same fused attention backward per segment and accumulate
/// into shared leaves (the cartridge rows) across segments.
pub const ForwardOptions = struct {
    start_layer: usize = 0,
    layer_count: ?usize = null,
    inject: ?Injection = null,
    cartridge: ?*const cartridge_mod.Cartridge = null,
    /// COMPOSED prefix (Cartridges at Scale): the parts' rows concatenate in
    /// order ahead of the tokens, which shift to RoPE positions
    /// `composedP(parts)..`; gradients flow into EVERY part's trainable rows
    /// through one concat per layer — the mixed-visibility joint-training
    /// seam. Mutually exclusive with `cartridge` (a single-part composition
    /// is op-for-op the single-cartridge path); plain-path only, like
    /// `cartridge`.
    cartridges: ?[]const *const cartridge_mod.Cartridge = null,
    capture: ?*KvCapture = null,
    packed_segments: ?[]const usize = null,
    engram: ?EngramOptions = null,
};

/// Engram graft seam (`ForwardOptions.engram`): before each layer whose id
/// is in the engram model's `layer_ids`, the layer's memory output is added
/// to the residual stream (`hidden += engram(hidden, rows)` — the reference
/// block order, ahead of attention). `rows[slot]` holds the precomputed
/// table-row indices for `tokens` at plan slot `slot`
/// (`HashPlan.compressInto` + `hashInto`; pure host work, once per
/// sequence). Plain-path only, and hashing depends only on token ids, so
/// it composes with `cartridge` (positions shift, ids don't). Rejected
/// with `packed_segments`: the ShortConv is causal over the packed row and
/// would leak across segment boundaries.
pub const EngramOptions = struct {
    model: *const engram_mod.Engram,
    rows: []const []const usize,
};

/// The seven adaptable projections, in fixed order. Index doubles as the
/// dropout-seed slot and the `Targets` field name.
const n_targets = 7;
const target_names = [n_targets][]const u8{ "q", "k", "v", "o", "gate", "up", "down" };

/// Everything one layer's forward needs besides the differentiable inputs.
/// Stored BY VALUE in checkpoint backward nodes: pointers/slices reference
/// model- or trainer-owned state that outlives the backward pass.
const LayerExtra = struct {
    layer: *const ModelLayer,
    config: qwen3.Config,
    rope_table: *const fucina.RopeTable,
    kv_head_for_head: []const usize,
    /// Per-projection dropout seeds for this (step, layer); null = eval.
    seeds: [n_targets]?u64,
    /// LoRA alpha / rank.
    scale: f32,
    dropout_p: f32,
    /// Absolute layer index — the capture sink's row slot.
    layer_i: usize = 0,
    /// This layer's trained KV prefix (plain path only; never checkpointed).
    cartridge_layer: ?*const cartridge_mod.LayerKv = null,
    /// Composed multi-cartridge prefix (plain path only; never checkpointed).
    cartridge_parts: ?[]const *const cartridge_mod.Cartridge = null,
    /// Token K/V row sink (plain path only; never checkpointed).
    capture: ?*KvCapture = null,
    /// Packed segment lengths (plain path only; never checkpointed).
    packed_segments: ?[]const usize = null,
};

/// Rope-table cache key: token positions run offset..offset+len-1.
const RopeKey = struct {
    offset: usize,
    len: usize,
};

/// Differentiable frozen linear: route through the plain `.value` tensor of
/// every `LinearWeight` variant (the packed fast paths are inference-only —
/// they reject gradients), tagged [out, in] as the frozen-RHS `dot` expects.
/// Fused-distill route gate: FUCINA_NO_FUSED_DISTILL=1 forces the composed
/// logits + `cartridge.distillLoss` tail (the A/B and emergency-revert
/// switch — the fused route matches it to f32 roundoff, not bitwise).
/// Read once, cached; `setFusedDistill` is the test hook.
var fused_distill_state = std.atomic.Value(u8).init(0); // 0 unread, 1 on, 2 off

pub fn setFusedDistill(on: ?bool) void {
    fused_distill_state.store(if (on) |o| (if (o) @as(u8, 1) else 2) else 0, .release);
}

fn fusedDistillEnabled() bool {
    var state = fused_distill_state.load(.acquire);
    if (state == 0) {
        // fucina.parallel.envFlag, NOT std.c.getenv: libc-free Linux
        // builds have no std.c.
        state = if (fucina.parallel.envFlag("FUCINA_NO_FUSED_DISTILL")) 2 else 1;
        fused_distill_state.store(state, .release);
    }
    return state == 1;
}

fn dotLinear(
    weight: *const LinearWeight,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    @setEvalBranchQuota(20_000);
    return switch (weight.*) {
        .q4_k => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .q5_k => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .q6_k => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .q8_0 => |*w| dotFrozen(&w.value, ctx, input, in_tag, out_tag),
        .ptqtp => |*w| blk: {
            var acc = try dotFrozen(&w.p1, ctx, input, in_tag, out_tag);
            inline for ([_][]const u8{ "p2", "p3" }) |plane_field| {
                if (@field(w, plane_field)) |*plane| {
                    errdefer acc.deinit();
                    var y = try dotFrozen(plane, ctx, input, in_tag, out_tag);
                    defer y.deinit();
                    const sum = try acc.add(ctx, &y);
                    acc.deinit();
                    acc = sum;
                }
            }
            break :blk acc;
        },
        inline else => |*w| dotFrozen(w, ctx, input, in_tag, out_tag),
    };
}

fn dotFrozen(
    weight: anytype,
    ctx: *ExecContext,
    input: anytype,
    comptime in_tag: Tag,
    comptime out_tag: Tag,
) !fucina.Tensor(.{ .seq, out_tag }) {
    var tagged = try weight.withTags(ctx, .{ out_tag, in_tag });
    defer tagged.deinit();
    return input.dot(ctx, &tagged, in_tag);
}

const QkvLinear = struct {
    q: fucina.Tensor(.{ .seq, .q }),
    k: fucina.Tensor(.{ .seq, .k }),
    v: fucina.Tensor(.{ .seq, .v }),

    fn deinit(self: *QkvLinear) void {
        self.v.deinit();
        self.k.deinit();
        self.q.deinit();
        self.* = undefined;
    }
};

/// Base QKV through the frozen projections — both union arms (mirrors
/// `AttentionProjection.project`, with the differentiable dot).
fn projectQkv(ctx: *ExecContext, layer: *const ModelLayer, input: *const Hidden, cfg: qwen3.Config) !QkvLinear {
    return switch (layer.attn_proj) {
        .separate => |*sep| blk: {
            var q = try dotLinear(&sep.q_proj, ctx, input, .embed, .q);
            errdefer q.deinit();
            var k = try dotLinear(&sep.k_proj, ctx, input, .embed, .k);
            errdefer k.deinit();
            const v = try dotLinear(&sep.v_proj, ctx, input, .embed, .v);
            break :blk .{ .q = q, .k = k, .v = v };
        },
        .fused => |*w| blk: {
            var qkv = try dotLinear(w, ctx, input, .embed, .qkv);
            defer qkv.deinit();
            break :blk try splitQkv(ctx, &qkv, cfg);
        },
    };
}

/// Replica of qwen3.zig's (private) fused-QKV split: zero-copy narrows.
fn splitQkv(ctx: *ExecContext, qkv: *const fucina.Tensor(.{ .seq, .qkv }), cfg: qwen3.Config) !QkvLinear {
    const q_dim = cfg.num_attention_heads * cfg.head_dim;
    const kv_dim = cfg.num_key_value_heads * cfg.head_dim;

    var q_view = try qkv.narrow(ctx, .qkv, 0, q_dim);
    defer q_view.deinit();
    var q = try q_view.withTags(ctx, .{ .seq, .q });
    errdefer q.deinit();

    var k_view = try qkv.narrow(ctx, .qkv, q_dim, kv_dim);
    defer k_view.deinit();
    var k = try k_view.withTags(ctx, .{ .seq, .k });
    errdefer k.deinit();

    var v_view = try qkv.narrow(ctx, .qkv, q_dim + kv_dim, kv_dim);
    defer v_view.deinit();
    const v = try v_view.withTags(ctx, .{ .seq, .v });
    return .{ .q = q, .k = k, .v = v };
}

const GateUpLinear = struct {
    gate: fucina.Tensor(.{ .seq, .ffn }),
    up: fucina.Tensor(.{ .seq, .ffn }),

    fn deinit(self: *GateUpLinear) void {
        self.up.deinit();
        self.gate.deinit();
        self.* = undefined;
    }
};

const DenseFfn = @FieldType(@FieldType(ModelLayer, "ffn"), "dense");

/// Base gate/up through the frozen projections — both union arms (mirrors
/// `FfnInputProjection.project`, with the differentiable dot).
fn projectGateUp(ctx: *ExecContext, dense: *const DenseFfn, input: *const Hidden, cfg: qwen3.Config) !GateUpLinear {
    return switch (dense.input_proj) {
        .separate => |*sep| blk: {
            var gate = try dotLinear(&sep.gate_proj, ctx, input, .embed, .ffn);
            errdefer gate.deinit();
            const up = try dotLinear(&sep.up_proj, ctx, input, .embed, .ffn);
            break :blk .{ .gate = gate, .up = up };
        },
        .fused => |*w| blk: {
            var gate_up = try dotLinear(w, ctx, input, .embed, .gate_up);
            defer gate_up.deinit();
            break :blk try splitGateUp(ctx, &gate_up, cfg);
        },
    };
}

/// Replica of qwen3.zig's (private) fused gate/up split.
fn splitGateUp(ctx: *ExecContext, gate_up: *const fucina.Tensor(.{ .seq, .gate_up }), cfg: qwen3.Config) !GateUpLinear {
    var gate_view = try gate_up.narrow(ctx, .gate_up, 0, cfg.intermediate_size);
    defer gate_view.deinit();
    var gate = try gate_view.withTags(ctx, .{ .seq, .ffn });
    errdefer gate.deinit();

    var up_view = try gate_up.narrow(ctx, .gate_up, cfg.intermediate_size, cfg.intermediate_size);
    defer up_view.deinit();
    const up = try up_view.withTags(ctx, .{ .seq, .ffn });
    return .{ .gate = gate, .up = up };
}

pub fn Trainer(comptime targets: Targets) type {
    return struct {
        model: *const qwen3.Model,
        allocator: Allocator,
        lora_config: lora.Config,
        /// alpha / rank, mirrored from the adapters for the checkpoint extra.
        scale: f32,
        /// Base seed: drives adapter init and the per-step dropout streams.
        seed: u64,
        /// One adapter set per transformer layer.
        adapters: []LayerAdapters,
        /// Every adapter A/B under its "layers.<i>.<target>.lora_{a,b}"
        /// checkpoint name: the registry owns the names and retains
        /// refcounted views of the tensors; optimizers registered via
        /// `registerAllParams` borrow both, so the trainer must outlive them.
        registry: ParamRegistry,
        /// RoPE tables keyed by (position offset, sequence length),
        /// heap-pinned and NEVER freed before `deinit`: checkpoint backward
        /// nodes hold raw pointers into this cache (`LayerExtra.rope_table`)
        /// and dereference them at recompute time, possibly after several
        /// intervening forwards. Positions are offset..offset+seq-1 (offset 0
        /// for plain training; a cartridge forward shifts tokens to offset p),
        /// so a table per key is immutable and reusable; memory is bounded by
        /// the number of DISTINCT keys seen (one [seq, head_dim] f32 sin/cos
        /// pair each).
        rope_tables: std.AutoHashMapUnmanaged(RopeKey, *fucina.RopeTable) = .empty,
        /// Per-forward rope tables for PACKED forwards (positions restart per
        /// segment, so a shared-length cache key does not exist). Appended by
        /// every packed forward and freed only by `freeTransientRope`/`deinit`
        /// — the rope backward reads the table, so free strictly BETWEEN an
        /// optimizer step and the next packed forward, never between a
        /// forward and its backward.
        transient_tables: std.ArrayListUnmanaged(*fucina.RopeTable) = .empty,
        /// Advances once per `loss` call; selects the dropout seed stream.
        step_counter: u64 = 0,
        /// Lazily built f32 view/copy of the frozen output projection for
        /// the fused distillation loss ([vocab, embed]): f32 heads retag a
        /// borrowed view, f16/bf16 heads widen ONCE (the copy is shared by
        /// every subsequent step); heads with no f32 route (quantized /
        /// ptqtp) stay unavailable and `distillLoss` serves the composed
        /// path. Constants — safe to build under an exec scope.
        fused_head: ?fucina.Tensor(.{ .vocab, .embed }) = null,
        fused_head_state: enum { unknown, ready, unavailable } = .unknown,
        /// Recompute-in-backward per layer (one checkpoint block per layer).
        checkpoint_layers: bool = false,

        const Self = @This();

        fn enabled(comptime t: usize) bool {
            return @field(targets, target_names[t]);
        }

        fn TargetAdapter(comptime t: usize) type {
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

        /// {in_dim, out_dim} of target `t`'s frozen projection.
        fn targetDims(cfg: qwen3.Config, comptime t: usize) [2]usize {
            const q_dim = cfg.num_attention_heads * cfg.head_dim;
            return switch (t) {
                0 => .{ cfg.hidden_size, q_dim },
                1, 2 => .{ cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim },
                3 => .{ q_dim, cfg.hidden_size },
                4, 5 => .{ cfg.hidden_size, cfg.intermediate_size },
                6 => .{ cfg.intermediate_size, cfg.hidden_size },
                else => unreachable,
            };
        }

        pub const n_enabled = blk: {
            var n: usize = 0;
            for (0..n_targets) |t| {
                if (enabled(t)) n += 1;
            }
            break :blk n;
        };

        /// Position of target `t`'s A tensor within the A/B tuple (B is +1).
        fn abIndex(comptime t: usize) usize {
            comptime {
                var j: usize = 0;
                for (0..t) |i| {
                    if (enabled(i)) j += 2;
                }
                return j;
            }
        }

        pub const LayerAdapters = struct {
            q: if (targets.q) TargetAdapter(0) else void,
            k: if (targets.k) TargetAdapter(1) else void,
            v: if (targets.v) TargetAdapter(2) else void,
            o: if (targets.o) TargetAdapter(3) else void,
            gate: if (targets.gate) TargetAdapter(4) else void,
            up: if (targets.up) TargetAdapter(5) else void,
            down: if (targets.down) TargetAdapter(6) else void,
        };

        const ab_ptr_types = blk: {
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

        /// The enabled adapters' A/B pointers, in target order — the shared
        /// currency of the plain and checkpointed layer paths.
        const AbTuple = std.meta.Tuple(&ab_ptr_types);

        const InputsTuple = std.meta.Tuple(&([_]type{*const Hidden} ++ ab_ptr_types));

        fn AbPtr(comptime i: usize) type {
            return ab_ptr_types[i];
        }

        pub fn init(ctx: *ExecContext, model: *const qwen3.Model, config: lora.Config, seed: u64) !Self {
            if (model.config.isMoe()) return Error.MoeUnsupported;
            const allocator = ctx.allocator;
            const n_layers = model.config.num_layers;

            const adapters = try allocator.alloc(LayerAdapters, n_layers);
            errdefer allocator.free(adapters);
            var built_layers: usize = 0;
            errdefer for (adapters[0..built_layers]) |*ads| deinitLayerAdaptersPartial(ads, n_enabled);
            for (adapters, 0..) |*ads, layer_i| {
                try initLayerAdapters(ctx, ads, model.config, config, seed, layer_i);
                built_layers += 1;
            }

            var registry = ParamRegistry.init(allocator);
            errdefer registry.deinit();
            for (adapters, 0..) |*ads, layer_i| {
                inline for (0..n_targets) |t| {
                    if (comptime enabled(t)) {
                        const ad = &@field(ads.*, target_names[t]);
                        const ab = .{ &ad.a, &ad.b };
                        inline for ([2][]const u8{ "lora_a", "lora_b" }, 0..) |suffix, which| {
                            const name = try std.fmt.allocPrint(
                                allocator,
                                "layers.{d}.{s}.{s}",
                                .{ layer_i, target_names[t], suffix },
                            );
                            defer allocator.free(name);
                            try registry.addParam(name, ab[which]);
                        }
                    }
                }
            }
            std.debug.assert(registry.parameterCount() == n_layers * n_enabled * 2);

            return .{
                .model = model,
                .allocator = allocator,
                .lora_config = config,
                .scale = config.alpha / @as(f32, @floatFromInt(config.rank)),
                .seed = seed,
                .adapters = adapters,
                .registry = registry,
            };
        }

        /// Free the packed-forward rope tables (see `transient_tables`).
        /// Call between optimizer steps; the tables of an un-backwarded
        /// packed forward must stay alive.
        pub fn freeTransientRope(self: *Self) void {
            for (self.transient_tables.items) |table| {
                table.deinit();
                self.allocator.destroy(table);
            }
            self.transient_tables.clearRetainingCapacity();
        }

        pub fn deinit(self: *Self) void {
            if (self.fused_head) |*head| head.deinit();
            self.freeTransientRope();
            self.transient_tables.deinit(self.allocator);
            var tables = self.rope_tables.valueIterator();
            while (tables.next()) |table| {
                table.*.deinit();
                self.allocator.destroy(table.*);
            }
            self.rope_tables.deinit(self.allocator);
            // Registry first: it retains views of the adapters' storage and
            // their GradState pointers, both torn down just below.
            self.registry.deinit();
            for (self.adapters) |*ads| deinitLayerAdaptersPartial(ads, n_enabled);
            self.allocator.free(self.adapters);
            self.* = undefined;
        }

        fn initLayerAdapters(
            ctx: *ExecContext,
            ads: *LayerAdapters,
            model_config: qwen3.Config,
            config: lora.Config,
            seed: u64,
            layer_i: usize,
        ) !void {
            var built: usize = 0;
            errdefer deinitLayerAdaptersPartial(ads, built);
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    const dims = targetDims(model_config, t);
                    @field(ads.*, target_names[t]) = try TargetAdapter(t).init(
                        ctx,
                        dims[0],
                        dims[1],
                        config,
                        rng.at(seed, layer_i * n_targets + t),
                    );
                    built += 1;
                }
            }
        }

        /// Deinit the first `built` enabled adapters (full teardown at
        /// `built == n_enabled`); the partial form serves init error paths.
        fn deinitLayerAdaptersPartial(ads: *LayerAdapters, built: usize) void {
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    const ordinal = comptime abIndex(t) / 2;
                    if (ordinal < built) @field(ads.*, target_names[t]).deinit();
                }
            }
        }

        /// Register every adapter A/B on `opt` (anything with `addParamNamed`)
        /// under the "layers.<i>.<target>" names. The trainer must outlive the
        /// optimizer (params and names are borrowed).
        pub fn registerAllParams(self: *Self, opt: anytype) !void {
            try self.registry.addParamsTo(opt);
        }

        /// Serialize all adapters as a clean safetensors state dict
        /// (name-matched on load, so target sets may be saved/loaded across
        /// trainers with the same `targets` and shapes). Trainer resume state
        /// such as `step_counter` belongs in the checkpoint directory's
        /// `trainer_state.json`, not in this portable tensor payload.
        pub fn saveAdapters(self: *const Self, writer: *std.Io.Writer) !void {
            try self.registry.saveStateDict(writer);
        }

        /// Load adapters saved by `saveAdapters` (strict: one-to-one match).
        pub fn loadAdapters(self: *Self, reader: *std.Io.Reader) !void {
            try self.loadAdaptersWithOptions(reader, .{});
        }

        /// `loadAdapters` with explicit `LoadOptions` — e.g. an `aliases`
        /// map to resume from a checkpoint written under older adapter paths.
        pub fn loadAdaptersWithOptions(self: *Self, reader: *std.Io.Reader, options: optim.LoadOptions) !void {
            try self.registry.loadStateDict(reader, options);
        }

        /// Mean cross-entropy over `tokens` (the model inputs) against
        /// `labels` (pre-shifted next tokens; `ignore_index` masks positions).
        /// MUST run under an open exec scope; the result is scope-owned.
        /// Advances the step counter (one dropout stream per call).
        /// Equivalent to `lossExt(ctx, tokens, labels, .{})`.
        pub fn loss(self: *Self, ctx: *ExecContext, tokens: []const usize, labels: []const usize) !fucina.Tensor(.{}) {
            return self.lossExt(ctx, tokens, labels, .{});
        }

        /// Loss knobs for gradient accumulation (defaults reproduce `loss`
        /// exactly). Backward WITHOUT `zeroGrad` ADDS into each param's
        /// persisted gradient, so N micro-batch lossExt+backward passes
        /// followed by ONE clip/step/zeroGrad implement an N-sequence batch.
        /// Normalize on the loss side: `.mean` + `loss_scale = 1.0/N` is
        /// mean-of-means (the true batch mean only for equal supervised-token
        /// counts); `.sum` + `loss_scale = 1.0/total_valid` (valid = labels
        /// != `ignore_index` across the window) is the exact token-weighted
        /// mean. See docs/TRAINING.md §4 "Gradient accumulation".
        pub const LossOptions = struct {
            /// CE reduction over the non-ignored positions.
            reduction: enum { mean, sum } = .mean,
            /// Multiplies the returned loss (and thus the gradients) via the
            /// differentiable `scale` op when != 1.
            loss_scale: f32 = 1,
        };

        /// `loss` with explicit reduction/scale (same exec-scope requirement,
        /// scope-owned result, and one step-counter advance per call).
        pub fn lossExt(self: *Self, ctx: *ExecContext, tokens: []const usize, labels: []const usize, options: LossOptions) !fucina.Tensor(.{}) {
            if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
            if (labels.len != tokens.len) return Error.LabelLengthMismatch;
            const step = self.step_counter;
            self.step_counter += 1;
            var logits = try self.forwardLogits(ctx, tokens, step);
            defer logits.deinit();
            return ceTail(ctx, &logits, labels, options);
        }

        /// `lossExt` with full `ForwardOptions` — the graft/cartridge loss
        /// entry: same CE tail, exec-scope requirement, scope-owned result,
        /// and one step-counter advance per call. Gradients flow into
        /// whatever the options attach (engram parameters, cartridge rows,
        /// LoRA adapters) through the frozen stack.
        pub fn lossForwardExt(self: *Self, ctx: *ExecContext, tokens: []const usize, labels: []const usize, fwd: ForwardOptions, options: LossOptions) !fucina.Tensor(.{}) {
            if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
            if (labels.len != tokens.len) return Error.LabelLengthMismatch;
            const step = self.step_counter;
            self.step_counter += 1;
            var x = try self.forwardHiddenImpl(ctx, tokens, step, fwd);
            defer x.deinit();
            var logits = try self.logitsTail(ctx, &x);
            defer logits.deinit();
            return ceTail(ctx, &logits, labels, options);
        }

        /// `lossExt` with a single-row embedding injection: the full-depth
        /// forward substitutes `injection.row` at `injection.pos` before the
        /// first layer, then applies the identical norm/projection/CE tail
        /// (same reduction/scale plumbing, same exec-scope requirement,
        /// scope-owned result, and one step-counter advance per call). When
        /// `injection.row` is a variable, backward accumulates its gradient —
        /// the CE gradient flows through the frozen stack into the row.
        pub fn lossInjected(self: *Self, ctx: *ExecContext, tokens: []const usize, labels: []const usize, injection: Injection, options: LossOptions) !fucina.Tensor(.{}) {
            if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
            if (labels.len != tokens.len) return Error.LabelLengthMismatch;
            const step = self.step_counter;
            self.step_counter += 1;
            var hidden = try self.forwardHiddenImpl(ctx, tokens, step, .{ .inject = injection });
            defer hidden.deinit();
            var logits = try self.logitsTail(ctx, &hidden);
            defer logits.deinit();
            return ceTail(ctx, &logits, labels, options);
        }

        /// The CE tail shared by `lossExt` and `lossInjected` (ignore_index
        /// masking, runtime reduction over the two comptime CE instantiations,
        /// optional differentiable scale).
        fn ceTail(ctx: *ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab }), labels: []const usize, options: LossOptions) !fucina.Tensor(.{}) {
            var ce = switch (options.reduction) {
                .mean => try logits.crossEntropyExt(ctx, .vocab, labels, .{
                    .ignore_index = ignore_index,
                    .reduction = .mean,
                }),
                .sum => try logits.crossEntropyExt(ctx, .vocab, labels, .{
                    .ignore_index = ignore_index,
                    .reduction = .sum,
                }),
            };
            if (options.loss_scale == 1) return ce;
            defer ce.deinit(); // scope-owned: a safe no-op, the graph survives
            return ce.scale(ctx, options.loss_scale);
        }

        /// Eval forward (dropout off, no step advance): the last position's
        /// logits as a caller-owned [1, vocab] constant. Runs under its own
        /// exec scope, so it composes with any surrounding training loop.
        pub fn evalLastLogits(self: *Self, ctx: *ExecContext, tokens: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
            return self.evalLastLogitsExt(ctx, tokens, .{});
        }

        /// Eval forward (dropout off, no step advance): the FULL sequence's
        /// logits as a caller-owned [seq, vocab] constant — the
        /// teacher-forced seam for scoring whole responses in one pass
        /// (token accuracy, likelihood rewards for gradient-free training).
        /// Runs under its own exec scope like `evalLastLogits`; the returned
        /// copy is ~seq x vocab x 4 bytes, so keep sequences short-ish.
        pub fn evalLogits(self: *Self, ctx: *ExecContext, tokens: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var hidden = try self.forwardHiddenImpl(ctx, tokens, null, .{});
            defer hidden.deinit();
            var logits = try self.logitsTail(ctx, &hidden);
            defer logits.deinit();
            // Deep-copy out of the scope: everything else dies at close.
            var value = try logits.value.clone(ctx.allocator);
            errdefer value.deinit();
            return fucina.Tensor(.{ .seq, .vocab }).fromTensor(ctx, value);
        }

        /// `evalLogits` with `ForwardOptions` — the cartridge student/teacher
        /// seam: pass `.cartridge` to score tokens behind a trained KV prefix
        /// (dropout off, no step advance, caller-owned [seq, vocab] constant).
        pub fn evalLogitsExt(self: *Self, ctx: *ExecContext, tokens: []const usize, opts: ForwardOptions) !fucina.Tensor(.{ .seq, .vocab }) {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var hidden = try self.forwardHiddenImpl(ctx, tokens, null, opts);
            defer hidden.deinit();
            var logits = try self.logitsTail(ctx, &hidden);
            defer logits.deinit();
            // Deep-copy out of the scope: everything else dies at close.
            var value = try logits.value.clone(ctx.allocator);
            errdefer value.deinit();
            return fucina.Tensor(.{ .seq, .vocab }).fromTensor(ctx, value);
        }

        /// `evalLogitsExt` restricted to the logits of `rows` (residual-row
        /// indices into `tokens`): the memory-bounded teacher-scoring seam —
        /// a [rows.len, vocab] copy instead of the full [seq, vocab] block
        /// (distillation only reads the rows preceding supervised tokens).
        pub fn evalLogitsRows(self: *Self, ctx: *ExecContext, tokens: []const usize, rows: []const usize, opts: ForwardOptions) !fucina.Tensor(.{ .seq, .vocab }) {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var hidden = try self.forwardHiddenImpl(ctx, tokens, null, opts);
            defer hidden.deinit();
            var picked = try hidden.gather(ctx, .seq, rows, .seq);
            defer picked.deinit();
            var logits = try self.logitsTail(ctx, &picked);
            defer logits.deinit();
            // Deep-copy out of the scope: everything else dies at close.
            var value = try logits.value.clone(ctx.allocator);
            errdefer value.deinit();
            return fucina.Tensor(.{ .seq, .vocab }).fromTensor(ctx, value);
        }

        /// Final-norm LAST hidden state of `tokens`, copied into `out`
        /// (`hidden_size` floats) — the retrieval-embedding primitive of
        /// cartridge fleets (`llm/cartridge_fleet.zig`: callers append the
        /// ids of `cartridge_fleet.embed_suffix` to the text ids first, and
        /// the index normalizes/centers downstream). Eval pass: dropout
        /// off, no step advance; runs under its own exec scope.
        pub fn embedLastHidden(self: *Self, ctx: *ExecContext, tokens: []const usize, out: []f32) !void {
            if (out.len != self.model.config.hidden_size) return Error.InvalidSequenceLength;
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var hidden = try self.forwardHiddenImpl(ctx, tokens, null, .{});
            defer hidden.deinit();
            const model = self.model;
            var normed = try hidden.rmsNormMul(ctx, .embed, &model.output_norm, model.config.rms_norm_eps);
            defer normed.deinit();
            var last = try normed.narrow(ctx, .seq, normed.dim(.seq) - 1, 1);
            defer last.deinit();
            @memcpy(out, try last.dataConst());
        }

        /// One eval forward over `tokens` (positions 0..len-1, no cartridge)
        /// that copies every layer's token K/V rows out of the graph — the
        /// cartridge initialization capture. Caller owns the result.
        pub fn captureKv(self: *Self, ctx: *ExecContext, tokens: []const usize) !KvCapture {
            const cfg = self.model.config;
            var cap = try KvCapture.init(
                self.allocator,
                cfg.num_layers,
                tokens.len * cfg.num_key_value_heads * cfg.head_dim,
            );
            errdefer cap.deinit();
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var hidden = try self.forwardHiddenImpl(ctx, tokens, null, .{ .capture = &cap });
            hidden.deinit(); // scope-owned borrow: safe no-op
            return cap;
        }

        /// Build a cartridge initialized from the model's OWN K/V rows for
        /// `tokens` at positions 0..p-1 (p = tokens.len) — the paper's
        /// winning "first p corpus tokens" initialization. `frozen_prefix`
        /// rows stay constant (1 = the attention-sink freeze). With zero
        /// training steps the result is behaviorally identical to actually
        /// prefilling `tokens` (see the equivalence test). Caller owns the
        /// cartridge; create it OUTSIDE any exec scope.
        pub fn initCartridge(self: *Self, ctx: *ExecContext, tokens: []const usize, frozen_prefix: usize) !cartridge_mod.Cartridge {
            var cap = try self.captureKv(ctx, tokens);
            defer cap.deinit();
            const cfg = self.model.config;
            const k_rows = try self.allocator.alloc([]const f32, cap.k_rows.len);
            defer self.allocator.free(k_rows);
            const v_rows = try self.allocator.alloc([]const f32, cap.v_rows.len);
            defer self.allocator.free(v_rows);
            for (k_rows, cap.k_rows) |*dst, src| dst.* = src;
            for (v_rows, cap.v_rows) |*dst, src| dst.* = src;
            return cartridge_mod.Cartridge.initFromRows(
                ctx,
                self.allocator,
                frozen_prefix,
                tokens.len,
                cfg.num_key_value_heads,
                cfg.head_dim,
                k_rows,
                v_rows,
            );
        }

        /// Cartridge training step loss: the teacher top-k distillation
        /// objective (`cartridge.distillLoss`) over this model's logits for
        /// `tokens` computed BEHIND `cart` (tokens at positions p..). Same
        /// exec-scope requirement, scope-owned result, and step-counter
        /// advance as `loss`; backward routes gradients into the cartridge's
        /// trainable rows only (the base model stays frozen — instantiate
        /// `Trainer(.{ .q = false, .v = false })` for pure cartridge
        /// training).
        /// `packed_segments` (lengths summing to `tokens.len`) trains several
        /// conversations in ONE forward/backward — target positions index
        /// the packed row, and the `.mean` reduction over all entries is the
        /// reference's packed-batch objective. Pair packed forwards with
        /// `freeTransientRope()` between optimizer steps.
        pub fn distillLoss(
            self: *Self,
            ctx: *ExecContext,
            tokens: []const usize,
            cart: *const cartridge_mod.Cartridge,
            distill_targets: cartridge_mod.DistillTargets,
            packed_segments: ?[]const usize,
            options: cartridge_mod.DistillOptions,
        ) !fucina.Tensor(.{}) {
            return self.distillLossExt(ctx, tokens, .{ .cartridge = cart, .packed_segments = packed_segments }, distill_targets, options);
        }

        /// `distillLoss` with full `ForwardOptions` — the composed-prefix
        /// entry (Cartridges at Scale): pass `.cartridges` to train several
        /// cartridges JOINTLY behind one forward (backward accumulates into
        /// every part's trainable rows), with `.packed_segments` batching
        /// conversations that share the same visibility set. Same exec-scope
        /// requirement, scope-owned result, and step-counter advance.
        pub fn distillLossExt(
            self: *Self,
            ctx: *ExecContext,
            tokens: []const usize,
            fwd: ForwardOptions,
            distill_targets: cartridge_mod.DistillTargets,
            options: cartridge_mod.DistillOptions,
        ) !fucina.Tensor(.{}) {
            if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
            const step = self.step_counter;
            self.step_counter += 1;
            var hidden = try self.forwardHiddenImpl(ctx, tokens, step, fwd);
            defer hidden.deinit();
            if (fusedDistillEnabled()) {
                if (self.fusedDistillHead(ctx)) |head| {
                    return self.distillLossFusedTail(ctx, &hidden, head, distill_targets, options);
                }
            }
            var logits = try self.logitsTail(ctx, &hidden);
            defer logits.deinit();
            return cartridge_mod.distillLoss(ctx, &logits, distill_targets, options);
        }

        /// The fused tail of `distillLoss`: final norm, then
        /// `linearDistillExt` — the output projection and the sparse
        /// teacher targets as ONE op, so the [seq, vocab] logits (and
        /// their log-softmax) never enter the graph and only the
        /// supervised rows are ever projected. Same objective as the
        /// composed tail (`cartridge.distillLoss` documents it); the two
        /// routes agree to f32 roundoff, pinned by a trainer test.
        fn distillLossFusedTail(
            self: *Self,
            ctx: *ExecContext,
            hidden: *const Hidden,
            head: *const fucina.Tensor(.{ .vocab, .embed }),
            distill_targets: cartridge_mod.DistillTargets,
            options: cartridge_mod.DistillOptions,
        ) !fucina.Tensor(.{}) {
            const model = self.model;
            var normed = try hidden.rmsNormMul(ctx, .embed, &model.output_norm, model.config.rms_norm_eps);
            defer normed.deinit();
            const n = distill_targets.positions.len;
            if (n == 0 or distill_targets.tokens.len != n or distill_targets.logprobs.len != n) return cartridge_mod.Error.InvalidTargets;
            const seq = normed.dim(.seq);
            const rows = try ctx.allocator.alloc(usize, n);
            defer ctx.allocator.free(rows);
            const probs = try ctx.allocator.alloc(f32, n);
            defer ctx.allocator.free(probs);
            for (rows, probs, distill_targets.positions, distill_targets.tokens, distill_targets.logprobs) |*row, *prob, pos, token, logprob| {
                if (pos == 0 or pos > seq) return cartridge_mod.Error.InvalidTargets;
                if (token >= model.config.vocab_size) return cartridge_mod.Error.InvalidTargets;
                row.* = pos - 1;
                prob.* = @exp(logprob);
            }
            return normed.linearDistillExt(ctx, head, rows, distill_targets.tokens, probs, .{
                .reduction = switch (options.reduction) {
                    .mean => .mean,
                    .sum => .sum,
                },
                .loss_scale = options.loss_scale,
            });
        }

        /// Get-or-build the f32 head for the fused distillation tail (see
        /// the `fused_head` field). Null = no f32 route for this head
        /// format; the caller falls back to the composed tail. Built with a
        /// deep copy OUT of any active exec scope (the evalLastLogitsExt
        /// pattern) — the head must outlive the step's scope.
        fn fusedDistillHead(self: *Self, ctx: *ExecContext) ?*const fucina.Tensor(.{ .vocab, .embed }) {
            if (self.fused_head_state == .unknown) {
                self.fused_head_state = .unavailable;
                switch (self.model.output) {
                    .f32 => |*w| self.adoptFusedHead(ctx, &w.value),
                    .f16 => |*w| self.widenFusedHead(ctx, w),
                    .bf16 => |*w| self.widenFusedHead(ctx, w),
                    else => {},
                }
            }
            return if (self.fused_head_state == .ready) &self.fused_head.? else null;
        }

        fn widenFusedHead(self: *Self, ctx: *ExecContext, w: anytype) void {
            var wide = w.to(ctx, .f32) catch return;
            defer wide.deinit();
            self.adoptFusedHead(ctx, &wide.value);
        }

        fn adoptFusedHead(self: *Self, ctx: *ExecContext, raw: anytype) void {
            var value = raw.clone(ctx.allocator) catch return;
            if (fucina.Tensor(.{ .vocab, .embed }).fromTensor(ctx, value)) |head| {
                self.fused_head = head;
                self.fused_head_state = .ready;
            } else |_| {
                value.deinit();
            }
        }

        /// `evalLastLogits` with `ForwardOptions` (injection / layer range) —
        /// the generation entry for injected prompts. NOTE with a truncated
        /// range the final norm + output projection still apply, just to the
        /// truncated residual; injected tensors passed via `opts` must be
        /// caller-owned (this runs under its own inner exec scope).
        pub fn evalLastLogitsExt(self: *Self, ctx: *ExecContext, tokens: []const usize, opts: ForwardOptions) !fucina.Tensor(.{ .seq, .vocab }) {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var hidden = try self.forwardHiddenImpl(ctx, tokens, null, opts);
            defer hidden.deinit();
            var logits = try self.logitsTail(ctx, &hidden);
            defer logits.deinit();
            var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
            defer last.deinit();
            // Deep-copy out of the scope: everything else dies at close.
            var value = try last.value.clone(ctx.allocator);
            errdefer value.deinit();
            return fucina.Tensor(.{ .seq, .vocab }).fromTensor(ctx, value);
        }

        /// Full-sequence trainable forward: embedding (frozen constant) →
        /// layers (plain or checkpointed, same math) → final norm → logits
        /// via the frozen output projection. `step` selects the dropout
        /// stream; null disables dropout (eval).
        fn forwardLogits(self: *Self, ctx: *ExecContext, tokens: []const usize, step: ?u64) !fucina.Tensor(.{ .seq, .vocab }) {
            var x = try self.forwardHiddenImpl(ctx, tokens, step, .{});
            defer x.deinit();
            return self.logitsTail(ctx, &x);
        }

        /// The norm + output-projection tail of `forwardLogits`, factored so
        /// the logits paths (`loss*`, `evalLastLogits*`) share one layer-stack
        /// body with the raw-residual `forwardHidden`.
        fn logitsTail(self: *Self, ctx: *ExecContext, hidden: *const Hidden) !fucina.Tensor(.{ .seq, .vocab }) {
            const model = self.model;
            var normed = try hidden.rmsNormMul(ctx, .embed, &model.output_norm, model.config.rms_norm_eps);
            defer normed.deinit();
            return dotLinear(&model.output, ctx, &normed, .embed, .vocab);
        }

        /// Raw-residual trainable forward: embedding (frozen constant) →
        /// optional single-row injection → layers [start_layer, start_layer +
        /// layer_count) → the RAW residual stream (no output norm, no output
        /// projection). `step` selects the dropout stream like `forwardLogits`
        /// (null = eval; the step counter is NOT advanced — callers that want
        /// per-call dropout streams manage it themselves).
        ///
        /// MUST run inside an open exec scope (like `loss`): the result and
        /// every intermediate are scope-owned borrows, so gradients survive
        /// until `backward()` and values must be copied out before the scope
        /// closes.
        pub fn forwardHidden(self: *Self, ctx: *ExecContext, tokens: []const usize, step: ?u64, opts: ForwardOptions) !Hidden {
            if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
            return self.forwardHiddenImpl(ctx, tokens, step, opts);
        }

        /// The shared layer-stack body (see `forwardHidden`). Callers own the
        /// scope discipline: `loss*` demands an active scope, `evalLastLogits*`
        /// opens its own.
        fn forwardHiddenImpl(self: *Self, ctx: *ExecContext, tokens: []const usize, step: ?u64, opts: ForwardOptions) !Hidden {
            if (tokens.len == 0) return Error.InvalidSequenceLength;
            const model = self.model;
            const cfg = model.config;
            if (cfg.isMoe()) return Error.MoeUnsupported;
            const n_layers = model.layers.len;
            if (opts.start_layer > n_layers) return Error.InvalidLayerRange;
            const layer_count = opts.layer_count orelse (n_layers - opts.start_layer);
            if (layer_count > n_layers - opts.start_layer) return Error.InvalidLayerRange;
            if (opts.inject) |inj| {
                if (inj.pos >= tokens.len or inj.row.dim(.seq) > tokens.len - inj.pos) return Error.InvalidInjection;
                if (inj.row.dim(.embed) != cfg.hidden_size) return Error.InvalidInjection;
            }
            if (opts.cartridge) |cart| {
                if (self.checkpoint_layers) return Error.CartridgeCheckpointUnsupported;
                if (opts.cartridges != null) return Error.InvalidCartridge;
                if (cart.layers.len != n_layers) return Error.InvalidCartridge;
                if (cart.kv_heads != cfg.num_key_value_heads or cart.head_dim != cfg.head_dim) return Error.InvalidCartridge;
            }
            if (opts.cartridges) |parts| {
                if (self.checkpoint_layers) return Error.CartridgeCheckpointUnsupported;
                if (parts.len == 0) return Error.InvalidCartridge;
                for (parts) |cart| {
                    if (cart.layers.len != n_layers) return Error.InvalidCartridge;
                    if (cart.kv_heads != cfg.num_key_value_heads or cart.head_dim != cfg.head_dim) return Error.InvalidCartridge;
                }
            }
            if (opts.capture) |cap| {
                if (self.checkpoint_layers) return Error.CartridgeCheckpointUnsupported;
                if (cap.k_rows.len != n_layers or cap.v_rows.len != n_layers) return Error.InvalidCapture;
                const row_len = tokens.len * cfg.num_key_value_heads * cfg.head_dim;
                for (cap.k_rows, cap.v_rows) |k, v| {
                    if (k.len != row_len or v.len != row_len) return Error.InvalidCapture;
                }
            }

            if (opts.engram) |eng| {
                if (self.checkpoint_layers) return Error.CartridgeCheckpointUnsupported;
                if (opts.packed_segments != null) return Error.InvalidPacking;
                const ecfg = &eng.model.plan.cfg;
                if (ecfg.hc_mult != 1 or ecfg.hidden_size != cfg.hidden_size) return Error.InvalidEngram;
                if (eng.rows.len != eng.model.layers.len) return Error.InvalidEngram;
                const want = tokens.len * ecfg.headsPerLayer();
                for (eng.rows) |layer_rows| {
                    if (layer_rows.len != want) return Error.InvalidEngram;
                }
                for (eng.model.plan.layer_ids) |id| {
                    if (id >= n_layers) return Error.InvalidEngram;
                }
            }

            if (opts.packed_segments) |seg_lens| {
                if (self.checkpoint_layers) return Error.CartridgeCheckpointUnsupported;
                if (opts.inject != null or opts.capture != null) return Error.InvalidPacking;
                if (seg_lens.len == 0) return Error.InvalidPacking;
                var total: usize = 0;
                for (seg_lens) |len| {
                    if (len == 0) return Error.InvalidPacking;
                    total += len;
                }
                if (total != tokens.len) return Error.InvalidPacking;
            }

            // A cartridge occupies positions 0..p-1; real tokens start at p
            // (a composition occupies 0..sum(p_i)-1).
            const position_offset = if (opts.cartridges) |parts|
                cartridge_mod.composedP(parts)
            else if (opts.cartridge) |cart|
                cart.p
            else
                0;
            const rope_table = if (opts.packed_segments) |seg_lens|
                try self.preparePackedRope(ctx, position_offset, tokens.len, seg_lens)
            else
                try self.prepareRope(ctx, position_offset, tokens.len);

            var x = try model.token_embedding.getRowsAs(ctx, tokens, .embed);
            errdefer x.deinit();

            if (opts.inject) |inj| {
                x = try ctx.replace(x, x.setSlice(ctx, .seq, inj.pos, inj.row));
            }

            for (model.layers[opts.start_layer..][0..layer_count], opts.start_layer..) |*layer, layer_i| {
                if (opts.engram) |eng| {
                    if (eng.model.plan.slotOf(layer_i)) |slot| {
                        // hidden += engram(hidden, rows): reference block
                        // order (before attention). Zero-copy retags bridge
                        // the trainer's .embed tag to the module's .d.
                        var q = try x.withTags(ctx, .{ .seq, .d });
                        defer q.deinit();
                        var mem = try eng.model.layers[slot].forwardResidual(ctx, &q, eng.rows[slot], null);
                        defer mem.deinit();
                        var mem_e = try mem.withTags(ctx, .{ .seq, .embed });
                        defer mem_e.deinit();
                        x = try ctx.replace(x, x.add(ctx, &mem_e));
                    }
                }
                const extra = LayerExtra{
                    .layer = layer,
                    .config = cfg,
                    .rope_table = rope_table,
                    .kv_head_for_head = model.kv_head_for_head,
                    .seeds = self.layerSeeds(step, layer_i),
                    .scale = self.scale,
                    .dropout_p = self.lora_config.dropout_p,
                    .layer_i = layer_i,
                    .cartridge_layer = if (opts.cartridge) |cart| &cart.layers[layer_i] else null,
                    .cartridge_parts = opts.cartridges,
                    .capture = opts.capture,
                    .packed_segments = opts.packed_segments,
                };
                const ads = &self.adapters[layer_i];
                if (self.checkpoint_layers) {
                    x = try ctx.replace(x, fucina.checkpointWithContext(ctx, LayerBlock.run, extra, checkpointInputs(&x, ads)));
                } else {
                    x = try ctx.replace(x, layerBody(ctx, extra, &x, abTuple(ads)));
                }
            }
            return x;
        }

        /// Rope table for a PACKED forward: positions restart at `offset`
        /// for every segment. Uncacheable (keyed by the whole length
        /// vector), so it goes on the transient list — alive until
        /// `freeTransientRope`.
        fn preparePackedRope(self: *Self, ctx: *ExecContext, offset: usize, total_len: usize, seg_lens: []const usize) !*const fucina.RopeTable {
            const cfg = self.model.config;
            const positions = try ctx.allocator.alloc(i32, total_len);
            defer ctx.allocator.free(positions);
            var idx: usize = 0;
            for (seg_lens) |len| {
                for (0..len) |j| {
                    positions[idx] = @intCast(offset + j);
                    idx += 1;
                }
            }

            const fresh = try self.allocator.create(fucina.RopeTable);
            errdefer self.allocator.destroy(fresh);
            fresh.* = try ctx.prepareRopeTable(positions, cfg.head_dim, cfg.rope_theta, false);
            errdefer fresh.deinit();
            try self.transient_tables.append(self.allocator, fresh);
            return fresh;
        }

        /// The trainer-owned RoPE table for positions offset..offset+seq-1:
        /// cached per (offset, length), built on first use, freed only in
        /// `deinit` (see the `rope_tables` field for why tables must never be
        /// freed earlier). Repeat-key forwards reuse the cached table.
        fn prepareRope(self: *Self, ctx: *ExecContext, offset: usize, seq_len: usize) !*const fucina.RopeTable {
            const key = RopeKey{ .offset = offset, .len = seq_len };
            if (self.rope_tables.get(key)) |table| return table;

            const cfg = self.model.config;
            const positions = try ctx.allocator.alloc(i32, seq_len);
            defer ctx.allocator.free(positions);
            for (positions, 0..) |*position, i| position.* = @intCast(offset + i);

            const fresh = try self.allocator.create(fucina.RopeTable);
            errdefer self.allocator.destroy(fresh);
            fresh.* = try ctx.prepareRopeTable(positions, cfg.head_dim, cfg.rope_theta, false);
            errdefer fresh.deinit();

            try self.rope_tables.put(self.allocator, key, fresh);
            return fresh;
        }

        /// "drop" domain separator so dropout streams never collide with the
        /// adapter-init streams derived from the same base seed.
        const dropout_domain: u64 = 0x64726f70_64726f70;

        fn layerSeeds(self: *const Self, step: ?u64, layer_i: usize) [n_targets]?u64 {
            var seeds: [n_targets]?u64 = [1]?u64{null} ** n_targets;
            const s = step orelse return seeds;
            const base = (s * @as(u64, self.model.config.num_layers) + layer_i) * n_targets;
            for (&seeds, 0..) |*seed, t| seed.* = rng.at(self.seed ^ dropout_domain, base + t);
            return seeds;
        }

        fn abTuple(ads: *const LayerAdapters) AbTuple {
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

        fn checkpointInputs(x: *const Hidden, ads: *const LayerAdapters) InputsTuple {
            var inputs: InputsTuple = undefined;
            inputs[0] = x;
            comptime var j: usize = 1;
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    inputs[j] = &@field(ads.*, target_names[t]).a;
                    inputs[j + 1] = &@field(ads.*, target_names[t]).b;
                    j += 2;
                }
            }
            return inputs;
        }

        /// A borrow-adapter over checkpoint-supplied A/B tensors: same
        /// scale/dropout as the trainer's adapters, gradients route to
        /// whatever GradStates the borrowed tensors carry.
        fn tempAdapter(comptime t: usize, abs: AbTuple, extra: LayerExtra) TargetAdapter(t) {
            const j = comptime abIndex(t);
            return .{
                .a = abs[j].*,
                .b = abs[j + 1].*,
                .scale = extra.scale,
                .dropout_p = extra.dropout_p,
            };
        }

        /// base + LoRA delta when target `t` is enabled; otherwise a zero-copy
        /// view of base (uniform ownership for the caller's defer).
        fn adapted(
            comptime t: usize,
            ctx: *ExecContext,
            abs: AbTuple,
            extra: LayerExtra,
            x: anytype,
            base: anytype,
        ) !@TypeOf(base.*) {
            if (comptime !enabled(t)) {
                return base.withTags(ctx, @TypeOf(base.*).axis_tags);
            }
            const ad = tempAdapter(t, abs, extra);
            var d = try ad.delta(ctx, x, extra.seeds[t]);
            defer d.deinit();
            return base.add(ctx, &d);
        }

        /// One transformer layer, op-for-op the inference attentionBlock +
        /// ffnBlock (full sequence, no KV cache, no last-query shortcut) with
        /// LoRA deltas added to the enabled projections. Runs under an exec
        /// scope on every path (training scope, eval scope, or the checkpoint
        /// inner scope), so defer-deinit is a safe no-op throughout.
        fn layerBody(ctx: *ExecContext, extra: LayerExtra, hidden: *const Hidden, abs: AbTuple) !Hidden {
            const cfg = extra.config;
            const layer = extra.layer;

            // Attention block.
            var attn_in = try hidden.rmsNormMul(ctx, .embed, &layer.attn_norm, cfg.rms_norm_eps);
            defer attn_in.deinit();

            var qkv = try projectQkv(ctx, layer, &attn_in, cfg);
            defer qkv.deinit();

            // LoRA deltas land on the per-projection slices AFTER any fused
            // split, so both attn_proj arms share this path.
            var q = try adapted(0, ctx, abs, extra, &attn_in, &qkv.q);
            defer q.deinit();
            var k = try adapted(1, ctx, abs, extra, &attn_in, &qkv.k);
            defer k.deinit();
            var v = try adapted(2, ctx, abs, extra, &attn_in, &qkv.v);
            defer v.deinit();

            var q3 = try q.split(ctx, .q, .{ .head, .d }, .{ cfg.num_attention_heads, cfg.head_dim });
            defer q3.deinit();
            var k3 = try k.split(ctx, .k, .{ .kv_head, .d }, .{ cfg.num_key_value_heads, cfg.head_dim });
            defer k3.deinit();
            var v3 = try v.split(ctx, .v, .{ .kv_head, .d }, .{ cfg.num_key_value_heads, cfg.head_dim });
            defer v3.deinit();

            var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, cfg.rms_norm_eps, extra.rope_table);
            defer q_rope.deinit();
            var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, cfg.rms_norm_eps, extra.rope_table);
            defer k_rope.deinit();

            if (extra.capture) |cap| {
                // Copy the freshly computed token K/V rows (post q/k-norm,
                // post-RoPE keys) — the exact rows a same-position prefill
                // would put in the KV cache, and the cartridge init payload.
                var k_contig = try k_rope.contiguous(ctx);
                defer k_contig.deinit();
                @memcpy(cap.k_rows[extra.layer_i], try k_contig.dataConst());
                var v_contig = try v3.contiguous(ctx);
                defer v_contig.deinit();
                @memcpy(cap.v_rows[extra.layer_i], try v_contig.dataConst());
            }

            const attn_scale = 1 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
            var attn = if (extra.packed_segments != null)
                try segmentedAttention(ctx, extra, &q_rope, &k_rope, &v3, attn_scale)
            else if (extra.cartridge_parts) |parts| blk: {
                // Composed prefix: one concat over every part's rows, so the
                // same fused backward routes gradients into ALL of them.
                var k_cat = try cartridge_mod.composedCatK(ctx, parts, extra.layer_i, &k_rope);
                defer k_cat.deinit();
                var v_cat = try cartridge_mod.composedCatV(ctx, parts, extra.layer_i, &v3);
                defer v_cat.deinit();
                break :blk try q_rope.groupedAttention(ctx, &k_cat, &v_cat, extra.kv_head_for_head, .attn, attn_scale, .{});
            } else if (extra.cartridge_layer) |cart_layer| blk: {
                // Trained KV prefix: every query attends the whole prefix,
                // then causally over the real tokens (the end-aligned
                // source_offset = p kernel); gradients reach the cartridge's
                // trainable rows through the concat.
                var k_cat = try cart_layer.catK(ctx, &k_rope);
                defer k_cat.deinit();
                var v_cat = try cart_layer.catV(ctx, &v3);
                defer v_cat.deinit();
                break :blk try q_rope.groupedAttention(ctx, &k_cat, &v_cat, extra.kv_head_for_head, .attn, attn_scale, .{});
            } else try q_rope.groupedAttention(ctx, &k_rope, &v3, extra.kv_head_for_head, .attn, attn_scale, .{});
            defer attn.deinit();

            var attn_out_base = try dotLinear(&layer.o_proj, ctx, &attn, .attn, .embed);
            defer attn_out_base.deinit();
            var attn_out = try adapted(3, ctx, abs, extra, &attn, &attn_out_base);
            defer attn_out.deinit();

            var h = try hidden.add(ctx, &attn_out);
            defer h.deinit();

            // FFN block (dense only; MoE configs are rejected up front).
            const dense = switch (layer.ffn) {
                .dense => |*d| d,
                .moe => return Error.MoeUnsupported,
            };

            var ffn_in = try h.rmsNormMul(ctx, .embed, &layer.ffn_norm, cfg.rms_norm_eps);
            defer ffn_in.deinit();

            var gate_up = try projectGateUp(ctx, dense, &ffn_in, cfg);
            defer gate_up.deinit();

            var gate = try adapted(4, ctx, abs, extra, &ffn_in, &gate_up.gate);
            defer gate.deinit();
            var up = try adapted(5, ctx, abs, extra, &ffn_in, &gate_up.up);
            defer up.deinit();

            var gated = try up.swiglu(ctx, &gate);
            defer gated.deinit();

            var down_base = try dotLinear(&dense.down_proj, ctx, &gated, .ffn, .embed);
            defer down_base.deinit();
            var down = try adapted(6, ctx, abs, extra, &gated, &down_base);
            defer down.deinit();

            return h.add(ctx, &down);
        }

        /// Packed attention: one fused-attention call per contiguous segment
        /// over zero-copy `.seq` narrows. No cross-segment key is ever
        /// visible — each segment's queries see the shared cartridge prefix
        /// (when present) plus their own causal rows, exactly like a lone
        /// sequence — and attention does no more work packed than unpacked
        /// (the mask zeroes cross-segment pairs by definition). The heavy
        /// GEMMs around this block stay packed; per-segment gradients flow
        /// through the existing fused backward and accumulate into shared
        /// leaves (the cartridge rows) across segments. Runs under an exec
        /// scope like the rest of `layerBody`.
        fn segmentedAttention(
            ctx: *ExecContext,
            extra: LayerExtra,
            q_rope: *const fucina.Tensor(.{ .seq, .head, .d }),
            k_rope: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
            v3: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
            attn_scale: f32,
        ) !fucina.Tensor(.{ .seq, .attn }) {
            const seg_lens = extra.packed_segments.?;
            const Out = fucina.Tensor(.{ .seq, .attn });

            const outs = try ctx.allocator.alloc(Out, seg_lens.len);
            defer ctx.allocator.free(outs);
            var built: usize = 0;
            errdefer for (outs[0..built]) |*out| out.deinit();

            var start: usize = 0;
            for (seg_lens, 0..) |len, seg_i| {
                var q_seg = try q_rope.narrow(ctx, .seq, start, len);
                defer q_seg.deinit();
                var k_seg = try k_rope.narrow(ctx, .seq, start, len);
                defer k_seg.deinit();
                var v_seg = try v3.narrow(ctx, .seq, start, len);
                defer v_seg.deinit();

                outs[seg_i] = if (extra.cartridge_parts) |parts| blk: {
                    var k_cat = try cartridge_mod.composedCatK(ctx, parts, extra.layer_i, &k_seg);
                    defer k_cat.deinit();
                    var v_cat = try cartridge_mod.composedCatV(ctx, parts, extra.layer_i, &v_seg);
                    defer v_cat.deinit();
                    break :blk try q_seg.groupedAttention(ctx, &k_cat, &v_cat, extra.kv_head_for_head, .attn, attn_scale, .{});
                } else if (extra.cartridge_layer) |cart_layer| blk: {
                    var k_cat = try cart_layer.catK(ctx, &k_seg);
                    defer k_cat.deinit();
                    var v_cat = try cart_layer.catV(ctx, &v_seg);
                    defer v_cat.deinit();
                    break :blk try q_seg.groupedAttention(ctx, &k_cat, &v_cat, extra.kv_head_for_head, .attn, attn_scale, .{});
                } else try q_seg.groupedAttention(ctx, &k_seg, &v_seg, extra.kv_head_for_head, .attn, attn_scale, .{});
                built += 1;
                start += len;
            }

            if (outs.len == 1) return outs[0];
            defer for (outs) |*out| out.deinit(); // scope-owned borrows: safe no-ops
            const rest = try ctx.allocator.alloc(*const Out, outs.len - 1);
            defer ctx.allocator.free(rest);
            for (rest, outs[1..]) |*ptr, *out| ptr.* = out;
            return outs[0].concat(ctx, .seq, rest);
        }

        /// Checkpoint-block wrapper of `layerBody` with the comptime arity the
        /// enabled `targets` dictate: (hidden, A/B per enabled target). The
        /// block signature must be concrete, so one wrapper per arity; all
        /// forward to the shared body.
        const LayerBlock = switch (n_enabled) {
            0 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden) !Hidden {
                    return layerBody(ctx, extra, h, .{});
                }
            },
            1 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0 });
                }
            },
            2 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1), a1: AbPtr(2), b1: AbPtr(3)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0, a1, b1 });
                }
            },
            3 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1), a1: AbPtr(2), b1: AbPtr(3), a2: AbPtr(4), b2: AbPtr(5)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0, a1, b1, a2, b2 });
                }
            },
            4 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1), a1: AbPtr(2), b1: AbPtr(3), a2: AbPtr(4), b2: AbPtr(5), a3: AbPtr(6), b3: AbPtr(7)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0, a1, b1, a2, b2, a3, b3 });
                }
            },
            5 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1), a1: AbPtr(2), b1: AbPtr(3), a2: AbPtr(4), b2: AbPtr(5), a3: AbPtr(6), b3: AbPtr(7), a4: AbPtr(8), b4: AbPtr(9)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0, a1, b1, a2, b2, a3, b3, a4, b4 });
                }
            },
            6 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1), a1: AbPtr(2), b1: AbPtr(3), a2: AbPtr(4), b2: AbPtr(5), a3: AbPtr(6), b3: AbPtr(7), a4: AbPtr(8), b4: AbPtr(9), a5: AbPtr(10), b5: AbPtr(11)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5 });
                }
            },
            7 => struct {
                fn run(ctx: *ExecContext, extra: LayerExtra, h: *const Hidden, a0: AbPtr(0), b0: AbPtr(1), a1: AbPtr(2), b1: AbPtr(3), a2: AbPtr(4), b2: AbPtr(5), a3: AbPtr(6), b3: AbPtr(7), a4: AbPtr(8), b4: AbPtr(9), a5: AbPtr(10), b5: AbPtr(11), a6: AbPtr(12), b6: AbPtr(13)) !Hidden {
                    return layerBody(ctx, extra, h, .{ a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6 });
                }
            },
            else => unreachable,
        };
    };
}

test {
    _ = @import("train_tests.zig");
    _ = @import("train_golden_tests.zig");
    _ = @import("train_cartridge_tests.zig");
    _ = @import("train_cartridge_compose_tests.zig");
}
