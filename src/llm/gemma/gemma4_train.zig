//! LoRA fine-tuning over frozen Gemma4 dense/shared projections.
//!
//! This is the conservative first training surface: attention and the shared
//! dense GeGLU FFN use differentiable frozen-RHS `dot` over the loaded GGUF
//! weights, with trainable LoRA adapters on selected projections. The base
//! weights remain frozen and quantized/f16/bf16 as loaded.
//!
//! Current exclusions are intentional:
//! - MoE layers require the model to retain raw expert blocks (`--experts=borrow`
//!   on CPU, or the raw representation used by Metal/Q4 expert builds). The
//!   packed CPU inference-only expert RHS is still rejected for training.
//! - shared-KV and PLE are rejected until their full-sequence training semantics
//!   are mirrored explicitly.

const std = @import("std");
const fucina = @import("fucina");
const gemma4 = @import("gemma4.zig");
const gemma_moe = @import("moe.zig");
const weights = @import("../weights.zig");

const Allocator = std.mem.Allocator;
const backend_mod = fucina.internal.backend_mod;
const ExecContext = fucina.ExecContext;
const LinearWeight = weights.LinearWeight;
const ParamRegistry = fucina.ParamRegistry;
const Tag = @TypeOf(.tag);
const lora = fucina.lora;
const optim = fucina.optim;
const rng = fucina.rng;

pub const Error = error{
    ExecScopeRequired,
    InvalidSequenceLength,
    LabelLengthMismatch,
    PleUnsupported,
    RawMoeWeightsRequired,
    SharedKvUnsupported,
};

pub const ignore_index: usize = std.math.maxInt(usize);

pub const Targets = struct {
    q: bool = true,
    k: bool = false,
    v: bool = true,
    o: bool = false,
    gate: bool = false,
    up: bool = false,
    down: bool = false,
};

const n_targets = 7;
const target_names = [n_targets][]const u8{ "q", "k", "v", "o", "gate", "up", "down" };

const Hidden = fucina.Tensor(.{ .seq, .embed });

const RopeTables = struct {
    swa: fucina.RopeTable,
    global: fucina.RopeTable,

    fn deinit(self: *RopeTables) void {
        self.global.deinit();
        self.swa.deinit();
        self.* = undefined;
    }
};

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

fn projectAttention(
    ctx: *ExecContext,
    layer: *const gemma4.Layer,
    input: *const Hidden,
    q_dim: usize,
    kv_dim: usize,
) !QkvLinear {
    return switch (layer.attn_proj) {
        .separate => |*sep| blk: {
            var q = try dotLinear(&sep.q_proj, ctx, input, .embed, .q);
            errdefer q.deinit();
            var k = try dotLinear(&sep.k_proj.?, ctx, input, .embed, .k);
            errdefer k.deinit();
            var v = if (sep.v_proj) |*w|
                try dotLinear(w, ctx, input, .embed, .v)
            else
                try k.withTags(ctx, .{ .seq, .v });
            errdefer v.deinit();
            break :blk .{ .q = q, .k = k, .v = v };
        },
        .fused => |*fused| switch (fused.kind) {
            .qk => blk: {
                var qk = try dotLinear(&fused.weight, ctx, input, .embed, .qk);
                defer qk.deinit();
                break :blk try splitQk(ctx, &qk, q_dim, kv_dim);
            },
            .qkv => blk: {
                var qkv = try dotLinear(&fused.weight, ctx, input, .embed, .qkv);
                defer qkv.deinit();
                break :blk try splitQkv(ctx, &qkv, q_dim, kv_dim);
            },
        },
    };
}

fn splitQk(
    ctx: *ExecContext,
    qk: *const fucina.Tensor(.{ .seq, .qk }),
    q_dim: usize,
    kv_dim: usize,
) !QkvLinear {
    var q_view = try qk.narrow(ctx, .qk, 0, q_dim);
    defer q_view.deinit();
    var q = try q_view.withTags(ctx, .{ .seq, .q });
    errdefer q.deinit();

    var k_view = try qk.narrow(ctx, .qk, q_dim, kv_dim);
    defer k_view.deinit();
    var k = try k_view.withTags(ctx, .{ .seq, .k });
    errdefer k.deinit();

    var v = try k.withTags(ctx, .{ .seq, .v });
    errdefer v.deinit();
    return .{ .q = q, .k = k, .v = v };
}

fn splitQkv(
    ctx: *ExecContext,
    qkv: *const fucina.Tensor(.{ .seq, .qkv }),
    q_dim: usize,
    kv_dim: usize,
) !QkvLinear {
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
    var v = try v_view.withTags(ctx, .{ .seq, .v });
    errdefer v.deinit();
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

fn projectGateUp(ctx: *ExecContext, layer: *const gemma4.Layer, input: *const Hidden) !GateUpLinear {
    var gate = try dotLinear(&layer.ffn_gate, ctx, input, .embed, .ffn);
    errdefer gate.deinit();
    var up = try dotLinear(&layer.ffn_up, ctx, input, .embed, .ffn);
    errdefer up.deinit();
    return .{ .gate = gate, .up = up };
}

const ExpertGuPart = enum { gate, up };

fn dotRawExpertGateUp(
    ctx: *ExecContext,
    gw: gemma_moe.RawExpertWeights,
    expert: usize,
    comptime part: ExpertGuPart,
    input: *const Hidden,
    hidden: usize,
    n_ff: usize,
) !fucina.Tensor(.{ .seq, .ffn }) {
    const bpr = hidden / 256;
    const row_off: usize = if (part == .gate) 0 else n_ff;
    const start = (expert * 2 * n_ff + row_off) * bpr;
    const len = n_ff * bpr;
    return switch (gw.gu) {
        .q6_k => |blocks| blk: {
            var w = try fucina.Tensor(.{ .dtype = .q6_k, .tags = .{ .ffn, .embed } }).fromBorrowedBlocks(ctx, .{ n_ff, hidden }, @constCast(blocks[start..][0..len]));
            defer w.deinit();
            break :blk input.dot(ctx, &w, .embed);
        },
        .q4_k => |blocks| blk: {
            var w = try fucina.Tensor(.{ .dtype = .q4_k, .tags = .{ .ffn, .embed } }).fromBorrowedBlocks(ctx, .{ n_ff, hidden }, @constCast(blocks[start..][0..len]));
            defer w.deinit();
            break :blk input.dot(ctx, &w, .embed);
        },
    };
}

fn dotRawExpertDown(
    ctx: *ExecContext,
    gw: gemma_moe.RawExpertWeights,
    expert: usize,
    input: *const fucina.Tensor(.{ .seq, .ffn }),
    hidden: usize,
    n_ff: usize,
) !Hidden {
    const bpr = n_ff / 32;
    const start = expert * hidden * bpr;
    var w = try fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .embed, .ffn } }).fromBorrowedBlocks(ctx, .{ hidden, n_ff }, @constCast(gw.dn_blocks[start..][0 .. hidden * bpr]));
    defer w.deinit();
    return input.dot(ctx, &w, .ffn);
}

fn rawExpertOutput(
    ctx: *ExecContext,
    gw: gemma_moe.RawExpertWeights,
    expert: usize,
    input: *const Hidden,
    hidden: usize,
    n_ff: usize,
) !Hidden {
    var gate = try dotRawExpertGateUp(ctx, gw, expert, .gate, input, hidden, n_ff);
    defer gate.deinit();
    var up = try dotRawExpertGateUp(ctx, gw, expert, .up, input, hidden, n_ff);
    defer up.deinit();
    var gated = try up.geglu(ctx, &gate);
    defer gated.deinit();
    return dotRawExpertDown(ctx, gw, expert, &gated, hidden, n_ff);
}

fn zeroHidden(ctx: *ExecContext, seq: usize, hidden: usize) !Hidden {
    var value = try ctx.zeros(&.{ seq, hidden });
    errdefer value.deinit();
    return Hidden.fromTensor(ctx, value);
}

pub fn Trainer(comptime targets: Targets) type {
    return struct {
        model: *const gemma4.Model,
        allocator: Allocator,
        lora_config: lora.Config,
        scale: f32,
        seed: u64,
        adapters: []LayerAdapters,
        /// Every adapter A/B under its "layers.<i>.<target>.lora_{a,b}"
        /// checkpoint name: the registry owns the names and retains
        /// refcounted views of the tensors; optimizers registered via
        /// `registerAllParams` borrow both, so the trainer must outlive them.
        registry: ParamRegistry,
        rope_tables: std.AutoHashMapUnmanaged(usize, *RopeTables) = .empty,
        step_counter: u64 = 0,

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

        fn targetDims(model: *const gemma4.Model, layer_i: usize, comptime t: usize) [2]usize {
            const cfg = model.config;
            const head_dim = model.geom.head_dim[layer_i];
            const q_dim = cfg.num_attention_heads * head_dim;
            const kv_dim = model.geom.kv_heads[layer_i] * head_dim;
            return switch (t) {
                0 => .{ cfg.hidden_size, q_dim },
                1, 2 => .{ cfg.hidden_size, kv_dim },
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
        const AbTuple = std.meta.Tuple(&ab_ptr_types);

        pub fn init(ctx: *ExecContext, model: *const gemma4.Model, config: lora.Config, seed: u64) !Self {
            if (model.ple != null) return Error.PleUnsupported;
            for (model.geom.has_kv) |has_kv| if (!has_kv) return Error.SharedKvUnsupported;
            for (model.layers) |*layer| {
                if (layer.moe) |*moe| {
                    if (moe.gpu_weights == null) return Error.RawMoeWeightsRequired;
                }
            }

            const allocator = ctx.allocator;
            const n_layers = model.config.num_layers;
            const adapters = try allocator.alloc(LayerAdapters, n_layers);
            errdefer allocator.free(adapters);
            var built_layers: usize = 0;
            errdefer for (adapters[0..built_layers]) |*ads| deinitLayerAdaptersPartial(ads, n_enabled);
            for (adapters, 0..) |*ads, layer_i| {
                try initLayerAdapters(ctx, ads, model, config, seed, layer_i);
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

        pub fn deinit(self: *Self) void {
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
            model: *const gemma4.Model,
            config: lora.Config,
            seed: u64,
            layer_i: usize,
        ) !void {
            var built: usize = 0;
            errdefer deinitLayerAdaptersPartial(ads, built);
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    const dims = targetDims(model, layer_i, t);
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

        fn deinitLayerAdaptersPartial(ads: *LayerAdapters, built: usize) void {
            inline for (0..n_targets) |t| {
                if (comptime enabled(t)) {
                    const ordinal = comptime abIndex(t) / 2;
                    if (ordinal < built) @field(ads.*, target_names[t]).deinit();
                }
            }
        }

        pub fn registerAllParams(self: *Self, opt: anytype) !void {
            try self.registry.addParamsTo(opt);
        }

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

        /// Mean cross-entropy over `tokens` against `labels` (`ignore_index`
        /// masks positions). MUST run under an open exec scope; the result is
        /// scope-owned. Advances the step counter (one dropout stream per
        /// call). Equivalent to `lossExt(ctx, tokens, labels, .{})`.
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

        pub fn evalLastLogits(self: *Self, ctx: *ExecContext, tokens: []const usize) !fucina.Tensor(.{ .seq, .vocab }) {
            const scope = ctx.openExecScope();
            defer ctx.closeExecScope(scope);
            var logits = try self.forwardLogits(ctx, tokens, null);
            defer logits.deinit();
            var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
            defer last.deinit();
            var value = try last.value.clone(ctx.allocator);
            errdefer value.deinit();
            return fucina.Tensor(.{ .seq, .vocab }).fromTensor(ctx, value);
        }

        fn forwardLogits(self: *Self, ctx: *ExecContext, tokens: []const usize, step: ?u64) !fucina.Tensor(.{ .seq, .vocab }) {
            if (tokens.len == 0) return Error.InvalidSequenceLength;
            if (self.model.ple != null) return Error.PleUnsupported;

            const model = self.model;
            const cfg = model.config;
            const rope = try self.prepareRope(ctx, tokens.len);

            var x = try model.token_embedding.getRowsAs(ctx, tokens, .embed);
            defer x.deinit();
            x = try ctx.replace(x, x.scale(ctx, @sqrt(@as(f32, @floatFromInt(cfg.hidden_size)))));

            for (model.layers, 0..) |*layer, layer_i| {
                if (!model.geom.has_kv[layer_i]) return Error.SharedKvUnsupported;
                if (layer.moe) |*moe| {
                    if (moe.gpu_weights == null) return Error.RawMoeWeightsRequired;
                }
                x = try ctx.replace(x, self.layerBody(ctx, layer, layer_i, rope, &x, self.layerSeeds(step, layer_i), &self.adapters[layer_i]));
                if (layer.out_scale) |s| x = try ctx.replace(x, x.scale(ctx, s));
            }

            var normed = try x.rmsNormMul(ctx, .embed, &model.output_norm, cfg.rms_norm_eps);
            defer normed.deinit();
            x.deinit();
            var logits = try dotLinear(&model.output, ctx, &normed, .embed, .vocab);
            if (cfg.final_logit_softcapping != 0) {
                const sc = cfg.final_logit_softcapping;
                var down = try logits.scale(ctx, 1.0 / sc);
                logits.deinit();
                defer down.deinit();
                var t = try down.tanh(ctx);
                defer t.deinit();
                return t.scale(ctx, sc);
            }
            return logits;
        }

        fn prepareRope(self: *Self, ctx: *ExecContext, seq_len: usize) !*const RopeTables {
            if (self.rope_tables.get(seq_len)) |tables| return tables;

            const cfg = self.model.config;
            const positions = try ctx.allocator.alloc(i32, seq_len);
            defer ctx.allocator.free(positions);
            for (positions, 0..) |*position, i| position.* = @intCast(i);

            const fresh = try self.allocator.create(RopeTables);
            errdefer self.allocator.destroy(fresh);
            const factors: ?[]const f32 = if (self.model.rope_freqs) |*t| try t.dataConst() else null;
            fresh.* = .{
                .swa = try ctx.prepareRopeTable(positions, cfg.head_dim_swa, cfg.rope_theta_swa, false),
                .global = try ctx.prepareRopeTableFactors(positions, cfg.head_dim_global, cfg.rope_theta, false, factors),
            };
            errdefer fresh.deinit();

            try self.rope_tables.put(self.allocator, seq_len, fresh);
            return fresh;
        }

        const dropout_domain: u64 = 0x67656d6d_61346472;

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

        fn tempAdapter(comptime t: usize, abs: AbTuple) TargetAdapter(t) {
            const j = comptime abIndex(t);
            return .{
                .a = abs[j].*,
                .b = abs[j + 1].*,
                .scale = 1,
                .dropout_p = 0,
            };
        }

        fn adapted(
            self: *Self,
            comptime t: usize,
            ctx: *ExecContext,
            abs: AbTuple,
            seeds: [n_targets]?u64,
            x: anytype,
            base: anytype,
        ) !@TypeOf(base.*) {
            if (comptime !enabled(t)) return base.withTags(ctx, @TypeOf(base.*).axis_tags);
            var ad = tempAdapter(t, abs);
            ad.scale = self.scale;
            ad.dropout_p = self.lora_config.dropout_p;
            var d = try ad.delta(ctx, x, seeds[t]);
            defer d.deinit();
            return base.add(ctx, &d);
        }

        fn moeFfn(
            self: *Self,
            ctx: *ExecContext,
            moe: *const gemma4.MoeFfn,
            attn_out: *const Hidden,
            moe_in: *const Hidden,
        ) !Hidden {
            const gw = moe.gpu_weights orelse return Error.RawMoeWeightsRequired;
            const cfg = self.model.config;
            const seq = moe_in.dim(.seq);
            const hidden = cfg.hidden_size;
            const top_k = cfg.num_experts_used;
            const n_expert = cfg.num_experts;
            const n_ff = cfg.moe_intermediate_size;

            var router_in = try attn_out.rmsNormMul(ctx, .embed, &moe.router_weight, cfg.rms_norm_eps);
            defer router_in.deinit();
            var logits = try dotLinear(&moe.router, ctx, &router_in, .embed, .expert);
            defer logits.deinit();

            var top = try logits.topK(ctx, .expert, top_k, .top);
            defer top.deinit();
            var top_weights = try top.values.softmax(ctx, .top, .{});
            defer top_weights.deinit();

            const selected_data = try top.indices.dataConst();
            const n_pairs = seq * top_k;
            const selected = try self.allocator.alloc(usize, n_pairs);
            defer self.allocator.free(selected);
            const scale_values = try self.allocator.alloc(f32, n_pairs);
            defer self.allocator.free(scale_values);
            for (selected_data, 0..) |raw, i| {
                const e: usize = @intCast(raw);
                if (e >= n_expert) return error.IndexOutOfBounds;
                selected[i] = e;
                scale_values[i] = moe.down_scale[e];
            }

            var scale_t = try fucina.Tensor(.{ .seq, .top }).fromSlice(ctx, .{ seq, top_k }, scale_values);
            defer scale_t.deinit();
            var weights_t = try top_weights.mul(ctx, &scale_t);
            defer weights_t.deinit();

            const rows = try self.allocator.alloc(usize, seq);
            defer self.allocator.free(rows);

            var out = try zeroHidden(ctx, seq, hidden);
            errdefer out.deinit();
            for (0..top_k) |slot| {
                var slot_out = try zeroHidden(ctx, seq, hidden);
                defer slot_out.deinit();

                for (0..n_expert) |expert| {
                    var m: usize = 0;
                    for (0..seq) |row| {
                        if (selected[row * top_k + slot] == expert) {
                            rows[m] = row;
                            m += 1;
                        }
                    }
                    if (m == 0) continue;

                    var gathered_rows = try moe_in.gather(ctx, .seq, rows[0..m], .row);
                    defer gathered_rows.deinit();
                    var gathered = try gathered_rows.withTags(ctx, .{ .seq, .embed });
                    defer gathered.deinit();
                    var expert_out = try rawExpertOutput(ctx, gw, expert, &gathered, hidden, n_ff);
                    defer expert_out.deinit();
                    slot_out = try ctx.replace(slot_out, slot_out.setRows(ctx, .seq, rows[0..m], &expert_out));
                }

                var weight_col = try weights_t.narrow(ctx, .top, slot, 1);
                defer weight_col.deinit();
                var weight_seq = try weight_col.squeeze(ctx, .top);
                defer weight_seq.deinit();
                var weight_b = try weight_seq.broadcastTo(ctx, .{ .seq, .embed }, .{ seq, hidden });
                defer weight_b.deinit();
                var weighted = try slot_out.mul(ctx, &weight_b);
                defer weighted.deinit();
                out = try ctx.replace(out, out.add(ctx, &weighted));
            }
            return out;
        }

        fn layerBody(
            self: *Self,
            ctx: *ExecContext,
            layer: *const gemma4.Layer,
            layer_i: usize,
            rope: *const RopeTables,
            hidden: *const Hidden,
            seeds: [n_targets]?u64,
            ads: *const LayerAdapters,
        ) !Hidden {
            const cfg = self.model.config;
            const geom = self.model.geom;
            const head_dim = geom.head_dim[layer_i];
            const n_head = cfg.num_attention_heads;
            const n_kv = geom.kv_heads[layer_i];
            const q_dim = n_head * head_dim;
            const kv_dim = n_kv * head_dim;
            const table = if (geom.is_swa[layer_i]) &rope.swa else &rope.global;
            const window: usize = if (geom.is_swa[layer_i]) cfg.sliding_window else 0;
            var kvhh: [gemma4.max_heads]usize = undefined;
            const heads_per_kv = n_head / n_kv;
            for (0..n_head) |h| kvhh[h] = h / heads_per_kv;
            const kv_head_for_head = kvhh[0..n_head];

            const abs = abTuple(ads);

            var attn_in = try hidden.rmsNormMul(ctx, .embed, &layer.attn_norm, cfg.rms_norm_eps);
            defer attn_in.deinit();

            var qkv = try projectAttention(ctx, layer, &attn_in, q_dim, kv_dim);
            defer qkv.deinit();

            var q = try self.adapted(0, ctx, abs, seeds, &attn_in, &qkv.q);
            defer q.deinit();
            var k = try self.adapted(1, ctx, abs, seeds, &attn_in, &qkv.k);
            defer k.deinit();
            var v = try self.adapted(2, ctx, abs, seeds, &attn_in, &qkv.v);
            defer v.deinit();

            var q3 = try q.split(ctx, .q, .{ .head, .d }, .{ n_head, head_dim });
            defer q3.deinit();
            var k3 = try k.split(ctx, .k, .{ .kv_head, .d }, .{ n_kv, head_dim });
            defer k3.deinit();
            var v3 = try v.split(ctx, .v, .{ .kv_head, .d }, .{ n_kv, head_dim });
            defer v3.deinit();

            var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, cfg.rms_norm_eps, table);
            defer q_rope.deinit();
            var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm.?, cfg.rms_norm_eps, table);
            defer k_rope.deinit();
            var v_norm = try v3.rmsNorm(ctx, .d, cfg.rms_norm_eps);
            defer v_norm.deinit();

            var attn = try q_rope.groupedAttention(ctx, &k_rope, &v_norm, kv_head_for_head, .attn, 1.0, .{ .window = window });
            defer attn.deinit();

            var attn_base = try dotLinear(&layer.o_proj, ctx, &attn, .attn, .embed);
            defer attn_base.deinit();
            var attn_out = try self.adapted(3, ctx, abs, seeds, &attn, &attn_base);
            defer attn_out.deinit();
            var h0 = try attn_out.rmsNormMul(ctx, .embed, &layer.attn_post_norm, cfg.rms_norm_eps);
            defer h0.deinit();
            var h = try hidden.add(ctx, &h0);
            defer h.deinit();

            var ffn_in = try h.rmsNormMul(ctx, .embed, &layer.ffn_norm, cfg.rms_norm_eps);
            defer ffn_in.deinit();
            var gate_up = try projectGateUp(ctx, layer, &ffn_in);
            defer gate_up.deinit();
            var gate = try self.adapted(4, ctx, abs, seeds, &ffn_in, &gate_up.gate);
            defer gate.deinit();
            var up = try self.adapted(5, ctx, abs, seeds, &ffn_in, &gate_up.up);
            defer up.deinit();
            var gated = try up.geglu(ctx, &gate);
            defer gated.deinit();
            var down_base = try dotLinear(&layer.ffn_down, ctx, &gated, .ffn, .embed);
            defer down_base.deinit();
            var mlp = try self.adapted(6, ctx, abs, seeds, &gated, &down_base);
            defer mlp.deinit();

            var combined: Hidden = if (layer.moe) |*moe| blk: {
                var mlp_post = try mlp.rmsNormMul(ctx, .embed, &moe.post_norm_1, cfg.rms_norm_eps);
                defer mlp_post.deinit();
                var moe_in = try h.rmsNormMul(ctx, .embed, &moe.pre_norm_2, cfg.rms_norm_eps);
                defer moe_in.deinit();
                var moe_out = try self.moeFfn(ctx, moe, &h, &moe_in);
                defer moe_out.deinit();
                var moe_post = try moe_out.rmsNormMul(ctx, .embed, &moe.post_norm_2, cfg.rms_norm_eps);
                defer moe_post.deinit();
                break :blk try mlp_post.add(ctx, &moe_post);
            } else try mlp.withTags(ctx, .{ .seq, .embed });
            defer combined.deinit();

            var post = try combined.rmsNormMul(ctx, .embed, &layer.ffn_post_norm, cfg.rms_norm_eps);
            defer post.deinit();
            return h.add(ctx, &post);
        }
    };
}

test {
    _ = @import("gemma4_train_tests.zig");
}
