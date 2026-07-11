//! nanochat GPT (karpathy/nanochat → Fucina), CPU fp32 parity port.
//!
//! Replicates `refs/nanochat/nanochat/gpt.py` `GPT.forward(idx, targets)` on the
//! training path (kv_cache=None) for ONE sequence at a time; the (B,T) oracle
//! batch is handled by the caller running each row independently and summing the
//! per-token losses (grad-accumulation semantics; the attention and CE kernels
//! are single-sequence). All math is f32.
//!
//! Notable reference features (gpt.py docstring): rotary embeddings, QK norm,
//! untied wte/lm_head, relu² MLP, norm after token embedding, no rmsnorm weights,
//! no linear bias, GQA. Reproduced here with facade ops only (no fucina.internal).
//!
//! Tag convention for the residual stream and projections:
//!   .seq  sequence positions            .d      model / head-feature dim
//!   .head query heads                   .kv_head key/value heads (GQA)
//!   .attn flattened attention output    .vocab  vocabulary
//!   .qo/.kvo   qkv projection outputs    .ff    MLP hidden (4·n_embd)
//!   .layer per-layer scalar axis         .one   length-1 scalar-param axis
//! `.d` is reused as both n_embd (residual stream) and head_dim (post-split) — the
//! length is per-tensor, only the name matters for a contraction.

const std = @import("std");
const fucina = @import("fucina");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const RopeTable = fucina.RopeTable;
const Tensor = fucina.Tensor;
const safetensors = fucina.safetensors;

/// gpt.py norm(): F.rms_norm(x,(d,)) with the eps torch actually uses =
/// torch.finfo(float32).eps = 2^-23 (discovered bitwise, init_d6.config.json).
const rms_eps: f32 = 1.1920928955078125e-07;

/// gpt.py rotary base (_precompute_rotary_embeddings theta).
const rope_theta: f32 = 100000.0;

/// nanochat's ignore_index=-1 (gpt.py:520) → Fucina's usize sentinel
/// (crossEntropyExt has no signed labels).
pub const ignore_index: usize = std.math.maxInt(usize);

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub const Config = struct {
    sequence_len: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
    /// Sliding-window pattern tiled across layers ("L"=full context, "S"=short).
    window_pattern: []const u8,

    /// The CPU-demo config from runs/runcpu.sh (nanochat_dump.py CONFIGS["d6"]).
    pub const d6 = Config{
        .sequence_len = 512,
        .vocab_size = 32768,
        .n_layer = 6,
        .n_head = 6,
        .n_kv_head = 6,
        .n_embd = 384,
        .window_pattern = "L",
    };

    /// Tiny config for the fast finite-diff gradcheck (CONFIGS["d2"]).
    pub const d2 = Config{
        .sequence_len = 32,
        .vocab_size = 256,
        .n_layer = 2,
        .n_head = 2,
        .n_kv_head = 2,
        .n_embd = 128,
        .window_pattern = "L",
    };

    pub fn headDim(self: Config) usize {
        return self.n_embd / self.n_head;
    }

    /// gpt.py has_ve(): alternating value-embedding layers, last layer included.
    pub fn hasVe(self: Config, layer_idx: usize) bool {
        return hasVeAt(layer_idx, self.n_layer);
    }

    /// gpt.py backout caches the residual at n_layer//2 (forward:497).
    pub fn backoutLayer(self: Config) usize {
        return self.n_layer / 2;
    }

    /// groupedAttention window for layer i, in Fucina's kernel convention.
    /// Fucina's `.window=W` attends [i-W+1, i] = W keys INCLUDING self
    /// (exec.zig §groupedCausalAttentionWindowed); the reference left bound
    /// excludes self (flash_attention.py _sdpa_attention masks
    /// `(row_idx - col_idx) <= window` = window+1 keys), so every arm returns
    /// the reference value + 1. gpt.py _compute_window_sizes maps 'L' (and the
    /// always-'L' final layer) to the FINITE (sequence_len, 0), not unlimited:
    /// identical to full causal while the context stays <= sequence_len (all
    /// training/eval shapes), and capping decode attention to the last
    /// sequence_len+1 keys once the KV cache grows past it.
    pub fn windowFor(self: Config, layer_idx: usize) usize {
        const long = self.sequence_len;
        if (layer_idx == self.n_layer - 1) return long + 1;
        const c = std.ascii.toUpper(self.window_pattern[layer_idx % self.window_pattern.len]);
        if (c == 'S') {
            // gpt.py: short_window = -(-long // 4 // 128) * 128 — BOTH divisions
            // are ceilings (differs from floor-then-ceil only when long % 4 != 0).
            const short_window = ((long + 3) / 4 + 127) / 128 * 128;
            return short_window + 1;
        }
        return long + 1; // 'L'
    }
};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// The GPT parameterized over the storage dtype of its trained matrices:
/// `.f32` (the default `Model`) or
/// `.bf16` — the Muon-routed transformer matrices (c_q/c_k/c_v/c_proj/
/// c_fc/c_proj_mlp/ve_gate) are stored 16-bit and train through f32
/// masters (gradients are f32; MuonAdamW steps the master and narrows
/// back). Embeddings (wte, value_embeds) stay f32 — they train through
/// gather — as do the head, gates, and scalars (AdamW-routed).
pub fn ModelOf(comptime dtype: fucina.DType) type {
    comptime {
        if (dtype != .f32 and dtype != .bf16)
            @compileError("nanochat matrix params support f32 or bf16");
    }
    return struct {
        const Self = @This();

        const WteT = Tensor(.{ .vocab, .d });
        const LmHeadT = Tensor(.{ .vocab, .d });
        const VeEmbT = Tensor(.{ .vocab, .kvo });
        const CqT = ParamTensor(dtype, .{ .qo, .d });
        const CkvT = ParamTensor(dtype, .{ .kvo, .d });
        const CProjAttnT = ParamTensor(dtype, .{ .d, .attn });
        const CfcT = ParamTensor(dtype, .{ .ff, .d });
        const CProjMlpT = ParamTensor(dtype, .{ .d, .ff });
        const VeGateT = ParamTensor(dtype, .{ .kv_head, .d }); // [n_kv_head, 12]
        const LayerScalarsT = Tensor(.{.layer});
        const SmearGateT = Tensor(.{ .one, .d }); // [1, 24]
        const OneScalarT = Tensor(.{.one});

        const Layer = struct {
            c_q: CqT,
            c_k: CkvT,
            c_v: CkvT,
            c_proj: CProjAttnT,
            c_fc: CfcT,
            c_proj_mlp: CProjMlpT,
            ve_gate: ?VeGateT, // present iff Config.hasVe(i)
        };

        fn deinitLayer(l: *Layer) void {
            l.c_q.deinit();
            l.c_k.deinit();
            l.c_v.deinit();
            l.c_proj.deinit();
            l.c_fc.deinit();
            l.c_proj_mlp.deinit();
            if (l.ve_gate) |*g| g.deinit();
        }

        allocator: Allocator,
        cfg: Config,
        padded_vocab: usize,

        wte: WteT,
        lm_head: LmHeadT,
        resid_lambdas: LayerScalarsT,
        x0_lambdas: LayerScalarsT,
        smear_gate: SmearGateT,
        smear_lambda: OneScalarT,
        backout_lambda: OneScalarT,
        layers: []Layer,
        value_embeds: []?VeEmbT, // per-layer, null where !hasVe

        /// GQA head map (len = n_head). Owned so the attention backward node's
        /// duped copy is not the only reference; also reused across forwards.
        kv_head_for_head: []usize,

        /// SDPA default softmax scale = 1/sqrt(head_dim). nanochat's CPU path runs
        /// F.scaled_dot_product_attention with no explicit scale (flash_attention.py
        /// _sdpa_attention), and applies the extra ×1.2 to q and k separately.
        attn_scale: f32,

        /// Rotary sin/cos table for positions 0..sequence_len-1, built once at init
        /// and shared by every full-sequence forward (which always runs at
        /// T == sequence_len in training/eval; other lengths derive an ad-hoc
        /// table). Each rope VJP clones the table it is given, so sharing is safe.
        rope_table: RopeTable,

        pub fn initFromSafetensors(cfg: Config, ctx: *ExecContext, allocator: Allocator, io: std.Io, path: []const u8) !Self {
            var file = try safetensors.File.load(allocator, io, path);
            defer file.deinit();

            const d = cfg.n_embd;
            const hd = cfg.headDim();
            const qo = cfg.n_head * hd;
            const kvo = cfg.n_kv_head * hd;
            const ff = 4 * d;
            const padded_vocab = (try file.tensor("transformer.wte.weight")).shape[0];

            var rope_table = try buildRopeTable(ctx, allocator, cfg.sequence_len, hd);
            errdefer rope_table.deinit();

            var wte = try loadParam(.{ .vocab, .d }, ctx, allocator, &file, "transformer.wte.weight", [_]usize{ padded_vocab, d });
            errdefer wte.deinit();
            var lm_head = try loadParam(.{ .vocab, .d }, ctx, allocator, &file, "lm_head.weight", [_]usize{ padded_vocab, d });
            errdefer lm_head.deinit();
            var resid_lambdas = try loadParam(.{.layer}, ctx, allocator, &file, "resid_lambdas", [_]usize{cfg.n_layer});
            errdefer resid_lambdas.deinit();
            var x0_lambdas = try loadParam(.{.layer}, ctx, allocator, &file, "x0_lambdas", [_]usize{cfg.n_layer});
            errdefer x0_lambdas.deinit();
            var smear_gate = try loadParam(.{ .one, .d }, ctx, allocator, &file, "smear_gate.weight", [_]usize{ 1, 24 });
            errdefer smear_gate.deinit();
            var smear_lambda = try loadParam(.{.one}, ctx, allocator, &file, "smear_lambda", [_]usize{1});
            errdefer smear_lambda.deinit();
            var backout_lambda = try loadParam(.{.one}, ctx, allocator, &file, "backout_lambda", [_]usize{1});
            errdefer backout_lambda.deinit();

            const layers = try allocator.alloc(Layer, cfg.n_layer);
            var layers_made: usize = 0;
            errdefer {
                for (layers[0..layers_made]) |*l| deinitLayer(l);
                allocator.free(layers);
            }
            const value_embeds = try allocator.alloc(?VeEmbT, cfg.n_layer);
            for (value_embeds) |*v| v.* = null;
            errdefer {
                for (value_embeds) |*v| if (v.*) |*ve| ve.deinit();
                allocator.free(value_embeds);
            }

            var name_buf: [64]u8 = undefined;
            for (0..cfg.n_layer) |i| {
                const l = &layers[i];
                l.ve_gate = null;
                l.c_q = try loadParamAs(dtype, .{ .qo, .d }, ctx, allocator, &file, try lname(&name_buf, i, "attn.c_q.weight"), [_]usize{ qo, d });
                errdefer l.c_q.deinit();
                l.c_k = try loadParamAs(dtype, .{ .kvo, .d }, ctx, allocator, &file, try lname(&name_buf, i, "attn.c_k.weight"), [_]usize{ kvo, d });
                errdefer l.c_k.deinit();
                l.c_v = try loadParamAs(dtype, .{ .kvo, .d }, ctx, allocator, &file, try lname(&name_buf, i, "attn.c_v.weight"), [_]usize{ kvo, d });
                errdefer l.c_v.deinit();
                l.c_proj = try loadParamAs(dtype, .{ .d, .attn }, ctx, allocator, &file, try lname(&name_buf, i, "attn.c_proj.weight"), [_]usize{ d, qo });
                errdefer l.c_proj.deinit();
                l.c_fc = try loadParamAs(dtype, .{ .ff, .d }, ctx, allocator, &file, try lname(&name_buf, i, "mlp.c_fc.weight"), [_]usize{ ff, d });
                errdefer l.c_fc.deinit();
                l.c_proj_mlp = try loadParamAs(dtype, .{ .d, .ff }, ctx, allocator, &file, try lname(&name_buf, i, "mlp.c_proj.weight"), [_]usize{ d, ff });
                errdefer l.c_proj_mlp.deinit();
                if (cfg.hasVe(i)) {
                    l.ve_gate = try loadParamAs(dtype, .{ .kv_head, .d }, ctx, allocator, &file, try lname(&name_buf, i, "attn.ve_gate.weight"), [_]usize{ cfg.n_kv_head, 12 });
                    errdefer if (l.ve_gate) |*g| g.deinit();
                    value_embeds[i] = try loadParam(.{ .vocab, .kvo }, ctx, allocator, &file, try veName(&name_buf, i), [_]usize{ padded_vocab, kvo });
                }
                layers_made = i + 1;
            }

            const kv_head_for_head = try allocator.alloc(usize, cfg.n_head);
            for (kv_head_for_head, 0..) |*h, qh| h.* = qh * cfg.n_kv_head / cfg.n_head;

            return .{
                .allocator = allocator,
                .cfg = cfg,
                .padded_vocab = padded_vocab,
                .wte = wte,
                .lm_head = lm_head,
                .resid_lambdas = resid_lambdas,
                .x0_lambdas = x0_lambdas,
                .smear_gate = smear_gate,
                .smear_lambda = smear_lambda,
                .backout_lambda = backout_lambda,
                .layers = layers,
                .value_embeds = value_embeds,
                .kv_head_for_head = kv_head_for_head,
                .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                .rope_table = rope_table,
            };
        }

        /// From-scratch initialization (gpt.py `init_weights`) using the Fucina
        /// splitmix64 RNG instead of torch's — bit-exact torch RNG parity is not a
        /// requirement: only the std/bounds and the deterministic per-`seed`
        /// reproducibility matter (the parity gates use `initFromSafetensors`).
        /// Each parameter draws from an independent
        /// `rng.at(seed, k)` stream so the whole init is a pure function of `seed`.
        ///
        /// Init spec (gpt.py:204-256): wte N(0,0.8); lm_head N(0,0.001); per block
        /// c_q/c_k/c_v U(±s) with s=√3·n_embd^-0.5, c_fc U(±0.4s), c_proj &
        /// mlp.c_proj zeros; value_embeds U(±s); ve_gate U(0,0.02);
        /// smear_gate U(0,0.02); smear_lambda 0; backout_lambda 0.2;
        /// resid_lambdas[i]=1.15−0.10·i/(n−1); x0_lambdas[i]=0.20−0.15·i/(n−1).
        pub fn initRandom(cfg: Config, ctx: *ExecContext, allocator: Allocator, seed: u64) !Self {
            const d = cfg.n_embd;
            const hd = cfg.headDim();
            const qo = cfg.n_head * hd;
            const kvo = cfg.n_kv_head * hd;
            const ff = 4 * d;
            const padded_vocab = paddedVocab(cfg.vocab_size);

            var rope_table = try buildRopeTable(ctx, allocator, cfg.sequence_len, hd);
            errdefer rope_table.deinit();

            // s = √3 · n_embd^-0.5 (uniform bound matching a normal of the same std).
            const s: f32 = @floatCast(@sqrt(3.0) * std.math.pow(f64, @floatFromInt(d), -0.5));
            var k: u64 = 0; // per-tensor stream selector

            var wte = try rngParam(.{ .vocab, .d }, ctx, allocator, [_]usize{ padded_vocab, d }, .{ .normal = .{ .mean = 0, .std = 0.8 } }, fucina.rng.at(seed, k));
            errdefer wte.deinit();
            k += 1;
            var lm_head = try rngParam(.{ .vocab, .d }, ctx, allocator, [_]usize{ padded_vocab, d }, .{ .normal = .{ .mean = 0, .std = 0.001 } }, fucina.rng.at(seed, k));
            errdefer lm_head.deinit();
            k += 1;

            // Per-layer scalars: closed-form (not RNG).
            var resid_lambdas: LayerScalarsT = undefined;
            var x0_lambdas: LayerScalarsT = undefined;
            {
                const rl = try allocator.alloc(f32, cfg.n_layer);
                defer allocator.free(rl);
                const xl = try allocator.alloc(f32, cfg.n_layer);
                defer allocator.free(xl);
                const denom: f32 = @floatFromInt(@max(cfg.n_layer - 1, 1));
                for (0..cfg.n_layer) |i| {
                    const fi: f32 = @floatFromInt(i);
                    rl[i] = 1.15 - 0.10 * fi / denom;
                    xl[i] = 0.20 - 0.15 * fi / denom;
                }
                resid_lambdas = try Tensor(.{.layer}).variableFromSlice(ctx, [_]usize{cfg.n_layer}, rl);
                x0_lambdas = try Tensor(.{.layer}).variableFromSlice(ctx, [_]usize{cfg.n_layer}, xl);
            }
            errdefer resid_lambdas.deinit();
            errdefer x0_lambdas.deinit();
            var smear_gate = try rngParam(.{ .one, .d }, ctx, allocator, [_]usize{ 1, 24 }, .{ .uniform = .{ .lo = 0, .hi = 0.02 } }, fucina.rng.at(seed, k));
            errdefer smear_gate.deinit();
            k += 1;
            var smear_lambda = try rngParam(.{.one}, ctx, allocator, [_]usize{1}, .{ .constant = 0 }, 0);
            errdefer smear_lambda.deinit();
            var backout_lambda = try rngParam(.{.one}, ctx, allocator, [_]usize{1}, .{ .constant = 0.2 }, 0);
            errdefer backout_lambda.deinit();

            const layers = try allocator.alloc(Layer, cfg.n_layer);
            var layers_made: usize = 0;
            errdefer {
                for (layers[0..layers_made]) |*l| deinitLayer(l);
                allocator.free(layers);
            }
            const value_embeds = try allocator.alloc(?VeEmbT, cfg.n_layer);
            for (value_embeds) |*v| v.* = null;
            errdefer {
                for (value_embeds) |*v| if (v.*) |*ve| ve.deinit();
                allocator.free(value_embeds);
            }

            for (0..cfg.n_layer) |i| {
                const l = &layers[i];
                l.ve_gate = null;
                l.c_q = try rngParamAs(dtype, .{ .qo, .d }, ctx, allocator, [_]usize{ qo, d }, .{ .uniform = .{ .lo = -s, .hi = s } }, fucina.rng.at(seed, k));
                errdefer l.c_q.deinit();
                k += 1;
                l.c_k = try rngParamAs(dtype, .{ .kvo, .d }, ctx, allocator, [_]usize{ kvo, d }, .{ .uniform = .{ .lo = -s, .hi = s } }, fucina.rng.at(seed, k));
                errdefer l.c_k.deinit();
                k += 1;
                l.c_v = try rngParamAs(dtype, .{ .kvo, .d }, ctx, allocator, [_]usize{ kvo, d }, .{ .uniform = .{ .lo = -s, .hi = s } }, fucina.rng.at(seed, k));
                errdefer l.c_v.deinit();
                k += 1;
                l.c_proj = try rngParamAs(dtype, .{ .d, .attn }, ctx, allocator, [_]usize{ d, qo }, .zeros, 0);
                errdefer l.c_proj.deinit();
                l.c_fc = try rngParamAs(dtype, .{ .ff, .d }, ctx, allocator, [_]usize{ ff, d }, .{ .uniform = .{ .lo = -s * 0.4, .hi = s * 0.4 } }, fucina.rng.at(seed, k));
                errdefer l.c_fc.deinit();
                k += 1;
                l.c_proj_mlp = try rngParamAs(dtype, .{ .d, .ff }, ctx, allocator, [_]usize{ d, ff }, .zeros, 0);
                errdefer l.c_proj_mlp.deinit();
                if (cfg.hasVe(i)) {
                    l.ve_gate = try rngParamAs(dtype, .{ .kv_head, .d }, ctx, allocator, [_]usize{ cfg.n_kv_head, 12 }, .{ .uniform = .{ .lo = 0, .hi = 0.02 } }, fucina.rng.at(seed, k));
                    errdefer if (l.ve_gate) |*g| g.deinit();
                    k += 1;
                    value_embeds[i] = try rngParam(.{ .vocab, .kvo }, ctx, allocator, [_]usize{ padded_vocab, kvo }, .{ .uniform = .{ .lo = -s, .hi = s } }, fucina.rng.at(seed, k));
                    k += 1;
                }
                layers_made = i + 1;
            }

            const kv_head_for_head = try allocator.alloc(usize, cfg.n_head);
            for (kv_head_for_head, 0..) |*h, qh| h.* = qh * cfg.n_kv_head / cfg.n_head;

            return .{
                .allocator = allocator,
                .cfg = cfg,
                .padded_vocab = padded_vocab,
                .wte = wte,
                .lm_head = lm_head,
                .resid_lambdas = resid_lambdas,
                .x0_lambdas = x0_lambdas,
                .smear_gate = smear_gate,
                .smear_lambda = smear_lambda,
                .backout_lambda = backout_lambda,
                .layers = layers,
                .value_embeds = value_embeds,
                .kv_head_for_head = kv_head_for_head,
                .attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
                .rope_table = rope_table,
            };
        }

        pub fn deinit(self: *Self) void {
            self.rope_table.deinit();
            self.wte.deinit();
            self.lm_head.deinit();
            self.resid_lambdas.deinit();
            self.x0_lambdas.deinit();
            self.smear_gate.deinit();
            self.smear_lambda.deinit();
            self.backout_lambda.deinit();
            for (self.layers) |*l| deinitLayer(l);
            for (self.value_embeds) |*v| if (v.*) |*ve| ve.deinit();
            self.allocator.free(self.layers);
            self.allocator.free(self.value_embeds);
            self.allocator.free(self.kv_head_for_head);
            self.* = undefined;
        }

        /// Full-sequence training/prefill forward for ONE sequence. Returns the
        /// post-softcap logits [.seq,.vocab_size]. When `trace` is non-null, every
        /// oracle intermediate is copied into it (nanochat_dump.py traced_forward).
        ///
        /// Must be called under an OPEN exec scope (intermediates are scope-owned).
        pub fn forward(self: *const Self, ctx: *ExecContext, token_ids: []const usize, trace: ?*Trace) !Tensor(.{ .seq, .vocab }) {
            var x = try self.forwardTrunk(ctx, token_ids, trace);

            // lm_head, crop padding, softcap 15 (gpt.py:511-515): 15·tanh(x/15)
            // as one fused elementwise pass with a transcendental-free VJP.
            var logits_full = try x.dot(ctx, &self.lm_head, .d); // [.seq, padded_vocab]
            var logits = try logits_full.narrow(ctx, .vocab, 0, self.cfg.vocab_size);
            try rec(trace, "logits_pre_softcap", &logits);
            var capped = try logits.softcap15(ctx);
            try rec(trace, "logits_post_softcap", &capped);
            return capped;
        }

        /// The residual trunk shared by `forward` and the checkpointed loss head:
        /// embed → smear → layer loop → backout → final rmsNorm, returning the
        /// normalized stream [.seq,.d]. Identical op sequence to the pre-refactor
        /// forward up to the lm_head.
        fn forwardTrunk(self: *const Self, ctx: *ExecContext, token_ids: []const usize, trace: ?*Trace) !Tensor(.{ .seq, .d }) {
            const cfg = self.cfg;
            const n = token_ids.len;
            const hd = cfg.headDim();
            std.debug.assert(n > 1); // training forward has T>1 (gpt.py:478)

            // Rotary table: positions 0..T-1, base 100000, inverse=true so the stock
            // .half kernel reproduces nanochat's y1=x1c+x2s, y2=-x1s+x2c
            // (gpt.py:57-65). feature_dim = head_dim. The init-time table covers the
            // training/eval length; ad-hoc lengths (short greedy previews) derive
            // their own, freed on return (each rope VJP clones the table it uses).
            var adhoc: ?RopeTable = null;
            defer if (adhoc) |*t| t.deinit();
            const table: *const RopeTable = if (n == cfg.sequence_len)
                &self.rope_table
            else blk: {
                adhoc = try buildRopeTable(ctx, self.allocator, n, hd);
                break :blk &adhoc.?;
            };

            var name_buf: [32]u8 = undefined;

            // 1. Embed + norm after token embedding (gpt.py:471-473).
            var x = try self.wte.gather(ctx, .vocab, token_ids, .seq); // [.seq,.d]
            x = try x.rmsNorm(ctx, .d, rms_eps);
            try rec(trace, "emb_norm", &x);

            // 2. Smear: mix previous token's embedding into the current position
            //    (gpt.py:479-480). gate = smear_lambda·σ(smear_gate(x[:,1:,:24])).
            {
                var x_cur = try x.narrow(ctx, .seq, 1, n - 1); // rows 1..
                var x_prev = try x.narrow(ctx, .seq, 0, n - 1); // rows ..-1
                var g_in = try x_cur.narrow(ctx, .d, 0, 24);
                var g_lin = try g_in.dot(ctx, &self.smear_gate, .d); // [.seq,.one]
                var g_sq = try g_lin.squeeze(ctx, .one); // [.seq]
                var g_sig = try g_sq.sigmoid(ctx);
                var lam = try self.smear_lambda.squeeze(ctx, .one); // scalar
                var gate = try g_sig.mul(ctx, &lam); // [.seq]
                var mixed_prev = try gate.mul(ctx, &x_prev); // broadcast .d
                var mixed = try x_cur.add(ctx, &mixed_prev);
                var row0 = try x.narrow(ctx, .seq, 0, 1);
                x = try row0.concat(ctx, .seq, &.{&mixed}); // [.seq,.d]
            }
            try rec(trace, "post_smear", &x);

            // x0 saved AFTER smear, before the layer loop (gpt.py:495).
            const x0 = x;
            try rec(trace, "x0", &x0);

            const backout_layer = cfg.backoutLayer();
            var x_backout: ?Tensor(.{ .seq, .d }) = null;

            for (0..cfg.n_layer) |i| {
                // a. residual mix: resid_lambdas[i]·x + x0_lambdas[i]·x0 (gpt.py:500).
                var rl = try (try self.resid_lambdas.narrow(ctx, .layer, i, 1)).squeeze(ctx, .layer);
                var xl = try (try self.x0_lambdas.narrow(ctx, .layer, i, 1)).squeeze(ctx, .layer);
                var t1 = try x.mul(ctx, &rl);
                var t2 = try x0.mul(ctx, &xl);
                x = try t1.add(ctx, &t2);
                try rec(trace, try std.fmt.bufPrint(&name_buf, "resid_in.{d}", .{i}), &x);

                const l = &self.layers[i];

                // b. block: attn(norm(x)) then mlp(norm(x)) (gpt.py Block.forward).
                var h = try x.rmsNorm(ctx, .d, rms_eps); // = norm(x), the attn input

                var q = try (try h.dot(ctx, &l.c_q, .d)).split(ctx, .qo, .{ .head, .d }, [_]usize{ cfg.n_head, hd });
                var k = try (try h.dot(ctx, &l.c_k, .d)).split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ cfg.n_kv_head, hd });
                var v = try (try h.dot(ctx, &l.c_v, .d)).split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ cfg.n_kv_head, hd });

                // Value residual (ResFormer, gpt.py:94-97): gate·value-embedding.
                if (self.value_embeds[i]) |ve_w| {
                    var ve = try (try ve_w.gather(ctx, .vocab, token_ids, .seq)).split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ cfg.n_kv_head, hd });
                    var g = try (try h.narrow(ctx, .d, 0, 12)).dot(ctx, &l.ve_gate.?, .d); // [.seq,.kv_head]
                    var gate = try (try g.sigmoid(ctx)).scale(ctx, 3.0); // 3·σ, range (0,3)
                    try rec(trace, try std.fmt.bufPrint(&name_buf, "ve_gate.{d}", .{i}), &gate);
                    var add_v = try gate.mul(ctx, &ve); // broadcast over .d
                    v = try v.add(ctx, &add_v);
                }

                // RoPE then QK-norm then ×1.2 (gpt.py:101-104) — order matters, so
                // this uses the un-fused rope+rmsNorm sequence.
                q = try q.rope(ctx, .seq, .d, table, .half);
                k = try k.rope(ctx, .seq, .d, table, .half);
                q = try q.rmsNorm(ctx, .d, rms_eps);
                k = try k.rmsNorm(ctx, .d, rms_eps);
                q = try q.scale(ctx, 1.2);
                k = try k.scale(ctx, 1.2);

                var y = try q.groupedAttention(ctx, &k, &v, self.kv_head_for_head, .attn, self.attn_scale, .{ .window = cfg.windowFor(i) });

                var attn_out = try y.dot(ctx, &l.c_proj, .attn); // [.seq,.d]
                try rec(trace, try std.fmt.bufPrint(&name_buf, "attn_out.{d}", .{i}), &attn_out);
                x = try x.add(ctx, &attn_out);

                // MLP: c_proj( relu(c_fc(norm(x)))² ) (gpt.py MLP.forward).
                var m = try x.rmsNorm(ctx, .d, rms_eps);
                var m_fc = try m.dot(ctx, &l.c_fc, .d); // [.seq,.ff]
                var m_relu = try m_fc.relu(ctx);
                var m_sq = try m_relu.mul(ctx, &m_relu); // relu²
                var mlp_out = try m_sq.dot(ctx, &l.c_proj_mlp, .ff); // [.seq,.d]
                try rec(trace, try std.fmt.bufPrint(&name_buf, "mlp_out.{d}", .{i}), &mlp_out);
                x = try x.add(ctx, &mlp_out);
                try rec(trace, try std.fmt.bufPrint(&name_buf, "block_out.{d}", .{i}), &x);

                if (i == backout_layer) x_backout = x;
            }

            // Backout: subtract the cached mid-layer residual (gpt.py:506-507).
            if (x_backout) |xb| {
                try rec(trace, "x_backout", &xb);
                var bl = try self.backout_lambda.squeeze(ctx, .one);
                var term = try xb.mul(ctx, &bl);
                x = try x.sub(ctx, &term);
            }
            try rec(trace, "pre_final_norm", &x);
            x = try x.rmsNorm(ctx, .d, rms_eps);
            try rec(trace, "post_final_norm", &x);
            return x;
        }

        /// Incremental KV-cache forward for one chunk of tokens starting at absolute
        /// position `pos0` (gpt.py forward with kv_cache != None). Returns the
        /// post-softcap logits [.seq,.vocab] for the chunk. Works for prefill (T>1,
        /// pos0=0) and single-token decode (T=1, pos0>0). f32 KV, no window here
        /// beyond the configured pattern. Requires n_kv_head == n_head is NOT needed
        /// (GQA handled by groupedAttention); the causal kernel's source_offset =
        /// kv_seq - q_seq aligns each query to its absolute position.
        pub fn forwardStep(self: *const Self, ctx: *ExecContext, cache: *Cache, token_ids: []const usize, pos0: usize) !Tensor(.{ .seq, .vocab }) {
            const cfg = self.cfg;
            const t = token_ids.len;
            const hd = cfg.headDim();
            std.debug.assert(pos0 == cache.len);

            if (cache.len + t > cache.cap) return error.CacheFull;

            const positions = try self.allocator.alloc(i32, t);
            defer self.allocator.free(positions);
            for (positions, 0..) |*p, i| p.* = @intCast(pos0 + i);
            var table = try ctx.prepareRopeTable(positions, hd, rope_theta, true);
            defer table.deinit();

            var x_norm = try (try self.wte.gather(ctx, .vocab, token_ids, .seq)).rmsNorm(ctx, .d, rms_eps);

            // Smear (gpt.py kv-cache branch, lines 483-492).
            var x: Tensor(.{ .seq, .d }) = undefined;
            if (t > 1) {
                // Prefill: same fast slice as training (row 0 unchanged).
                var x_cur = try x_norm.narrow(ctx, .seq, 1, t - 1);
                var x_prev = try x_norm.narrow(ctx, .seq, 0, t - 1);
                var g_lin = try (try x_cur.narrow(ctx, .d, 0, 24)).dot(ctx, &self.smear_gate, .d);
                var g_sig = try (try g_lin.squeeze(ctx, .one)).sigmoid(ctx);
                var lam = try self.smear_lambda.squeeze(ctx, .one);
                var gate = try g_sig.mul(ctx, &lam);
                var mixed = try x_cur.add(ctx, &(try gate.mul(ctx, &x_prev)));
                var row0 = try x_norm.narrow(ctx, .seq, 0, 1);
                x = try row0.concat(ctx, .seq, &.{&mixed});
            } else if (cache.has_prev) {
                // Decode: single token mixes with the cached previous embedding.
                var g_lin = try (try x_norm.narrow(ctx, .d, 0, 24)).dot(ctx, &self.smear_gate, .d);
                var g_sig = try (try g_lin.squeeze(ctx, .one)).sigmoid(ctx);
                var lam = try self.smear_lambda.squeeze(ctx, .one);
                var gate = try g_sig.mul(ctx, &lam); // [.seq(1)]
                // Borrow the old cached embedding (a manual constant, not scope-owned,
                // so deinit it; the mul's backward node clones it via cloneView). The
                // prev buffer is only overwritten below — no backward runs here.
                var prev = try Tensor(.{.d}).fromBorrowedSlice(ctx, [_]usize{cfg.n_embd}, cache.prev_norm_emb);
                defer prev.deinit();
                x = try x_norm.add(ctx, &(try gate.mul(ctx, &prev)));
            } else {
                x = x_norm;
            }

            // Update the cached previous embedding to this chunk's last normed row.
            {
                var last = try x_norm.narrow(ctx, .seq, t - 1, 1);
                try last.copyTo(cache.prev_norm_emb);
                cache.has_prev = true;
            }

            const x0 = x;
            const backout_layer = cfg.backoutLayer();
            var x_backout: ?Tensor(.{ .seq, .d }) = null;
            const per = cfg.n_kv_head * hd;
            const base = cache.len;
            const new_len = base + t;

            for (0..cfg.n_layer) |i| {
                var rl = try (try self.resid_lambdas.narrow(ctx, .layer, i, 1)).squeeze(ctx, .layer);
                var xl = try (try self.x0_lambdas.narrow(ctx, .layer, i, 1)).squeeze(ctx, .layer);
                x = try (try x.mul(ctx, &rl)).add(ctx, &(try x0.mul(ctx, &xl)));

                const l = &self.layers[i];
                var h = try x.rmsNorm(ctx, .d, rms_eps);
                var q = try (try h.dot(ctx, &l.c_q, .d)).split(ctx, .qo, .{ .head, .d }, [_]usize{ cfg.n_head, hd });
                var k = try (try h.dot(ctx, &l.c_k, .d)).split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ cfg.n_kv_head, hd });
                var v = try (try h.dot(ctx, &l.c_v, .d)).split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ cfg.n_kv_head, hd });

                if (self.value_embeds[i]) |ve_w| {
                    var ve = try (try ve_w.gather(ctx, .vocab, token_ids, .seq)).split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ cfg.n_kv_head, hd });
                    var g = try (try h.narrow(ctx, .d, 0, 12)).dot(ctx, &l.ve_gate.?, .d);
                    var gate = try (try g.sigmoid(ctx)).scale(ctx, 3.0);
                    v = try v.add(ctx, &(try gate.mul(ctx, &ve)));
                }

                q = try (try q.rope(ctx, .seq, .d, &table, .half)).rmsNorm(ctx, .d, rms_eps);
                k = try (try k.rope(ctx, .seq, .d, &table, .half)).rmsNorm(ctx, .d, rms_eps);
                q = try q.scale(ctx, 1.2);
                k = try k.scale(ctx, 1.2);

                // Append K/V for this chunk to the persistent cache, then attend the
                // whole cache. The causal kernel's source_offset = new_len - t = pos0
                // gives absolute-position causality.
                try k.copyTo(cache.k[i][base * per ..][0 .. t * per]);
                try v.copyTo(cache.v[i][base * per ..][0 .. t * per]);
                // Manual constants (not scope-owned): deinit them. The attention
                // backward node clones k/v via cloneView, so early deinit is safe;
                // the borrowed cache buffers are freed by Cache.deinit, not here.
                var k_all = try Tensor(.{ .seq, .kv_head, .d }).fromBorrowedSlice(ctx, [_]usize{ new_len, cfg.n_kv_head, hd }, cache.k[i][0 .. new_len * per]);
                defer k_all.deinit();
                var v_all = try Tensor(.{ .seq, .kv_head, .d }).fromBorrowedSlice(ctx, [_]usize{ new_len, cfg.n_kv_head, hd }, cache.v[i][0 .. new_len * per]);
                defer v_all.deinit();

                var y = try q.groupedAttention(ctx, &k_all, &v_all, self.kv_head_for_head, .attn, self.attn_scale, .{ .window = cfg.windowFor(i) });

                x = try x.add(ctx, &(try y.dot(ctx, &l.c_proj, .attn)));

                var m = try x.rmsNorm(ctx, .d, rms_eps);
                var m_relu = try (try m.dot(ctx, &l.c_fc, .d)).relu(ctx);
                var mlp_out = try (try m_relu.mul(ctx, &m_relu)).dot(ctx, &l.c_proj_mlp, .ff);
                x = try x.add(ctx, &mlp_out);
                if (i == backout_layer) x_backout = x;
            }
            cache.len = new_len;

            if (x_backout) |xb| {
                var bl = try self.backout_lambda.squeeze(ctx, .one);
                x = try x.sub(ctx, &(try xb.mul(ctx, &bl)));
            }
            x = try x.rmsNorm(ctx, .d, rms_eps);
            var logits = try (try x.dot(ctx, &self.lm_head, .d)).narrow(ctx, .vocab, 0, cfg.vocab_size);
            return logits.softcap15(ctx);
        }

        /// Mean cross-entropy over this single sequence's non-ignored targets
        /// (gpt.py:520 with reduction='mean' on one row).
        pub fn loss(self: *const Self, ctx: *ExecContext, token_ids: []const usize, targets: []const isize) !Tensor(.{}) {
            const labels = try self.allocator.alloc(usize, targets.len);
            defer self.allocator.free(labels); // crossEntropyExt dupes labels
            toLabels(targets, labels);
            const logits = try self.forward(ctx, token_ids, null);
            return logits.crossEntropyExt(ctx, .vocab, labels, .{ .reduction = .mean, .ignore_index = ignore_index });
        }

        /// Per-token cross-entropy [.seq] (reduction='none'); ignored positions = 0.
        pub fn lossNone(self: *const Self, ctx: *ExecContext, token_ids: []const usize, targets: []const isize) !Tensor(.{.seq}) {
            const labels = try self.allocator.alloc(usize, targets.len);
            defer self.allocator.free(labels);
            toLabels(targets, labels);
            const logits = try self.forward(ctx, token_ids, null);
            return logits.crossEntropyExt(ctx, .vocab, labels, .{ .reduction = .none, .ignore_index = ignore_index });
        }

        /// Summed cross-entropy over this sequence's non-ignored targets (the
        /// numerator of the batch mean). Used by the (B,T) grad-accumulation path.
        pub fn lossSum(self: *const Self, ctx: *ExecContext, token_ids: []const usize, labels: []const usize) !Tensor(.{}) {
            const logits = try self.forward(ctx, token_ids, null);
            return logits.crossEntropyExt(ctx, .vocab, labels, .{ .reduction = .sum, .ignore_index = ignore_index });
        }
    };
}

pub const Model = ModelOf(.f32);

/// f32 KV cache for the incremental `forwardStep` path. Stores per-layer seq-major
/// [len, n_kv_head, head_dim] K/V plus the previous normed embedding used by the
/// single-token decode smear.
pub const Cache = struct {
    allocator: Allocator,
    cap: usize,
    k: [][]f32,
    v: [][]f32,
    len: usize,
    prev_norm_emb: []f32,
    has_prev: bool,

    pub fn init(allocator: Allocator, cfg: Config, capacity: usize) !Cache {
        const per = cfg.n_kv_head * cfg.headDim();
        const k = try allocator.alloc([]f32, cfg.n_layer);
        errdefer allocator.free(k);
        const v = try allocator.alloc([]f32, cfg.n_layer);
        errdefer allocator.free(v);
        var made: usize = 0;
        errdefer for (0..made) |i| {
            allocator.free(k[i]);
            allocator.free(v[i]);
        };
        for (0..cfg.n_layer) |i| {
            k[i] = try allocator.alloc(f32, capacity * per);
            v[i] = try allocator.alloc(f32, capacity * per);
            made = i + 1;
        }
        const prev = try allocator.alloc(f32, cfg.n_embd);
        return .{ .allocator = allocator, .cap = capacity, .k = k, .v = v, .len = 0, .prev_norm_emb = prev, .has_prev = false };
    }

    pub fn deinit(self: *Cache) void {
        for (self.k) |buf| self.allocator.free(buf);
        for (self.v) |buf| self.allocator.free(buf);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.prev_norm_emb);
        self.* = undefined;
    }
};

fn toLabels(targets: []const isize, labels: []usize) void {
    for (targets, labels) |t, *l| l.* = if (t < 0) ignore_index else @intCast(t);
}

/// Rotary table for positions 0..seq_len-1 (rope_theta base, inverse=true so
/// the stock .half kernel reproduces nanochat's rotation — gpt.py:57-65).
fn buildRopeTable(ctx: *ExecContext, allocator: Allocator, seq_len: usize, head_dim: usize) !RopeTable {
    const positions = try allocator.alloc(i32, seq_len);
    defer allocator.free(positions);
    for (positions, 0..) |*p, i| p.* = @intCast(i);
    return ctx.prepareRopeTable(positions, head_dim, rope_theta, true);
}

fn lname(buf: []u8, layer: usize, comptime suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "transformer.h.{d}." ++ suffix, .{layer});
}

fn veName(buf: []u8, layer: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "value_embeds.{d}.weight", .{layer});
}

/// The facade type for a param of `tags` at matrix dtype `dt`: the plain f32
/// facade for `.f32` (identical to the pre-ModelOf types), the typed 16-bit
/// facade otherwise.
fn ParamTensor(comptime dt: fucina.DType, comptime tags: anytype) type {
    if (dt == .f32) return Tensor(tags);
    return Tensor(.{ .dtype = dt, .tags = tags });
}

fn loadParam(
    comptime tags: anytype,
    ctx: *ExecContext,
    allocator: Allocator,
    file: *const safetensors.File,
    name: []const u8,
    shape: anytype,
) !Tensor(tags) {
    return loadParamAs(.f32, tags, ctx, allocator, file, name, shape);
}

/// Matrix-dtype-aware load: f32 targets require F32 entries (unchanged);
/// bf16 targets accept BF16 (raw bits — the trainer's own checkpoints) or
/// F32 (narrowed on load — the f32 init/golden files).
fn loadParamAs(
    comptime dt: fucina.DType,
    comptime tags: anytype,
    ctx: *ExecContext,
    allocator: Allocator,
    file: *const safetensors.File,
    name: []const u8,
    shape: anytype,
) !ParamTensor(dt, tags) {
    const info = try file.tensor(name);
    if (info.shape.len != shape.len) return error.ShapeMismatch;
    var n: usize = 1;
    inline for (shape, 0..) |s, i| {
        if (info.shape[i] != s) return error.ShapeMismatch;
        n *= s;
    }
    if (comptime dt == .f32) {
        if (info.dtype != .F32) return error.UnexpectedDtype;
        const tmp = try allocator.alloc(f32, n);
        defer allocator.free(tmp);
        @memcpy(std.mem.sliceAsBytes(tmp), info.data[0 .. n * 4]);
        return ParamTensor(dt, tags).variableFromSlice(ctx, shape, tmp);
    }
    const tmp = try allocator.alloc(u16, n);
    defer allocator.free(tmp);
    switch (info.dtype) {
        .BF16 => @memcpy(std.mem.sliceAsBytes(tmp), info.data[0 .. n * 2]),
        .F32 => for (tmp, 0..) |*v, i| {
            const bits = std.mem.readInt(u32, info.data[i * 4 ..][0..4], .little);
            v.* = fucina.f32ToBf16(@bitCast(bits));
        },
        else => return error.UnexpectedDtype,
    }
    return ParamTensor(dt, tags).variableFromSlice(ctx, shape, tmp);
}

fn rec(trace: ?*Trace, name: []const u8, t: anytype) !void {
    if (trace) |tr| try tr.record(name, t);
}

/// gpt.py pads vocab up to a multiple of 64 (GPT.__init__ pad_vocab_size_to=64).
pub fn paddedVocab(vocab_size: usize) usize {
    return ((vocab_size + 63) / 64) * 64;
}

/// gpt.py has_ve(layer_idx, n_layer): alternating value-embedding layers, last
/// layer included. Shared by Config.hasVe and train.zig's scaling-param count.
pub fn hasVeAt(layer_idx: usize, n_layer: usize) bool {
    return layer_idx % 2 == (n_layer - 1) % 2;
}

/// A per-parameter fill rule for `initRandom`.
const RngFill = union(enum) {
    normal: struct { mean: f32, std: f32 },
    uniform: struct { lo: f32, hi: f32 },
    zeros,
    constant: f32,
};

/// Allocate a scratch buffer, fill it per `fill` (seeded by `seed`), and build a
/// leaf parameter tensor from it (variableFromSlice copies, so the scratch is
/// freed on return).
fn rngParam(
    comptime tags: anytype,
    ctx: *ExecContext,
    allocator: Allocator,
    shape: anytype,
    fill: RngFill,
    seed: u64,
) !Tensor(tags) {
    return rngParamAs(.f32, tags, ctx, allocator, shape, fill, seed);
}

/// `rngParam` at a matrix dtype: the fill always runs in f32 (the RNG
/// streams stay identical across dtypes) and narrows once at construction.
fn rngParamAs(
    comptime dt: fucina.DType,
    comptime tags: anytype,
    ctx: *ExecContext,
    allocator: Allocator,
    shape: anytype,
    fill: RngFill,
    seed: u64,
) !ParamTensor(dt, tags) {
    var n: usize = 1;
    inline for (shape) |dim| n *= dim;
    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    switch (fill) {
        .normal => |p| fucina.rng.normalFill(seed, buf, p.mean, p.std),
        .uniform => |p| fucina.rng.uniformFill(seed, buf, p.lo, p.hi),
        .zeros => @memset(buf, 0),
        .constant => |v| @memset(buf, v),
    }
    if (comptime dt == .f32) {
        return ParamTensor(dt, tags).variableFromSlice(ctx, shape, buf);
    }
    const nbuf = try allocator.alloc(u16, n);
    defer allocator.free(nbuf);
    for (nbuf, buf) |*v, x| v.* = fucina.f32ToBf16(x);
    return ParamTensor(dt, tags).variableFromSlice(ctx, shape, nbuf);
}

// ---------------------------------------------------------------------------
// Trace: owned copies of forward intermediates for oracle comparison.
// ---------------------------------------------------------------------------

pub const Trace = struct {
    pub const Entry = struct {
        name: []u8,
        data: []f32,
        shape: []usize,
    };

    allocator: Allocator,
    entries: std.ArrayList(Entry),

    pub fn init(allocator: Allocator) Trace {
        return .{ .allocator = allocator, .entries = .empty };
    }

    pub fn deinit(self: *Trace) void {
        for (self.entries.items) |e| {
            self.allocator.free(e.name);
            self.allocator.free(e.data);
            self.allocator.free(e.shape);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    fn record(self: *Trace, name: []const u8, t: anytype) !void {
        const raw = t.asRawTensor();
        const len = raw.len();
        const data = try self.allocator.alloc(f32, len);
        errdefer self.allocator.free(data);
        try t.copyTo(data);
        const shape = try self.allocator.dupe(usize, raw.shape.slice());
        errdefer self.allocator.free(shape);
        const nm = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(nm);
        try self.entries.append(self.allocator, .{ .name = nm, .data = data, .shape = shape });
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    std.testing.refAllDecls(@This());
}

comptime {
    _ = @import("model_tests.zig");
}
