//! Zig-native SHINE training step (arXiv 2602.06358): the full
//! differentiable chain — Meta-LoRA encoder pass with per-layer memory
//! capture, M2P, "rl" slicing, adapted conversation pass, masked CE — as
//! ONE autograd graph, so the loss backpropagates through the generated
//! adapter into the metanetwork and the Meta LoRA.
//!
//! Composition: the base model's frozen projections run through the
//! trainer's differentiable frozen dots (train.zig `dotLinear` +
//! `FrozenCache`), the LoRA side paths and the M2P reuse shine.zig's
//! facade code with gradient-carrying variables in place of constants
//! (same types, same ops), and the memory capture / adapter slicing are
//! graph-preserving views. Memory tokens stay constant zeros, matching
//! the reference training exactly (they are excluded from its optimizer
//! and never move).
//!
//! Every `loss` call must run under an exec scope (the caller opens it,
//! like the LoRA trainer's tests); defer-deinit inside is then a safe
//! no-op and the retained graph is freed at scope close.
//!
//! With `checkpoint_layers` every base layer (both passes) runs as one
//! `fucina.checkpointWithContext` block: only the layer-boundary hiddens
//! stay retained, which is what lets packed multi-example steps fit in
//! RAM, and the block's grad-free first forward makes its GEMMs GPU-
//! eligible. Step-transient state the backward still reads (rope tables,
//! segment vectors) then lives on the trainer until `freeTransient`.
const std = @import("std");
const fucina = @import("fucina");
const weights = @import("fucina").weights;
const qwen3 = @import("../../qwen3/model.zig");
const qwen3_train = @import("../../qwen3/train.zig");
const shine = @import("shine.zig");
const cartridge_mod = @import("../../text/cartridge.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;

pub const Error = shine.Error;

const Hidden = fucina.Tensor(.{ .seq, .embed });
const ATensor = fucina.Tensor(.{ .lin, .lora_r });
const BTensor = fucina.Tensor(.{ .lora_r, .lout });

/// Constant context of one packed-layer block, stored by value in the
/// checkpoint node: everything it points at is model-, trainer-, or
/// step-retained (see `freeTransient`) and is treated as frozen.
const LayerExtra = struct {
    model: *const qwen3.Model,
    cache: ?*qwen3_train.FrozenCache,
    layer: *const qwen3.Layer,
    segs: []const usize,
    rope: *const fucina.RopeTable,
    /// Offset step into the pack's `.lora_r` axis per segment: the pair
    /// rank for per-segment packs, 0 when every segment shares one pair
    /// (the Meta LoRA, or a single-example pack).
    pair_stride: usize,
    /// One segment's pair rank (the `.lora_r` slice length).
    pair_rank: usize,
};

/// The block's differentiable inputs: the hidden pack plus (a, b) per
/// module — arity fixed at comptime for ANY pack width, the tuple-form
/// checkpoint contract.
const InputsTuple = std.meta.Tuple(&([_]type{*const Hidden} ++ ([_]type{ *const ATensor, *const BTensor } ** 7)));

fn packInputs(x: *const Hidden, pack: *const shine.LayerLora) InputsTuple {
    var inputs: InputsTuple = undefined;
    inputs[0] = x;
    inline for (shine.modules, 0..) |module, mi| {
        inputs[1 + 2 * mi] = &pack.pair(module).a;
        inputs[2 + 2 * mi] = &pack.pair(module).b;
    }
    return inputs;
}

/// Checkpoint-block wrapper of `layerBody`: rebuilds each segment's pair
/// set as `.lora_r` slices of the packed inputs (pure views), so one
/// block type serves the encoder pass, the single-example conversation
/// pass, and packed conversation passes.
const LayerBlock = struct {
    fn run(ctx: *ExecContext, extra: LayerExtra, inputs: InputsTuple) !Hidden {
        const a = ctx.allocator();
        const lora = try a.alloc(shine.LayerLora, extra.segs.len);
        defer a.free(lora);
        var built: usize = 0;
        defer for (lora[0..built]) |*l| l.deinit(); // scope-owned views: no-ops
        for (lora, 0..) |*l, i| {
            inline for (shine.modules, 0..) |module, mi| {
                var pa = try inputs[1 + 2 * mi].narrow(ctx, .lora_r, i * extra.pair_stride, extra.pair_rank);
                errdefer pa.deinit();
                const pb = try inputs[2 + 2 * mi].narrow(ctx, .lora_r, i * extra.pair_stride, extra.pair_rank);
                @field(l.*, @tagName(module)) = .{ .a = pa, .b = pb };
            }
            built = i + 1;
        }
        const ptrs = try a.alloc(*const shine.LayerLora, lora.len);
        defer a.free(ptrs);
        for (ptrs, lora) |*ptr, *l| ptr.* = l;
        return layerBody(ctx, extra, inputs[0], ptrs);
    }
};

/// Values for one M2P block, as raw row-major slices (the golden test's
/// construction surface; a randn initializer can fill the same shape).
pub const M2pBlockValues = struct {
    in_w: []const f32, // [3H, H]
    in_b: []const f32,
    out_w: []const f32, // [H, H]
    out_b: []const f32,
    ff1_w: []const f32, // [FF, H]
    ff1_b: []const f32,
    ff2_w: []const f32, // [H, FF]
    ff2_b: []const f32,
    norm1_w: []const f32,
    norm1_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
};

pub const ShineTrainer = struct {
    allocator: Allocator,
    model: *const qwen3.Model,
    config: shine.Config,
    frozen_cache: qwen3_train.FrozenCache = .{},
    /// See train.zig `widen_frozen` — same trade, same default.
    widen_frozen: bool = true,
    /// Recompute-in-backward per base layer (the ag checkpoint component,
    /// see the module doc). Off by default; packed steps want it on.
    checkpoint_layers: bool = false,
    /// Step-transient rope tables and segment vectors: the rope backward
    /// and the checkpoint recompute read them during backward, so they
    /// live until `freeTransient`.
    transient_tables: std.ArrayListUnmanaged(*fucina.RopeTable) = .empty,
    transient_segs: std.ArrayListUnmanaged([]usize) = .empty,
    /// Lazily built no-adapter qwen3 trainer: `lossCartridge`'s
    /// conversation pass runs ITS cartridge forward (the same gated
    /// machinery a distilled cartridge trains and serves through), with
    /// the generated prefix rows borrowed in as graph tensors.
    conv_trainer: ?qwen3_train.Trainer(.{}) = null,
    /// Leaf registry (built by `initRandom`/`buildRegistry`): the optimizer
    /// and save/load surface, one named entry per trainable tensor.
    /// Optimizers registered via `registerAllParams` borrow the entries, so
    /// the trainer must outlive them.
    registry: ?fucina.ParamRegistry = null,
    /// Trainable leaves: the Meta LoRA pairs, the M2P blocks, and the
    /// positional embeddings. All fucina.Tensor variables.
    metalora: shine.LoraSet,
    m2p: []shine.M2pBlock,
    layer_pe: fucina.Tensor(.{ .layer, .embed }),
    token_pe: fucina.Tensor(.{ .mem, .embed }),

    pub fn deinit(self: *ShineTrainer) void {
        if (self.registry) |*registry| registry.deinit(); // retains leaf views: first
        if (self.conv_trainer) |*trainer| trainer.deinit();
        self.freeTransient();
        self.transient_tables.deinit(self.allocator);
        self.transient_segs.deinit(self.allocator);
        self.frozen_cache.deinit(self.allocator);
        self.token_pe.deinit();
        self.layer_pe.deinit();
        for (self.m2p) |*block| block.deinit(self.allocator);
        self.allocator.free(self.m2p);
        self.metalora.deinit();
        self.* = undefined;
    }

    fn frozenCache(self: *ShineTrainer) ?*qwen3_train.FrozenCache {
        return if (self.widen_frozen) &self.frozen_cache else null;
    }

    /// Free the step-transient rope tables and segment vectors. Call
    /// between optimizer steps, never between a forward and its backward
    /// (the rope backward and the checkpoint recompute read them).
    pub fn freeTransient(self: *ShineTrainer) void {
        for (self.transient_tables.items) |table| {
            table.deinit();
            self.allocator.destroy(table);
        }
        self.transient_tables.clearRetainingCapacity();
        for (self.transient_segs.items) |segs| self.allocator.free(segs);
        self.transient_segs.clearRetainingCapacity();
    }

    /// A step-retained segment-length vector (rides the checkpoint
    /// `extra`, so it must outlive backward — see `freeTransient`).
    fn retainedSegs(self: *ShineTrainer, n: usize) ![]usize {
        const segs = try self.allocator.alloc(usize, n);
        errdefer self.allocator.free(segs);
        try self.transient_segs.append(self.allocator, segs);
        return segs;
    }

    /// One SHINE step's scalar loss (mean CE over unmasked labels; labels
    /// are PRE-SHIFTED next tokens with `train.ignore_index` masking, the
    /// LoRA trainer's convention). Backward flows into every variable.
    pub fn loss(
        self: *ShineTrainer,
        ctx: *ExecContext,
        evidence_ids: []const usize,
        input_ids: []const usize,
        labels: []const usize,
    ) !fucina.Tensor(.{}) {
        return self.lossPacked(ctx, &.{.{ .evidence = evidence_ids, .input = input_ids, .labels = labels }});
    }

    /// Fresh trainable leaves from a seeded PRNG: Meta-LoRA A ~ N(0, 0.02)
    /// with B = 0 (the standard LoRA start — the encoder side path begins
    /// exactly at the base model), M2P weights ~ N(0, 0.02) with zero
    /// biases and unit norms, positional embeddings ~ N(0, 0.02). Builds
    /// the leaf registry. Same seed, same leaves — bitwise.
    pub fn initRandom(ctx: *ExecContext, allocator: Allocator, model: *const qwen3.Model, config: shine.Config, seed: u64) !ShineTrainer {
        try config.validate(model.config);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const base = model.config;
        const h = config.hidden_size;

        const fill = struct {
            fn noise(rand: std.Random, values: []f32) void {
                for (values) |*v| v.* = rand.floatNorm(f32) * 0.02;
            }
        };

        const layers = try allocator.alloc(shine.LayerLora, base.num_layers);
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*layer| layer.deinit();
            allocator.free(layers);
        }
        for (layers) |*layer| {
            var pairs: [7]shine.LoraPair = undefined;
            var pair_built: usize = 0;
            errdefer for (pairs[0..pair_built]) |*pair| pair.deinit();
            inline for (shine.modules, 0..) |_, gi| {
                const dims = shine.moduleDims(base, shine.modules[gi]);
                const a_values = try allocator.alloc(f32, dims[0] * config.metalora_r);
                defer allocator.free(a_values);
                fill.noise(random, a_values);
                const b_values = try allocator.alloc(f32, config.metalora_r * dims[1]);
                defer allocator.free(b_values);
                @memset(b_values, 0);
                pairs[gi] = try pairVariables(ctx, dims[0], config.metalora_r, dims[1], a_values, b_values);
                pair_built = gi + 1;
            }
            layer.* = .{ .q = pairs[0], .k = pairs[1], .v = pairs[2], .o = pairs[3], .gate = pairs[4], .up = pairs[5], .down = pairs[6] };
            built += 1;
        }
        var metalora = shine.LoraSet{ .allocator = allocator, .layers = layers };
        errdefer metalora.deinit();

        const m2p = try allocator.alloc(shine.M2pBlock, config.m2p_layers);
        var m2p_built: usize = 0;
        errdefer {
            for (m2p[0..m2p_built]) |*block| block.deinit(allocator);
            allocator.free(m2p);
        }
        const ff = config.m2p_ffn;
        for (m2p) |*block| {
            const lens = [_]usize{ 3 * h * h, 3 * h, h * h, h, ff * h, ff, h * ff, h, h, h, h, h };
            // weight, bias alternating; norms live in the last four slots.
            const is_weight = [_]bool{ true, false, true, false, true, false, true, false, false, false, false, false };
            var bufs: [12][]f32 = undefined;
            var bufs_built: usize = 0;
            defer for (bufs[0..bufs_built]) |values| allocator.free(values);
            inline for (0..12) |pi| {
                bufs[pi] = try allocator.alloc(f32, lens[pi]);
                bufs_built = pi + 1;
                if (is_weight[pi]) fill.noise(random, bufs[pi]) else @memset(bufs[pi], 0);
            }
            @memset(bufs[8], 1); // norm1.weight
            @memset(bufs[10], 1); // norm2.weight
            block.* = try m2pBlockVariables(ctx, config, .{
                .in_w = bufs[0],
                .in_b = bufs[1],
                .out_w = bufs[2],
                .out_b = bufs[3],
                .ff1_w = bufs[4],
                .ff1_b = bufs[5],
                .ff2_w = bufs[6],
                .ff2_b = bufs[7],
                .norm1_w = bufs[8],
                .norm1_b = bufs[9],
                .norm2_w = bufs[10],
                .norm2_b = bufs[11],
            });
            m2p_built += 1;
        }

        const lp = try allocator.alloc(f32, base.num_layers * h);
        defer allocator.free(lp);
        fill.noise(random, lp);
        const tp = try allocator.alloc(f32, config.num_mem_token * h);
        defer allocator.free(tp);
        fill.noise(random, tp);
        var layer_pe = try fucina.Tensor(.{ .layer, .embed }).variableFromSlice(ctx, .{ base.num_layers, h }, lp);
        errdefer layer_pe.deinit();
        var token_pe = try fucina.Tensor(.{ .mem, .embed }).variableFromSlice(ctx, .{ config.num_mem_token, h }, tp);
        errdefer token_pe.deinit();

        var trainer = ShineTrainer{
            .allocator = allocator,
            .model = model,
            .config = config,
            .metalora = metalora,
            .m2p = m2p,
            .layer_pe = layer_pe,
            .token_pe = token_pe,
        };
        try trainer.buildRegistry();
        return trainer;
    }

    /// Build the leaf registry over the trainer's tensors (no-op when
    /// already built). Names: `metalora.<layer>.<module>.{a,b}`,
    /// `m2p.<block>.<field>`, `layer_pe`, `token_pe`.
    pub fn buildRegistry(self: *ShineTrainer) !void {
        if (self.registry != null) return;
        var registry = fucina.ParamRegistry.init(self.allocator);
        errdefer registry.deinit();
        var name_buf: [64]u8 = undefined;
        for (self.metalora.layers, 0..) |*layer, i| {
            inline for (shine.modules) |module| {
                const pair = &@field(layer.*, @tagName(module));
                try registry.addParam(try std.fmt.bufPrint(&name_buf, "metalora.{d}.{s}.a", .{ i, @tagName(module) }), &pair.a);
                try registry.addParam(try std.fmt.bufPrint(&name_buf, "metalora.{d}.{s}.b", .{ i, @tagName(module) }), &pair.b);
            }
        }
        for (self.m2p, 0..) |*block, i| {
            inline for (.{ "in_w", "in_b", "out_w", "out_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b", "norm1_w", "norm1_b", "norm2_w", "norm2_b" }) |field| {
                try registry.addParam(try std.fmt.bufPrint(&name_buf, "m2p.{d}.{s}", .{ i, field }), &@field(block.*, field));
            }
        }
        try registry.addParam("layer_pe", &self.layer_pe);
        try registry.addParam("token_pe", &self.token_pe);
        self.registry = registry;
    }

    /// Register every trainable leaf with an optimizer (`optim.AdamW` et
    /// al.) — the finetune/cartridge wiring. The trainer must outlive the
    /// optimizer.
    pub fn registerAllParams(self: *ShineTrainer, opt: anytype) !void {
        try self.buildRegistry();
        try self.registry.?.addParamsTo(opt);
    }

    /// Serialize every leaf as a safetensors state dict (name-matched on
    /// load — the checkpoint surface, like the cartridge's `saveState`).
    pub fn saveParams(self: *ShineTrainer, writer: *std.Io.Writer) !void {
        try self.buildRegistry();
        try self.registry.?.saveStateDict(writer);
    }

    /// Load a state dict saved by `saveParams` (strict one-to-one match).
    pub fn loadParams(self: *ShineTrainer, reader: *std.Io.Reader) !void {
        try self.buildRegistry();
        try self.registry.?.loadStateDict(reader, .{});
    }

    /// One SHINE step with the CARTRIDGE readout: the encoder + M2P run
    /// exactly as in the LoRA mode, the generated block is sliced into a
    /// per-layer KV prefix (graph views), and the conversation is the
    /// qwen3 trainer's own cartridge forward over that prefix — the same
    /// code path a distilled cartridge trains through, so RoPE offsets,
    /// prefix attention, and the CE convention are shared, and the loss
    /// backpropagates through the generated rows into the M2P and the
    /// Meta LoRA. Labels are PRE-SHIFTED next tokens with
    /// `train.ignore_index` masking, the house convention.
    pub fn lossCartridge(
        self: *ShineTrainer,
        ctx: *ExecContext,
        evidence_ids: []const usize,
        input_ids: []const usize,
        labels: []const usize,
    ) !fucina.Tensor(.{}) {
        if (self.config.cartridge_rows == 0) return shine.Error.InvalidShineConfig;
        var plain = try self.generatedBlock(ctx, evidence_ids);
        defer plain.deinit();
        var cart = try sliceCartridgeViews(ctx, self.model.config, self.config, &plain);
        defer cart.deinit(); // graph views are scope-owned: deinit is a no-op walk
        const conv = try self.convTrainer(ctx);
        return conv.lossForwardExt(ctx, input_ids, labels, .{ .cartridge = &cart }, .{});
    }

    /// Encoder pass + M2P for ONE example (the packed path's b = 1 walk):
    /// per-layer memory captures stacked into the `[layer, mem, embed]`
    /// generated block, graph-linked to every trainable leaf. Public for
    /// eval harnesses (run it under `fucina.noGrad` to generate without
    /// building a graph).
    pub fn generatedBlock(self: *ShineTrainer, ctx: *ExecContext, evidence_ids: []const usize) !fucina.Tensor(.{ .layer, .mem, .embed }) {
        if (evidence_ids.len == 0) return qwen3.Error.InvalidSequenceLength;
        const a = ctx.allocator();
        const base = self.model.config;
        const mem = self.config.num_mem_token;

        const enc_segs = try self.retainedSegs(1);
        enc_segs[0] = evidence_ids.len + mem;
        const enc_rope = try self.segmentRope(ctx, enc_segs, enc_segs[0]);

        const example = Example{ .evidence = evidence_ids, .input = &.{}, .labels = &.{} };
        var x = try self.packedEncoderInput(ctx, &.{example});
        defer x.deinit();

        const captures = try a.alloc(fucina.Tensor(.{ .seq, .embed }), base.num_layers);
        defer a.free(captures);
        var captured: usize = 0;
        defer for (captures[0..captured]) |*view| view.deinit();
        for (self.model.layers, 0..) |*layer, layer_i| {
            const extra = LayerExtra{
                .model = self.model,
                .cache = self.frozenCache(),
                .layer = layer,
                .segs = enc_segs,
                .rope = enc_rope,
                .pair_stride = 0,
                .pair_rank = self.config.metalora_r,
            };
            x = try ctx.replace(x, self.layerForward(ctx, extra, &x, &self.metalora.layers[layer_i]));
            captures[layer_i] = try x.narrow(ctx, .seq, evidence_ids.len, mem);
            captured += 1;
        }

        const rest = try a.alloc(*const fucina.Tensor(.{ .seq, .embed }), base.num_layers - 1);
        defer a.free(rest);
        for (rest, 1..) |*ptr, layer_i| ptr.* = &captures[layer_i];
        var stacked = try captures[0].stack(ctx, .layer, 0, rest);
        defer stacked.deinit();
        var memory = try stacked.withTags(ctx, .{ .layer, .mem, .embed });
        defer memory.deinit();
        return self.m2pTrain(ctx, &memory);
    }

    /// The lazily built no-adapter conversation trainer (public for eval
    /// harnesses that score with and without prefixes through the same
    /// forward the loss uses).
    pub fn convTrainer(self: *ShineTrainer, ctx: *ExecContext) !*qwen3_train.Trainer(.{}) {
        if (self.conv_trainer == null) {
            var trainer = try qwen3_train.Trainer(.{}).init(ctx, self.model, .{ .rank = 1, .alpha = 1 }, 0);
            trainer.widen_frozen = self.widen_frozen;
            self.conv_trainer = trainer;
        }
        return &self.conv_trainer.?;
    }

    /// Packed multi-example step: every base GEMM runs ONCE over the
    /// concatenated segments (m = sum of lengths — the work that clears
    /// the GPU dispatch crossover), while RoPE positions, attention, the
    /// memory captures, and the adapter side paths stay per segment. The
    /// loss is the token-mean over the whole pack. With one example this
    /// is exactly the single-example step.
    pub fn lossPacked(self: *ShineTrainer, ctx: *ExecContext, examples: []const Example) !fucina.Tensor(.{}) {
        const b_count = examples.len;
        if (b_count == 0) return qwen3.Error.InvalidSequenceLength;
        const a = ctx.allocator();
        const base = self.model.config;
        const mem = self.config.num_mem_token;

        // ---- Packed encoder pass ----
        const enc_segs = try self.retainedSegs(b_count);
        var enc_total: usize = 0;
        for (enc_segs, examples) |*seg, ex| {
            if (ex.evidence.len == 0) return qwen3.Error.InvalidSequenceLength;
            seg.* = ex.evidence.len + mem;
            enc_total += seg.*;
        }
        const enc_rope = try self.segmentRope(ctx, enc_segs, enc_total);

        var x = try self.packedEncoderInput(ctx, examples);
        defer x.deinit();

        // Per-(layer, example) memory views, graph-linked (scope-retained).
        // They are narrows of the layer BOUNDARIES — exactly what the
        // checkpoint retains, so capturing costs no extra memory.
        const captures = try a.alloc(fucina.Tensor(.{ .seq, .embed }), base.num_layers * b_count);
        defer a.free(captures);
        var captured: usize = 0;
        defer for (captures[0..captured]) |*view| view.deinit();

        for (self.model.layers, 0..) |*layer, layer_i| {
            // Every segment shares the layer's Meta-LoRA pair: stride 0.
            const extra = LayerExtra{
                .model = self.model,
                .cache = self.frozenCache(),
                .layer = layer,
                .segs = enc_segs,
                .rope = enc_rope,
                .pair_stride = 0,
                .pair_rank = self.config.metalora_r,
            };
            x = try ctx.replace(x, self.layerForward(ctx, extra, &x, &self.metalora.layers[layer_i]));
            var off: usize = 0;
            for (examples, 0..) |ex, bi| {
                captures[layer_i * b_count + bi] = try x.narrow(ctx, .seq, off + ex.evidence.len, mem);
                captured += 1;
                off += enc_segs[bi];
            }
        }

        // ---- Per-example M2P + slicing (small next to the base GEMMs) ----
        const generated = try a.alloc(shine.LoraSet, b_count);
        var gen_built: usize = 0;
        defer {
            for (generated[0..gen_built]) |*set| set.deinit();
            a.free(generated);
        }
        const rest = try a.alloc(*const fucina.Tensor(.{ .seq, .embed }), base.num_layers - 1);
        defer a.free(rest);
        for (0..b_count) |bi| {
            for (rest, 1..) |*ptr, layer_i| ptr.* = &captures[layer_i * b_count + bi];
            var stacked = try captures[bi].stack(ctx, .layer, 0, rest);
            defer stacked.deinit();
            var memory = try stacked.withTags(ctx, .{ .layer, .mem, .embed });
            defer memory.deinit();
            var plain = try self.m2pTrain(ctx, &memory);
            defer plain.deinit();
            generated[bi] = try sliceViews(ctx, base, self.config, &plain);
            gen_built = bi + 1;
        }

        // ---- Packed adapted conversation pass ----
        const conv_segs = try self.retainedSegs(b_count);
        var conv_total: usize = 0;
        for (conv_segs, examples) |*seg, ex| {
            if (ex.input.len == 0 or ex.input.len != ex.labels.len) return qwen3.Error.InvalidSequenceLength;
            seg.* = ex.input.len;
            conv_total += seg.*;
        }
        const conv_rope = try self.segmentRope(ctx, conv_segs, conv_total);

        const all_inputs = try a.alloc(usize, conv_total);
        defer a.free(all_inputs);
        const all_labels = try a.alloc(usize, conv_total);
        defer a.free(all_labels);
        {
            var at: usize = 0;
            for (examples) |ex| {
                @memcpy(all_inputs[at..][0..ex.input.len], ex.input);
                @memcpy(all_labels[at..][0..ex.labels.len], ex.labels);
                at += ex.input.len;
            }
        }

        var cx = try self.model.token_embedding.getRowsAs(ctx, all_inputs, .embed);
        defer cx.deinit();
        for (self.model.layers, 0..) |*layer, layer_i| {
            var pack_owned: shine.LayerLora = undefined;
            var have_pack = false;
            defer if (have_pack) pack_owned.deinit(); // scope-owned views: no-op
            const pack: *const shine.LayerLora = if (b_count == 1) &generated[0].layers[layer_i] else blk: {
                pack_owned = try packLayer(ctx, generated, layer_i);
                have_pack = true;
                break :blk &pack_owned;
            };
            const extra = LayerExtra{
                .model = self.model,
                .cache = self.frozenCache(),
                .layer = layer,
                .segs = conv_segs,
                .rope = conv_rope,
                .pair_stride = if (b_count == 1) 0 else self.config.lora_r,
                .pair_rank = self.config.lora_r,
            };
            cx = try ctx.replace(cx, self.layerForward(ctx, extra, &cx, pack));
        }
        var final = try cx.rmsNormMul(ctx, .embed, &self.model.output_norm, base.rms_norm_eps);
        defer final.deinit();
        var logits = try qwen3_train.dotLinear(self.frozenCache(), &self.model.output, ctx, &final, .embed, .vocab);
        defer logits.deinit();
        return logits.crossEntropy(ctx, .vocab, all_labels, .{
            .ignore_index = qwen3_train.ignore_index,
            .reduction = .mean,
        });
    }

    /// Per-segment RoPE table (each segment restarts at position 0),
    /// heap-owned and step-retained (see `freeTransient`).
    fn segmentRope(self: *ShineTrainer, ctx: *ExecContext, segs: []const usize, total: usize) !*const fucina.RopeTable {
        const base = self.model.config;
        const positions = try ctx.allocator().alloc(i32, total);
        defer ctx.allocator().free(positions);
        var at: usize = 0;
        for (segs) |seg| {
            for (0..seg) |i| {
                positions[at] = @intCast(i);
                at += 1;
            }
        }
        const fresh = try self.allocator.create(fucina.RopeTable);
        errdefer self.allocator.destroy(fresh);
        fresh.* = try ctx.prepareRopeTable(.{ .positions = .{ .explicit = positions }, .feature_dim = base.head_dim, .freqs = .{ .theta = .{ .base = base.rope_theta } } });
        errdefer fresh.deinit();
        try self.transient_tables.append(self.allocator, fresh);
        return fresh;
    }

    /// One packed base layer, plain or recompute-in-backward by
    /// `checkpoint_layers` — the same block either way; the checkpoint
    /// retains only the boundary hidden.
    fn layerForward(self: *ShineTrainer, ctx: *ExecContext, extra: LayerExtra, x: *const Hidden, pack: *const shine.LayerLora) !Hidden {
        if (self.checkpoint_layers)
            return fucina.checkpointWithContext(ctx, LayerBlock.run, extra, packInputs(x, pack));
        return LayerBlock.run(ctx, extra, packInputs(x, pack));
    }

    /// evidence_0 ++ mem ++ evidence_1 ++ mem ++ ... as one embedded pack
    /// (memory rows are constant zeros, the reference's trained state).
    fn packedEncoderInput(self: *ShineTrainer, ctx: *ExecContext, examples: []const Example) !fucina.Tensor(.{ .seq, .embed }) {
        const a = ctx.allocator();
        const base = self.model.config;
        const mem = self.config.num_mem_token;

        var mem_rows = try fucina.Tensor(.{ .seq, .embed }).zeros(ctx, .{ mem, base.hidden_size });
        defer mem_rows.deinit();

        const pieces = try a.alloc(fucina.Tensor(.{ .seq, .embed }), examples.len);
        defer a.free(pieces);
        var built: usize = 0;
        defer for (pieces[0..built]) |*piece| piece.deinit();
        for (pieces, examples) |*piece, ex| {
            piece.* = try self.model.token_embedding.getRowsAs(ctx, ex.evidence, .embed);
            built += 1;
        }

        const others = try a.alloc(*const fucina.Tensor(.{ .seq, .embed }), 2 * examples.len - 1);
        defer a.free(others);
        var oi: usize = 0;
        others[oi] = &mem_rows;
        oi += 1;
        for (pieces[1..]) |*piece| {
            others[oi] = piece;
            oi += 1;
            others[oi] = &mem_rows;
            oi += 1;
        }
        return pieces[0].concat(ctx, .seq, others[0..oi]);
    }

    /// M2P with gradient flow into the positional embeddings: size-1-axis
    /// broadcasts instead of the inference path's precombined PE buffer.
    fn m2pTrain(self: *ShineTrainer, ctx: *ExecContext, memory: *const fucina.Tensor(.{ .layer, .mem, .embed })) !fucina.Tensor(.{ .layer, .mem, .embed }) {
        var lp = try self.layer_pe.insertAxis(ctx, .mem, 1);
        defer lp.deinit();
        var tp = try self.token_pe.insertAxis(ctx, .layer, 0);
        defer tp.deinit();
        var with_lp = try memory.add(ctx, &lp);
        defer with_lp.deinit();
        var x = try with_lp.add(ctx, &tp);
        defer x.deinit();

        var out = try x.withTags(ctx, .{ .layer, .mem, .embed });
        errdefer out.deinit();
        for (0..self.m2p.len) |i| {
            out = try ctx.replace(out, shine.m2pStage(self.config, self.m2p, ctx, i, &out));
        }
        return out;
    }
};

/// One base layer over a PACK of segments — pure in (`extra`, inputs), the
/// checkpoint block contract: every frozen GEMM and every projection runs
/// once over the whole pack; attention is causal per segment; the LoRA
/// side paths use segment `i`'s pair set on segment `i`'s rows.
/// `pairs.len == segs.len`; one segment covering the whole sequence
/// reproduces the single-example layer exactly.
fn layerBody(
    ctx: *ExecContext,
    extra: LayerExtra,
    input: *const Hidden,
    pairs: []const *const shine.LayerLora,
) !Hidden {
    const base = extra.model.config;
    const layer = extra.layer;
    const segs = extra.segs;
    const rope_table = extra.rope;
    const attn_scale = 1 / @sqrt(@as(f32, @floatFromInt(base.head_dim)));
    const cache = extra.cache;

    var normed = try input.rmsNormMul(ctx, .embed, &layer.attn_norm, base.rms_norm_eps);
    defer normed.deinit();

    var qkv = try qwen3_train.projectQkv(cache, ctx, layer, &normed, base);
    defer qkv.deinit();
    qkv.q = try ctx.replace(qkv.q, addLoraSeg(ctx, &qkv.q, &normed, pairs, segs, .q, .q));
    qkv.k = try ctx.replace(qkv.k, addLoraSeg(ctx, &qkv.k, &normed, pairs, segs, .k, .k));
    qkv.v = try ctx.replace(qkv.v, addLoraSeg(ctx, &qkv.v, &normed, pairs, segs, .v, .v));

    var q3 = try qkv.q.split(ctx, .q, .{ .head, .d }, .{ base.num_attention_heads, base.head_dim });
    defer q3.deinit();
    var k3 = try qkv.k.split(ctx, .k, .{ .kv_head, .d }, .{ base.num_key_value_heads, base.head_dim });
    defer k3.deinit();
    var v3 = try qkv.v.split(ctx, .v, .{ .kv_head, .d }, .{ base.num_key_value_heads, base.head_dim });
    defer v3.deinit();

    var q_rope = try q3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.q_norm, base.rms_norm_eps, rope_table);
    defer q_rope.deinit();
    var k_rope = try k3.rmsNormMulRopeHalfPrepared(ctx, .seq, .d, &layer.k_norm, base.rms_norm_eps, rope_table);
    defer k_rope.deinit();

    var attn = try segmentAttention(ctx, extra.model, &q_rope, &k_rope, &v3, segs, attn_scale);
    defer attn.deinit();

    var attn_out = try qwen3_train.dotLinear(cache, &layer.o_proj, ctx, &attn, .attn, .embed);
    defer attn_out.deinit();
    attn_out = try ctx.replace(attn_out, addLoraSeg(ctx, &attn_out, &attn, pairs, segs, .o, .embed));

    var with_attn = try input.add(ctx, &attn_out);
    defer with_attn.deinit();

    var ffn_in = try with_attn.rmsNormMul(ctx, .embed, &layer.ffn_norm, base.rms_norm_eps);
    defer ffn_in.deinit();

    const dense = switch (layer.ffn) {
        .dense => |*d| d,
        .moe => return Error.MoeNotSupported,
    };
    var gate_up = try qwen3_train.projectGateUp(cache, ctx, dense, &ffn_in, base);
    defer gate_up.deinit();
    gate_up.gate = try ctx.replace(gate_up.gate, addLoraSeg(ctx, &gate_up.gate, &ffn_in, pairs, segs, .gate, .ffn));
    gate_up.up = try ctx.replace(gate_up.up, addLoraSeg(ctx, &gate_up.up, &ffn_in, pairs, segs, .up, .ffn));

    var gated = try gate_up.up.swiglu(ctx, &gate_up.gate);
    defer gated.deinit();

    var down = try qwen3_train.dotLinear(cache, &dense.down_proj, ctx, &gated, .ffn, .embed);
    defer down.deinit();
    down = try ctx.replace(down, addLoraSeg(ctx, &down, &gated, pairs, segs, .down, .embed));

    return with_attn.add(ctx, &down);
}

/// Causal attention per segment over the packed rows, concatenated back
/// in order — the packed-segment mask decomposition.
fn segmentAttention(
    ctx: *ExecContext,
    model: *const qwen3.Model,
    q: *const fucina.Tensor(.{ .seq, .head, .d }),
    k: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
    v: *const fucina.Tensor(.{ .seq, .kv_head, .d }),
    segs: []const usize,
    attn_scale: f32,
) !fucina.Tensor(.{ .seq, .attn }) {
    if (segs.len == 1) return q.groupedAttention(ctx, k, v, model.kv_head_for_head, .attn, attn_scale, .{});
    const a = ctx.allocator();
    const parts = try a.alloc(fucina.Tensor(.{ .seq, .attn }), segs.len);
    defer a.free(parts);
    var built: usize = 0;
    defer for (parts[0..built]) |*part| part.deinit();
    var off: usize = 0;
    for (segs, 0..) |seg, i| {
        var qs = try q.narrow(ctx, .seq, off, seg);
        defer qs.deinit();
        var ks = try k.narrow(ctx, .seq, off, seg);
        defer ks.deinit();
        var vs = try v.narrow(ctx, .seq, off, seg);
        defer vs.deinit();
        parts[i] = try qs.groupedAttention(ctx, &ks, &vs, model.kv_head_for_head, .attn, attn_scale, .{});
        built = i + 1;
        off += seg;
    }
    const others = try a.alloc(*const fucina.Tensor(.{ .seq, .attn }), segs.len - 1);
    defer a.free(others);
    for (others, parts[1..]) |*ptr, *part| ptr.* = part;
    return parts[0].concat(ctx, .seq, others);
}

/// Add each segment's LoRA delta (its own pair set) to the packed base
/// projection output. One segment takes the whole-tensor fast path.
fn addLoraSeg(
    ctx: *ExecContext,
    proj: anytype,
    x: anytype,
    pairs: []const *const shine.LayerLora,
    segs: []const usize,
    comptime module: shine.Module,
    comptime out_tag: @TypeOf(.tag),
) !@TypeOf(proj.*) {
    if (segs.len == 1) {
        var delta = try shine.loraDelta(ctx, x, pairs[0].pair(module), out_tag);
        defer delta.deinit();
        return proj.add(ctx, &delta);
    }
    const a = ctx.allocator();
    const parts = try a.alloc(fucina.Tensor(.{ .seq, out_tag }), segs.len);
    defer a.free(parts);
    var built: usize = 0;
    defer for (parts[0..built]) |*part| part.deinit();
    var off: usize = 0;
    for (segs, 0..) |seg, i| {
        var xs = try x.narrow(ctx, .seq, off, seg);
        defer xs.deinit();
        parts[i] = try shine.loraDelta(ctx, &xs, pairs[i].pair(module), out_tag);
        built = i + 1;
        off += seg;
    }
    const others = try a.alloc(*const fucina.Tensor(.{ .seq, out_tag }), segs.len - 1);
    defer a.free(others);
    for (others, parts[1..]) |*ptr, *part| ptr.* = part;
    var delta = try parts[0].concat(ctx, .seq, others);
    defer delta.deinit();
    return proj.add(ctx, &delta);
}

/// One layer's generated adapters packed for the block: per module, the B
/// per-segment pairs concatenated along `.lora_r` (A [in, B*r], B
/// [B*r, out]); segment `i`'s pair is the [i*r, r) slice. Concat of graph
/// views — gradients route back to each example's M2P output. Must run
/// under an exec scope (partial results lean on scope cleanup).
fn packLayer(ctx: *ExecContext, generated: []const shine.LoraSet, layer_i: usize) !shine.LayerLora {
    const a = ctx.allocator();
    var out: shine.LayerLora = undefined;
    const rest_a = try a.alloc(*const ATensor, generated.len - 1);
    defer a.free(rest_a);
    const rest_b = try a.alloc(*const BTensor, generated.len - 1);
    defer a.free(rest_b);
    inline for (shine.modules) |module| {
        const first = generated[0].layers[layer_i].pair(module);
        for (rest_a, rest_b, generated[1..]) |*pa, *pb, *set| {
            const pair = set.layers[layer_i].pair(module);
            pa.* = &pair.a;
            pb.* = &pair.b;
        }
        var packed_a = try first.a.concat(ctx, .lora_r, rest_a);
        errdefer packed_a.deinit();
        const packed_b = try first.b.concat(ctx, .lora_r, rest_b);
        @field(out, @tagName(module)) = .{ .a = packed_a, .b = packed_b };
    }
    return out;
}

pub const Example = struct {
    evidence: []const usize,
    input: []const usize,
    labels: []const usize,
};

/// "rl" slicing as GRAPH-PRESERVING views: scale by sqrt(scale)
/// (differentiable), then per layer flatten the (mem * embed) row and cut
/// A [in, r] / B [r, out] reshaped views in module order. The returned
/// pairs stay connected to the M2P output.
pub fn sliceViews(
    ctx: *ExecContext,
    base: qwen3.Config,
    config: shine.Config,
    plain: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !shine.LoraSet {
    const allocator = ctx.allocator();
    const r = config.lora_r;

    var scaled = try plain.scale(ctx, @sqrt(config.scale));
    defer scaled.deinit();

    const layers = try allocator.alloc(shine.LayerLora, base.num_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*layer| layer.deinit();
        allocator.free(layers);
    }
    const row_len = config.num_mem_token * config.hidden_size;
    for (layers, 0..) |*layer, layer_i| {
        var row = try scaled.narrow(ctx, .layer, layer_i, 1);
        defer row.deinit();
        var flat = try row.reshape(ctx, .{.flat}, .{row_len});
        defer flat.deinit();
        var at: usize = 0;
        inline for (shine.modules) |module| {
            const dims = shine.moduleDims(base, module);
            var a_flat = try flat.narrow(ctx, .flat, at, dims[0] * r);
            defer a_flat.deinit();
            var a = try a_flat.reshape(ctx, .{ .lin, .lora_r }, .{ dims[0], r });
            errdefer a.deinit();
            at += dims[0] * r;
            var b_flat = try flat.narrow(ctx, .flat, at, r * dims[1]);
            defer b_flat.deinit();
            var b = try b_flat.reshape(ctx, .{ .lora_r, .lout }, .{ r, dims[1] });
            errdefer b.deinit();
            at += r * dims[1];
            @field(layer, @tagName(module)) = .{ .a = a, .b = b };
        }
        std.debug.assert(at == row_len);
        built += 1;
    }
    return .{ .allocator = allocator, .layers = layers };
}

/// Cartridge readout as GRAPH-PRESERVING views: scale by `sqrt(scale)`,
/// then per layer cut the flat `M * hidden` row into `rows` K rows and
/// `rows` V rows reshaped to the cartridge `[seq, kv_head, d]` layout. The
/// result is assembled as a STANDARD `models.text.cartridge.Cartridge` (empty
/// registry, no sink) whose rows stay connected to the M2P output, so a
/// forward through it backpropagates into the hypernetwork. Scope-owned:
/// use under an exec scope and `deinit` normally (tensor deinits are
/// scope no-ops).
pub fn sliceCartridgeViews(
    ctx: *ExecContext,
    base: qwen3.Config,
    config: shine.Config,
    plain: *const fucina.Tensor(.{ .layer, .mem, .embed }),
) !cartridge_mod.Cartridge {
    if (config.cartridge_rows == 0) return shine.Error.InvalidShineConfig;
    const allocator = ctx.allocator();
    const rows = config.cartridge_rows;
    const kv_heads = base.num_key_value_heads;
    const head_dim = base.head_dim;
    const kv_dim = kv_heads * head_dim;
    const row_len = config.num_mem_token * config.hidden_size;

    var scaled = try plain.scale(ctx, @sqrt(config.scale));
    defer scaled.deinit();

    const layers = try allocator.alloc(cartridge_mod.LayerKv, base.num_layers);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*layer| layer.deinit();
        allocator.free(layers);
    }
    for (layers, 0..) |*layer, layer_i| {
        var row = try scaled.narrow(ctx, .layer, layer_i, 1);
        defer row.deinit();
        var flat = try row.reshape(ctx, .{.flat}, .{row_len});
        defer flat.deinit();
        var k_flat = try flat.narrow(ctx, .flat, 0, rows * kv_dim);
        defer k_flat.deinit();
        var k = try k_flat.reshape(ctx, .{ .seq, .kv_head, .d }, .{ rows, kv_heads, head_dim });
        errdefer k.deinit();
        var v_flat = try flat.narrow(ctx, .flat, rows * kv_dim, rows * kv_dim);
        defer v_flat.deinit();
        const v = try v_flat.reshape(ctx, .{ .seq, .kv_head, .d }, .{ rows, kv_heads, head_dim });
        layer.* = .{ .k_sink = null, .v_sink = null, .k = k, .v = v };
        built += 1;
    }
    return .{
        .allocator = allocator,
        .layers = layers,
        .p = rows,
        .frozen_prefix = 0,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .registry = fucina.ParamRegistry.init(allocator),
    };
}

/// Build a gradient-carrying LoRA pair from raw values (a: [in, r] row-major,
/// b: [r, out]).
pub fn pairVariables(ctx: *ExecContext, in_dim: usize, r: usize, out_dim: usize, a_values: []const f32, b_values: []const f32) !shine.LoraPair {
    var a = try fucina.Tensor(.{ .lin, .lora_r }).variableFromSlice(ctx, .{ in_dim, r }, a_values);
    errdefer a.deinit();
    var b = try fucina.Tensor(.{ .lora_r, .lout }).variableFromSlice(ctx, .{ r, out_dim }, b_values);
    errdefer b.deinit();
    return .{ .a = a, .b = b };
}

/// Build a gradient-carrying M2P block from raw values.
pub fn m2pBlockVariables(ctx: *ExecContext, config: shine.Config, values: M2pBlockValues) !shine.M2pBlock {
    const hidden = config.hidden_size;
    const ff = config.m2p_ffn;
    var in_w = try fucina.Tensor(.{ .proj, .embed }).variableFromSlice(ctx, .{ 3 * hidden, hidden }, values.in_w);
    errdefer in_w.deinit();
    var in_b = try fucina.Tensor(.{.proj}).variableFromSlice(ctx, .{3 * hidden}, values.in_b);
    errdefer in_b.deinit();
    var out_w = try fucina.Tensor(.{ .eout, .hmerge }).variableFromSlice(ctx, .{ hidden, hidden }, values.out_w);
    errdefer out_w.deinit();
    var out_b = try fucina.Tensor(.{.eout}).variableFromSlice(ctx, .{hidden}, values.out_b);
    errdefer out_b.deinit();
    var ff1_w = try fucina.Tensor(.{ .m2pffn, .embed }).variableFromSlice(ctx, .{ ff, hidden }, values.ff1_w);
    errdefer ff1_w.deinit();
    var ff1_b = try fucina.Tensor(.{.m2pffn}).variableFromSlice(ctx, .{ff}, values.ff1_b);
    errdefer ff1_b.deinit();
    var ff2_w = try fucina.Tensor(.{ .eout, .m2pffn }).variableFromSlice(ctx, .{ hidden, ff }, values.ff2_w);
    errdefer ff2_w.deinit();
    var ff2_b = try fucina.Tensor(.{.eout}).variableFromSlice(ctx, .{hidden}, values.ff2_b);
    errdefer ff2_b.deinit();
    var norm1_w = try fucina.Tensor(.{.embed}).variableFromSlice(ctx, .{hidden}, values.norm1_w);
    errdefer norm1_w.deinit();
    var norm1_b = try fucina.Tensor(.{.embed}).variableFromSlice(ctx, .{hidden}, values.norm1_b);
    errdefer norm1_b.deinit();
    var norm2_w = try fucina.Tensor(.{.embed}).variableFromSlice(ctx, .{hidden}, values.norm2_w);
    errdefer norm2_w.deinit();
    var norm2_b = try fucina.Tensor(.{.embed}).variableFromSlice(ctx, .{hidden}, values.norm2_b);
    errdefer norm2_b.deinit();
    return .{
        .in_w = in_w,
        .in_b = in_b,
        .out_w = out_w,
        .out_b = out_b,
        .ff1_w = ff1_w,
        .ff1_b = ff1_b,
        .ff2_w = ff2_w,
        .ff2_b = ff2_b,
        .norm1_w = norm1_w,
        .norm1_b = norm1_b,
        .norm2_w = norm2_w,
        .norm2_b = norm2_b,
    };
}

test {
    _ = @import("shine_train_golden_tests.zig");
    _ = @import("shine_cartridge_tests.zig");
}
