//! Cartridges: trainable KV-cache prefixes (arXiv 2506.06266, reference
//! implementation HazyResearch/cartridges).
//!
//! A cartridge compresses a long corpus into the KV cache of a virtual
//! p-token prefix, trained offline by self-study distillation and served as
//! a reusable prefix at a fraction of the ICL cache size. Per transformer
//! layer it is a pair of `[p, kv_head, d]` key/value tensors living in the
//! SAME space as KV-cache rows — keys post q/k-norm and post-RoPE, rotated
//! at positions `0..p-1` and never rotated again; real tokens sit at
//! positions `p..`. Attention over `concat(cartridge, tokens)` with the
//! end-aligned causal kernel (`source_offset = kv_seq - q_seq`) reproduces
//! the reference mask exactly: every query sees the whole cartridge, and is
//! causal over the real tokens.
//!
//! Reference semantics pinned here (paper Sec 3.2/4.2 + repo train.py,
//! cache.py):
//! - The first `frozen_prefix` rows (default 1: the BOS attention sink) are
//!   frozen constants — training them destabilizes the run (paper App A.1).
//!   The remaining `p - frozen_prefix` rows are leaf variables.
//! - The distillation objective is the teacher top-k cross-entropy
//!   `mean(-p_teacher * log q_student)` over sparse (position, token,
//!   logprob) entries: the student log-probability for the target token at
//!   packed position `pos` is read from logits row `pos - 1`. Truncated
//!   teacher tail mass is dropped, NOT renormalized, and entries are
//!   averaged uniformly (rows with more retained entries weigh more) —
//!   identical gradients to forward KL(teacher ‖ student).
//!
//! Ownership: cartridge tensors are long-lived parameters like LoRA A/B —
//! create them OUTSIDE any exec scope and keep them alive across steps;
//! `distillLoss` and the concat helpers are composite ops and must run
//! under an open exec scope when the result will be `backward()`'d.

const std = @import("std");
const fucina = @import("fucina");
const kv_cache = @import("kv_cache.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;

pub const Error = error{
    InvalidCartridge,
    InvalidTargets,
    ExecScopeRequired,
};

/// Cartridge K/V tensors carry KV-cache row layout: [seq, kv_head, d].
pub const Kv = fucina.Tensor(.{ .seq, .kv_head, .d });

/// Persisted form of the optional serving-time draft reference: the corpus
/// token ids as a 1-D i64 tensor (torch-conventional index dtype).
pub const DraftReference = fucina.Tensor(.{ .dtype = .i64, .tags = .{.token} });

/// One layer's cartridge: optional frozen sink rows (constants) followed by
/// trainable rows (variables). Concatenation order is sink ++ trainable ++
/// tokens, so the sink occupies positions 0..frozen-1 like the reference's
/// `cat([frozen_K, trainable_K, new_K])`.
pub const LayerKv = struct {
    /// Frozen attention-sink rows, or null when `frozen_prefix == 0`.
    k_sink: ?Kv,
    v_sink: ?Kv,
    /// Trainable rows (leaf variables).
    k: Kv,
    v: Kv,

    /// concat(sink?, trainable, tokens) along .seq — the full key sequence
    /// for `groupedAttention`. Scope-owned like any op result.
    pub fn catK(self: *const LayerKv, ctx: *ExecContext, k_tokens: *const Kv) !Kv {
        if (self.k_sink) |*sink| return sink.concat(ctx, .seq, &.{ &self.k, k_tokens });
        return self.k.concat(ctx, .seq, &.{k_tokens});
    }

    /// Value-side counterpart of `catK`.
    pub fn catV(self: *const LayerKv, ctx: *ExecContext, v_tokens: *const Kv) !Kv {
        if (self.v_sink) |*sink| return sink.concat(ctx, .seq, &.{ &self.v, v_tokens });
        return self.v.concat(ctx, .seq, &.{v_tokens});
    }

    /// The full [p, kv_head, d] key prefix (sink ++ trainable) — the serving
    /// payload. Without a sink this is a zero-copy view of the trainable
    /// rows (uniform ownership for the caller's deinit).
    pub fn fullK(self: *const LayerKv, ctx: *ExecContext) !Kv {
        if (self.k_sink) |*sink| return sink.concat(ctx, .seq, &.{&self.k});
        return self.k.withTags(ctx, Kv.axis_tags);
    }

    /// Value-side counterpart of `fullK`.
    pub fn fullV(self: *const LayerKv, ctx: *ExecContext) !Kv {
        if (self.v_sink) |*sink| return sink.concat(ctx, .seq, &.{&self.v});
        return self.v.withTags(ctx, Kv.axis_tags);
    }

    pub fn deinit(self: *LayerKv) void {
        if (self.k_sink) |*sink| sink.deinit();
        if (self.v_sink) |*sink| sink.deinit();
        self.k.deinit();
        self.v.deinit();
        self.* = undefined;
    }
};

/// A whole-model cartridge: one `LayerKv` per transformer layer.
pub const Cartridge = struct {
    allocator: Allocator,
    layers: []LayerKv,
    /// Total prefix rows p (sink + trainable). Real tokens sit at p...
    p: usize,
    /// Leading rows excluded from training (the attention-sink freeze).
    frozen_prefix: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Every tensor under its "layers.<i>.{k,v}[_sink]" checkpoint name:
    /// sinks register as frozen entries (saved/loaded, skipped by
    /// optimizers), trainable rows as parameters. Optimizers registered via
    /// `registerParams` borrow the entries, so the cartridge must outlive
    /// them.
    registry: fucina.ParamRegistry,
    /// Optional corpus token ids embedded in the artifact (the
    /// "draft_reference" entry, set by `setDraftReference`): the SERVING
    /// speculation reference — a suffix automaton built over these tokens
    /// once at load drafts the cartridge's corpus-grounded answers
    /// (docs/CARTRIDGES.md, docs/SPECULATIVE.md). Decoded here on load.
    /// Owned.
    draft_reference: ?[]usize = null,
    /// The persisted i64 view of `draft_reference` (frozen registry entry).
    draft_reference_tensor: ?DraftReference = null,

    /// Build from captured per-layer K/V rows — `k_rows[l]`/`v_rows[l]` hold
    /// `p * kv_heads * head_dim` floats in KV-cache row order (position-major,
    /// keys already q/k-normed + roped at positions 0..p-1). This is the
    /// paper's winning initialization when the rows come from a real forward
    /// over the corpus' first p tokens; it is also the test seam (any values).
    pub fn initFromRows(
        ctx: *ExecContext,
        allocator: Allocator,
        frozen_prefix: usize,
        p: usize,
        kv_heads: usize,
        head_dim: usize,
        k_rows: []const []const f32,
        v_rows: []const []const f32,
    ) !Cartridge {
        const kv_heads_per = try allocator.alloc(usize, k_rows.len);
        defer allocator.free(kv_heads_per);
        const head_dim_per = try allocator.alloc(usize, k_rows.len);
        defer allocator.free(head_dim_per);
        @memset(kv_heads_per, kv_heads);
        @memset(head_dim_per, head_dim);
        return initFromRowsVaried(ctx, allocator, frozen_prefix, p, kv_heads_per, head_dim_per, k_rows, v_rows);
    }

    /// `initFromRows` with PER-LAYER KV geometry — models like gemma-4 mix
    /// layer shapes (e.g. 8-head/256-dim local-SWA layers with 2-head/512-dim
    /// globals). Each layer's tensors carry their true shapes; the struct's
    /// `kv_heads`/`head_dim` metadata records layer 0.
    pub fn initFromRowsVaried(
        ctx: *ExecContext,
        allocator: Allocator,
        frozen_prefix: usize,
        p: usize,
        kv_heads: []const usize,
        head_dims: []const usize,
        k_rows: []const []const f32,
        v_rows: []const []const f32,
    ) !Cartridge {
        if (p == 0 or frozen_prefix >= p) return Error.InvalidCartridge;
        if (k_rows.len == 0 or k_rows.len != v_rows.len) return Error.InvalidCartridge;
        if (kv_heads.len != k_rows.len or head_dims.len != k_rows.len) return Error.InvalidCartridge;
        for (k_rows, v_rows, kv_heads, head_dims) |k_layer, v_layer, heads, dim| {
            const row = heads * dim;
            if (row == 0 or k_layer.len != p * row or v_layer.len != p * row) return Error.InvalidCartridge;
        }

        const layers = try allocator.alloc(LayerKv, k_rows.len);
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*layer| layer.deinit();

        const train_rows = p - frozen_prefix;
        for (layers, k_rows, v_rows, kv_heads, head_dims) |*layer, k_layer, v_layer, heads, dim| {
            const row = heads * dim;
            const sink_len = frozen_prefix * row;
            var k_sink: ?Kv = null;
            errdefer if (k_sink) |*sink| sink.deinit();
            var v_sink: ?Kv = null;
            errdefer if (v_sink) |*sink| sink.deinit();
            if (frozen_prefix > 0) {
                k_sink = try Kv.fromSlice(ctx, .{ frozen_prefix, heads, dim }, k_layer[0..sink_len]);
                v_sink = try Kv.fromSlice(ctx, .{ frozen_prefix, heads, dim }, v_layer[0..sink_len]);
            }
            var k = try Kv.variableFromSlice(ctx, .{ train_rows, heads, dim }, k_layer[sink_len..]);
            errdefer k.deinit();
            const v = try Kv.variableFromSlice(ctx, .{ train_rows, heads, dim }, v_layer[sink_len..]);
            layer.* = .{ .k_sink = k_sink, .v_sink = v_sink, .k = k, .v = v };
            built += 1;
        }

        var registry = try buildRegistry(allocator, layers);
        errdefer registry.deinit();

        return .{
            .allocator = allocator,
            .layers = layers,
            .p = p,
            .frozen_prefix = frozen_prefix,
            .kv_heads = kv_heads[0],
            .head_dim = head_dims[0],
            .registry = registry,
        };
    }

    /// Gaussian-init cartridge (the paper's random-vector ablation baseline;
    /// mainly a test/bring-up seam — corpus-token capture wins by 25 points).
    pub fn initRandom(
        ctx: *ExecContext,
        allocator: Allocator,
        n_layers: usize,
        frozen_prefix: usize,
        p: usize,
        kv_heads: usize,
        head_dim: usize,
        seed: u64,
        std_dev: f32,
    ) !Cartridge {
        if (p == 0 or frozen_prefix >= p or n_layers == 0) return Error.InvalidCartridge;
        const layer_len = p * kv_heads * head_dim;
        const scratch = try allocator.alloc(f32, 2 * layer_len);
        defer allocator.free(scratch);

        const layers = try allocator.alloc(LayerKv, n_layers);
        errdefer allocator.free(layers);
        var built: usize = 0;
        errdefer for (layers[0..built]) |*layer| layer.deinit();

        const row = kv_heads * head_dim;
        const sink_len = frozen_prefix * row;
        const train_rows = p - frozen_prefix;
        for (layers, 0..) |*layer, layer_i| {
            const k_data = scratch[0..layer_len];
            const v_data = scratch[layer_len..];
            fucina.rng.normalFill(fucina.rng.at(seed, 2 * layer_i), k_data, 0, std_dev);
            fucina.rng.normalFill(fucina.rng.at(seed, 2 * layer_i + 1), v_data, 0, std_dev);

            var k_sink: ?Kv = null;
            errdefer if (k_sink) |*sink| sink.deinit();
            var v_sink: ?Kv = null;
            errdefer if (v_sink) |*sink| sink.deinit();
            if (frozen_prefix > 0) {
                k_sink = try Kv.fromSlice(ctx, .{ frozen_prefix, kv_heads, head_dim }, k_data[0..sink_len]);
                v_sink = try Kv.fromSlice(ctx, .{ frozen_prefix, kv_heads, head_dim }, v_data[0..sink_len]);
            }
            var k = try Kv.variableFromSlice(ctx, .{ train_rows, kv_heads, head_dim }, k_data[sink_len..]);
            errdefer k.deinit();
            const v = try Kv.variableFromSlice(ctx, .{ train_rows, kv_heads, head_dim }, v_data[sink_len..]);
            layer.* = .{ .k_sink = k_sink, .v_sink = v_sink, .k = k, .v = v };
            built += 1;
        }

        var registry = try buildRegistry(allocator, layers);
        errdefer registry.deinit();

        return .{
            .allocator = allocator,
            .layers = layers,
            .p = p,
            .frozen_prefix = frozen_prefix,
            .kv_heads = kv_heads,
            .head_dim = head_dim,
            .registry = registry,
        };
    }

    fn buildRegistry(allocator: Allocator, layers: []LayerKv) !fucina.ParamRegistry {
        var registry = fucina.ParamRegistry.init(allocator);
        errdefer registry.deinit();
        var name_buf: [64]u8 = undefined;
        for (layers, 0..) |*layer, layer_i| {
            if (layer.k_sink) |*sink| {
                try registry.addParam(try std.fmt.bufPrint(&name_buf, "layers.{d}.k_sink", .{layer_i}), sink);
            }
            if (layer.v_sink) |*sink| {
                try registry.addParam(try std.fmt.bufPrint(&name_buf, "layers.{d}.v_sink", .{layer_i}), sink);
            }
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "layers.{d}.k", .{layer_i}), &layer.k);
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "layers.{d}.v", .{layer_i}), &layer.v);
        }
        return registry;
    }

    /// Register the trainable rows on an optimizer (anything with
    /// `addParamNamed`); sinks are frozen entries and are skipped. The
    /// cartridge must outlive the optimizer (params and names are borrowed).
    pub fn registerParams(self: *const Cartridge, opt: anytype) !void {
        try self.registry.addParamsTo(opt);
    }

    pub fn zeroGrad(self: *Cartridge) void {
        self.registry.zeroGrad();
    }

    /// Serialize every row (sinks included) as a clean safetensors state
    /// dict. Geometry is fully recoverable from the entry names and shapes.
    pub fn saveState(self: *const Cartridge, writer: *std.Io.Writer) !void {
        try self.registry.saveStateDict(writer);
    }

    /// Load a state dict saved by `saveState` into this cartridge (strict:
    /// one-to-one name and shape match, so the receiving cartridge must be
    /// built with the same geometry).
    pub fn loadState(self: *Cartridge, reader: *std.Io.Reader) !void {
        try self.registry.loadStateDict(reader, .{});
    }

    /// Rebuild a cartridge from `saveState` bytes without knowing the
    /// geometry up front: layer count, p, frozen prefix, and kv dims are
    /// recovered from the safetensors header, then the strict loader
    /// overwrites every row.
    pub fn initFromStateDict(ctx: *ExecContext, allocator: Allocator, bytes: []const u8) !Cartridge {
        var file = try fucina.safetensors.File.parse(allocator, bytes);
        defer file.deinit();
        const k0 = file.maybeTensor("layers.0.k") orelse return Error.InvalidCartridge;
        if (k0.shape.len != 3) return Error.InvalidCartridge;
        const train_rows = k0.shape[0];
        const frozen: usize = if (file.maybeTensor("layers.0.k_sink")) |sink| blk: {
            if (sink.shape.len != 3) return Error.InvalidCartridge;
            break :blk sink.shape[0];
        } else 0;
        const p = frozen + train_rows;
        var n_layers: usize = 1;
        var name_buf: [64]u8 = undefined;
        while (true) : (n_layers += 1) {
            const name = std.fmt.bufPrint(&name_buf, "layers.{d}.k", .{n_layers}) catch unreachable;
            if (file.maybeTensor(name) == null) break;
        }

        // Per-layer geometry straight from the header (layers may vary,
        // e.g. gemma-4's mixed SWA/global shapes); the strict loader then
        // overwrites every row.
        const kv_heads = try allocator.alloc(usize, n_layers);
        defer allocator.free(kv_heads);
        const head_dims = try allocator.alloc(usize, n_layers);
        defer allocator.free(head_dims);
        var zeros: std.ArrayListUnmanaged(f32) = .empty;
        defer zeros.deinit(allocator);
        const layer_rows = try allocator.alloc([]const f32, n_layers);
        defer allocator.free(layer_rows);
        var max_row_len: usize = 0;
        for (0..n_layers) |layer_i| {
            const name = std.fmt.bufPrint(&name_buf, "layers.{d}.k", .{layer_i}) catch unreachable;
            const info = file.maybeTensor(name) orelse return Error.InvalidCartridge;
            if (info.shape.len != 3 or info.shape[0] != train_rows) return Error.InvalidCartridge;
            kv_heads[layer_i] = info.shape[1];
            head_dims[layer_i] = info.shape[2];
            max_row_len = @max(max_row_len, p * info.shape[1] * info.shape[2]);
        }
        try zeros.appendNTimes(allocator, 0, max_row_len);
        for (layer_rows, kv_heads, head_dims) |*rows, heads, dim| {
            rows.* = zeros.items[0 .. p * heads * dim];
        }

        var cart = try initFromRowsVaried(ctx, allocator, frozen, p, kv_heads, head_dims, layer_rows, layer_rows);
        errdefer cart.deinit();
        if (file.maybeTensor("draft_reference")) |ref| {
            // Register a placeholder of the persisted length so the strict
            // loader has a one-to-one destination, then decode the real ids.
            if (ref.shape.len != 1 or ref.shape[0] == 0) return Error.InvalidCartridge;
            const zero_ids = try allocator.alloc(usize, ref.shape[0]);
            defer allocator.free(zero_ids);
            @memset(zero_ids, 0);
            try cart.setDraftReference(ctx, zero_ids);
        }
        var reader = std.Io.Reader.fixed(bytes);
        try cart.loadState(&reader);
        try cart.decodeDraftReference();
        return cart;
    }

    /// Embed the corpus token ids as the artifact's serving-time draft
    /// reference (persisted under "draft_reference" as a frozen i64 entry).
    /// Call once, before `saveState`. The serving side builds its
    /// speculation index from these tokens ONCE at load — nothing is
    /// constructed per call.
    pub fn setDraftReference(self: *Cartridge, ctx: *ExecContext, tokens: []const usize) !void {
        if (self.draft_reference != null or tokens.len == 0) return Error.InvalidCartridge;
        const ids = try self.allocator.dupe(usize, tokens);
        errdefer self.allocator.free(ids);
        const values = try self.allocator.alloc(i64, tokens.len);
        defer self.allocator.free(values);
        for (values, tokens) |*value, token| value.* = @intCast(token);
        var tensor = try DraftReference.fromSlice(ctx, .{tokens.len}, values);
        errdefer tensor.deinit();
        try self.registry.addParam("draft_reference", &tensor);
        self.draft_reference_tensor = tensor;
        self.draft_reference = ids;
    }

    /// Refresh `draft_reference` from the persisted tensor after a
    /// `loadState` overwrote its bytes.
    fn decodeDraftReference(self: *Cartridge) !void {
        const tensor = &(self.draft_reference_tensor orelse return);
        const values = try tensor.dataConst();
        const ids = self.draft_reference.?;
        std.debug.assert(ids.len == values.len);
        for (ids, values) |*id, value| id.* = @intCast(value);
    }

    /// Serve the cartridge: write all p rows of every layer into an EMPTY
    /// `KvCache` (converted to the cache dtype) and advance it to p, so a
    /// normal `forwardStep` continues at position p — exactly the layout the
    /// cartridge was trained at. The decode loop then treats the prefix like
    /// any cached p-token prompt.
    pub fn writeToCache(self: *const Cartridge, ctx: *ExecContext, cache: *kv_cache.KvCache) !void {
        if (cache.len != 0) return Error.InvalidCartridge;
        if (cache.kv_heads.len != self.layers.len) return Error.InvalidCartridge;
        // Pure data movement: no autograd nodes for the serving concat.
        var no_grad = fucina.noGrad();
        defer no_grad.close();
        for (self.layers, 0..) |*layer, layer_i| {
            var k_full = try layer.fullK(ctx);
            defer k_full.deinit();
            var v_full = try layer.fullV(ctx);
            defer v_full.deinit();
            try cache.appendLayer(ctx, layer_i, &k_full, &v_full);
        }
        cache.advance(self.p);
    }

    pub fn deinit(self: *Cartridge) void {
        // Registry first: it retains views of the layers' storage and their
        // GradState pointers, both torn down just below.
        self.registry.deinit();
        for (self.layers) |*layer| layer.deinit();
        self.allocator.free(self.layers);
        if (self.draft_reference_tensor) |*tensor| tensor.deinit();
        if (self.draft_reference) |ids| self.allocator.free(ids);
        self.* = undefined;
    }
};

/// Per-layer host copies of the post-q/k-norm, post-RoPE keys and the
/// cache-layout values of one forward pass — `[seq * kv_heads * head_dim]`
/// floats per layer in KV-cache row order, the exact payload
/// `Cartridge.initFromRows` consumes (the paper's corpus-token
/// initialization). Fill by passing the struct as a trainer's
/// `ForwardOptions.capture`; the trainers' `captureKv` wraps the flow.
pub const KvCapture = struct {
    k_rows: [][]f32,
    v_rows: [][]f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator, n_layers: usize, row_len: usize) !KvCapture {
        const row_lens = try allocator.alloc(usize, n_layers);
        defer allocator.free(row_lens);
        @memset(row_lens, row_len);
        return initVaried(allocator, row_lens);
    }

    /// `init` with a per-layer row length — heterogeneous KV geometry
    /// (mixed kv_heads/head_dim across layers, e.g. gemma-4).
    pub fn initVaried(allocator: Allocator, row_lens: []const usize) !KvCapture {
        const k_rows = try allocator.alloc([]f32, row_lens.len);
        errdefer allocator.free(k_rows);
        const v_rows = try allocator.alloc([]f32, row_lens.len);
        errdefer allocator.free(v_rows);
        var built: usize = 0;
        errdefer for (0..built) |i| {
            allocator.free(k_rows[i]);
            allocator.free(v_rows[i]);
        };
        for (k_rows, v_rows, row_lens) |*k, *v, row_len| {
            k.* = try allocator.alloc(f32, row_len);
            errdefer allocator.free(k.*);
            v.* = try allocator.alloc(f32, row_len);
            built += 1;
        }
        return .{ .k_rows = k_rows, .v_rows = v_rows, .allocator = allocator };
    }

    pub fn deinit(self: *KvCapture) void {
        for (self.k_rows, self.v_rows) |k, v| {
            self.allocator.free(k);
            self.allocator.free(v);
        }
        self.allocator.free(self.k_rows);
        self.allocator.free(self.v_rows);
        self.* = undefined;
    }
};

/// Incrementally builds `DistillTargets` from teacher logits rows — the
/// host-side counterpart of the reference's server-returned top-k logprobs
/// truncated at 0.99 cumulative mass (clients/base.py `flatten`).
pub const TargetsBuilder = struct {
    allocator: Allocator,
    positions: std.ArrayList(usize) = .empty,
    tokens: std.ArrayList(usize) = .empty,
    logprobs: std.ArrayList(f32) = .empty,

    pub fn init(allocator: Allocator) TargetsBuilder {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *TargetsBuilder) void {
        self.positions.deinit(self.allocator);
        self.tokens.deinit(self.allocator);
        self.logprobs.deinit(self.allocator);
        self.* = undefined;
    }

    /// Append the teacher's top-k entries for the target token at packed
    /// student position `target_pos` (>= 1), reading `teacher_row` — the
    /// teacher's raw logits row PREDICTING that token. Keeps the most
    /// probable tokens in descending order until cumulative probability
    /// reaches `min_prob_mass` (the crossing entry included), capped at
    /// `top_k` entries; the dropped tail is NOT renormalized.
    pub fn appendRow(self: *TargetsBuilder, target_pos: usize, teacher_row: []const f32, top_k: usize, min_prob_mass: f32) !void {
        if (target_pos == 0 or top_k == 0 or teacher_row.len == 0) return Error.InvalidTargets;

        // Stable log-softmax constants for the row.
        var max: f32 = teacher_row[0];
        for (teacher_row) |x| max = @max(max, x);
        var sum_exp: f32 = 0;
        for (teacher_row) |x| sum_exp += @exp(x - max);
        const log_z = max + @log(sum_exp);

        // Top-k by logit via a fixed-size insertion buffer (k is tiny).
        const k = @min(top_k, teacher_row.len);
        const best = try self.allocator.alloc(usize, k);
        defer self.allocator.free(best);
        var filled: usize = 0;
        for (teacher_row, 0..) |x, token| {
            var at = filled;
            while (at > 0 and x > teacher_row[best[at - 1]]) at -= 1;
            if (at >= k) continue;
            if (filled < k) filled += 1;
            var j = filled - 1;
            while (j > at) : (j -= 1) best[j] = best[j - 1];
            best[at] = token;
        }

        var mass: f32 = 0;
        for (best[0..filled]) |token| {
            const logprob = teacher_row[token] - log_z;
            try self.positions.append(self.allocator, target_pos);
            try self.tokens.append(self.allocator, token);
            try self.logprobs.append(self.allocator, logprob);
            mass += @exp(logprob);
            if (mass >= min_prob_mass) break;
        }
    }

    /// The tensor-side counterpart of `appendRow`: append one row's targets
    /// from PRE-EXTRACTED descending top-k logits (`Tensor.topK` along the
    /// vocab axis — its lowest-index tie-break IS `appendRow`'s scan order)
    /// plus the row's log-partition (`logsumexp` along vocab), so the
    /// vocab-wide passes run as core ops and only `[rows, k]` values/indices
    /// and `[rows]` log-partitions reach the host. Same semantics:
    /// `logprob_j = values_j − log_z`, appended until `min_prob_mass`
    /// cumulative probability with the crossing entry included. Selection is
    /// identical to `appendRow`; logprobs may differ from it in the last
    /// ulps (the core reduction's summation order) — pinned by a unit test.
    pub fn appendTopKRow(
        self: *TargetsBuilder,
        target_pos: usize,
        values: []const f32,
        indices: []const i64,
        log_z: f32,
        min_prob_mass: f32,
    ) !void {
        if (target_pos == 0 or values.len == 0 or values.len != indices.len) return Error.InvalidTargets;
        var mass: f32 = 0;
        for (values, indices) |value, token| {
            const logprob = value - log_z;
            try self.positions.append(self.allocator, target_pos);
            try self.tokens.append(self.allocator, @intCast(token));
            try self.logprobs.append(self.allocator, logprob);
            mass += @exp(logprob);
            if (mass >= min_prob_mass) break;
        }
    }

    /// Append one already-extracted entry — the merge path when packing
    /// several conversations' targets into one packed row (positions shift
    /// by the segment start).
    pub fn appendEntry(self: *TargetsBuilder, target_pos: usize, token: usize, logprob: f32) !void {
        if (target_pos == 0) return Error.InvalidTargets;
        try self.positions.append(self.allocator, target_pos);
        try self.tokens.append(self.allocator, token);
        try self.logprobs.append(self.allocator, logprob);
    }

    /// Borrowed view over the accumulated entries.
    pub fn targets(self: *const TargetsBuilder) DistillTargets {
        return .{
            .positions = self.positions.items,
            .tokens = self.tokens.items,
            .logprobs = self.logprobs.items,
        };
    }
};

/// Sparse teacher targets for `distillLoss`: parallel arrays of top-k
/// entries, multiple entries per supervised position (the reference stores
/// flat (token_idx, token_id, logprob) triples truncated at 0.99 cumulative
/// mass). `positions[i]` is the packed-sequence index of the TARGET token —
/// the student's prediction is read from logits row `positions[i] - 1`.
pub const DistillTargets = struct {
    positions: []const usize,
    tokens: []const usize,
    logprobs: []const f32,
};

pub const DistillOptions = struct {
    /// `.mean` (reference train.py) averages over all retained entries;
    /// `.sum` composes with external normalization for gradient
    /// accumulation (see qwen3 train LossOptions).
    reduction: enum { mean, sum } = .mean,
    /// Multiplies the returned loss (and thus the gradients) when != 1.
    loss_scale: f32 = 1,
};

/// Teacher top-k distillation loss over full-sequence logits:
/// `mean_i(-exp(logprob_i) * log_softmax(logits)[positions_i - 1, tokens_i])`.
/// Differentiable in `logits`; entries sharing a row accumulate through the
/// duplicate-index gather gradient. MUST run inside an open exec scope
/// (`ctx.openExecScope()`) — the composite's intermediates rely on scope
/// adoption to keep the graph alive until `backward()`, exactly like
/// `qwen3.train.Trainer.loss`; the result is a scope-owned borrow.
pub fn distillLoss(
    ctx: *ExecContext,
    logits: *const fucina.Tensor(.{ .seq, .vocab }),
    targets: DistillTargets,
    options: DistillOptions,
) !fucina.Tensor(.{}) {
    if (!ctx.execScopeActive()) return Error.ExecScopeRequired;
    const n = targets.positions.len;
    if (n == 0 or targets.tokens.len != n or targets.logprobs.len != n) return Error.InvalidTargets;
    const seq = logits.dim(.seq);
    const vocab = logits.dim(.vocab);
    for (targets.positions, targets.tokens) |pos, token| {
        if (pos == 0 or pos > seq) return Error.InvalidTargets;
        if (token >= vocab) return Error.InvalidTargets;
    }

    const flat_indices = try ctx.allocator.alloc(usize, n);
    defer ctx.allocator.free(flat_indices);
    const neg_weights = try ctx.allocator.alloc(f32, n);
    defer ctx.allocator.free(neg_weights);
    for (flat_indices, neg_weights, targets.positions, targets.tokens, targets.logprobs) |*idx, *w, pos, token, logprob| {
        idx.* = (pos - 1) * vocab + token;
        w.* = -@exp(logprob);
    }

    var logq = try logits.logSoftmax(ctx, .vocab);
    defer logq.deinit();
    var flat = try logq.flatten(ctx, .flat);
    defer flat.deinit();
    var picked = try flat.gather(ctx, .flat, flat_indices, .entry);
    defer picked.deinit();
    var weights = try fucina.Tensor(.{.entry}).fromSlice(ctx, .{n}, neg_weights);
    defer weights.deinit();
    var weighted = try picked.mul(ctx, &weights);
    defer weighted.deinit();

    var reduced = switch (options.reduction) {
        .mean => try weighted.mean(ctx, .entry),
        .sum => try weighted.sum(ctx, .entry),
    };
    if (options.loss_scale == 1) return reduced;
    defer reduced.deinit(); // scope-owned: safe no-op, the graph survives
    return reduced.scale(ctx, options.loss_scale);
}

test {
    _ = @import("cartridge_tests.zig");
    _ = @import("cartridge_golden_tests.zig");
}
