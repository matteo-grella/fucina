//! Engram: conditional n-gram memory (arXiv 2601.07372, reference
//! implementation deepseek-ai/Engram — engram_demo_v1.py, pinned in
//! tools/fetch_refs.sh).
//!
//! Engram buys model capacity with LOOKUP instead of FLOPs: per selected
//! layer, suffix n-grams of the (compressed) token ids are hashed by K
//! multiplicative-XOR heads into prime-sized embedding tables, and the
//! retrieved rows — gated per hyper-connection stream by an RMS-normed
//! key·query dot — are added to the residual stream after a dilated causal
//! depthwise short convolution. Because every table address is a pure
//! function of token ids, all lookups for a sequence are known BEFORE the
//! forward pass: tables can live out-of-core and be prefetched with zero
//! speculation (the ExpertStore composition; see docs/ENGRAM.md).
//!
//! Reference semantics pinned here (demo v1):
//! - Token compression: raw ids map through a normalization lookup table
//!   (NFKC/lowercase/whitespace dedup of the tokenizer vocab). The table is
//!   an INPUT to this module (`lookup`); identity when absent.
//! - Per layer, per n-gram order n in `2..=max_ngram_size`, the mix is
//!   `xor_k(shift_k(ids) *% mult[k])` for `k < n` over int64 with wrapping
//!   multiply, where `shift_k` prepends k pad ids; per head j the row index
//!   is `mix mod prime[layer][n][j]` (FLOORED mod — numpy `%` semantics —
//!   so negative wrapped mixes stay in range).
//! - Head table sizes are consecutive DISTINCT primes searched upward from
//!   `engram_vocab_size[n-2] - 1`, with the seen-set GLOBAL across the
//!   layer/order/head iteration (reference `calculate_vocab_size_across_layers`).
//! - Multipliers are odd int64 draws in `[1, 2·half_bound)` seeded per
//!   layer (`seed + 10007·layer_id`). The reference draws them from
//!   numpy's PCG64; Fucina generates its own (std) draws natively and
//!   accepts injected multipliers for bit-parity with reference artifacts.
//!   Multipliers are part of the checkpoint either way.
//! - Per stream g: `gate = sigmoid(signed_sqrt((rms(key_g)·rms(query_g)) /
//!   sqrt(d)))` with `signed_sqrt(x) = sign(x)·sqrt(clamp_min(|x|, 1e-6))`;
//!   `value = gate_g · value_proj(emb)`; output = `value + ShortConv(value)`
//!   where ShortConv = per-stream RMSNorm -> depthwise causal conv1d
//!   (kernel_size taps, dilation = max_ngram_size, no bias) -> SiLU.
//!   The caller adds the residual (`hidden += engram(hidden, ids)`).
//!
//! Ownership: Layer/Engram tensors are long-lived parameters like LoRA A/B —
//! create them OUTSIDE any exec scope and keep them alive across steps;
//! `Layer.forward` is a composite op and must run under an open exec scope
//! when the result will be `backward()`'d (Error.ExecScopeRequired).

const std = @import("std");
const fucina = @import("fucina");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;

pub const Error = error{
    InvalidConfig,
    InvalidHashInput,
    ExecScopeRequired,
};

/// Hidden states enter and leave as `[seq, stream, d]` — `stream` is the
/// hyper-connection stream axis (`hc_mult`); standard residual-stream
/// models use `hc_mult == 1` (see `forwardResidual`).
pub const Hidden = fucina.Tensor(.{ .seq, .stream, .d });

/// i64 table-row indices `[seq, head]` (row-major = the `hashInto` flat
/// order), offsets included — the tensor-op counterpart of
/// `HashPlan.hashInto`.
pub const HashRows = fucina.Tensor(.{ .dtype = .i64, .tags = .{ .seq, .head } });

const I64Seq = fucina.Tensor(.{ .dtype = .i64, .tags = .{.seq} });
const I64Head = fucina.Tensor(.{ .dtype = .i64, .tags = .{.head} });
const I64SeqHead = HashRows;

/// The multi-head embedding table `[table_rows, head_dim]`.
pub const Table = fucina.Tensor(.{ .row, .hd });
/// Key/value projections `[hidden, engram_hidden]` (torch Linear layout).
pub const Proj = fucina.Tensor(.{ .d, .eh });
/// Bias / RMSNorm weight vectors `[hidden]`.
pub const Vec = fucina.Tensor(.{.d});
/// ShortConv depthwise kernel `[hc_mult*hidden, kernel_size]`.
pub const ConvKernel = fucina.Tensor(.{ .channel, .tap });

/// Geometry + numerics of the Engram module family. `engram_vocab_size`
/// holds the base table size per n-gram order (`len == max_ngram_size - 1`,
/// order 2 first); actual head table sizes are the primes searched from
/// these bases.
pub const Config = struct {
    /// Backbone model width (the residual stream dimension).
    hidden_size: usize,
    /// Hyper-connection streams sharing one Engram (1 = plain residual).
    hc_mult: usize = 1,
    /// Largest n-gram order; orders 2..=max participate.
    max_ngram_size: usize = 3,
    /// Total embedding width contributed per n-gram order.
    n_embed_per_ngram: usize,
    /// Hash heads per order; per-head embed dim = n_embed_per_ngram / heads.
    n_head_per_ngram: usize,
    /// Base table size per order (reference `engram_vocab_size`).
    engram_vocab_size: []const usize,
    /// ShortConv taps.
    kernel_size: usize = 4,
    /// ShortConv dilation; null = max_ngram_size (the reference default).
    dilation: ?usize = null,
    /// Pad id in RAW token space (mapped through `lookup` when present).
    pad_id: i64 = 0,
    /// RMSNorm eps of the gate norms (torch nn.RMSNorm default:
    /// finfo(f32).eps).
    norm_eps: f32 = 1.1920929e-07,
    /// RMSNorm eps of the ShortConv input norms (reference: 1e-5).
    conv_norm_eps: f32 = 1e-5,

    pub fn validate(self: *const Config) Error!void {
        if (self.hidden_size == 0 or self.hc_mult == 0) return Error.InvalidConfig;
        if (self.max_ngram_size < 2) return Error.InvalidConfig;
        if (self.engram_vocab_size.len != self.max_ngram_size - 1) return Error.InvalidConfig;
        if (self.n_head_per_ngram == 0) return Error.InvalidConfig;
        if (self.n_embed_per_ngram % self.n_head_per_ngram != 0) return Error.InvalidConfig;
        if (self.kernel_size == 0) return Error.InvalidConfig;
        if (self.dilation) |d| {
            if (d == 0) return Error.InvalidConfig;
        }
        for (self.engram_vocab_size) |v| {
            if (v < 3) return Error.InvalidConfig;
        }
    }

    pub fn ngramOrders(self: *const Config) usize {
        return self.max_ngram_size - 1;
    }

    /// Hash heads per layer: orders × heads-per-order.
    pub fn headsPerLayer(self: *const Config) usize {
        return self.ngramOrders() * self.n_head_per_ngram;
    }

    /// Flattened retrieved-embedding width per token.
    pub fn engramHidden(self: *const Config) usize {
        return self.ngramOrders() * self.n_embed_per_ngram;
    }

    pub fn headDim(self: *const Config) usize {
        return self.n_embed_per_ngram / self.n_head_per_ngram;
    }

    pub fn dilationOrDefault(self: *const Config) usize {
        return self.dilation orelse self.max_ngram_size;
    }

    pub fn convChannels(self: *const Config) usize {
        return self.hidden_size * self.hc_mult;
    }
};

fn isPrime(candidate: usize) bool {
    if (candidate < 2) return false;
    if (candidate % 2 == 0) return candidate == 2;
    var f: usize = 3;
    while (f * f <= candidate) : (f += 2) {
        if (candidate % f == 0) return false;
    }
    return true;
}

/// Per-model hash geometry: layer multipliers, per-head prime table sizes,
/// per-head offsets into each layer's concatenated table, and the optional
/// token-compression lookup. Pure integer data — no tensors — so it is
/// cheaply shareable across the trainer, the serving path, and a prefetcher.
pub const HashPlan = struct {
    allocator: Allocator,
    cfg: Config,
    /// Owned copy of cfg.engram_vocab_size (cfg's slice points here).
    vocab_sizes: []usize,
    /// Layers carrying an Engram, in reference iteration order.
    layer_ids: []usize,
    /// Odd int64 hash multipliers, `[layer][max_ngram]` flattened.
    multipliers: []i64,
    /// Per-head prime table sizes, `[layer][order][head]` flattened.
    head_mods: []i64,
    /// Per-head row offsets into the layer's concatenated table.
    head_offsets: []usize,
    /// Total table rows per layer (sum of that layer's head primes).
    table_rows: []usize,
    /// Optional raw-id -> compressed-id lookup (owned copy).
    lookup: ?[]i64,
    /// Pad id in compressed space (pad_id mapped through lookup).
    pad_compressed: i64,

    /// Native construction: multipliers drawn from a std PRNG seeded
    /// `seed + 10007 * layer_id` per layer (deterministic across
    /// platforms; NOT bit-equal to the reference's numpy PCG64 draws —
    /// use `initWithMultipliers` to reproduce reference artifacts).
    pub fn init(
        allocator: Allocator,
        cfg: Config,
        layer_ids: []const usize,
        seed: u64,
        lookup: ?[]const i64,
    ) !HashPlan {
        try cfg.validate();
        const compressed = compressedVocab(cfg, lookup);
        const half_bound: i64 = @max(1, @divTrunc(@divTrunc(std.math.maxInt(i64), @as(i64, @intCast(compressed))), 2));

        const multipliers = try allocator.alloc(i64, layer_ids.len * cfg.max_ngram_size);
        errdefer allocator.free(multipliers);
        for (layer_ids, 0..) |layer_id, slot| {
            var prng = std.Random.DefaultPrng.init(seed +% 10007 *% @as(u64, @intCast(layer_id)));
            const random = prng.random();
            for (0..cfg.max_ngram_size) |k| {
                const draw = random.intRangeLessThan(i64, 0, half_bound);
                multipliers[slot * cfg.max_ngram_size + k] = draw * 2 + 1;
            }
        }
        const plan = try initWithMultipliers(allocator, cfg, layer_ids, multipliers, lookup);
        allocator.free(multipliers);
        return plan;
    }

    /// Construction from explicit multipliers (`[layer][max_ngram]`
    /// flattened, layer order = `layer_ids` order) — the parity path for
    /// reference artifacts and the state-dict load path. Prime table sizes
    /// are always derived (they are a pure function of the config).
    pub fn initWithMultipliers(
        allocator: Allocator,
        cfg: Config,
        layer_ids: []const usize,
        multipliers: []const i64,
        lookup: ?[]const i64,
    ) !HashPlan {
        try cfg.validate();
        if (layer_ids.len == 0) return Error.InvalidConfig;
        if (multipliers.len != layer_ids.len * cfg.max_ngram_size) return Error.InvalidConfig;
        for (multipliers) |m| {
            if (@mod(m, 2) != 1) return Error.InvalidConfig;
        }

        const orders = cfg.ngramOrders();
        const heads = cfg.n_head_per_ngram;
        const per_layer = orders * heads;

        const vocab_sizes = try allocator.dupe(usize, cfg.engram_vocab_size);
        errdefer allocator.free(vocab_sizes);
        const ids = try allocator.dupe(usize, layer_ids);
        errdefer allocator.free(ids);
        const mults = try allocator.dupe(i64, multipliers);
        errdefer allocator.free(mults);
        const head_mods = try allocator.alloc(i64, layer_ids.len * per_layer);
        errdefer allocator.free(head_mods);
        const head_offsets = try allocator.alloc(usize, layer_ids.len * per_layer);
        errdefer allocator.free(head_offsets);
        const table_rows = try allocator.alloc(usize, layer_ids.len);
        errdefer allocator.free(table_rows);

        // The reference's GLOBAL seen-set across layers x orders x heads:
        // every head everywhere gets a distinct prime.
        var seen = std.AutoHashMap(usize, void).init(allocator);
        defer seen.deinit();
        for (0..layer_ids.len) |slot| {
            var offset: usize = 0;
            for (0..orders) |order| {
                var start = cfg.engram_vocab_size[order] - 1;
                for (0..heads) |head| {
                    var candidate = start + 1;
                    while (!isPrime(candidate) or seen.contains(candidate)) candidate += 1;
                    try seen.put(candidate, {});
                    const h = slot * per_layer + order * heads + head;
                    head_mods[h] = @intCast(candidate);
                    head_offsets[h] = offset;
                    offset += candidate;
                    start = candidate;
                }
            }
            table_rows[slot] = offset;
        }

        var owned_lookup: ?[]i64 = null;
        errdefer if (owned_lookup) |l| allocator.free(l);
        var pad_compressed = cfg.pad_id;
        if (lookup) |table| {
            owned_lookup = try allocator.dupe(i64, table);
            if (cfg.pad_id < 0 or @as(usize, @intCast(cfg.pad_id)) >= table.len) return Error.InvalidConfig;
            pad_compressed = table[@intCast(cfg.pad_id)];
        }

        var cfg_owned = cfg;
        cfg_owned.engram_vocab_size = vocab_sizes;

        return .{
            .allocator = allocator,
            .cfg = cfg_owned,
            .vocab_sizes = vocab_sizes,
            .layer_ids = ids,
            .multipliers = mults,
            .head_mods = head_mods,
            .head_offsets = head_offsets,
            .table_rows = table_rows,
            .lookup = owned_lookup,
            .pad_compressed = pad_compressed,
        };
    }

    pub fn deinit(self: *HashPlan) void {
        self.allocator.free(self.vocab_sizes);
        self.allocator.free(self.layer_ids);
        self.allocator.free(self.multipliers);
        self.allocator.free(self.head_mods);
        self.allocator.free(self.head_offsets);
        self.allocator.free(self.table_rows);
        if (self.lookup) |l| self.allocator.free(l);
        self.* = undefined;
    }

    fn compressedVocab(cfg: Config, lookup: ?[]const i64) usize {
        _ = cfg;
        const table = lookup orelse return 1;
        var max_id: i64 = 0;
        for (table) |v| max_id = @max(max_id, v);
        return @intCast(max_id + 1);
    }

    /// Slot of `layer_id` inside the plan arrays, or null when the layer
    /// carries no Engram.
    pub fn slotOf(self: *const HashPlan, layer_id: usize) ?usize {
        for (self.layer_ids, 0..) |id, slot| {
            if (id == layer_id) return slot;
        }
        return null;
    }

    /// Map raw token ids into compressed space (identity without a lookup).
    /// Negative ids pass through untouched (the reference's mask contract).
    pub fn compressInto(self: *const HashPlan, raw: []const i64, out: []i64) Error!void {
        if (out.len != raw.len) return Error.InvalidHashInput;
        const table = self.lookup orelse {
            @memcpy(out, raw);
            return;
        };
        for (raw, out) |id, *dst| {
            if (id < 0) {
                dst.* = id;
                continue;
            }
            if (@as(usize, @intCast(id)) >= table.len) return Error.InvalidHashInput;
            dst.* = table[@intCast(id)];
        }
    }

    /// The hash proper, host path: COMPRESSED ids `[T]` -> flat table-row
    /// indices `[T * headsPerLayer]` (`seq`-major, head-minor: head index
    /// `(order-2)·heads + j`), per-head offsets included — ready for
    /// `Tensor.gather` along the table's `.row` axis. Bit-exact to the
    /// reference: wrapping i64 multiply, XOR, floored mod.
    pub fn hashInto(self: *const HashPlan, slot: usize, ids: []const i64, out: []usize) Error!void {
        const cfg = &self.cfg;
        const orders = cfg.ngramOrders();
        const heads = cfg.n_head_per_ngram;
        const per_layer = orders * heads;
        if (slot >= self.layer_ids.len) return Error.InvalidHashInput;
        if (out.len != ids.len * per_layer) return Error.InvalidHashInput;

        const mults = self.multipliers[slot * cfg.max_ngram_size ..][0..cfg.max_ngram_size];
        for (0..ids.len) |t| {
            // mix accumulates order n = k_used + 1 as k grows: after
            // xor-ing shift_k the mix covers tokens t, t-1, .., t-k.
            var mix: i64 = shiftedId(self, ids, t, 0) *% mults[0];
            for (1..cfg.max_ngram_size) |k| {
                mix ^= shiftedId(self, ids, t, k) *% mults[k];
                const order = k - 1; // n = k + 1 => order index n - 2
                for (0..heads) |head| {
                    const h = order * heads + head;
                    const modulus = self.head_mods[slot * per_layer + h];
                    const row = @mod(mix, modulus);
                    out[t * per_layer + h] = @as(usize, @intCast(row)) + self.head_offsets[slot * per_layer + h];
                }
            }
        }
    }

    fn shiftedId(self: *const HashPlan, ids: []const i64, t: usize, k: usize) i64 {
        return if (t >= k) ids[t - k] else self.pad_compressed;
    }

    /// The hash as tensor ops — the API-first counterpart of `hashInto`,
    /// composed from the integer ops (`mul` wraps, `bitXor`, floored `mod`,
    /// broadcast `add`): returns the same flat i64 row indices. Used by the
    /// parity tests; the host path is the serving fast lane.
    pub fn hashTensor(self: *const HashPlan, ctx: *ExecContext, slot: usize, ids: []const i64) !HashRows {
        const cfg = &self.cfg;
        const orders = cfg.ngramOrders();
        const heads = cfg.n_head_per_ngram;
        const per_layer = orders * heads;
        if (slot >= self.layer_ids.len) return Error.InvalidHashInput;
        if (ids.len == 0) return Error.InvalidHashInput;

        const t_len = ids.len;
        const shift_buf = try self.allocator.alloc(i64, t_len);
        defer self.allocator.free(shift_buf);

        const mults = self.multipliers[slot * cfg.max_ngram_size ..][0..cfg.max_ngram_size];

        var mix: ?I64Seq = null;
        defer if (mix) |*m| m.deinit();
        var per_order = try self.allocator.alloc(I64SeqHead, orders);
        var produced: usize = 0;
        defer {
            for (per_order[0..produced]) |*p| p.deinit();
            self.allocator.free(per_order);
        }

        for (0..cfg.max_ngram_size) |k| {
            for (0..t_len) |t| shift_buf[t] = shiftedId(self, ids, t, k);
            var shifted = try I64Seq.fromSlice(ctx, .{t_len}, shift_buf);
            defer shifted.deinit();
            const mult_arr = [_]i64{mults[k]};
            var mult = try I64Seq.fromSlice(ctx, .{1}, &mult_arr);
            defer mult.deinit();
            var product = try shifted.mul(ctx, &mult);
            if (mix) |*m| {
                defer product.deinit();
                const next = try m.bitXor(ctx, &product);
                m.deinit();
                mix = next;
            } else {
                mix = product;
            }

            if (k == 0) continue;
            const order = k - 1;
            const mods_slice = self.head_mods[slot * per_layer + order * heads ..][0..heads];
            var mods = try I64Head.fromSlice(ctx, .{heads}, mods_slice);
            defer mods.deinit();
            const offs_buf = try self.allocator.alloc(i64, heads);
            defer self.allocator.free(offs_buf);
            for (offs_buf, self.head_offsets[slot * per_layer + order * heads ..][0..heads]) |*dst, off| dst.* = @intCast(off);
            var offs = try I64Head.fromSlice(ctx, .{heads}, offs_buf);
            defer offs.deinit();

            var rows = try mix.?.mod(ctx, &mods);
            defer rows.deinit();
            per_order[produced] = try rows.add(ctx, &offs);
            produced += 1;
        }

        // concat orders along .head — row-major [seq][head] is exactly
        // hashInto's flat order.
        if (produced == 1) {
            const joined = per_order[0];
            produced = 0;
            return joined;
        }
        const rest = try self.allocator.alloc(*const I64SeqHead, produced - 1);
        defer self.allocator.free(rest);
        for (rest, per_order[1..produced]) |*ptr, *p| ptr.* = p;
        return per_order[0].concat(ctx, .head, rest);
    }
};

pub const InitOptions = struct {
    /// Zero-initialize the value projection (weight + bias): the module's
    /// output is exactly zero, so grafting it onto a frozen pretrained
    /// model is bitwise identity at step 0 while gradients still reach
    /// every parameter through the value path.
    graft_zero_init: bool = false,
    /// Embedding table init scale (reference nn.Embedding: N(0, 1)).
    table_std: f32 = 1.0,
};

/// One layer's Engram parameters + forward. All tensors are long-lived
/// leaf variables (create outside exec scopes).
pub const Layer = struct {
    allocator: Allocator,
    cfg: Config,
    /// Multi-head embedding table `[table_rows, head_dim]` — the reference's
    /// single concatenated nn.Embedding.
    table: Table,
    /// Per-stream gate key projections `[d, engram_hidden]` + bias.
    key_w: []Proj,
    key_b: []Vec,
    /// Gate norms: norm1 (key side) and norm2 (query side), per stream.
    norm_key: []Vec,
    norm_query: []Vec,
    /// Shared value projection `[d, engram_hidden]` + bias.
    value_w: Proj,
    value_b: Vec,
    /// ShortConv: depthwise kernel `[hc_mult·d, kernel_size]` (no bias) and
    /// per-stream input norms.
    conv_w: ConvKernel,
    conv_norm: []Vec,

    /// Random initialization (torch module defaults: embedding N(0,·),
    /// linear Kaiming-uniform bounds U(±1/sqrt(fan_in)), norms ones,
    /// conv U(±1/sqrt(taps))).
    pub fn initRandom(
        ctx: *ExecContext,
        allocator: Allocator,
        cfg: Config,
        table_rows: usize,
        seed: u64,
        opts: InitOptions,
    ) !Layer {
        try cfg.validate();
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        const g = cfg.hc_mult;
        const d = cfg.hidden_size;
        const eh = cfg.engramHidden();
        const hd = cfg.headDim();

        var self: Layer = undefined;
        self.allocator = allocator;
        self.cfg = cfg;

        const scratch_len = @max(table_rows * hd, @max(d * eh, cfg.convChannels() * cfg.kernel_size));
        const scratch = try allocator.alloc(f32, scratch_len);
        defer allocator.free(scratch);

        for (scratch[0 .. table_rows * hd]) |*v| v.* = randomNormal(random, opts.table_std);
        self.table = try Table.variableFromSlice(ctx, .{ table_rows, hd }, scratch[0 .. table_rows * hd]);
        errdefer self.table.deinit();

        const proj_bound = 1.0 / @sqrt(@as(f32, @floatFromInt(eh)));
        // Simple init failure contract: free the slices; tensors created so
        // far leak only on OOM mid-init (accepted: init is a startup path).
        self.key_w = try allocator.alloc(Proj, g);
        errdefer allocator.free(self.key_w);
        self.key_b = try allocator.alloc(Vec, g);
        errdefer allocator.free(self.key_b);
        self.norm_key = try allocator.alloc(Vec, g);
        errdefer allocator.free(self.norm_key);
        self.norm_query = try allocator.alloc(Vec, g);
        errdefer allocator.free(self.norm_query);
        self.conv_norm = try allocator.alloc(Vec, g);
        errdefer allocator.free(self.conv_norm);
        for (0..g) |i| {
            for (scratch[0 .. d * eh]) |*v| v.* = randomUniform(random, proj_bound);
            self.key_w[i] = try Proj.variableFromSlice(ctx, .{ d, eh }, scratch[0 .. d * eh]);
            for (scratch[0..d]) |*v| v.* = randomUniform(random, proj_bound);
            self.key_b[i] = try Vec.variableFromSlice(ctx, .{d}, scratch[0..d]);
            for (scratch[0..d]) |*v| v.* = 1;
            self.norm_key[i] = try Vec.variableFromSlice(ctx, .{d}, scratch[0..d]);
            self.norm_query[i] = try Vec.variableFromSlice(ctx, .{d}, scratch[0..d]);
            self.conv_norm[i] = try Vec.variableFromSlice(ctx, .{d}, scratch[0..d]);
        }

        if (opts.graft_zero_init) {
            for (scratch[0 .. d * eh]) |*v| v.* = 0;
        } else {
            for (scratch[0 .. d * eh]) |*v| v.* = randomUniform(random, proj_bound);
        }
        self.value_w = try Proj.variableFromSlice(ctx, .{ d, eh }, scratch[0 .. d * eh]);
        if (opts.graft_zero_init) {
            for (scratch[0..d]) |*v| v.* = 0;
        } else {
            for (scratch[0..d]) |*v| v.* = randomUniform(random, proj_bound);
        }
        self.value_b = try Vec.variableFromSlice(ctx, .{d}, scratch[0..d]);

        const channels = cfg.convChannels();
        const conv_bound = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.kernel_size)));
        for (scratch[0 .. channels * cfg.kernel_size]) |*v| v.* = randomUniform(random, conv_bound);
        self.conv_w = try ConvKernel.variableFromSlice(ctx, .{ channels, cfg.kernel_size }, scratch[0 .. channels * cfg.kernel_size]);

        return self;
    }

    pub fn deinit(self: *Layer) void {
        self.table.deinit();
        for (self.key_w) |*t| t.deinit();
        for (self.key_b) |*t| t.deinit();
        for (self.norm_key) |*t| t.deinit();
        for (self.norm_query) |*t| t.deinit();
        for (self.conv_norm) |*t| t.deinit();
        self.allocator.free(self.key_w);
        self.allocator.free(self.key_b);
        self.allocator.free(self.norm_key);
        self.allocator.free(self.norm_query);
        self.allocator.free(self.conv_norm);
        self.value_w.deinit();
        self.value_b.deinit();
        self.conv_w.deinit();
        self.* = undefined;
    }

    /// Register every parameter under `<prefix>.<name>` (bufPrint names —
    /// the registry copies them).
    pub fn registerInto(self: *Layer, registry: *fucina.ParamRegistry, comptime prefix_fmt: []const u8, prefix_args: anytype) !void {
        var name_buf: [96]u8 = undefined;
        var prefix_buf: [64]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buf, prefix_fmt, prefix_args);
        try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.table", .{prefix}), &self.table);
        for (0..self.cfg.hc_mult) |i| {
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.key_w.{d}", .{ prefix, i }), &self.key_w[i]);
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.key_b.{d}", .{ prefix, i }), &self.key_b[i]);
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.norm_key.{d}", .{ prefix, i }), &self.norm_key[i]);
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.norm_query.{d}", .{ prefix, i }), &self.norm_query[i]);
            try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.conv_norm.{d}", .{ prefix, i }), &self.conv_norm[i]);
        }
        try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.value_w", .{prefix}), &self.value_w);
        try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.value_b", .{prefix}), &self.value_b);
        try registry.addParam(try std.fmt.bufPrint(&name_buf, "{s}.conv_w", .{prefix}), &self.conv_w);
    }

    /// The Engram side-branch output for one sequence: `hidden` is
    /// `[seq, stream, d]`, `rows` the flat table-row indices from
    /// `HashPlan.hashInto` (`seq * headsPerLayer`), `conv_state` the
    /// optional `dilation·(taps−1)` ShortConv rows preceding this chunk
    /// (streaming decode). Returns `[seq, stream, d]`; the caller adds the
    /// residual. Composite op: requires an open exec scope when any
    /// parameter (or `hidden`) requires grad.
    pub fn forward(
        self: *const Layer,
        ctx: *ExecContext,
        hidden: *const Hidden,
        rows: []const usize,
        conv_state: ?[]const f32,
    ) !Hidden {
        const cfg = &self.cfg;
        const g_count = cfg.hc_mult;
        const d = cfg.hidden_size;
        const eh = cfg.engramHidden();
        const heads = cfg.headsPerLayer();
        const seq = hidden.dim(.seq);
        if (hidden.dim(.stream) != g_count or hidden.dim(.d) != d) return Error.InvalidHashInput;
        if (rows.len != seq * heads) return Error.InvalidHashInput;
        // Same gate the composed library ops apply (select/stack/reshape):
        // parameters are variables, so a training forward needs an open
        // exec scope; a `fucina.noGrad()` serving forward does not.
        const wants_grad = hidden.requiresGrad() or self.table.requiresGrad();
        if (wants_grad and fucina.isGradEnabled() and !ctx.execScopeActive()) return Error.ExecScopeRequired;

        // Retrieved memory: gather rows, then flatten heads into the
        // engram-hidden axis: [seq*heads, hd] -> [seq, eh].
        var gathered = try self.table.gather(ctx, .row, rows, .th);
        defer gathered.deinit();
        var emb = try gathered.reshape(ctx, .{ .seq, .eh }, .{ seq, eh });
        defer emb.deinit();

        // Shared value projection: [seq, d] + bias.
        var value_lin = try emb.dot(ctx, &self.value_w, .eh);
        defer value_lin.deinit();
        var value = try value_lin.add(ctx, &self.value_b);
        defer value.deinit();

        const inv_sqrt_d = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

        var gated = try self.allocator.alloc(fucina.Tensor(.{ .seq, .d }), g_count);
        var gated_count: usize = 0;
        defer {
            for (gated[0..gated_count]) |*t| t.deinit();
            self.allocator.free(gated);
        }
        var normed_conv = try self.allocator.alloc(fucina.Tensor(.{ .seq, .channel }), g_count);
        var normed_count: usize = 0;
        defer {
            for (normed_conv[0..normed_count]) |*t| t.deinit();
            self.allocator.free(normed_conv);
        }

        for (0..g_count) |g| {
            // Gate: sigmoid(signed_sqrt((rms(key) . rms(query)) / sqrt(d))).
            var key_lin = try emb.dot(ctx, &self.key_w[g], .eh);
            defer key_lin.deinit();
            var key = try key_lin.add(ctx, &self.key_b[g]);
            defer key.deinit();
            var normed_key = try key.rmsNormMul(ctx, .d, &self.norm_key[g], cfg.norm_eps);
            defer normed_key.deinit();

            var query = try hidden.select(ctx, .stream, @intCast(g));
            defer query.deinit();
            var normed_query = try query.rmsNormMul(ctx, .d, &self.norm_query[g], cfg.norm_eps);
            defer normed_query.deinit();

            var prod = try normed_key.mul(ctx, &normed_query);
            defer prod.deinit();
            var dot_raw = try prod.sum(ctx, .d);
            defer dot_raw.deinit();
            var dot_scaled = try dot_raw.scale(ctx, inv_sqrt_d);
            defer dot_scaled.deinit();

            var sgn = try dot_scaled.sign(ctx);
            defer sgn.deinit();
            var mag = try dot_scaled.abs(ctx);
            defer mag.deinit();
            var mag_floor = try mag.clampMin(ctx, 1e-6);
            defer mag_floor.deinit();
            var root = try mag_floor.sqrt(ctx);
            defer root.deinit();
            var signed = try root.mul(ctx, &sgn);
            defer signed.deinit();
            var gate = try signed.sigmoid(ctx);
            defer gate.deinit();

            // value_g = gate ⊙ value (gate broadcast over .d).
            gated[g] = try value.mul(ctx, &gate);
            gated_count += 1;

            // ShortConv input: per-stream RMSNorm, retagged to the conv
            // channel block g·d..(g+1)·d.
            var cn = try gated[g].rmsNormMul(ctx, .d, &self.conv_norm[g], cfg.conv_norm_eps);
            defer cn.deinit();
            normed_conv[g] = try cn.withTags(ctx, .{ .seq, .channel });
            normed_count += 1;
        }

        // Depthwise causal conv over the concatenated stream channels.
        var conv_in: fucina.Tensor(.{ .seq, .channel }) = undefined;
        if (g_count == 1) {
            conv_in = normed_conv[0];
            normed_count = 0;
        } else {
            const rest = try self.allocator.alloc(*const fucina.Tensor(.{ .seq, .channel }), g_count - 1);
            defer self.allocator.free(rest);
            for (rest, normed_conv[1..g_count]) |*ptr, *t| ptr.* = t;
            conv_in = try normed_conv[0].concat(ctx, .channel, rest);
        }
        defer conv_in.deinit();
        var conv_out = try conv_in.causalDepthwiseConv1d(ctx, .seq, .channel, .tap, &self.conv_w, cfg.dilationOrDefault(), conv_state);
        defer conv_out.deinit();
        var conv_act = try conv_out.silu(ctx);
        defer conv_act.deinit();

        // output_g = value_g + conv_g; stack streams back into the middle
        // axis.
        var outs = try self.allocator.alloc(fucina.Tensor(.{ .seq, .d }), g_count);
        var outs_count: usize = 0;
        defer {
            for (outs[0..outs_count]) |*t| t.deinit();
            self.allocator.free(outs);
        }
        for (0..g_count) |g| {
            var window = try conv_act.narrow(ctx, .channel, g * d, d);
            defer window.deinit();
            var branch = try window.withTags(ctx, .{ .seq, .d });
            defer branch.deinit();
            outs[g] = try gated[g].add(ctx, &branch);
            outs_count += 1;
        }
        // stack copies its inputs (concat of inserted-axis views), so the
        // per-stream outputs stay ours to deinit; g_count == 1 stacks with
        // zero others.
        const rest = try self.allocator.alloc(*const fucina.Tensor(.{ .seq, .d }), g_count - 1);
        defer self.allocator.free(rest);
        for (rest, outs[1..g_count]) |*ptr, *t| ptr.* = t;
        return outs[0].stack(ctx, .stream, 1, rest);
    }

    /// Convenience for plain residual-stream models (`hc_mult == 1`):
    /// `[seq, d]` in, Engram output `[seq, d]` out (caller adds it to the
    /// residual).
    pub fn forwardResidual(
        self: *const Layer,
        ctx: *ExecContext,
        hidden: *const fucina.Tensor(.{ .seq, .d }),
        rows: []const usize,
        conv_state: ?[]const f32,
    ) !fucina.Tensor(.{ .seq, .d }) {
        if (self.cfg.hc_mult != 1) return Error.InvalidConfig;
        var expanded = try hidden.insertAxis(ctx, .stream, 1);
        defer expanded.deinit();
        var out = try self.forward(ctx, &expanded, rows, conv_state);
        defer out.deinit();
        var squeezed = try out.select(ctx, .stream, 0);
        errdefer squeezed.deinit();
        return squeezed;
    }
};

fn randomNormal(random: std.Random, std_dev: f32) f32 {
    return random.floatNorm(f32) * std_dev;
}

fn randomUniform(random: std.Random, bound: f32) f32 {
    return (random.float(f32) * 2 - 1) * bound;
}

/// A whole-model Engram: the shared `HashPlan` plus one `Layer` per entry
/// of `layer_ids`, with every parameter (and the multipliers, as a frozen
/// i64 entry) in a `ParamRegistry` for optimizers and state-dict
/// persistence.
pub const Engram = struct {
    allocator: Allocator,
    plan: HashPlan,
    layers: []Layer,
    registry: fucina.ParamRegistry,
    /// Frozen i64 view of plan.multipliers (persisted geometry).
    multipliers_tensor: fucina.Tensor(.{ .dtype = .i64, .tags = .{.idx} }),

    pub fn init(
        ctx: *ExecContext,
        allocator: Allocator,
        cfg: Config,
        layer_ids: []const usize,
        seed: u64,
        lookup: ?[]const i64,
        opts: InitOptions,
    ) !Engram {
        var plan = try HashPlan.init(allocator, cfg, layer_ids, seed, lookup);
        errdefer plan.deinit();
        return initFromPlan(ctx, allocator, plan, seed, opts);
    }

    /// Takes ownership of `plan`.
    pub fn initFromPlan(ctx: *ExecContext, allocator: Allocator, plan: HashPlan, seed: u64, opts: InitOptions) !Engram {
        var layers = try allocator.alloc(Layer, plan.layer_ids.len);
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*l| l.deinit();
            allocator.free(layers);
        }
        for (0..plan.layer_ids.len) |slot| {
            layers[slot] = try Layer.initRandom(ctx, allocator, plan.cfg, plan.table_rows[slot], seed +% 0x9E3779B97F4A7C15 *% @as(u64, @intCast(slot + 1)), opts);
            built += 1;
        }

        var registry = fucina.ParamRegistry.init(allocator);
        errdefer registry.deinit();
        var multipliers_tensor = try fucina.Tensor(.{ .dtype = .i64, .tags = .{.idx} }).fromSlice(ctx, .{plan.multipliers.len}, plan.multipliers);
        errdefer multipliers_tensor.deinit();
        try registry.addParam("engram.multipliers", &multipliers_tensor);
        for (layers, plan.layer_ids) |*layer, layer_id| {
            try layer.registerInto(&registry, "engram.layers.{d}", .{layer_id});
        }

        return .{
            .allocator = allocator,
            .plan = plan,
            .layers = layers,
            .registry = registry,
            .multipliers_tensor = multipliers_tensor,
        };
    }

    pub fn deinit(self: *Engram) void {
        self.registry.deinit();
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        self.multipliers_tensor.deinit();
        self.plan.deinit();
        self.* = undefined;
    }

    pub fn layerFor(self: *Engram, layer_id: usize) ?*Layer {
        const slot = self.plan.slotOf(layer_id) orelse return null;
        return &self.layers[slot];
    }

    /// Register the trainable parameters on an optimizer (frozen entries —
    /// the multipliers — are skipped).
    pub fn registerParams(self: *const Engram, opt: anytype) !void {
        try self.registry.addParamsTo(opt);
    }

    pub fn zeroGrad(self: *Engram) void {
        self.registry.zeroGrad();
    }

    pub fn saveStateDict(self: *const Engram, writer: anytype) !void {
        try self.registry.saveStateDict(writer);
    }

    pub fn loadStateDict(self: *Engram, reader: *std.Io.Reader, options: fucina.state_dict.LoadOptions) !void {
        try self.registry.loadStateDict(reader, options);
        // Adopt persisted multipliers into the plan (geometry travels with
        // the checkpoint; primes re-derive from the config).
        const data = try self.multipliers_tensor.dataConst();
        if (data.len == self.plan.multipliers.len) {
            @memcpy(self.plan.multipliers, data);
        }
    }
};

test {
    _ = @import("engram_tests.zig");
    _ = @import("engram_golden_tests.zig");
}
