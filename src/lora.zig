//! LoRA (Low-Rank Adaptation) adapters over the autograd facade.
//!
//! For a frozen linear weight W: [out, in] — f32, f16, bf16, or a
//! block-quantized constant, anything `dot` accepts as a frozen RHS — an
//! adapter learns the additive update
//!
//!     y = base(x) + (alpha / r) * dropout(x) · A^T · B^T
//!
//! with A: [r, in] (kaiming-uniform, the PyTorch nn.Linear / LoRA-A init via
//! `rng.kaimingUniformFill`) and B: [out, r] (zeros — the initial delta is
//! exactly zero). Only A and B train; the base weight stays frozen (f16/bf16/
//! quantized RHS dots route gradients to the f32 LHS only,
//! ConstRhsDotBackward, and constants carry no GradState at all).
//!
//! Tags: the adapter is generic over the model's axis tags, mirroring how the
//! rest of the runtime is generic over the facade. The rank axis uses the
//! reserved tag `.lora_r` — tags are open enum literals (`tags.Tag =
//! @TypeOf(.tag)`), so no registry edit is needed; the tag is reserved here by
//! convention and rejected on adapter inputs.
//!
//! Lifetime contract (the same composite-op contract as `fucina.checkpoint`,
//! see docs/TRAINING.md): `delta`/`apply` build a multi-op chain and release their
//! interior tensors on return. Under an open exec scope
//! (`ctx.openExecScope()`) those releases are no-ops and the scope keeps the
//! whole graph alive — training (anything that calls `backward()` through the
//! result) MUST run inside a scope, or backward walks freed graph nodes
//! (GradState is single-owner). Without a scope the result is forward-only
//! (eval), the usual inference deinit-ASAP contract. The adapter's own A/B are
//! explicitly created variables: caller-owned, never scope-adopted.

const std = @import("std");

const ag = @import("ag.zig");
const exec_mod = @import("exec.zig");
const optim = @import("optim.zig");
const rng = @import("rng.zig");
const tags_mod = @import("tags.zig");
const tensor_mod = @import("tensor.zig");

const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const Tensor = ag.Tensor;
const TensorError = tensor_mod.TensorError;

/// The reserved rank axis tag every adapter contracts over.
pub const rank_tag: Tag = .lora_r;

pub const LoraError = error{
    InvalidRank,
    InvalidDropout,
};

/// Adapter hyperparameters. The effective delta scaling is `alpha / rank`
/// (the standard LoRA parameterization, so `alpha` transfers across ranks).
pub const Config = struct {
    rank: usize,
    alpha: f32,
    dropout_p: f32 = 0,
};

const AdapterConfig = Config;

/// A LoRA adapter for frozen linears mapping `in_tag` features to `out_tag`
/// features. `in_tag` and `out_tag` must be distinct and neither may be the
/// reserved `.lora_r` rank tag.
pub fn Adapter(comptime in_tag: Tag, comptime out_tag: Tag) type {
    comptime {
        if (tags_mod.tagEqual(in_tag, out_tag)) @compileError("LoRA adapter requires distinct in/out tags");
        if (tags_mod.tagEqual(in_tag, rank_tag) or tags_mod.tagEqual(out_tag, rank_tag)) {
            @compileError("the .lora_r tag is reserved for the LoRA rank axis");
        }
    }

    return struct {
        pub const ATensor = Tensor(.{ rank_tag, in_tag });
        pub const BTensor = Tensor(.{ out_tag, rank_tag });
        pub const Config = AdapterConfig;

        /// A: [rank, in] trainable variable, kaiming-uniform initialized.
        a: ATensor,
        /// B: [out, rank] trainable variable, zero initialized.
        b: BTensor,
        /// alpha / rank, applied to the low-rank product.
        scale: f32,
        /// Inverted-dropout probability on x inside `delta` (train mode only).
        dropout_p: f32,

        const Self = @This();

        /// Create an adapter for a frozen [out_dim, in_dim] weight. `seed`
        /// drives A's kaiming-uniform fill deterministically (same seed →
        /// bitwise-identical A); B starts as exact zeros so the initial delta
        /// is exactly zero. A and B are caller-owned variables: pair with
        /// `deinit`, and keep them alive for as long as any optimizer or
        /// state-dict entry borrows them.
        pub fn init(ctx: *ExecContext, in_dim: usize, out_dim: usize, config: AdapterConfig, seed: u64) !Self {
            if (config.rank < 1 or config.rank > @min(in_dim, out_dim)) return LoraError.InvalidRank;
            if (!(config.dropout_p >= 0 and config.dropout_p < 1)) return LoraError.InvalidDropout;

            var a = blk: {
                var value = try ctx.emptyRank(2, .{ config.rank, in_dim });
                errdefer value.deinit();
                rng.kaimingUniformFill(seed, value.data(), in_dim);
                break :blk try ATensor.variable(ctx, value);
            };
            errdefer a.deinit();

            const b = blk: {
                var value = try ctx.zeros(&.{ out_dim, config.rank });
                errdefer value.deinit();
                break :blk try BTensor.variable(ctx, value);
            };

            return .{
                .a = a,
                .b = b,
                .scale = config.alpha / @as(f32, @floatFromInt(config.rank)),
                .dropout_p = config.dropout_p,
            };
        }

        pub fn deinit(self: *Self) void {
            self.a.deinit();
            self.b.deinit();
            self.* = undefined;
        }

        /// Result type of `delta`/`apply` for an input pointer type `XPtr`:
        /// x's tags with `in_tag` contracted away and `out_tag` appended — for
        /// the usual trailing-feature layout (.., in_tag) that is exactly x's
        /// type with `in_tag` replaced by `out_tag`. Also performs the
        /// comptime input validation.
        pub fn Delta(comptime XPtr: type) type {
            const X = InputTensor(XPtr);
            const xa_tags = tags_mod.dotResultTags(X.axis_tags, ATensor.axis_tags, in_tag);
            return Tensor(tags_mod.dotResultTags(xa_tags, BTensor.axis_tags, rank_tag));
        }

        /// delta(x) = scale * dropout(x) · A^T · B^T.
        ///
        /// `dropout_seed` selects the mode: pass a fresh per-step seed (e.g.
        /// `rng.at`-derived per step/layer) to train — it is only consumed
        /// when `dropout_p > 0` — or `null` for eval, which skips dropout
        /// entirely (identical to the p == 0 identity path).
        ///
        /// Composite-op contract: call under an exec scope when the result
        /// will be backward()'d (see the module doc).
        pub fn delta(self: *const Self, ctx: *ExecContext, x: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x)) {
            // Eval (null seed) reuses dropout's p == 0 zero-copy identity
            // path, so both modes run the same op chain.
            const p: f32 = if (dropout_seed != null) self.dropout_p else 0;
            var dropped = try x.dropout(ctx, p, dropout_seed orelse 0);
            defer dropped.deinit();
            var xa = try dropped.dot(ctx, &self.a, in_tag);
            defer xa.deinit();
            var xab = try xa.dot(ctx, &self.b, rank_tag);
            defer xab.deinit();
            return xab.scale(ctx, self.scale);
        }

        /// apply(x, base) = base + delta(x). `base` must be an f32 facade
        /// tensor carrying exactly the delta's tags (x with in → out), e.g.
        /// the frozen-path output `x.dot(ctx, &w, in_tag)`. Same exec-scope
        /// contract as `delta`.
        pub fn apply(self: *const Self, ctx: *ExecContext, x: anytype, base: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x)) {
            const Base = facadePointee(@TypeOf(base), "LoRA apply base");
            comptime {
                if (Base.dtype != .f32) @compileError("LoRA apply base must be f32 (frozen-weight dots already produce f32)");
                if (!tags_mod.tagsEqual(Base.axis_tags, Delta(@TypeOf(x)).axis_tags)) {
                    @compileError("LoRA apply base must carry the delta's tags (x with in_tag replaced by out_tag)");
                }
            }
            var d = try self.delta(ctx, x, dropout_seed);
            defer d.deinit();
            return base.add(ctx, &d);
        }

        /// Register A and B on an optimizer (anything exposing
        /// `addParamNamed`, e.g. `optim.AdamW`) as "<prefix>.lora_a" /
        /// "<prefix>.lora_b". The adapter must outlive the optimizer (params
        /// and names are borrowed).
        pub fn registerParams(self: *Self, opt: anytype, comptime name_prefix: []const u8) !void {
            try opt.addParamNamed(&self.a, name_prefix ++ ".lora_a");
            try opt.addParamNamed(&self.b, name_prefix ++ ".lora_b");
        }

        /// State-dict entries "<prefix>.lora_a" / "<prefix>.lora_b" for
        /// `optim.saveStateDict`. Borrowed views: the adapter must outlive
        /// the entries.
        pub fn namedTensors(self: *const Self, comptime name_prefix: []const u8) ![2]optim.NamedTensor {
            return .{
                try optim.NamedTensor.of(name_prefix ++ ".lora_a", &self.a),
                try optim.NamedTensor.of(name_prefix ++ ".lora_b", &self.b),
            };
        }

        /// Mutable counterpart of `namedTensors` for `optim.loadStateDict`.
        pub fn namedTensorsMut(self: *Self, comptime name_prefix: []const u8) ![2]optim.NamedTensorMut {
            return .{
                try optim.NamedTensorMut.of(name_prefix ++ ".lora_a", &self.a),
                try optim.NamedTensorMut.of(name_prefix ++ ".lora_b", &self.b),
            };
        }

        /// Merge the adapter into an f32 weight IN PLACE: w += scale * B·A,
        /// with w: [out_tag, in_tag] row-major.
        ///
        /// Mutability contract: this goes through the facade's `data()` gate,
        /// which only grants mutable access to NO-GRAD tensors (variables
        /// return error.MutableDataRequiresNoGrad). That is exactly the right
        /// fence here — a frozen base weight is a constant, and merging into a
        /// weight that participates in an autograd graph would silently
        /// invalidate recorded forwards — so in-place mutation fits the
        /// facade contract and no return-new variant is needed for f32.
        ///
        /// Quantized bases are NOT supported: merging needs f32 → K-quant
        /// block ENCODERS, and the runtime only implements decode/dequant
        /// kernels for the block formats (re-quantization would also silently
        /// degrade the base). Dequantize to f32 (`.to(ctx, .f32)`) and merge
        /// into that copy instead.
        pub fn mergeInto(self: *const Self, ctx: *ExecContext, w: anytype) !void {
            const info = @typeInfo(@TypeOf(w));
            comptime {
                if (info != .pointer or info.pointer.size != .one or info.pointer.is_const) {
                    @compileError("mergeInto expects a mutable pointer to an f32 facade weight tensor");
                }
            }
            const W = facadePointee(@TypeOf(w), "LoRA merge weight");
            comptime {
                if (W.dtype != .f32) @compileError("mergeInto requires an f32 weight; use mergeF16 for f16 bases");
                if (!tags_mod.tagsEqual(W.axis_tags, .{ out_tag, in_tag })) {
                    @compileError("mergeInto weight must have tags { out_tag, in_tag }");
                }
            }
            if (w.dim(out_tag) != self.b.dim(out_tag) or w.dim(in_tag) != self.a.dim(in_tag)) {
                return TensorError.ShapeMismatch;
            }
            const w_data = try w.data();
            try self.addScaledDeltaW(ctx, w_data);
        }

        /// f16-base merge helper: widens w to f32, merges (`mergeInto` math),
        /// and casts back, returning a NEW f16 tensor of the caller's type.
        /// Return-new rather than in-place because the f32 accumulate cannot
        /// happen inside f16 storage without a round-trip anyway.
        pub fn mergeF16(self: *const Self, ctx: *ExecContext, w: anytype) !facadePointee(@TypeOf(w), "LoRA merge weight") {
            const W = facadePointee(@TypeOf(w), "LoRA merge weight");
            comptime {
                if (W.dtype != .f16) @compileError("mergeF16 requires an f16 weight; use mergeInto for f32 bases");
                if (!tags_mod.tagsEqual(W.axis_tags, .{ out_tag, in_tag })) {
                    @compileError("mergeF16 weight must have tags { out_tag, in_tag }");
                }
            }
            if (w.dim(out_tag) != self.b.dim(out_tag) or w.dim(in_tag) != self.a.dim(in_tag)) {
                return TensorError.ShapeMismatch;
            }
            var wide = try ctx.castTyped(.f16, .f32, w.asRawTensor());
            defer wide.deinit();
            try self.addScaledDeltaW(ctx, wide.data());
            var back = try ctx.castTyped(.f32, .f16, &wide);
            errdefer back.deinit();
            return W.fromTensor(ctx, back);
        }

        /// w_data += scale * (B·A), w_data row-major [out, in]. A/B are
        /// always contiguous (fresh buffers, updated in place), so the raw
        /// matmul applies directly.
        fn addScaledDeltaW(self: *const Self, ctx: *ExecContext, w_data: []f32) !void {
            var ba = try ctx.matmul2D(self.b.asRawTensor(), self.a.asRawTensor());
            defer ba.deinit();
            const ba_data = ba.dataConst();
            std.debug.assert(w_data.len == ba_data.len);
            for (w_data, ba_data) |*wi, di| wi.* += self.scale * di;
        }

        /// Comptime validation shared by `Delta`: x must be an f32 autograd
        /// facade tensor carrying `in_tag` and neither `out_tag` (which would
        /// turn the B contraction into a batch axis) nor the reserved rank tag.
        fn InputTensor(comptime XPtr: type) type {
            const X = facadePointee(XPtr, "LoRA input");
            comptime {
                if (X.dtype != .f32) @compileError("LoRA input must be an f32 facade tensor");
                if (tags_mod.tagIndex(X.axis_tags, in_tag) == null) {
                    @compileError("LoRA input is missing the adapter's in tag");
                }
                if (tags_mod.tagIndex(X.axis_tags, out_tag) != null) {
                    @compileError("LoRA input must not carry the adapter's out tag");
                }
                if (tags_mod.tagIndex(X.axis_tags, rank_tag) != null) {
                    @compileError("LoRA input must not carry the reserved .lora_r tag");
                }
            }
            return X;
        }
    };
}

/// Strip a single-item pointer down to the facade tensor type it addresses,
/// with a readable error (mirrors checkpoint.zig's FacadeOf).
fn facadePointee(comptime Ptr: type, comptime what: []const u8) type {
    const info = @typeInfo(Ptr);
    if (info != .pointer or info.pointer.size != .one) {
        @compileError(what ++ " must be a single-item pointer to a facade tensor, got " ++ @typeName(Ptr));
    }
    const T = info.pointer.child;
    if (@typeInfo(T) != .@"struct" or !@hasDecl(T, "dtype") or !@hasDecl(T, "axis_tags")) {
        @compileError(what ++ " must be a facade tensor, got " ++ @typeName(T));
    }
    return T;
}

test {
    _ = @import("lora_tests.zig");
}
