//! f32 tensor methods: fused grouped causal attention. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const dtype_mod = @import("../../../dtype.zig");
const exec_mod = @import("../../../exec.zig");
const tags_mod = @import("../../../tags.zig");
const backward_attention = @import("../../backward/attention.zig");

const BlockQ8_0 = dtype_mod.BlockQ8_0;
const TensorError = tensor_mod.TensorError;
const ExecContext = exec_mod.ExecContext;
const Tag = tags_mod.Tag;
const tagsEqual = tags_mod.tagsEqual;
const GroupedCausalAttentionBackward = backward_attention.GroupedCausalAttentionBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const AttentionMask = ag_tensor.AttentionMask;
        const attentionKvRepr = ag_tensor.attentionKvRepr;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const rowStatsAlloc = plumbing.rowStatsAlloc;
        const finishOp = plumbing.finishOp;
        const finishNoGrad = plumbing.finishNoGrad;
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// Grouped (GQA) attention over every KV representation, `q` = `self`
        /// with tags `.{ .seq, .head, .d }`. The KV representation is
        /// comptime-dispatched from `@TypeOf(k)` into the exec entry's
        /// `KvView`:
        ///
        ///   - `*Tensor(.{ .seq, .kv_head, .d })` (f32)        → f32 kernel
        ///   - f16 Tensor with the same tags                   → f16 kernel (decode KV cache)
        ///   - `[]const BlockQ8_0`                             → q8_0 raw-block cache,
        ///     layout `[kv_seq, kv_heads, d/32]` (see `ExecContext.groupedAttention`'s `.q8` view)
        ///   - `[]const []const f16`                           → ragged multi-stream decode
        ///   - `[]const []const BlockQ8_0`                     → ragged multi-stream decode (q8_0)
        ///
        /// `opts` is a comptime-validated struct literal; unknown fields are
        /// compile errors:
        ///
        ///   - `.mask = .causal` (default) | `.bidirectional` — f32/f16 KV only
        ///     (bidirectional = every query attends every key: the
        ///     block-diffusion canvas / OmniVoice encoder attention; its SWA
        ///     reach is realized by narrowing the K/V views — no
        ///     bidirectional window exists by design).
        ///   - `.window = w` — runtime sliding window, 0 = full causal
        ///     (query p attends [max(0, p-w+1), p]); causal only.
        ///   - `.bias = &b` — rank-2 `[q_seq, kv_seq]` additive f32 bias added
        ///     to the scaled scores pre-softmax (ggml_soft_max_ext semantics,
        ///     NOT -inf masking); bidirectional + f32 KV only.
        ///   - `.kv_seq = n, .kv_heads = h` — REQUIRED for the q8_0-block
        ///     repr (raw blocks carry no shape).
        ///   - `.lens = lens, .kv_heads = h` — REQUIRED for the multi-stream
        ///     reprs. There, q's `.seq` tag is reinterpreted as the STREAM
        ///     axis: exactly one query row per stream, row `s` attending all
        ///     `lens[s]` cached positions of `k[s]`/`v[s]`; per-stream
        ///     results are bit-identical to N single-stream f16/q8 calls.
        ///
        /// Gradient matrix (not readable off the call site):
        ///
        ///   - f32 KV: full q/k/v backward (windowed re-masks to the window).
        ///   - f16 KV: q-grad only — K/V are cache constants, widened to f32
        ///     once and run through the f32 kernel + backward.
        ///   - q8_0 KV: q-grad only, causal only — the cache is dequantized
        ///     to f32 once (no bidirectional q8 path exists).
        ///   - `.bias` present: inference-only; ANY grad-requiring operand
        ///     (q, k, v, or the bias) returns `error.UnsupportedGradient` —
        ///     the shared backward re-derives the softmax without a bias.
        ///   - multi-stream: inference-only (`error.UnsupportedGradient`).
        pub fn groupedAttention(
            self: *const Self,
            ctx: *ExecContext,
            k: anytype,
            v: anytype,
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            opts: anytype,
        ) !Tensor(.{ .seq, out_tag }) {
            comptime if (!tagsEqual(tags, .{ .seq, .head, .d })) {
                @compileError("groupedAttention requires q tags .{ .seq, .head, .d }");
            };
            const repr = comptime attentionKvRepr(@TypeOf(k), "k");
            comptime {
                const v_repr = attentionKvRepr(@TypeOf(v), "v");
                if (repr != v_repr) @compileError("groupedAttention: k is a ." ++ @tagName(repr) ++ " KV but v is a ." ++ @tagName(v_repr) ++ " KV");
            }
            const O = @TypeOf(opts);
            comptime if (@typeInfo(O) != .@"struct") {
                @compileError("groupedAttention: opts must be a struct literal, e.g. .{} or .{ .window = w }");
            };
            const mask: AttentionMask = comptime if (@hasField(O, "mask")) opts.mask else .causal;
            comptime {
                // Every option must be a field of the exec entry's
                // AttentionOptions or one of the repr-specific shape fields:
                // a misspelled option (`.windw`) must be a compile error,
                // never silently-full-causal attention.
                for (@typeInfo(O).@"struct".fields) |field| {
                    const shape_field = std.mem.eql(u8, field.name, "kv_seq") or
                        std.mem.eql(u8, field.name, "kv_heads") or
                        std.mem.eql(u8, field.name, "lens");
                    if (!shape_field and !@hasField(exec_mod.AttentionOptions, field.name)) {
                        @compileError("groupedAttention: unknown option ." ++ field.name ++ " for the ." ++ @tagName(repr) ++ " KV representation");
                    }
                }
                // Repr-dependent constraints, each expressed once; the exec
                // entry answers the same combinations with
                // error.UnsupportedAttentionVariant at runtime.
                if (@hasField(O, "stats_out")) {
                    @compileError("groupedAttention: softmax stats capture is wired by the autograd record, not an option here");
                }
                if (mask == .bidirectional and @hasField(O, "window")) {
                    @compileError("groupedAttention: no bidirectional windowed kernel exists — realize SWA reach by narrowing the K/V views");
                }
                if (@hasField(O, "bias") and mask != .bidirectional) {
                    @compileError("groupedAttention: .bias requires .mask = .bidirectional (the only additive-bias kernel)");
                }
                if (@hasField(O, "bias") and repr != .f32_kv) {
                    @compileError("groupedAttention: .bias requires the .f32_kv KV representation (the only additive-bias kernel)");
                }
                if (@hasField(O, "mask") and (repr == .q8_kv or repr == .multi_f16_kv or repr == .multi_q8_kv)) {
                    @compileError("groupedAttention: the ." ++ @tagName(repr) ++ " KV representation is causal-only");
                }
                if (@hasField(O, "window") and (repr == .multi_f16_kv or repr == .multi_q8_kv)) {
                    @compileError("groupedAttention: the multi-stream KV reprs have no window (each stream attends its full cached length)");
                }
                switch (repr) {
                    .f32_kv, .f16_kv => if (@hasField(O, "kv_seq") or @hasField(O, "kv_heads") or @hasField(O, "lens")) {
                        @compileError("groupedAttention: ." ++ @tagName(repr) ++ " tensors carry their own shape; .kv_seq/.kv_heads/.lens apply to the raw-block and multi-stream KV reprs only");
                    },
                    .q8_kv => {
                        if (!@hasField(O, "kv_seq") or !@hasField(O, "kv_heads")) {
                            @compileError("groupedAttention: the q8_0-block KV repr requires .kv_seq and .kv_heads (raw blocks carry no shape; layout [kv_seq, kv_heads, d/32])");
                        }
                        if (@hasField(O, "lens")) {
                            @compileError("groupedAttention: .lens applies to the multi-stream KV reprs only");
                        }
                    },
                    .multi_f16_kv, .multi_q8_kv => {
                        if (!@hasField(O, "lens") or !@hasField(O, "kv_heads")) {
                            @compileError("groupedAttention: the multi-stream KV reprs require .lens and .kv_heads (q's .seq tag is the stream axis)");
                        }
                        if (@hasField(O, "kv_seq")) {
                            @compileError("groupedAttention: .kv_seq applies to the q8_0-block KV repr only");
                        }
                    },
                }
            }
            const window: usize = if (comptime @hasField(O, "window")) opts.window else 0;
            switch (comptime repr) {
                .f32_kv => {
                    if (comptime @hasField(O, "bias")) {
                        const bias = opts.bias;
                        const bias_ptr = tensorObjectPtrFrom(@TypeOf(bias), &bias);
                        comptime if (TensorObject(@TypeOf(bias)).axis_tags.len != 2) {
                            @compileError("groupedAttention: .bias must be a rank-2 [q_seq, kv_seq] tensor");
                        };
                        if (self.requiresGrad() or k.requiresGrad() or v.requiresGrad() or bias_ptr.requiresGrad()) {
                            return error.UnsupportedGradient;
                        }
                        var value = try ctx.groupedAttention(self.asRawTensor(), .{ .f32 = .{ .k = k.asRawTensor(), .v = v.asRawTensor() } }, kv_head_for_head, scale_value, .{ .mask = .bidirectional, .bias = bias_ptr.asRawTensor() });
                        errdefer value.deinit();
                        return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                    }
                    const wants_grad = self.requiresGrad() or k.requiresGrad() or v.requiresGrad();
                    // Forward-saved softmax {max, sum_exp} for the VJP's
                    // one-pass probability rebuild; the capture is write-only,
                    // so the output is bitwise identical either way.
                    const row_stats = try rowStatsAlloc(ctx, wants_grad, kv_head_for_head.len * self.asRawTensor().shape.at(0));
                    defer if (row_stats) |stats| ctx.allocator.free(stats);
                    var value = try ctx.groupedAttention(self.asRawTensor(), .{ .f32 = .{ .k = k.asRawTensor(), .v = v.asRawTensor() } }, kv_head_for_head, scale_value, .{ .mask = mask, .window = window, .stats_out = row_stats });
                    errdefer value.deinit();
                    return finishOp(.{ .seq, out_tag }, ctx, value, wants_grad, GroupedCausalAttentionBackward, .{
                        ctx.allocator,
                        self.grad_state,
                        k.grad_state,
                        v.grad_state,
                        self.asRawTensor(),
                        k.asRawTensor(),
                        v.asRawTensor(),
                        kv_head_for_head,
                        scale_value,
                        window,
                        mask == .causal,
                        row_stats orelse &[_]f32{},
                        &value,
                    });
                },
                .f16_kv => {
                    if (self.requiresGrad()) {
                        return f16KvAttentionWithGrad(self, ctx, k, v, kv_head_for_head, out_tag, scale_value, window, mask == .causal);
                    }
                    var value = try ctx.groupedAttention(self.asRawTensor(), .{ .f16 = .{ .k = k.asRawTensor(), .v = v.asRawTensor() } }, kv_head_for_head, scale_value, .{ .mask = mask, .window = window });
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .q8_kv => {
                    if (self.requiresGrad()) {
                        return q8KvAttentionWithGrad(self, ctx, k, v, opts.kv_seq, opts.kv_heads, kv_head_for_head, out_tag, scale_value, window);
                    }
                    var value = try ctx.groupedAttention(self.asRawTensor(), .{ .q8 = .{ .k = k, .v = v, .kv_seq = opts.kv_seq, .kv_heads = opts.kv_heads } }, kv_head_for_head, scale_value, .{ .window = window });
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .multi_f16_kv => {
                    if (self.requiresGrad()) return error.UnsupportedGradient;
                    var value = try ctx.groupedAttention(self.asRawTensor(), .{ .multi_f16 = .{ .k = k, .v = v, .lens = opts.lens, .kv_heads = opts.kv_heads } }, kv_head_for_head, scale_value, .{});
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
                .multi_q8_kv => {
                    if (self.requiresGrad()) return error.UnsupportedGradient;
                    var value = try ctx.groupedAttention(self.asRawTensor(), .{ .multi_q8 = .{ .k = k, .v = v, .lens = opts.lens, .kv_heads = opts.kv_heads } }, kv_head_for_head, scale_value, .{});
                    errdefer value.deinit();
                    return finishNoGrad(.{ .seq, out_tag }, ctx, value);
                },
            }
        }

        /// Gradient path for q8_0-KV attention: dequantize the constant
        /// cache to f32 once, then the f32 kernel + backward (q-grad only),
        /// mirroring `f16KvAttentionWithGrad`.
        fn q8KvAttentionWithGrad(
            self: *const Self,
            ctx: *ExecContext,
            k_blocks: []const BlockQ8_0,
            v_blocks: []const BlockQ8_0,
            kv_seq: usize,
            kv_heads: usize,
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            window: usize,
        ) !Tensor(.{ .seq, out_tag }) {
            const block_size = dtype_mod.q8_0_block_size;
            if (kv_seq * kv_heads == 0 or k_blocks.len % (kv_seq * kv_heads) != 0) return TensorError.InvalidShape;
            if (v_blocks.len != k_blocks.len) return TensorError.InvalidShape;
            const d = (k_blocks.len / (kv_seq * kv_heads)) * block_size;
            var k32 = try ctx.empty(.f32, .{ kv_seq, kv_heads, d });
            defer k32.deinit();
            try ctx.dequantizeQ8_0RowsInto(k32.data(), k_blocks);
            var v32 = try ctx.empty(.f32, .{ kv_seq, kv_heads, d });
            defer v32.deinit();
            try ctx.dequantizeQ8_0RowsInto(v32.data(), v_blocks);
            const row_stats = try rowStatsAlloc(ctx, true, kv_head_for_head.len * self.asRawTensor().shape.at(0));
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            var value = try ctx.groupedAttention(self.asRawTensor(), .{ .f32 = .{ .k = &k32, .v = &v32 } }, kv_head_for_head, scale_value, .{ .window = window, .stats_out = row_stats });
            errdefer value.deinit();
            return finishOp(.{ .seq, out_tag }, ctx, value, true, GroupedCausalAttentionBackward, .{
                ctx.allocator,
                self.grad_state,
                null,
                null,
                self.asRawTensor(),
                &k32,
                &v32,
                kv_head_for_head,
                scale_value,
                window,
                true,
                row_stats orelse &[_]f32{},
                &value,
            });
        }
        // (q8KvAttentionWithGrad above stays causal-only: no bidirectional
        // q8_0-cache user exists — the diffusion canvas runs on f16 KV.)

        /// Gradient path for f16-KV attention: K/V are constants (the cache),
        /// so only q-grad flows. Widens K/V to f32 once and runs the f32
        /// kernel + backward; the f16 fast path stays grad-free.
        fn f16KvAttentionWithGrad(
            self: *const Self,
            ctx: *ExecContext,
            k: *const Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }),
            v: *const Tensor(.{ .dtype = .f16, .tags = .{ .seq, .kv_head, .d } }),
            kv_head_for_head: []const usize,
            comptime out_tag: Tag,
            scale_value: f32,
            window: usize,
            causal: bool,
        ) !Tensor(.{ .seq, out_tag }) {
            var k32 = try ctx.cast(.f16, .f32, k.asRawTensor());
            defer k32.deinit();
            var v32 = try ctx.cast(.f16, .f32, v.asRawTensor());
            defer v32.deinit();
            const row_stats = try rowStatsAlloc(ctx, true, kv_head_for_head.len * self.asRawTensor().shape.at(0));
            defer if (row_stats) |stats| ctx.allocator.free(stats);
            const mask: AttentionMask = if (causal) .causal else .bidirectional;
            var value = try ctx.groupedAttention(self.asRawTensor(), .{ .f32 = .{ .k = &k32, .v = &v32 } }, kv_head_for_head, scale_value, .{ .mask = mask, .window = window, .stats_out = row_stats });
            errdefer value.deinit();
            return finishOp(.{ .seq, out_tag }, ctx, value, true, GroupedCausalAttentionBackward, .{
                ctx.allocator,
                self.grad_state,
                null,
                null,
                self.asRawTensor(),
                &k32,
                &v32,
                kv_head_for_head,
                scale_value,
                window,
                causal,
                row_stats orelse &[_]f32{},
                &value,
            });
        }
    };
}
