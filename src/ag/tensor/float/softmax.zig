//! f32 tensor methods: softmax family. A mixin over the ag
//! FloatTensor struct; aliased back onto it in ../../tensor.zig.

const std = @import("std");
const tensor_mod = @import("../../../tensor.zig");
const exec_mod = @import("../../../exec.zig");
const dtype_mod = @import("../../../dtype.zig");
const tag_ops = @import("../../../tag_ops.zig");
const tags_mod = @import("../../../tags.zig");
const backward_softmax = @import("../../backward/softmax.zig");

const RawTensor = tensor_mod.Tensor;
const ExecContext = exec_mod.ExecContext;
const SoftmaxExtOptions = exec_mod.SoftmaxExtOptions;
const Tag = tags_mod.Tag;
const removeTag = tags_mod.removeTag;
const broadcastTensorTo = tag_ops.broadcastTensorTo;
const LogsumexpBackward = backward_softmax.LogsumexpBackward;
const LogSoftmaxBackward = backward_softmax.LogSoftmaxBackward;
const SoftmaxBackward = backward_softmax.SoftmaxBackward;
const SoftmaxExtBackward = backward_softmax.SoftmaxExtBackward;

pub fn Ops(comptime Self: type) type {
    return struct {
        const tags = Self.axis_tags;
        const tag_rank = Self.tag_count;
        const tag_count = Self.tag_count;
        const axis = Self.axis;
        const ag_tensor = Self.ag_root;
        const Tensor = ag_tensor.Tensor;
        const plumbing = @import("../plumbing.zig").Mod(ag_tensor);
        const recordsGrad = plumbing.recordsGrad;
        const finishTypedNoGrad = plumbing.finishTypedNoGrad;
        const finishNoGrad = plumbing.finishNoGrad;
        const rawShapeArray = plumbing.rawShapeArray;
        const rawShapeArrayOf = plumbing.rawShapeArrayOf;
        const cloneInverseRopeTable = plumbing.cloneInverseRopeTable;
        const finishOp = plumbing.finishOp;
        const dtype = Self.dtype;
        /// The f32 branch is the differentiable one; every other dtype takes
        /// the constant tail.
        const differentiable = dtype == .f32;
        const reduced_dtype = dtype_mod.outputDType(.reduction, dtype);
        const TensorObject = plumbing.TensorObject;
        const tensorObjectPtrFrom = plumbing.tensorObjectPtrFrom;

        /// Softmax over `tag`, with optional fused extensions selected at
        /// comptime from the `options` struct literal (unknown fields are
        /// compile errors): `.scale` (logit multiplier), `.max_bias` (ALiBi
        /// slope base, needs `.head_tag`), `.sinks` (per-head attention
        /// sinks, needs `.head_tag`), `.causal = .{ .query_tag,
        /// .source_offset }`, `.mask` (additive tag-broadcast tensor; must
        /// not require grad). An empty `.{}` routes to the lean plain kernel
        /// (the backward is already unified at the exec layer).
        /// Log-sum-exp over `tag` (torch.logsumexp): `log(Σ exp(x))` with
        /// `tag` removed — a FUSED single-pass kernel (SIMD max scan +
        /// vexpf sum per row, task-parallel over rows like `softmax`; no
        /// materialized intermediates, no exec-scope requirement). Rows
        /// whose max is ±inf are shifted by 0 instead, so an all(-inf) row
        /// yields -inf and a row containing +inf yields +inf (the torch
        /// convention) rather than NaN. Differentiable: the backward is
        /// the row softmax of the saved INPUT times the upstream gradient
        /// (`exp(x − lse)·g`; only the input is retained).
        pub fn logsumexp(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Tensor(.{ .dtype = reduced_dtype, .tags = removeTag(tags, tag) }) {
            const result_tags = removeTag(tags, tag);
            const reduce_axis = comptime axis(tag);
            var value = try ctx.logsumexp(dtype, tag_rank, self.asRawTensor(), reduce_axis);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Tensor(.{ .dtype = reduced_dtype, .tags = result_tags }), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(result_tags, ctx, value);
            const Record = LogsumexpBackward(tags, reduce_axis);
            var saved_input = try (&self.value).cloneView();
            errdefer saved_input.deinit();
            return finishOp(result_tags, ctx, value, Record{
                .parents = .{self.grad_state},
                .input = saved_input,
            });
        }

        /// Log-softmax over `tag` (torch.log_softmax): `x − logsumexp(x)`
        /// broadcast, shape-preserving — the same FUSED kernel family as
        /// `logsumexp` (two SIMD passes per row, task-parallel; no
        /// exec-scope requirement) with the same non-finite max handling.
        /// Prefer `crossEntropy` when the next step is an NLL loss (fused
        /// with the loss, saved-stats backward). Differentiable: the
        /// backward is the saved-output identity `g − exp(y)·Σg`.
        pub fn logSoftmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag) !Self {
            const scan_axis = comptime axis(tag);
            var value = try ctx.logSoftmax(dtype, tag_rank, self.asRawTensor(), scan_axis);
            errdefer value.deinit();
            if (comptime !differentiable) return finishTypedNoGrad(Tensor(.{ .dtype = dtype, .tags = tags }), ctx, value, self.requiresGrad());
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = LogSoftmaxBackward(tags, scan_axis);
            var saved_output = try (&value).cloneView();
            errdefer saved_output.deinit();
            return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state}, .output = saved_output });
        }

        pub fn softmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self {
            const Options = @TypeOf(options);
            comptime {
                if (@typeInfo(Options) != .@"struct") @compileError("softmax: options must be a struct literal, e.g. .{} or .{ .scale = s }");
                const allowed = [_][]const u8{ "scale", "max_bias", "sinks", "head_tag", "causal", "mask" };
                for (@typeInfo(Options).@"struct".fields) |field| {
                    var known = false;
                    for (allowed) |name| {
                        if (std.mem.eql(u8, field.name, name)) known = true;
                    }
                    if (!known) @compileError("softmax: unknown option ." ++ field.name);
                }
            }
            const softmax_axis = comptime axis(tag);
            if (comptime @typeInfo(Options).@"struct".fields.len == 0) {
                var value = try ctx.softmax(dtype, tag_rank, self.asRawTensor(), softmax_axis);
                errdefer value.deinit();
                if (comptime !differentiable) return finishTypedNoGrad(Tensor(.{ .dtype = dtype, .tags = tags }), ctx, value, self.requiresGrad());
                if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
                const Record = SoftmaxBackward(tags, softmax_axis);
                var saved_output = try (&value).cloneView();
                errdefer saved_output.deinit();
                return finishOp(tags, ctx, value, Record{ .parents = .{self.grad_state}, .output = saved_output });
            }
            if (comptime !differentiable) @compileError("softmax options (scale, mask, sinks, causal, ...) are the f32 ext kernel's; cast to f32 first");
            const scale_value: f32 = if (comptime @hasField(Options, "scale")) options.scale else 1;
            const max_bias: f32 = if (comptime @hasField(Options, "max_bias")) options.max_bias else 0;
            const sinks: ?[]const f32 = if (comptime @hasField(Options, "sinks")) options.sinks else null;
            const head_axis: ?usize = comptime if (@hasField(Options, "head_tag")) axis(options.head_tag) else null;
            const causal_query_axis: ?usize = comptime if (@hasField(Options, "causal")) blk: {
                const Causal = @TypeOf(options.causal);
                for (@typeInfo(Causal).@"struct".fields) |field| {
                    if (!std.mem.eql(u8, field.name, "query_tag") and !std.mem.eql(u8, field.name, "source_offset"))
                        @compileError("softmax: unknown .causal option ." ++ field.name);
                }
                if (!@hasField(Causal, "query_tag")) @compileError("softmax: the .causal option requires .query_tag");
                break :blk axis(options.causal.query_tag);
            } else null;
            const causal_source_offset: usize = if (comptime @hasField(Options, "causal")) blk: {
                const Causal = @TypeOf(options.causal);
                break :blk if (comptime @hasField(Causal, "source_offset")) options.causal.source_offset else 0;
            } else 0;

            var mask_view: ?RawTensor = null;
            defer if (mask_view) |*mask| mask.deinit();
            if (comptime @hasField(Options, "mask")) {
                const mask_ptr = tensorObjectPtrFrom(@TypeOf(options.mask), &options.mask);
                if (mask_ptr.requiresGrad()) return error.UnsupportedGradient;
                const Mask = TensorObject(@TypeOf(options.mask));
                mask_view = try broadcastTensorTo(.f32, Mask.axis_tags, mask_ptr.asRawTensor(), tags, self.shape());
            }

            var value = try ctx.softmaxExt(
                tag_rank,
                self.asRawTensor(),
                softmax_axis,
                SoftmaxExtOptions{
                    .mask = if (mask_view) |*mask| mask else null,
                    .sinks = sinks,
                    .scale = scale_value,
                    .max_bias = max_bias,
                    .head_axis = head_axis,
                    .causal_query_axis = causal_query_axis,
                    .causal_source_offset = causal_source_offset,
                },
            );
            errdefer value.deinit();
            if (!recordsGrad(self.requiresGrad())) return finishNoGrad(tags, ctx, value);
            const Record = SoftmaxExtBackward(tags, softmax_axis);
            var saved_output = try (&value).cloneView();
            errdefer saved_output.deinit();
            return finishOp(tags, ctx, value, Record{
                .parents = .{self.grad_state},
                .output = saved_output,
                .scale = scale_value,
            });
        }
    };
}
