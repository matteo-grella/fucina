//! Elemental ops: lift comptime scalar functions to differentiable
//! elementwise tensor ops over the tagged facade. A convenience tier above
//! `customVjp` (`custom.zig`): the user writes scalar forward/backward
//! rules only, and this adapter owns the buffer plumbing, strided-input
//! materialization, tag-driven broadcasting, and broadcast-gradient
//! reduction, delegating graph wiring (input/output views, `needs_grad`
//! pruning, exec-scope adoption) to `customVjp`.
//!
//! Unary `Op` contract (checked at comptime):
//!
//! ```zig
//! const Op = struct {
//!     pub fn forward(x: f32, extra: Extra) f32 { ... }
//!     /// Returns the propagated gradient dL/dx, not the local dy/dx.
//!     pub fn backward(x: f32, y: f32, grad_y: f32, extra: Extra) f32 { ... }
//! };
//! ```
//!
//! Binary `Op` contract: `forward(a, b, extra)` plus `backwardA`/`backwardB`
//! returning dL/da and dL/db evaluated at the broadcast result shape; the
//! adapter sum-reduces each back to its operand's tags/shape (the same rule
//! as the built-in pointwise backwards). `extra` is captured by value with
//! the `customVjp` lifetime contract: pointees must outlive backward.
//!
//! Optional SIMD bodies: an `Op` may additionally declare vector twins —
//! `forwardVec`/`backwardVec` (unary) or `forwardVec`/`backwardAVec` +
//! `backwardBVec` (binary) — over `fucina.simd.Vf32` lanes; the kernels
//! then run those on full vectors and the scalar rules on the tail (the
//! same lane/tail split as the built-in SIMD kernels). Bodies containing
//! transcendentals want this: the scalar rules compile to one libm call
//! per element, where a vector body can use `fucina.simd.vexpf` and
//! friends. The scalar and vector rules must compute the same function;
//! lane/tail rounding may differ exactly as it does in the built-ins.
//!
//! The loops chunk across the worker team above the elementwise length
//! threshold; writes are disjoint and every element is a pure function of
//! its inputs, so results are bitwise identical for any thread count
//! (serial included).
const std = @import("std");
const exec_mod = @import("../exec.zig");
const tensor_mod = @import("../tensor.zig");
const tags_mod = @import("../tags.zig");
const tag_ops = @import("../tagged.zig");
const parallel = @import("../parallel.zig");
const custom = @import("custom.zig");
const vector_common = @import("../backend/vector/common.zig");

const Vec = vector_common.Vf32;
const vec_len = vector_common.vector_len;
const backward_mod = @import("backward.zig");

const ExecContext = exec_mod.ExecContext;
const RawTensor = tensor_mod.Tensor;
const rawRank = tags_mod.rawRank;
const pointwiseResultTags = tags_mod.pointwiseResultTags;
const contiguousForRead = backward_mod.contiguousForRead;
const reduceGradientToTags = backward_mod.reduceGradientToTags;

/// `elementalUnary` entry: `TensorT` is the caller's f32 facade tensor type
/// (input and output). Delegates to `customVjp` with a generated Spec.
pub fn unary(
    comptime TensorT: type,
    ctx: *ExecContext,
    comptime Op: type,
    extra: anytype,
    input: *const TensorT,
) !TensorT {
    comptime {
        if (!@hasDecl(Op, "forward"))
            @compileError("elementalUnary: Op must declare `pub fn forward(x: f32, extra) f32`");
        if (!@hasDecl(Op, "backward"))
            @compileError("elementalUnary: Op must declare `pub fn backward(x: f32, y: f32, grad_y: f32, extra) f32` returning dL/dx");
    }
    return custom.customVjp(ctx, UnarySpec(TensorT, Op, @TypeOf(extra)), extra, .{input});
}

/// `elementalBinary` entry: `OutputT` is the facade tensor type of the
/// broadcast result (`pointwiseResultTags(left_tags, right_tags)`), computed
/// by the facade caller so this module never names the facade constructor.
pub fn binary(
    comptime OutputT: type,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    ctx: *ExecContext,
    comptime Op: type,
    extra: anytype,
    left: anytype,
    right: anytype,
) !OutputT {
    comptime {
        if (!@hasDecl(Op, "forward"))
            @compileError("elementalBinary: Op must declare `pub fn forward(a: f32, b: f32, extra) f32`");
        if (!@hasDecl(Op, "backwardA"))
            @compileError("elementalBinary: Op must declare `pub fn backwardA(a, b, y, grad_y, extra) f32` returning dL/da");
        if (!@hasDecl(Op, "backwardB"))
            @compileError("elementalBinary: Op must declare `pub fn backwardB(a, b, y, grad_y, extra) f32` returning dL/db");
    }
    return custom.customVjp(ctx, BinarySpec(OutputT, left_tags, right_tags, Op, @TypeOf(extra)), extra, .{ left, right });
}

fn UnarySpec(comptime TensorT: type, comptime Op: type, comptime Extra: type) type {
    return struct {
        pub const Output = TensorT;

        pub fn forward(ctx: *ExecContext, extra: Extra, inputs: []const *const RawTensor) !RawTensor {
            var x = try contiguousForRead(ctx, inputs[0]);
            defer x.deinit();
            var out = try ctx.empty(inputs[0].shape.slice());
            errdefer out.deinit();
            const Body = struct {
                out: []f32,
                x: []const f32,
                extra: Extra,
                fn at(self: *const @This(), i: usize) void {
                    self.out[i] = Op.forward(self.x[i], self.extra);
                }
                fn span(self: *const @This(), start: usize, end: usize) void {
                    var i = start;
                    if (comptime @hasDecl(Op, "forwardVec")) {
                        while (i + vec_len <= end) : (i += vec_len) {
                            self.out[i..][0..vec_len].* = Op.forwardVec(@as(Vec, self.x[i..][0..vec_len].*), self.extra);
                        }
                    }
                    while (i < end) : (i += 1) self.at(i);
                }
            };
            dispatchElemental(Body, .{ .out = out.data(), .x = x.dataConst(), .extra = extra }, ctx);
            return out;
        }

        pub fn backward(
            ctx: *ExecContext,
            extra: Extra,
            inputs: []const *const RawTensor,
            output: *const RawTensor,
            gy: *const RawTensor,
            needs_grad: []const bool,
            out: []?RawTensor,
        ) !void {
            if (!needs_grad[0]) return;
            var x = try contiguousForRead(ctx, inputs[0]);
            defer x.deinit();
            var y = try contiguousForRead(ctx, output);
            defer y.deinit();
            var g = try contiguousForRead(ctx, gy);
            defer g.deinit();
            var gx = try ctx.empty(inputs[0].shape.slice());
            errdefer gx.deinit();
            const Body = struct {
                gx: []f32,
                x: []const f32,
                y: []const f32,
                g: []const f32,
                extra: Extra,
                fn at(self: *const @This(), i: usize) void {
                    self.gx[i] = Op.backward(self.x[i], self.y[i], self.g[i], self.extra);
                }
                fn span(self: *const @This(), start: usize, end: usize) void {
                    var i = start;
                    if (comptime @hasDecl(Op, "backwardVec")) {
                        while (i + vec_len <= end) : (i += vec_len) {
                            self.gx[i..][0..vec_len].* = Op.backwardVec(
                                @as(Vec, self.x[i..][0..vec_len].*),
                                @as(Vec, self.y[i..][0..vec_len].*),
                                @as(Vec, self.g[i..][0..vec_len].*),
                                self.extra,
                            );
                        }
                    }
                    while (i < end) : (i += 1) self.at(i);
                }
            };
            dispatchElemental(Body, .{ .gx = gx.data(), .x = x.dataConst(), .y = y.dataConst(), .g = g.dataConst(), .extra = extra }, ctx);
            out[0] = gx;
        }
    };
}

fn BinarySpec(
    comptime OutputT: type,
    comptime left_tags: anytype,
    comptime right_tags: anytype,
    comptime Op: type,
    comptime Extra: type,
) type {
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    return struct {
        pub const Output = OutputT;

        pub fn forward(ctx: *ExecContext, extra: Extra, inputs: []const *const RawTensor) !RawTensor {
            const result_shape = try tag_ops.pointwiseShape(result_tags, left_tags, inputs[0], right_tags, inputs[1]);
            var a = try broadcastContiguous(left_tags, inputs[0], result_shape, ctx);
            defer a.deinit();
            var b = try broadcastContiguous(right_tags, inputs[1], result_shape, ctx);
            defer b.deinit();
            var out = try ctx.empty(result_shape[0..]);
            errdefer out.deinit();
            const Body = struct {
                out: []f32,
                a: []const f32,
                b: []const f32,
                extra: Extra,
                fn at(self: *const @This(), i: usize) void {
                    self.out[i] = Op.forward(self.a[i], self.b[i], self.extra);
                }
                fn span(self: *const @This(), start: usize, end: usize) void {
                    var i = start;
                    if (comptime @hasDecl(Op, "forwardVec")) {
                        while (i + vec_len <= end) : (i += vec_len) {
                            self.out[i..][0..vec_len].* = Op.forwardVec(
                                @as(Vec, self.a[i..][0..vec_len].*),
                                @as(Vec, self.b[i..][0..vec_len].*),
                                self.extra,
                            );
                        }
                    }
                    while (i < end) : (i += 1) self.at(i);
                }
            };
            dispatchElemental(Body, .{ .out = out.data(), .a = a.dataConst(), .b = b.dataConst(), .extra = extra }, ctx);
            return out;
        }

        pub fn backward(
            ctx: *ExecContext,
            extra: Extra,
            inputs: []const *const RawTensor,
            output: *const RawTensor,
            gy: *const RawTensor,
            needs_grad: []const bool,
            out: []?RawTensor,
        ) !void {
            const result_shape = try tag_ops.pointwiseShape(result_tags, left_tags, inputs[0], right_tags, inputs[1]);
            var a = try broadcastContiguous(left_tags, inputs[0], result_shape, ctx);
            defer a.deinit();
            var b = try broadcastContiguous(right_tags, inputs[1], result_shape, ctx);
            defer b.deinit();
            var y = try contiguousForRead(ctx, output);
            defer y.deinit();
            var g = try contiguousForRead(ctx, gy);
            defer g.deinit();

            const Body = struct {
                grad: []f32,
                a: []const f32,
                b: []const f32,
                y: []const f32,
                g: []const f32,
                extra: Extra,
                which: enum { a, b },
                fn at(self: *const @This(), i: usize) void {
                    self.grad[i] = switch (self.which) {
                        .a => Op.backwardA(self.a[i], self.b[i], self.y[i], self.g[i], self.extra),
                        .b => Op.backwardB(self.a[i], self.b[i], self.y[i], self.g[i], self.extra),
                    };
                }
                fn span(self: *const @This(), start: usize, end: usize) void {
                    var i = start;
                    if (comptime @hasDecl(Op, "backwardAVec") and @hasDecl(Op, "backwardBVec")) {
                        while (i + vec_len <= end) : (i += vec_len) {
                            const av: Vec = self.a[i..][0..vec_len].*;
                            const bv: Vec = self.b[i..][0..vec_len].*;
                            const yv: Vec = self.y[i..][0..vec_len].*;
                            const gv: Vec = self.g[i..][0..vec_len].*;
                            self.grad[i..][0..vec_len].* = switch (self.which) {
                                .a => Op.backwardAVec(av, bv, yv, gv, self.extra),
                                .b => Op.backwardBVec(av, bv, yv, gv, self.extra),
                            };
                        }
                    }
                    while (i < end) : (i += 1) self.at(i);
                }
            };

            if (needs_grad[0]) {
                var full = try ctx.empty(result_shape[0..]);
                defer full.deinit();
                dispatchElemental(Body, .{ .grad = full.data(), .a = a.dataConst(), .b = b.dataConst(), .y = y.dataConst(), .g = g.dataConst(), .extra = extra, .which = .a }, ctx);
                out[0] = try reduceGradientToTags(result_tags, left_tags, ctx, &full, shapeOf(rawRank(left_tags.len), inputs[0]));
            }
            if (needs_grad[1]) {
                var full = try ctx.empty(result_shape[0..]);
                defer full.deinit();
                dispatchElemental(Body, .{ .grad = full.data(), .a = a.dataConst(), .b = b.dataConst(), .y = y.dataConst(), .g = g.dataConst(), .extra = extra, .which = .b }, ctx);
                out[1] = try reduceGradientToTags(result_tags, right_tags, ctx, &full, shapeOf(rawRank(right_tags.len), inputs[1]));
            }
        }

        fn broadcastContiguous(
            comptime source_tags: anytype,
            source: *const RawTensor,
            result_shape: [rawRank(result_tags.len)]usize,
            ctx: *ExecContext,
        ) !RawTensor {
            if (comptime result_tags.len == 0) return contiguousForRead(ctx, source);
            var view = try tag_ops.broadcastTensorTo(source_tags, source, result_tags, result_shape);
            defer view.deinit();
            return contiguousForRead(ctx, &view);
        }
    };
}

fn shapeOf(comptime rank: usize, t: *const RawTensor) [rank]usize {
    var out: [rank]usize = undefined;
    inline for (0..rank) |i| out[i] = t.shape.at(i);
    return out;
}

/// Run `body.at(i)` for every i in [0, len): serially below the elementwise
/// threshold or without a worker pool, else split into per-worker ranges
/// over the hot team. Disjoint pure writes — bitwise thread-count neutral.
fn dispatchElemental(comptime Body: type, body: Body, ctx: *ExecContext) void {
    const Task = struct {
        body: Body,
        start: usize,
        end: usize,
        fn run(task: *const @This()) void {
            task.body.span(task.start, task.end);
        }
    };
    const len = bodyLen(Body, body);
    const base: Task = .{ .body = body, .start = 0, .end = len };
    if (len >= parallel.vector_elementwise_len_threshold) {
        if (ctx.workPool()) |pool| {
            const task_count = @min(parallel.cpuThreadCount(parallel.vector_max_threads), len);
            var tasks: [parallel.vector_max_threads]Task = undefined;
            for (0..task_count) |task_i| {
                tasks[task_i] = base;
                tasks[task_i].start = task_i * len / task_count;
                tasks[task_i].end = (task_i + 1) * len / task_count;
            }
            pool.parallelChunks(Task, tasks[0..task_count], Task.run);
            return;
        }
    }
    Task.run(&base);
}

fn bodyLen(comptime Body: type, body: Body) usize {
    // The first field of every Body is its output slice; its length is the
    // logical element count.
    const fields = @typeInfo(Body).@"struct".fields;
    return @field(body, fields[0].name).len;
}

test {
    _ = @import("elemental_tests.zig");
}
