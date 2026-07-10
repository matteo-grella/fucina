//! einsum vs hand-written contraction pipelines at representative shapes.
//!
//! Since `dot` delegates to the einsum lowering, most dot/einsum pairs are
//! same-kernel parity baselines (any spread is noise): attention scores,
//! GQA multi-free scores, pre-merged multi-axis contraction, the LoRA
//! three-operand chain, and the multi-free forward+backward. The remaining
//! advantage case is equation-level: nt_double_trans, where einsum's free
//! choice of output orientation turns a materialize-then-GEMM into one
//! zero-copy plain GEMM.
const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;

const Allocator = std.mem.Allocator;

var benchmark_io: std.Io = undefined;
var benchmark_allocator_mode: bench_alloc.AllocatorMode = .smp;

const Result = struct {
    iterations: usize,
    ns_per_op: u64,
    checksum: f64,
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    benchmark_io = init.io;
    // Default to the SMP allocator (this is a perf bench, not an
    // overhead-accounting bench); `debug` opts back in via argv.
    for (args) |arg| {
        if (try bench_alloc.parseAllocatorModeArg(arg)) |parsed| benchmark_allocator_mode = parsed;
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try stdout.writeAll("case,mode,iters,ns_per_op,checksum\n");

    try printResult(stdout, "attn_scores", "dot", try runAttnScoresDot(20));
    try printResult(stdout, "attn_scores", "einsum", try runAttnScoresEinsum(20));
    try printResult(stdout, "gqa_scores", "dot", try runGqaScoresDot(20));
    try printResult(stdout, "gqa_scores", "einsum", try runGqaScoresEinsum(20));
    try printResult(stdout, "nt_double_trans", "dot", try runDoubleTransDot(20));
    try printResult(stdout, "nt_double_trans", "einsum", try runDoubleTransEinsum(20));
    try printResult(stdout, "multi_contract", "dot_premerged", try runMultiContractPremerged(20));
    try printResult(stdout, "multi_contract", "einsum", try runMultiContractEinsum(20));
    try printResult(stdout, "lora_chain", "dot_chain", try runLoraChainDot(50));
    try printResult(stdout, "lora_chain", "einsum_many", try runLoraChainEinsumMany(50));
    try printResult(stdout, "fwd_bwd_multi_free", "dot", try runMultiFreeBackwardDot(10));
    try printResult(stdout, "fwd_bwd_multi_free", "einsum", try runMultiFreeBackwardEinsum(10));
}

fn printResult(writer: anytype, case: []const u8, mode: []const u8, result: Result) !void {
    try writer.print("{s},{s},{d},{d},{d:.6}\n", .{ case, mode, result.iterations, result.ns_per_op, result.checksum });
}

const Bench = struct {
    benchmark_allocator: bench_alloc.BenchmarkAllocator,
    ctx: fucina.ExecContext,

    fn init() Bench {
        return .{ .benchmark_allocator = bench_alloc.BenchmarkAllocator.init(benchmark_allocator_mode), .ctx = undefined };
    }

    fn start(self: *Bench) *fucina.ExecContext {
        self.ctx.init(self.benchmark_allocator.allocator());
        return &self.ctx;
    }

    fn deinit(self: *Bench) void {
        self.ctx.deinit();
        self.benchmark_allocator.deinit();
    }
};

fn filledTensor(ctx: *fucina.ExecContext, comptime spec: anytype, comptime rank: usize, shape: [rank]usize, seed: usize) !fucina.Tensor(spec) {
    var value = try ctx.emptyRank(rank, shape);
    fillPattern(&value, seed);
    return fucina.Tensor(spec).fromTensor(ctx, value);
}

fn filledVariable(ctx: *fucina.ExecContext, comptime spec: anytype, comptime rank: usize, shape: [rank]usize, seed: usize) !fucina.Tensor(spec) {
    var elems: usize = 1;
    for (shape) |d| elems *= d;
    const data = try ctx.allocator.alloc(f32, elems);
    defer ctx.allocator.free(data);
    for (data, 0..) |*v, i| {
        const mixed = (i * 17 + seed * 31) % 97;
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast(mixed)) - 48)) * 0.0025;
    }
    return fucina.Tensor(spec).variableFromSlice(ctx, shape, data);
}

fn fillPattern(t: anytype, seed: usize) void {
    for (t.data(), 0..) |*value, i| {
        const mixed = (i * 17 + seed * 31) % 97;
        const centered: i32 = @as(i32, @intCast(mixed)) - 48;
        value.* = @as(f32, @floatFromInt(centered)) * 0.0025;
    }
}

fn measure(iterations: usize, comptime stepFn: anytype, args: anytype) !Result {
    var checksum: f64 = 0;
    for (0..2) |_| checksum += try @call(.auto, stepFn, args);
    checksum = 0;
    var timer = try Timer.start(benchmark_io);
    for (0..iterations) |_| {
        checksum += try @call(.auto, stepFn, args);
    }
    const elapsed = timer.read();
    return .{ .iterations = iterations, .ns_per_op = elapsed / iterations, .checksum = checksum };
}

// -- attention scores: q[b,h,i,d] x k[b,h,j,d] -> [b,h,i,j] (parity: both direct bmmTransB)

const attn_b = 2;
const attn_h = 8;
const attn_i = 256;
const attn_j = 256;
const attn_d = 64;

fn attnScoresDotStep(ctx: *fucina.ExecContext, q: *const fucina.Tensor(.{ .b, .h, .i, .d }), k: *const fucina.Tensor(.{ .b, .h, .j, .d })) !f64 {
    var scores = try q.dot(ctx, k, .d);
    defer scores.deinit();
    return scores.asRawTensor().dataConst()[0];
}

fn attnScoresEinsumStep(ctx: *fucina.ExecContext, q: *const fucina.Tensor(.{ .b, .h, .i, .d }), k: *const fucina.Tensor(.{ .b, .h, .j, .d })) !f64 {
    var scores = try q.einsum(ctx, k, .{ .b, .h, .i, .j });
    defer scores.deinit();
    return scores.asRawTensor().dataConst()[0];
}

fn runAttnScoresDot(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var q = try filledTensor(ctx, .{ .b, .h, .i, .d }, 4, .{ attn_b, attn_h, attn_i, attn_d }, 1);
    defer q.deinit();
    var k = try filledTensor(ctx, .{ .b, .h, .j, .d }, 4, .{ attn_b, attn_h, attn_j, attn_d }, 2);
    defer k.deinit();
    return measure(iterations, attnScoresDotStep, .{ ctx, &q, &k });
}

fn runAttnScoresEinsum(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var q = try filledTensor(ctx, .{ .b, .h, .i, .d }, 4, .{ attn_b, attn_h, attn_i, attn_d }, 1);
    defer q.deinit();
    var k = try filledTensor(ctx, .{ .b, .h, .j, .d }, 4, .{ attn_b, attn_h, attn_j, attn_d }, 2);
    defer k.deinit();
    return measure(iterations, attnScoresEinsumStep, .{ ctx, &q, &k });
}

// -- GQA scores: q[b,g,i,d] x k[b,j,d] -> [b,g,i,j]: multi-free m = g*i
// merged, k picks bmmTransB by orientation probe — zero copies on both
// spellings (dot delegates to the same lowering).

const gqa_b = 2;
const gqa_g = 4;
const gqa_i = 256;
const gqa_j = 256;
const gqa_d = 64;

fn gqaScoresDotStep(ctx: *fucina.ExecContext, q: *const fucina.Tensor(.{ .b, .g, .i, .d }), k: *const fucina.Tensor(.{ .b, .j, .d })) !f64 {
    var scores = try q.dot(ctx, k, .d);
    defer scores.deinit();
    return scores.asRawTensor().dataConst()[0];
}

fn gqaScoresEinsumStep(ctx: *fucina.ExecContext, q: *const fucina.Tensor(.{ .b, .g, .i, .d }), k: *const fucina.Tensor(.{ .b, .j, .d })) !f64 {
    var scores = try q.einsum(ctx, k, .{ .b, .g, .i, .j });
    defer scores.deinit();
    return scores.asRawTensor().dataConst()[0];
}

fn runGqaScoresDot(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var q = try filledTensor(ctx, .{ .b, .g, .i, .d }, 4, .{ gqa_b, gqa_g, gqa_i, gqa_d }, 3);
    defer q.deinit();
    var k = try filledTensor(ctx, .{ .b, .j, .d }, 3, .{ gqa_b, gqa_j, gqa_d }, 4);
    defer k.deinit();
    return measure(iterations, gqaScoresDotStep, .{ ctx, &q, &k });
}

fn runGqaScoresEinsum(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var q = try filledTensor(ctx, .{ .b, .g, .i, .d }, 4, .{ gqa_b, gqa_g, gqa_i, gqa_d }, 3);
    defer q.deinit();
    var k = try filledTensor(ctx, .{ .b, .j, .d }, 3, .{ gqa_b, gqa_j, gqa_d }, 4);
    defer k.deinit();
    return measure(iterations, gqaScoresEinsumStep, .{ ctx, &q, &k });
}

// -- double-transposed layouts: x[k,m] x y[n,k] over .k. dot's canonical
// [m,n] output can satisfy only one trans orientation, so one operand
// materializes; einsum asked for the [n,m] orientation runs as one plain
// GEMM with zero copies (downstream tag consumers are layout-agnostic).

const ddt_m = 512;
const ddt_n = 512;
const ddt_k = 512;

fn doubleTransDotStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .k, .m }), y: *const fucina.Tensor(.{ .n, .k })) !f64 {
    var out = try x.dot(ctx, y, .k);
    defer out.deinit();
    return out.asRawTensor().dataConst()[0];
}

fn doubleTransEinsumStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .k, .m }), y: *const fucina.Tensor(.{ .n, .k })) !f64 {
    var out = try x.einsum(ctx, y, .{ .n, .m });
    defer out.deinit();
    return out.asRawTensor().dataConst()[0];
}

fn runDoubleTransDot(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledTensor(ctx, .{ .k, .m }, 2, .{ ddt_k, ddt_m }, 5);
    defer x.deinit();
    var y = try filledTensor(ctx, .{ .n, .k }, 2, .{ ddt_n, ddt_k }, 6);
    defer y.deinit();
    return measure(iterations, doubleTransDotStep, .{ ctx, &x, &y });
}

fn runDoubleTransEinsum(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledTensor(ctx, .{ .k, .m }, 2, .{ ddt_k, ddt_m }, 5);
    defer x.deinit();
    var y = try filledTensor(ctx, .{ .n, .k }, 2, .{ ddt_n, ddt_k }, 6);
    defer y.deinit();
    return measure(iterations, doubleTransEinsumStep, .{ ctx, &x, &y });
}

// -- multi-axis contraction: x[i,k1,k2] x y[j,k1,k2] -> [i,j]. dot cannot
// express it; the hand-written alternative contracts pre-merged [i,K]x[j,K]
// tensors. Parity expected: einsum's contract-group merge is a free reshape.

const mc_i = 256;
const mc_j = 256;
const mc_k1 = 32;
const mc_k2 = 64;

fn multiContractPremergedStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .i, .k }), y: *const fucina.Tensor(.{ .j, .k })) !f64 {
    var out = try x.dot(ctx, y, .k);
    defer out.deinit();
    return out.asRawTensor().dataConst()[0];
}

fn multiContractEinsumStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .i, .k1, .k2 }), y: *const fucina.Tensor(.{ .j, .k1, .k2 })) !f64 {
    var out = try x.einsum(ctx, y, .{ .i, .j });
    defer out.deinit();
    return out.asRawTensor().dataConst()[0];
}

fn runMultiContractPremerged(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledTensor(ctx, .{ .i, .k }, 2, .{ mc_i, mc_k1 * mc_k2 }, 7);
    defer x.deinit();
    var y = try filledTensor(ctx, .{ .j, .k }, 2, .{ mc_j, mc_k1 * mc_k2 }, 8);
    defer y.deinit();
    return measure(iterations, multiContractPremergedStep, .{ ctx, &x, &y });
}

fn runMultiContractEinsum(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledTensor(ctx, .{ .i, .k1, .k2 }, 3, .{ mc_i, mc_k1, mc_k2 }, 7);
    defer x.deinit();
    var y = try filledTensor(ctx, .{ .j, .k1, .k2 }, 3, .{ mc_j, mc_k1, mc_k2 }, 8);
    defer y.deinit();
    return measure(iterations, multiContractEinsumStep, .{ ctx, &x, &y });
}

// -- LoRA delta chain: x[s,i] · A[r,i] · B[o,r] -> [s,o]. einsumMany's fold
// must match the hand-chained dots (parity: same two GEMMs).

const lora_s = 256;
const lora_i = 1024;
const lora_r = 16;
const lora_o = 1024;

fn loraChainDotStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .s, .i }), a: *const fucina.Tensor(.{ .r, .i }), b: *const fucina.Tensor(.{ .o, .r })) !f64 {
    var xa = try x.dot(ctx, a, .i);
    defer xa.deinit();
    var out = try xa.dot(ctx, b, .r);
    defer out.deinit();
    return out.asRawTensor().dataConst()[0];
}

fn loraChainEinsumManyStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .s, .i }), a: *const fucina.Tensor(.{ .r, .i }), b: *const fucina.Tensor(.{ .o, .r })) !f64 {
    var out = try fucina.einsumMany(ctx, .{ .s, .o }, .{ x, a, b });
    defer out.deinit();
    return out.asRawTensor().dataConst()[0];
}

fn runLoraChainDot(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledTensor(ctx, .{ .s, .i }, 2, .{ lora_s, lora_i }, 9);
    defer x.deinit();
    var a = try filledTensor(ctx, .{ .r, .i }, 2, .{ lora_r, lora_i }, 10);
    defer a.deinit();
    var b = try filledTensor(ctx, .{ .o, .r }, 2, .{ lora_o, lora_r }, 11);
    defer b.deinit();
    return measure(iterations, loraChainDotStep, .{ ctx, &x, &a, &b });
}

fn runLoraChainEinsumMany(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledTensor(ctx, .{ .s, .i }, 2, .{ lora_s, lora_i }, 9);
    defer x.deinit();
    var a = try filledTensor(ctx, .{ .r, .i }, 2, .{ lora_r, lora_i }, 10);
    defer a.deinit();
    var b = try filledTensor(ctx, .{ .o, .r }, 2, .{ lora_o, lora_r }, 11);
    defer b.deinit();
    return measure(iterations, loraChainEinsumManyStep, .{ ctx, &x, &a, &b });
}

// -- forward+backward with two left free axes: x[i,j,k] x w[n,k]. Both
// spellings share EinsumBackward's GEMM-lowered gradients (this case used to
// take DotBackward's broadcast pointwise-mul fallback, ~350x slower).

const bw_i = 32;
const bw_j = 32;
const bw_k = 128;
const bw_n = 128;

fn multiFreeBackwardDotStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .i, .j, .k }), w: *const fucina.Tensor(.{ .n, .k })) !f64 {
    var y = try x.dot(ctx, w, .k);
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    try loss.backward(ctx);
    const g = (try w.gradView(ctx)).?;
    var grad = g;
    defer grad.deinit();
    const checksum = grad.asRawTensor().dataConst()[0];
    x.zeroGrad();
    w.zeroGrad();
    return checksum;
}

fn multiFreeBackwardEinsumStep(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{ .i, .j, .k }), w: *const fucina.Tensor(.{ .n, .k })) !f64 {
    var y = try x.einsum(ctx, w, .{ .i, .j, .n });
    defer y.deinit();
    var loss = try y.sumAll(ctx);
    defer loss.deinit();
    try loss.backward(ctx);
    const g = (try w.gradView(ctx)).?;
    var grad = g;
    defer grad.deinit();
    const checksum = grad.asRawTensor().dataConst()[0];
    x.zeroGrad();
    w.zeroGrad();
    return checksum;
}

fn runMultiFreeBackwardDot(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledVariable(ctx, .{ .i, .j, .k }, 3, .{ bw_i, bw_j, bw_k }, 12);
    defer x.deinit();
    var w = try filledVariable(ctx, .{ .n, .k }, 2, .{ bw_n, bw_k }, 13);
    defer w.deinit();
    return measure(iterations, multiFreeBackwardDotStep, .{ ctx, &x, &w });
}

fn runMultiFreeBackwardEinsum(iterations: usize) !Result {
    var bench = Bench.init();
    defer bench.deinit();
    const ctx = bench.start();
    var x = try filledVariable(ctx, .{ .i, .j, .k }, 3, .{ bw_i, bw_j, bw_k }, 12);
    defer x.deinit();
    var w = try filledVariable(ctx, .{ .n, .k }, 2, .{ bw_n, bw_k }, 13);
    defer w.deinit();
    return measure(iterations, multiFreeBackwardEinsumStep, .{ ctx, &x, &w });
}
