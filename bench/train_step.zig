//! End-to-end GPT training-step benchmark (`zig build bench-train-step`).
//!
//! One full autograd training step — embed, transformer blocks (rmsNorm,
//! RoPE, causal attention, SwiGLU MLP), final norm, untied unembed,
//! cross-entropy, backward, AdamW — on a fixed synthetic sequence, timed per
//! step with the loss printed. `--dump <dir>` writes the seeded weights, the
//! token ids, and the rope table so `tools/torch_train_step.py` can run the
//! IDENTICAL model in PyTorch eager: matching loss trajectories prove both
//! sides compute the same thing; the per-step wall clock is the comparison.
//!
//! Run in ReleaseFast:
//!   zig build bench-train-step -Doptimize=ReleaseFast -- --prod-allocator --dump /tmp/train-dump

const std = @import("std");
const bench_alloc = @import("alloc.zig");
const Timer = @import("timer.zig").Timer;
const bench_raw = @import("bench_raw");
const fucina = bench_raw;
const optim = bench_raw.optim;

const ExecContext = bench_raw.ExecContext;
const Tensor = fucina.Tensor;
const RopeTable = bench_raw.RopeTable;

const vocab: usize = 16384;
const seq_len: usize = 1024;
const d_model: usize = 512;
const n_head: usize = 8;
const head_dim: usize = d_model / n_head;
const ffn: usize = 1536;
const n_layer: usize = 6;
const attn_scale: f32 = 0.125; // 1/sqrt(head_dim)
const rms_eps: f32 = 1e-5;
const rope_theta: f32 = 10000.0;
const init_std: f32 = 0.02;
const warmup_steps: usize = 2;
const default_timed_steps: usize = 10;

/// --composed-swiglu: the two-op silu+mul MLP gate instead of the fused
/// elemental op — the fusion A/B lever.
var composed_swiglu: bool = false;

/// --composed-residual: the two-op dot + add residual instead of the fused
/// addDot (beta=1 GEMM) — the epilogue A/B lever.
var composed_residual: bool = false;

/// --inference: eval-mode forward only (fucina.noGrad — no graph nodes, no
/// backward, no optimizer), the loss scalar as the cross-framework value pin.
var inference_mode: bool = false;

const Layer = struct {
    c_q: Tensor(.{ .d, .qo }),
    c_k: Tensor(.{ .d, .kvo }),
    c_v: Tensor(.{ .d, .kvo }),
    c_proj: Tensor(.{ .attn, .d }),
    w_gate: Tensor(.{ .d, .ff }),
    w_up: Tensor(.{ .d, .ff }),
    w_down: Tensor(.{ .ff, .d }),
};

const Model = struct {
    wte: Tensor(.{ .vocab, .d }),
    w_lm: Tensor(.{ .vocab, .d }),
    layers: [n_layer]Layer,
    rope_table: RopeTable,
    kv_map: [n_head]usize,
};

const Dumper = struct {
    io: std.Io,
    dir: ?std.Io.Dir,
    manifest: std.ArrayList(u8),
    allocator: std.mem.Allocator,
    first: bool = true,

    fn writeTensor(self: *Dumper, name: []const u8, values: []const f32, shape: []const usize) !void {
        const dir = self.dir orelse return;
        var name_buf: [128]u8 = undefined;
        const file_name = try std.fmt.bufPrint(&name_buf, "{s}.bin", .{name});
        try dir.writeFile(self.io, .{ .sub_path = file_name, .data = std.mem.sliceAsBytes(values) });
        if (!self.first) try self.manifest.appendSlice(self.allocator, ",\n");
        self.first = false;
        try self.manifest.print(self.allocator, "  {{\"name\": \"{s}\", \"file\": \"{s}\", \"shape\": [", .{ name, file_name });
        for (shape, 0..) |dim, i| {
            if (i > 0) try self.manifest.appendSlice(self.allocator, ", ");
            try self.manifest.print(self.allocator, "{d}", .{dim});
        }
        try self.manifest.appendSlice(self.allocator, "]}");
    }

    fn finish(self: *Dumper) !void {
        const dir = self.dir orelse return;
        var payload: std.ArrayList(u8) = .empty;
        defer payload.deinit(self.allocator);
        try payload.appendSlice(self.allocator, "[\n");
        try payload.appendSlice(self.allocator, self.manifest.items);
        try payload.appendSlice(self.allocator, "\n]\n");
        try dir.writeFile(self.io, .{ .sub_path = "manifest.json", .data = payload.items });
    }
};

fn initParam(
    comptime T: type,
    ctx: *ExecContext,
    allocator: std.mem.Allocator,
    random: std.Random,
    dumper: *Dumper,
    name: []const u8,
    shape: [2]usize,
) !T {
    const data = try allocator.alloc(f32, shape[0] * shape[1]);
    defer allocator.free(data);
    for (data) |*value| value.* = random.floatNorm(f32) * init_std;
    try dumper.writeTensor(name, data, &shape);
    return T.variableFromSlice(ctx, shape, data);
}

/// SwiGLU gate as ONE elemental op — silu(gate)·up fused into a single
/// forward pass and one pass per operand gradient, with SIMD bodies
/// (the scalar rules alone would pay a libm expf per element). Replaces
/// the composed `.unary(.silu)` + `.mul` pair.
const SwigluOp = struct {
    const simd = bench_raw.simd;

    pub fn forward(gate_value: f32, up_value: f32, extra: void) f32 {
        _ = extra;
        const s = 1 / (1 + @exp(-gate_value));
        return gate_value * s * up_value;
    }

    pub fn forwardVec(gate_vec: simd.Vf32, up_vec: simd.Vf32, extra: void) simd.Vf32 {
        _ = extra;
        return gate_vec * simd.sigmoidVec(gate_vec) * up_vec;
    }

    pub fn backwardA(gate_value: f32, up_value: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = y;
        _ = extra;
        const s = 1 / (1 + @exp(-gate_value));
        return grad_y * up_value * s * (1 + gate_value * (1 - s));
    }

    pub fn backwardAVec(gate_vec: simd.Vf32, up_vec: simd.Vf32, y: simd.Vf32, grad_y: simd.Vf32, extra: void) simd.Vf32 {
        _ = y;
        _ = extra;
        const one: simd.Vf32 = @splat(1);
        const s = simd.sigmoidVec(gate_vec);
        return grad_y * up_vec * s * (one + gate_vec * (one - s));
    }

    pub fn backwardB(gate_value: f32, up_value: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = up_value;
        _ = y;
        _ = extra;
        const s = 1 / (1 + @exp(-gate_value));
        return grad_y * gate_value * s;
    }

    pub fn backwardBVec(gate_vec: simd.Vf32, up_vec: simd.Vf32, y: simd.Vf32, grad_y: simd.Vf32, extra: void) simd.Vf32 {
        _ = up_vec;
        _ = y;
        _ = extra;
        return grad_y * gate_vec * simd.sigmoidVec(gate_vec);
    }
};

/// One forward for BOTH regimes — the exec-scope design makes this legal:
/// every intermediate deinits at its last use, which under a training scope
/// is a no-op borrow release (the scope keeps the autograd graph alive for
/// backward), and without a scope (inference) returns each transient to the
/// buffer pool immediately, so the same few hot buffers recycle across ops
/// and layers.
fn forwardLoss(ctx: *ExecContext, model: *const Model, input_ids: []const usize, labels: []const usize) !Tensor(.{}) {
    var x = try model.wte.gather(ctx, .vocab, input_ids, .seq);
    for (0..n_layer) |i| {
        const l = &model.layers[i];

        var h = try x.rmsNorm(ctx, .d, rms_eps);
        defer h.deinit();
        var q_proj = try h.dot(ctx, &l.c_q, .d);
        defer q_proj.deinit();
        var k_proj = try h.dot(ctx, &l.c_k, .d);
        defer k_proj.deinit();
        var v_proj = try h.dot(ctx, &l.c_v, .d);
        defer v_proj.deinit();
        var q_split = try q_proj.split(ctx, .qo, .{ .head, .d }, [_]usize{ n_head, head_dim });
        defer q_split.deinit();
        var k_split = try k_proj.split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ n_head, head_dim });
        defer k_split.deinit();
        var v = try v_proj.split(ctx, .kvo, .{ .kv_head, .d }, [_]usize{ n_head, head_dim });
        defer v.deinit();
        var q = try q_split.rope(ctx, .seq, .d, &model.rope_table, .half);
        defer q.deinit();
        var k = try k_split.rope(ctx, .seq, .d, &model.rope_table, .half);
        defer k.deinit();
        var y = try q.groupedAttention(ctx, &k, &v, &model.kv_map, .attn, attn_scale, .{});
        defer y.deinit();
        var x_attn = if (composed_residual) blk: {
            var attn_out = try y.dot(ctx, &l.c_proj, .attn);
            defer attn_out.deinit();
            break :blk try x.add(ctx, &attn_out);
        } else try x.addDot(ctx, y, l.c_proj, .attn);
        x.deinit();

        var m = try x_attn.rmsNorm(ctx, .d, rms_eps);
        defer m.deinit();
        var gate = try m.dot(ctx, &l.w_gate, .d);
        defer gate.deinit();
        var up = try m.dot(ctx, &l.w_up, .d);
        defer up.deinit();
        var gated = if (composed_swiglu) blk: {
            var silu_out = try gate.unary(ctx, .silu);
            defer silu_out.deinit();
            break :blk try silu_out.mul(ctx, &up);
        } else try gate.elementalBinary(ctx, up, SwigluOp, {});
        defer gated.deinit();
        x = if (composed_residual) blk: {
            var mlp_out = try gated.dot(ctx, &l.w_down, .ff);
            defer mlp_out.deinit();
            break :blk try x_attn.add(ctx, &mlp_out);
        } else try x_attn.addDot(ctx, gated, l.w_down, .ff);
        x_attn.deinit();
    }
    var x_norm = try x.rmsNorm(ctx, .d, rms_eps);
    x.deinit();
    defer x_norm.deinit();
    return x_norm.linearCrossEntropy(ctx, &model.w_lm, labels, .{ .reduction = .mean });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var mode: bench_alloc.AllocatorMode = .smp;
    var dump_path: ?[]const u8 = null;
    var timed_steps: usize = default_timed_steps;
    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        if (std.mem.eql(u8, args[arg_i], "--dump") and arg_i + 1 < args.len) {
            arg_i += 1;
            dump_path = args[arg_i];
        } else if (std.mem.eql(u8, args[arg_i], "--composed-swiglu")) {
            composed_swiglu = true;
        } else if (std.mem.eql(u8, args[arg_i], "--composed-residual")) {
            composed_residual = true;
        } else if (std.mem.eql(u8, args[arg_i], "--inference")) {
            inference_mode = true;
        } else if (std.mem.eql(u8, args[arg_i], "--steps") and arg_i + 1 < args.len) {
            arg_i += 1;
            timed_steps = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (try bench_alloc.parseAllocatorModeArg(args[arg_i])) |parsed| {
            mode = parsed;
        }
    }
    var bench_allocator = bench_alloc.BenchmarkAllocator.init(mode);
    const allocator = bench_allocator.allocator();

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var dumper = Dumper{
        .io = init.io,
        .dir = if (dump_path) |path|
            try std.Io.Dir.cwd().createDirPathOpen(init.io, path, .{})
        else
            null,
        .manifest = .empty,
        .allocator = allocator,
    };
    defer dumper.manifest.deinit(allocator);
    defer if (dumper.dir) |*dir| dir.close(init.io);

    var prng = std.Random.DefaultPrng.init(0x7ea1);
    const random = prng.random();

    var model: Model = undefined;
    for (0..n_head) |i| model.kv_map[i] = i;
    {
        const positions = try allocator.alloc(i32, seq_len);
        defer allocator.free(positions);
        for (positions, 0..) |*p, i| p.* = @intCast(i);
        model.rope_table = try ctx.prepareRopeTable(positions, head_dim, rope_theta, false);
    }
    defer model.rope_table.deinit();
    try dumper.writeTensor("rope_sin", model.rope_table.sinValues(), &.{ seq_len, head_dim / 2 });
    try dumper.writeTensor("rope_cos", model.rope_table.cosValues(), &.{ seq_len, head_dim / 2 });

    var name_buf: [128]u8 = undefined;
    model.wte = try initParam(Tensor(.{ .vocab, .d }), &ctx, allocator, random, &dumper, "wte", .{ vocab, d_model });
    model.w_lm = try initParam(Tensor(.{ .vocab, .d }), &ctx, allocator, random, &dumper, "w_lm", .{ vocab, d_model });
    for (0..n_layer) |i| {
        model.layers[i] = .{
            .c_q = try initParam(Tensor(.{ .d, .qo }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.c_q", .{i}), .{ d_model, n_head * head_dim }),
            .c_k = try initParam(Tensor(.{ .d, .kvo }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.c_k", .{i}), .{ d_model, n_head * head_dim }),
            .c_v = try initParam(Tensor(.{ .d, .kvo }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.c_v", .{i}), .{ d_model, n_head * head_dim }),
            .c_proj = try initParam(Tensor(.{ .attn, .d }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.c_proj", .{i}), .{ n_head * head_dim, d_model }),
            .w_gate = try initParam(Tensor(.{ .d, .ff }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.w_gate", .{i}), .{ d_model, ffn }),
            .w_up = try initParam(Tensor(.{ .d, .ff }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.w_up", .{i}), .{ d_model, ffn }),
            .w_down = try initParam(Tensor(.{ .ff, .d }), &ctx, allocator, random, &dumper, try std.fmt.bufPrint(&name_buf, "layers.{d}.w_down", .{i}), .{ ffn, d_model }),
        };
    }
    defer {
        model.wte.deinit();
        model.w_lm.deinit();
        for (0..n_layer) |i| {
            model.layers[i].c_q.deinit();
            model.layers[i].c_k.deinit();
            model.layers[i].c_v.deinit();
            model.layers[i].c_proj.deinit();
            model.layers[i].w_gate.deinit();
            model.layers[i].w_up.deinit();
            model.layers[i].w_down.deinit();
        }
    }

    // Fixed synthetic sequence: ids[t] feeds the model, ids[t+1] is the label.
    const ids = try allocator.alloc(usize, seq_len + 1);
    defer allocator.free(ids);
    for (ids) |*id| id.* = random.uintLessThan(usize, vocab);
    {
        const ids_f32 = try allocator.alloc(f32, seq_len + 1);
        defer allocator.free(ids_f32);
        for (ids_f32, ids) |*out, id| out.* = @floatFromInt(id);
        try dumper.writeTensor("token_ids", ids_f32, &.{seq_len + 1});
    }
    try dumper.finish();
    const input_ids = ids[0..seq_len];
    const labels = ids[1 .. seq_len + 1];

    var opt = optim.AdamW.init(allocator, .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-8, .weight_decay = 0 });
    defer opt.deinit();
    try opt.addParam(&model.wte);
    try opt.addParam(&model.w_lm);
    for (0..n_layer) |i| {
        try opt.addParam(&model.layers[i].c_q);
        try opt.addParam(&model.layers[i].c_k);
        try opt.addParam(&model.layers[i].c_v);
        try opt.addParam(&model.layers[i].c_proj);
        try opt.addParam(&model.layers[i].w_gate);
        try opt.addParam(&model.layers[i].w_up);
        try opt.addParam(&model.layers[i].w_down);
    }

    try stdout.print(
        "gpt train step — backend={s} layers={d} d={d} heads={d} ffn={d} vocab={d} seq={d}\n",
        .{ @tagName(fucina.active_backend_kind), n_layer, d_model, n_head, ffn, vocab, seq_len },
    );

    const losses = try allocator.alloc(f32, warmup_steps + timed_steps);
    defer allocator.free(losses);
    const step_ms = try allocator.alloc(f64, warmup_steps + timed_steps);
    defer allocator.free(step_ms);
    var fwd_total: f64 = 0;
    var bwd_total: f64 = 0;
    var opt_total: f64 = 0;
    for (0..warmup_steps + timed_steps) |step_i| {
        var timer = try Timer.start(init.io);
        if (inference_mode) {
            var ng = fucina.noGrad();
            var loss = try forwardLoss(&ctx, &model, input_ids, labels);
            const loss_value = try loss.item();
            loss.deinit();
            ng.close();
            const infer_ns = timer.read();
            losses[step_i] = loss_value;
            step_ms[step_i] = @as(f64, @floatFromInt(infer_ns)) / 1e6;
            if (step_i >= warmup_steps) fwd_total += step_ms[step_i];
            try stdout.print("step {d:>2}  loss {d:.6}  {d:>8.2} ms\n", .{ step_i, loss_value, step_ms[step_i] });
            try stdout.flush();
            continue;
        }
        const scope = ctx.openExecScope();
        const loss = try forwardLoss(&ctx, &model, input_ids, labels);
        const fwd_value = try loss.item(); // forces the forward to settle
        const fwd_ns = timer.read();
        try loss.backward(&ctx);
        const bwd_ns = timer.read();
        ctx.closeExecScope(scope);
        try opt.step(&ctx);
        opt.zeroGrad();
        const ns = timer.read();
        losses[step_i] = fwd_value;
        step_ms[step_i] = @as(f64, @floatFromInt(ns)) / 1e6;
        if (step_i >= warmup_steps) {
            fwd_total += @as(f64, @floatFromInt(fwd_ns)) / 1e6;
            bwd_total += @as(f64, @floatFromInt(bwd_ns - fwd_ns)) / 1e6;
            opt_total += @as(f64, @floatFromInt(ns - bwd_ns)) / 1e6;
        }
        try stdout.print("step {d:>2}  loss {d:.6}  {d:>8.2} ms\n", .{ step_i, fwd_value, step_ms[step_i] });
        try stdout.flush();
    }
    const steps_f: f64 = @floatFromInt(timed_steps);
    try stdout.print("sections: fwd {d:.1} ms, bwd {d:.1} ms, opt+zero {d:.1} ms\n", .{ fwd_total / steps_f, bwd_total / steps_f, opt_total / steps_f });

    var total: f64 = 0;
    for (step_ms[warmup_steps..]) |ms| total += ms;
    try stdout.print("timed mean ({d} steps after {d} warmup): {d:.2} ms/step\n", .{ timed_steps, warmup_steps, total / @as(f64, @floatFromInt(timed_steps)) });

    if (dumper.dir) |dir| {
        var payload: std.ArrayList(u8) = .empty;
        defer payload.deinit(allocator);
        try payload.appendSlice(allocator, "{\"losses\": [");
        for (losses, 0..) |value, i| {
            if (i > 0) try payload.appendSlice(allocator, ", ");
            try payload.print(allocator, "{d}", .{value});
        }
        try payload.appendSlice(allocator, "], \"step_ms\": [");
        for (step_ms, 0..) |value, i| {
            if (i > 0) try payload.appendSlice(allocator, ", ");
            try payload.print(allocator, "{d}", .{value});
        }
        try payload.print(allocator, "], \"warmup_steps\": {d}}}\n", .{warmup_steps});
        try dir.writeFile(init.io, .{ .sub_path = "fucina_results.json", .data = payload.items });
    }
}
