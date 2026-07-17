//! Qwen3.5 (`qwen35` hybrid Gated-DeltaNet) harness.
//!
//! Loads a `qwen35` GGUF, prints the derived config + per-kind block counts
//! (full-attention vs DeltaNet-linear blocks), and runs the hybrid forward
//! pass (conv1d + DeltaNet scan + multi-section RoPE): whole-sequence logits
//! with a top-5 readout, `--logits-out` dumps for external parity checks,
//! `--decode` incremental-decode equivalence, and a `--bench` pp/tg sweep.
//!
//!   zig build qwen35 -- <model.gguf> [--info]
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const Config = llm.qwen35.model.Config;
const ForwardProfile = llm.qwen35.model.ForwardProfile;
const LinearScanMode = llm.qwen35.model.LinearScanMode;
const Model = llm.qwen35.model.Model;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build qwen35 -- <model.gguf> [<token-ids>] [--info] [--decode] [--profile] [--bench R [--gen N]] [--logits-out PATH] [--linear-scan chunked|recurrent]\n", .{});
        return;
    }

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try fucina.gguf.File.loadMmap(allocator, init.io, args[1]);
    const config = try Config.fromGguf(&file);

    printConfig(stdout, config) catch {};

    // Parse args: a comma-separated token-id list and/or --logits-out <path>.
    // Runs before the weight load so --info can answer from GGUF metadata alone.
    var token_buf: [4096]usize = undefined;
    var tokens: []const usize = &.{ 9707, 11, 1879 };
    var logits_out: ?[]const u8 = null;
    var decode_check = false;
    var bench_reps: usize = 0;
    var gen_count: usize = 128;
    var profile_enabled = false;
    var linear_scan: LinearScanMode = .chunked;
    var ai: usize = 2;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--info")) {
            file.deinit();
            return;
        }
        if (std.mem.eql(u8, args[ai], "--decode")) {
            decode_check = true;
        } else if (std.mem.eql(u8, args[ai], "--profile")) {
            profile_enabled = true;
        } else if (std.mem.eql(u8, args[ai], "--bench")) {
            ai += 1;
            if (ai >= args.len) return error.MissingBenchReps;
            bench_reps = try std.fmt.parseInt(usize, args[ai], 10);
        } else if (std.mem.eql(u8, args[ai], "--gen")) {
            ai += 1;
            if (ai >= args.len) return error.MissingGenCount;
            gen_count = try std.fmt.parseInt(usize, args[ai], 10);
        } else if (std.mem.eql(u8, args[ai], "--linear-scan")) {
            ai += 1;
            if (ai >= args.len) return error.MissingLinearScanMode;
            linear_scan = try parseLinearScanMode(args[ai]);
        } else if (std.mem.eql(u8, args[ai], "--logits-out")) {
            ai += 1;
            if (ai >= args.len) return error.MissingLogitsPath;
            logits_out = args[ai];
        } else {
            tokens = try parseTokenList(args[ai], &token_buf);
        }
    }

    const t0 = nowNs(init.io);
    var model = try Model.loadGgufFromFile(&ctx, &file, config);
    defer model.deinit();
    file.deinit();
    const load_ns = nowNs(init.io) - t0;

    const counts = model.blockCounts();
    try stdout.print("loaded: {d} blocks ({d} full-attn, {d} DeltaNet-linear)  in {d:.3} s\n", .{
        config.num_layers, counts.attn, counts.linear, seconds(load_ns),
    });

    if (bench_reps > 0) {
        try runBench(init.io, stdout, &ctx, &model, bench_reps, gen_count, profile_enabled, linear_scan);
        return;
    }

    // Whole-sequence forward → last-token logits.
    var profile: ForwardProfile = .{};
    const fwd_start = nowNs(init.io);
    var logits = if (profile_enabled or linear_scan != .chunked) blk: {
        var cache = try model.initCache(&ctx, tokens.len + 8);
        defer cache.deinit();
        break :blk if (profile_enabled)
            try model.forwardStepProfiledWithScanMode(&ctx, &cache, tokens, 0, linear_scan, init.io, &profile)
        else
            try model.forwardStepWithScanMode(&ctx, &cache, tokens, 0, linear_scan);
    } else try model.forwardLastLogits(&ctx, tokens);
    defer logits.deinit();
    const fwd_ns = nowNs(init.io) - fwd_start;

    var top = try logits.topK(&ctx, .vocab, 5, .top);
    defer top.deinit();
    try stdout.print("forward: {d} tokens in {d:.3} s  top:", .{ tokens.len, seconds(fwd_ns) });
    const top_values = try top.values.dataConst();
    const top_indices = try top.indices.dataConst();
    for (top_values, top_indices) |value, index| {
        try stdout.print(" {d}:{d:.3}", .{ index, value });
    }
    try stdout.print("\n", .{});
    if (profile_enabled) try printProfile(stdout, &profile, "profile");

    if (logits_out) |path| {
        const data = try logits.dataConst();
        var file_w = try std.Io.Dir.cwd().createFile(init.io, path, .{});
        defer file_w.close(init.io);
        var buffer: [64 * 1024]u8 = undefined;
        var writer = file_w.writer(init.io, &buffer);
        defer writer.interface.flush() catch {};
        try writer.interface.writeAll(std.mem.sliceAsBytes(data));
        try stdout.print("logits: {s} ({d} f32)\n", .{ path, data.len });
    }

    // Incremental-decode equivalence: prefill the first token, then decode the
    // rest one at a time through the streaming cache (KV + recurrent state).
    // The final logits must match the whole-sequence forward (argmax-aligned;
    // small mean|Δ| from f16 KV in the attention layers).
    if (decode_check and tokens.len >= 2) {
        var cache = try model.initCache(&ctx, tokens.len + 8);
        defer cache.deinit();
        const dec_start = nowNs(init.io);
        var step = try model.forwardStepWithScanMode(&ctx, &cache, tokens[0..1], 0, linear_scan);
        for (1..tokens.len) |i| {
            step.deinit();
            step = try model.forwardStepWithScanMode(&ctx, &cache, tokens[i .. i + 1], i, linear_scan);
        }
        const dec_ns = nowNs(init.io) - dec_start;
        defer step.deinit();

        const ref = try logits.dataConst();
        const dec = try step.dataConst();
        var ra: usize = 0;
        var da: usize = 0;
        var sum: f64 = 0;
        for (ref, dec, 0..) |rv, dv, i| {
            if (rv > ref[ra]) ra = i;
            if (dv > dec[da]) da = i;
            sum += @abs(rv - dv);
        }
        try stdout.print("decode: {d} steps in {d:.3} s  argmax whole={d} decode={d} match={}  mean_abs_diff={d:.5}\n", .{
            tokens.len, seconds(dec_ns), ra, da, ra == da, sum / @as(f64, @floatFromInt(ref.len)),
        });
    }
}

fn argmaxLast(ctx: *fucina.ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab })) !usize {
    var last = try logits.narrow(ctx, .seq, logits.dim(.seq) - 1, 1);
    defer last.deinit();
    var index = try last.argmax(ctx, .vocab);
    defer index.deinit();
    return @intCast(try index.item());
}

/// Prefill (pp) throughput sweep over prompt lengths + decode (tg) throughput,
/// best-of-`reps` (warm), via the streaming cache — comparable to `llama-bench`.
fn runBench(
    io: std.Io,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    reps: usize,
    gen: usize,
    profile_enabled: bool,
    linear_scan: LinearScanMode,
) !void {
    var buf: [4096]usize = undefined;
    const lengths = [_]usize{ 1, 32, 128, 512 };

    var cap: usize = gen;
    for (lengths) |n| cap = @max(cap, n + 1);
    var cache = try model.initCache(ctx, cap + 8);
    defer cache.deinit();

    try stdout.print("qwen35 bench (best-of-{d}, linear-scan={s}):\n", .{ reps, @tagName(linear_scan) });
    fucina.internal.gpu.traceReset();
    for (lengths) |n| {
        for (0..n) |i| buf[i] = (i * 131 + 7) % 200000 + 1;
        var best: f64 = std.math.inf(f64);
        var best_profile: ForwardProfile = .{};
        for (0..reps) |_| {
            cache.reset();
            var profile: ForwardProfile = .{};
            const t0 = nowNs(io);
            var logits = if (profile_enabled)
                try model.forwardStepProfiledWithScanMode(ctx, &cache, buf[0..n], 0, linear_scan, io, &profile)
            else
                try model.forwardStepWithScanMode(ctx, &cache, buf[0..n], 0, linear_scan);
            const dt = seconds(nowNs(io) - t0);
            logits.deinit();
            if (dt < best) {
                best = dt;
                best_profile = profile;
            }
        }
        try stdout.print("  pp{d:<4} {d:8.1} tok/s  ({d:.2} ms)\n", .{ n, @as(f64, @floatFromInt(n)) / best, best * 1000.0 });
        if (profile_enabled) try printProfile(stdout, &best_profile, "    profile");
    }
    // Decode: prefill a short prompt, then time `gen` single-token steps.
    var best_tg: f64 = std.math.inf(f64);
    for (0..reps) |_| {
        cache.reset();
        var logits = try model.forwardStepWithScanMode(ctx, &cache, &.{ 9707, 11, 1879 }, 0, linear_scan);
        const t0 = nowNs(io);
        var steps: usize = 0;
        while (steps < gen and cache.len() < cache.kv.capacity) : (steps += 1) {
            const next = try argmaxLast(ctx, &logits);
            const fresh = try model.forwardStepWithScanMode(ctx, &cache, &.{next}, cache.len(), linear_scan);
            logits.deinit();
            logits = fresh;
        }
        const per = seconds(nowNs(io) - t0) / @as(f64, @floatFromInt(steps));
        logits.deinit();
        best_tg = @min(best_tg, per);
    }
    try stdout.print("  tg{d:<4} {d:8.1} tok/s  ({d:.2} ms/tok)\n", .{ gen, 1.0 / best_tg, best_tg * 1000.0 });
    fucina.internal.gpu.traceDump();
}

fn printProfile(stdout: *std.Io.Writer, p: *const ForwardProfile, prefix: []const u8) !void {
    const total = p.total_ns;
    try stdout.print(
        "{s}: total={d:.2} ms tokens={d} layers(attn/linear/ffn)={d}/{d}/{d}\n",
        .{ prefix, millis(total), p.tokens, p.attn_layers, p.linear_layers, p.ffn_layers },
    );
    try stdout.print(
        "{s}: embed={d:.2} ({d:.1}%) prep={d:.2} ({d:.1}%) ffn={d:.2} ({d:.1}%) final={d:.2} ({d:.1}%)\n",
        .{
            prefix,
            millis(p.embed_ns),
            pct(p.embed_ns, total),
            millis(p.prep_ns),
            pct(p.prep_ns, total),
            millis(p.ffn_ns),
            pct(p.ffn_ns, total),
            millis(p.final_ns),
            pct(p.final_ns, total),
        },
    );
    try stdout.print(
        "{s}: attn total={d:.2} ({d:.1}%) proj={d:.2} sdpa={d:.2} rope={d:.2} out={d:.2}\n",
        .{
            prefix,
            millis(p.attn_total_ns),
            pct(p.attn_total_ns, total),
            millis(p.attn_proj_ns),
            millis(p.attn_sdpa_ns),
            millis(p.attn_rope_ns),
            millis(p.attn_out_ns),
        },
    );
    try stdout.print(
        "{s}: linear total={d:.2} ({d:.1}%) qkv={d:.2} conv={d:.2} gate={d:.2} scan={d:.2} out={d:.2}\n",
        .{
            prefix,
            millis(p.linear_total_ns),
            pct(p.linear_total_ns, total),
            millis(p.linear_qkv_ns),
            millis(p.linear_conv_ns),
            millis(p.linear_gate_ns),
            millis(p.linear_scan_ns),
            millis(p.linear_out_ns),
        },
    );
    const linear_other =
        p.linear_total_ns -
        (p.linear_norm_ns + p.linear_qkv_ns + p.linear_conv_ns + p.linear_gate_ns + p.linear_scan_ns + p.linear_out_ns);
    try stdout.print(
        "{s}: linear detail norm={d:.2} z={d:.2} alpha={d:.2} beta={d:.2} other={d:.2}\n",
        .{
            prefix,
            millis(p.linear_norm_ns),
            millis(p.linear_z_ns),
            millis(p.linear_alpha_ns),
            millis(p.linear_beta_ns),
            millis(linear_other),
        },
    );
    const ffn_other =
        p.ffn_ns -
        (p.ffn_norm_ns + p.ffn_gate_up_ns + p.ffn_gate_ns + p.ffn_up_ns + p.ffn_act_ns + p.ffn_down_ns + p.ffn_residual_ns);
    try stdout.print(
        "{s}: ffn detail norm={d:.2} gate_up={d:.2} gate={d:.2} up={d:.2} act={d:.2} down={d:.2} residual={d:.2} other={d:.2}\n",
        .{
            prefix,
            millis(p.ffn_norm_ns),
            millis(p.ffn_gate_up_ns),
            millis(p.ffn_gate_ns),
            millis(p.ffn_up_ns),
            millis(p.ffn_act_ns),
            millis(p.ffn_down_ns),
            millis(p.ffn_residual_ns),
            millis(ffn_other),
        },
    );
    const top_other =
        p.total_ns -
        (p.prep_ns + p.embed_ns + p.attn_total_ns + p.linear_total_ns + p.ffn_ns + p.final_ns);
    try stdout.print("{s}: top other={d:.2} ({d:.1}%)\n", .{ prefix, millis(top_other), pct(top_other, total) });
}

fn parseTokenList(arg: []const u8, buf: []usize) ![]const usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, arg, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " ");
        if (trimmed.len == 0) continue;
        if (n >= buf.len) return error.TooManyTokens;
        buf[n] = try std.fmt.parseInt(usize, trimmed, 10);
        n += 1;
    }
    if (n == 0) return error.NoTokens;
    return buf[0..n];
}

fn parseLinearScanMode(arg: []const u8) !LinearScanMode {
    if (std.mem.eql(u8, arg, "chunked")) return .chunked;
    if (std.mem.eql(u8, arg, "recurrent")) return .recurrent;
    return error.InvalidLinearScanMode;
}

fn printConfig(stdout: *std.Io.Writer, c: Config) !void {
    try stdout.print(
        \\qwen35 config:
        \\  vocab={d} hidden={d} ffn={d} layers={d}
        \\  attn: heads={d} kv={d} head_dim={d} rope_n_rot={d} sections={d}/{d}/{d}/{d} full_attn_interval={d}
        \\  ssm: d_conv={d} d_inner={d} d_state={d} dt_rank={d} n_group={d}
        \\  rope_base={d:.1} eps={e}
        \\
    , .{
        c.vocab_size,              c.hidden_size,         c.intermediate_size, c.num_layers,
        c.num_attention_heads,     c.num_key_value_heads, c.head_dim,          c.rope_n_rot,
        c.rope_sections[0],        c.rope_sections[1],    c.rope_sections[2],  c.rope_sections[3],
        c.full_attention_interval, c.ssm_d_conv,          c.ssm_d_inner,       c.ssm_d_state,
        c.ssm_dt_rank,             c.ssm_n_group,         c.rope_theta,        c.rms_norm_eps,
    });
}

fn millis(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn pct(part: i128, total: i128) f64 {
    if (total <= 0) return 0;
    return 100.0 * @as(f64, @floatFromInt(part)) / @as(f64, @floatFromInt(total));
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}
