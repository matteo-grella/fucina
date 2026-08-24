//! Gate C stage 2: free-running evaluation of SubQ vs dense attention.
//!
//! Teacher-forced agreement cannot detect the failure class that fixed-budget
//! sparse attention exhibits in the literature (repetition collapse, drift
//! into loops under free-running generation). This tool runs both arms
//! FREE-RUNNING: after the SubQ self-calibration window (teacher-forced from
//! the dense arm's stream), each arm feeds its own greedy tokens.
//!
//! Modes:
//!   generate  run both arms, write generated streams (u32), print native
//!             repetition-loop metrics per stream;
//!   judge     teacher-forced mean NLL of a stream's generated segment under
//!             the DENSE model (the reference model judges the continuation).
//!
//! Pre-registered stage-2 gates (2026-08-14, before the first battery):
//!   G-FR1 loops: fraction of runs with a >=32-token periodic run in the
//!         generation: subq <= dense + 10 percentage points;
//!   G-FR2 quality: dense-judged NLL of subq continuations <= dense-judged
//!         NLL of dense continuations + 0.15 nats (mean over runs);
//!   G-FR3 free-running single-needle retrieval: subq answers correctly in
//!         >= 8/9 of the cases where dense does;
//!   G-FR4 multi-needle and variable tracking: subq correct on >= 80% of
//!         the cases where dense is correct.

const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const QModel = llm.qwen3.model;

pub fn main(init: std.process.Init) !void {
    // smp_allocator, not the process arena: the arena never frees, and the
    // two arms' KV caches plus packed copies would coexist (~13GB at 32K).
    const allocator = std.heap.smp_allocator;
    const arena = init.arena.allocator();
    _ = arena;
    const args = try init.minimal.args.toSlice(allocator);
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 3) {
        try stdout.print(
            "usage: zig build eval-subq-freerun -Doptimize=ReleaseFast -- <model.gguf> generate " ++
                "--tokens-u32 FILE --prefill N --gen M --out PREFIX [--calibrate C] [--calib-tol F] [--rebuild R]\n" ++
                "   or: ... <model.gguf> judge --tokens-u32 PROMPT --prefill N --stream FILE [--skip C]\n",
            .{},
        );
        return error.MissingArguments;
    }
    const mode = args[2];
    var tokens_path: ?[]const u8 = null;
    var stream_path: ?[]const u8 = null;
    var out_prefix: ?[]const u8 = null;
    var prefill: usize = 1024;
    var gen: usize = 384;
    var calibrate_n: usize = 32;
    var calib_tol: f32 = 0.025;
    var rebuild: usize = 512;
    var skip: usize = 32;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--tokens-u32")) {
            i += 1;
            tokens_path = args[i];
        } else if (std.mem.eql(u8, arg, "--stream")) {
            i += 1;
            stream_path = args[i];
        } else if (std.mem.eql(u8, arg, "--out")) {
            i += 1;
            out_prefix = args[i];
        } else if (std.mem.eql(u8, arg, "--prefill")) {
            i += 1;
            prefill = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--gen")) {
            i += 1;
            gen = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--calibrate")) {
            i += 1;
            calibrate_n = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--calib-tol")) {
            i += 1;
            calib_tol = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--rebuild")) {
            i += 1;
            rebuild = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--skip")) {
            i += 1;
            skip = try std.fmt.parseInt(usize, args[i], 10);
        } else return error.UnknownArgument;
    }

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);
    defer file.deinit();
    const config = try QModel.Config.fromGguf(&file);
    var model = try QModel.Model.loadGgufFromFile(&ctx, &file, config);
    defer model.deinit();

    const prompt = try loadTokens(init.io, allocator, tokens_path.?, prefill, config.vocab_size);
    defer allocator.free(prompt);
    // If the token file extends past the prefill, the calibration window is
    // teacher-forced on the REAL continuation for both arms (identical true
    // context up to the free-running boundary); otherwise the subq arm
    // follows the dense arm's stream.
    const ext: ?[]usize = loadTokens(init.io, allocator, tokens_path.?, prefill + calibrate_n, config.vocab_size) catch null;
    defer if (ext) |e| allocator.free(e);

    if (std.mem.eql(u8, mode, "judge")) {
        const stream = try loadTokensAll(init.io, allocator, stream_path.?, config.vocab_size);
        defer allocator.free(stream);
        var kv = try model.initCache(&ctx, prefill + stream.len + 8);
        defer kv.deinit();
        var logits = try model.forwardStep(&ctx, &kv, prompt, 0);
        var pos: usize = prefill;
        var nll_sum: f64 = 0;
        var judged: usize = 0;
        for (stream, 0..) |token, s_i| {
            if (s_i >= skip) {
                nll_sum += try tokenNll(&logits, token, allocator);
                judged += 1;
            }
            logits.deinit();
            const one = [_]usize{token};
            logits = try model.forwardStep(&ctx, &kv, &one, pos);
            pos += 1;
        }
        logits.deinit();
        try stdout.print(
            "judge stream={s} judged_tokens={d} mean_nll={d:.4}\n",
            .{ stream_path.?, judged, nll_sum / @as(f64, @floatFromInt(judged)) },
        );
        return;
    }

    if (!std.mem.eql(u8, mode, "generate")) return error.UnknownMode;
    try stdout.print(
        "model={s} prefill={d} gen={d} calibrate={d} tol={d} rebuild={d}\n",
        .{ args[1], prefill, gen, calibrate_n, calib_tol, rebuild },
    );
    try stdout.flush();

    var streams: [2][]usize = undefined;
    streams[0] = try allocator.alloc(usize, gen);
    streams[1] = try allocator.alloc(usize, gen);

    for (0..2) |arm| {
        var kv = try model.initCache(&ctx, prefill + gen + 8);
        defer kv.deinit();
        var sq = try llm.research.subq.State.init(
            allocator,
            config.num_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            .{ .rebuild_interval = rebuild },
        );
        defer sq.deinit();

        var logits = try model.forwardStep(&ctx, &kv, prompt, 0);
        // Installed AFTER the dense prefill (both arms prefill identically):
        // the subq arm decodes through the research seam, the dense arm
        // runs the stock kernels with the seam cleared.
        model.attention_override = if (arm == 1) llm.research.subq.attentionOverride(&sq) else null;
        defer model.attention_override = null;
        var pos: usize = prefill;
        if (arm == 1 and calibrate_n > 0) try sq.startCalibration(calib_tol);
        for (0..gen) |step| {
            if (arm == 1 and calibrate_n > 0 and step == calibrate_n) sq.finishCalibration();
            const next = try argmax(&logits, allocator);
            streams[arm][step] = next;
            // Calibration window: the subq arm follows the dense stream
            // (teacher-forced) so both arms share an identical context up to
            // the calibration boundary (subq outputs are exact there); past
            // it, each arm feeds its own token — true free-running.
            const feed = if (step < calibrate_n)
                (if (ext) |e| e[prefill + step] else if (arm == 1) streams[0][step] else next)
            else
                next;
            logits.deinit();
            const one = [_]usize{feed};
            logits = try model.forwardStep(&ctx, &kv, &one, pos);
            pos += 1;
        }
        logits.deinit();

        const name = try std.fmt.allocPrint(allocator, "{s}.{s}.u32", .{ out_prefix.?, if (arm == 0) "dense" else "subq" });
        defer allocator.free(name);
        const dumped = try allocator.alloc(u32, gen);
        defer allocator.free(dumped);
        for (dumped, streams[arm]) |*dst, src| dst.* = @intCast(src);
        var out_file = try std.Io.Dir.cwd().createFile(init.io, name, .{});
        defer out_file.close(init.io);
        try out_file.writeStreamingAll(init.io, std.mem.sliceAsBytes(dumped));

        const seg = streams[arm][calibrate_n..];
        const metrics = loopMetrics(seg);
        try stdout.print(
            "{s}: gen={d} judged_segment={d} longest_periodic_run={d} period={d} loop32={}\n",
            .{ if (arm == 0) "dense" else "subq ", gen, seg.len, metrics.run, metrics.period, metrics.run >= 32 },
        );
        try stdout.flush();
    }

    var agree: usize = 0;
    for (streams[0][calibrate_n..], streams[1][calibrate_n..]) |a, b| {
        if (a == b) agree += 1;
    }
    try stdout.print(
        "free-running token overlap after calibration: {d}/{d} (divergence expected; loops and judge NLL are the gates)\n",
        .{ agree, gen - calibrate_n },
    );
}

const LoopMetrics = struct { run: usize, period: usize };

/// Longest consecutive run where the sequence repeats with some period
/// p <= 64: the native repetition-collapse detector.
fn loopMetrics(seg: []const usize) LoopMetrics {
    var best: LoopMetrics = .{ .run = 0, .period = 0 };
    var p: usize = 1;
    while (p <= 64 and p < seg.len) : (p += 1) {
        var run: usize = 0;
        for (p..seg.len) |i| {
            if (seg[i] == seg[i - p]) {
                run += 1;
                if (run > best.run) best = .{ .run = run, .period = p };
            } else run = 0;
        }
    }
    return best;
}

fn argmax(logits: *const fucina.Tensor(.{ .seq, .vocab }), allocator: std.mem.Allocator) !usize {
    const vocab = logits.dim(.vocab);
    const flat = try allocator.alloc(f32, logits.dim(.seq) * vocab);
    defer allocator.free(flat);
    try logits.copyTo(flat);
    const last = flat[(logits.dim(.seq) - 1) * vocab ..][0..vocab];
    var best: usize = 0;
    for (last, 0..) |value, vi| {
        if (value > last[best]) best = vi;
    }
    return best;
}

fn loadTokens(io: std.Io, allocator: std.mem.Allocator, path: []const u8, count: usize, vocab: usize) ![]usize {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(raw);
    if (raw.len / 4 < count) return error.NotEnoughTokens;
    const out = try allocator.alloc(usize, count);
    for (out, 0..) |*t, ti| {
        const v = std.mem.readInt(u32, raw[ti * 4 ..][0..4], .little);
        if (v >= vocab) return error.InvalidTokenId;
        t.* = v;
    }
    return out;
}

fn loadTokensAll(io: std.Io, allocator: std.mem.Allocator, path: []const u8, vocab: usize) ![]usize {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(raw);
    const out = try allocator.alloc(usize, raw.len / 4);
    for (out, 0..) |*t, ti| {
        const v = std.mem.readInt(u32, raw[ti * 4 ..][0..4], .little);
        if (v >= vocab) return error.InvalidTokenId;
        t.* = v;
    }
    return out;
}

fn tokenNll(logits: *const fucina.Tensor(.{ .seq, .vocab }), target: usize, allocator: std.mem.Allocator) !f64 {
    const vocab = logits.dim(.vocab);
    const flat = try allocator.alloc(f32, logits.dim(.seq) * vocab);
    defer allocator.free(flat);
    try logits.copyTo(flat);
    const last = flat[(logits.dim(.seq) - 1) * vocab ..][0..vocab];
    var maxv: f32 = last[0];
    for (last) |v| maxv = @max(maxv, v);
    var lse: f64 = 0;
    for (last) |v| lse += @exp(@as(f64, v - maxv));
    return @as(f64, maxv) + @log(lse) - @as(f64, last[target]);
}
