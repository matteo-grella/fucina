//! Decode benchmark: dense vs SubQ attention on a Qwen3 GGUF (f16 KV).
//!
//! Prefills `--prefill` tokens densely from a `--tokens-u32` dump, then runs
//! `--decode` greedy steps twice — once through the ordinary dense decode
//! path and once through `forwardStepSubq` — reporting wall-clock tok/s and
//! the greedy-token agreement between the two runs. Research measurement
//! tool; per-head thresholds load from the Gate C calibration JSON.

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

    if (args.len < 2) {
        try stdout.print(
            "usage: zig build bench-subq -Doptimize=ReleaseFast -- <model.gguf> " ++
                "--tokens-u32 FILE --prefill N --decode M [--taus PATH] [--tau F] [--rebuild N]\n",
            .{},
        );
        return error.MissingModelPath;
    }
    var tokens_path: ?[]const u8 = null;
    var taus_path: ?[]const u8 = null;
    var prefill: usize = 1024;
    var decode: usize = 64;
    var tau_default: f32 = 0.05;
    var rebuild: usize = 128;
    var cluster: usize = 128;
    var rank: usize = 2;
    var calibrate_n: usize = 0;
    var calib_tol: f32 = 0.025;
    var serial: bool = false;
    var hier: bool = false;
    var packed_q8: bool = false;
    var packed_f16: bool = false;
    var save_dense_ref: ?[]const u8 = null;
    var dense_ref: ?[]const u8 = null;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--tokens-u32")) {
            i += 1;
            tokens_path = args[i];
        } else if (std.mem.eql(u8, arg, "--taus")) {
            i += 1;
            taus_path = args[i];
        } else if (std.mem.eql(u8, arg, "--prefill")) {
            i += 1;
            prefill = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--decode")) {
            i += 1;
            decode = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--tau")) {
            i += 1;
            tau_default = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--rebuild")) {
            i += 1;
            rebuild = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--cluster")) {
            i += 1;
            cluster = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--rank")) {
            i += 1;
            rank = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--calibrate")) {
            i += 1;
            calibrate_n = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--calib-tol")) {
            i += 1;
            calib_tol = try std.fmt.parseFloat(f32, args[i]);
        } else if (std.mem.eql(u8, arg, "--packed-q8")) {
            packed_q8 = true;
        } else if (std.mem.eql(u8, arg, "--packed-f16")) {
            packed_f16 = true;
        } else if (std.mem.eql(u8, arg, "--serial")) {
            serial = true;
        } else if (std.mem.eql(u8, arg, "--hier")) {
            hier = true;
        } else if (std.mem.eql(u8, arg, "--save-dense-ref")) {
            i += 1;
            save_dense_ref = args[i];
        } else if (std.mem.eql(u8, arg, "--dense-ref")) {
            i += 1;
            dense_ref = args[i];
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

    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, tokens_path.?, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(raw);
    const total_tokens = raw.len / @sizeOf(u32);
    if (total_tokens < prefill) return error.NotEnoughTokens;
    const prompt = try allocator.alloc(usize, prefill);
    defer allocator.free(prompt);
    for (prompt, 0..) |*t, ti| {
        t.* = std.mem.readInt(u32, raw[ti * 4 ..][0..4], .little);
    }

    try stdout.print(
        "model={s} prefill={d} decode={d} tau_default={d} taus={s} rebuild={d} cluster={d} rank={d}\n",
        .{ args[1], prefill, decode, tau_default, taus_path orelse "(default)", rebuild, cluster, rank },
    );
    try stdout.flush();

    var generated_dense = try allocator.alloc(usize, decode);
    var generated_subq = try allocator.alloc(usize, decode);
    var dense_ns: u64 = 0;
    var subq_ns: u64 = 0;

    // Dense-reference persistence: iteration gates re-measure only the subq
    // arm against a stored dense token stream and timing.
    var mode_start: usize = 0;
    if (dense_ref) |path| {
        const ref_raw = try std.Io.Dir.cwd().readFileAlloc(init.io, path, allocator, .limited(64 * 1024 * 1024));
        defer allocator.free(ref_raw);
        const hdr = std.mem.bytesAsSlice(u64, ref_raw[0..16]);
        if (hdr[1] != decode) return error.DenseRefMismatch;
        dense_ns = hdr[0];
        const toks = std.mem.bytesAsSlice(u32, ref_raw[16..]);
        for (generated_dense, 0..) |*g, gi| g.* = toks[gi];
        mode_start = 1;
        try stdout.print("dense (ref): {d} steps = {d:.2} tok/s\n", .{ decode - calibrate_n, @as(f64, @floatFromInt(decode - calibrate_n)) * 1e9 / @as(f64, @floatFromInt(dense_ns)) });
        try stdout.flush();
    }
    for (mode_start..2) |mode| {
        var kv = try model.initKvCache(&ctx, prefill + decode + 8);
        defer kv.deinit();
        var sq = try llm.subq.State.init(
            allocator,
            config.num_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            .{ .tau_default = tau_default, .rebuild_interval = rebuild, .cluster_size = cluster, .rank = rank, .packed_format = if (packed_q8) .q8_0 else if (packed_f16) .f16 else (llm.subq.Config{}).packed_format, .hierarchical = hier },
        );
        defer sq.deinit();
        sq.serial = serial;
        if (mode == 1) try stdout.print("packed format: {s}\n", .{@tagName(sq.config.packed_format)});
        try stdout.flush();
        if (mode == 1) {
            if (taus_path) |path| try sq.loadTausJson(init.io, path);
        }

        var logits = try model.forwardStep(&ctx, &kv, prompt, 0);
        var pos: usize = prefill;
        if (mode == 1 and calibrate_n > 0) try sq.startCalibration(calib_tol);
        var t0 = std.Io.Clock.awake.now(init.io).nanoseconds;
        var first_step_ns: u64 = 0;
        for (0..decode) |step| {
            if (mode == 1 and calibrate_n > 0 and step == calibrate_n) {
                sq.finishCalibration();
                var tau_min: f32 = 1.0;
                var tau_max: f32 = 0.0;
                for (sq.taus) |t| {
                    tau_min = @min(tau_min, t);
                    tau_max = @max(tau_max, t);
                }
                try stdout.print("calibrated taus in [{d:.3}, {d:.3}]\n", .{ tau_min, tau_max });
                try stdout.flush();
            }
            if (step == calibrate_n) t0 = std.Io.Clock.awake.now(init.io).nanoseconds;
            const next = try argmax(&logits, allocator);
            logits.deinit();
            if (mode == 0) generated_dense[step] = next else generated_subq[step] = next;
            // Teacher-forced comparison: the subq arm advances on the DENSE
            // run's token stream so per-step argmax agreement is measurable
            // without trajectory forking.
            const feed = if (mode == 0) next else generated_dense[step];
            const one = [_]usize{feed};
            const step_t0 = std.Io.Clock.awake.now(init.io).nanoseconds;
            logits = if (mode == 0)
                try model.forwardStep(&ctx, &kv, &one, pos)
            else
                try model.forwardStepSubq(&ctx, &kv, &one, pos, &sq);
            if (step == 0) first_step_ns = @intCast(std.Io.Clock.awake.now(init.io).nanoseconds - step_t0);
            pos += 1;
        }
        const elapsed: u64 = @intCast(std.Io.Clock.awake.now(init.io).nanoseconds - t0);
        logits.deinit();
        const timed_steps = decode - calibrate_n;
        _ = timed_steps;
        if (mode == 0) dense_ns = elapsed else subq_ns = elapsed;
        if (mode == 0) {
            if (save_dense_ref) |path| {
                const blob = try allocator.alloc(u8, 16 + generated_dense.len * 4);
                defer allocator.free(blob);
                const hdr: [2]u64 = .{ elapsed, decode };
                @memcpy(blob[0..16], std.mem.sliceAsBytes(hdr[0..]));
                const toks = std.mem.bytesAsSlice(u32, blob[16..]);
                for (generated_dense, 0..) |g, gi| toks[gi] = @intCast(g);
                var out_file = try std.Io.Dir.cwd().createFile(init.io, path, .{});
                defer out_file.close(init.io);
                try out_file.writeStreamingAll(init.io, blob);
            }
        }
        if (mode == 1) {
            try stdout.print("subq rebuilds {d} (first-step storm {d:.3} s)\n", .{ sq.rebuild_count, @as(f64, @floatFromInt(first_step_ns)) / 1e9 });
            const calls = @max(sq.stat_calls, 1);
            try stdout.print("subq per-call rows: opened {d:.1} exact {d:.1} scored {d:.1} (calls {d})\n", .{
                @as(f64, @floatFromInt(sq.stat_opened_rows)) / @as(f64, @floatFromInt(calls)),
                @as(f64, @floatFromInt(sq.stat_exact_rows)) / @as(f64, @floatFromInt(calls)),
                @as(f64, @floatFromInt(sq.stat_scored)) / @as(f64, @floatFromInt(calls)),
                calls,
            });
        }
        try stdout.print(
            "{s}: {d} steps in {d:.3} s = {d:.2} tok/s\n",
            .{
                if (mode == 0) "dense" else "subq ",
                decode - calibrate_n,
                @as(f64, @floatFromInt(elapsed)) / 1e9,
                @as(f64, @floatFromInt(decode - calibrate_n)) * 1e9 / @as(f64, @floatFromInt(elapsed)),
            },
        );
        try stdout.flush();
    }

    var agree: usize = 0;
    for (generated_dense[calibrate_n..], generated_subq[calibrate_n..]) |a, b| {
        if (a == b) agree += 1;
    }
    try stdout.print(
        "greedy agreement {d}/{d} ({d:.1}%)  speedup {d:.3}x\n",
        .{
            agree,
            decode - calibrate_n,
            @as(f64, @floatFromInt(agree)) * 100.0 / @as(f64, @floatFromInt(decode - calibrate_n)),
            @as(f64, @floatFromInt(dense_ns)) / @as(f64, @floatFromInt(subq_ns)),
        },
    );
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
