//! Qwen3 PTQTP decoration harness: load a GGUF (any source dtype Fucina can
//! decode — f16, Q4_K, Q8_0, ... — proving the method is source-agnostic),
//! measure teacher-forced NLL/perplexity on a text file BEFORE decoration,
//! decorate every eligible layer linear with dual trit-planes in place
//! (`Model.decoratePtqtp` -> `LinearWeight.toPtqtp`; originals are dropped),
//! measure NLL again on the SAME tokens, and finish with a greedy completion
//! plus decode timing. One process, one model load, direct deltas.
//!
//! The NLL pass runs `forwardStepAllLogits` (the speculative-verify seam:
//! full-position logits at prefill speed) in 128-token chunks through the
//! deployed inference path — the decorated forward is two stock TQ2_0
//! matmuls + add per linear. Embeddings, lm_head, and norms stay in their
//! source precision (the paper quantizes linear projections only).
//!
//! `--save FILE` persists the decorated model as a GGUF (one standalone
//! TQ2_0 tensor per trit-plane, everything else byte-verbatim; docs/PTQTP.md)
//! so the ~90 s 1.7B K=3 solve runs once: the saved file loads through the
//! ordinary qwen3 loaders — this example, the chat CLI, speculation — with
//! plane pair-detection, no re-decoration. Loading a saved file here with
//! `--planes 0 --nll FILE` must reproduce the decorated "nll after" exactly.
//!
//! Examples:
//!   zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf --nll refs/wiki.txt
//!   zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_M.gguf --nll refs/wiki.txt
//!   zig build ptqtp-qwen3 -- models/Qwen3-0.6B-f16.gguf --planes 0   (undecorated baseline only)
//!   zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-1.7B-BF16.gguf --planes 3 --save models/qwen3-1.7b-ptqtp-k3.gguf
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const ExecContext = fucina.ExecContext;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print(
            "usage: zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- <model.gguf> " ++
                "[--planes 0|1|2|3] [--nll FILE] [--nll-tokens N] [--down-planes N] [--o-planes N] [--head-planes N] " ++
                "[--skip-first N] [--skip-last N] [--save FILE] [--prompt TEXT] [--max-new N]\n",
            .{},
        );
        return error.MissingModelPath;
    }

    var planes: usize = 2;
    var nll_path: ?[]const u8 = null;
    var nll_tokens: usize = 512;
    var skip_first: usize = 0;
    var skip_last: usize = 0;
    var down_planes: ?u8 = null;
    var o_planes: ?u8 = null;
    var head_planes: ?u8 = null;
    var save_path: ?[]const u8 = null;
    var prompt_text: []const u8 = "The capital of Italy is";
    var max_new: usize = 48;

    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        if (argValue(args, &arg_i, "--planes")) |v| {
            planes = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--nll")) |v| {
            nll_path = v;
        } else if (argValue(args, &arg_i, "--nll-tokens")) |v| {
            nll_tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--down-planes")) |v| {
            down_planes = try std.fmt.parseInt(u8, v, 10);
        } else if (argValue(args, &arg_i, "--o-planes")) |v| {
            o_planes = try std.fmt.parseInt(u8, v, 10);
        } else if (argValue(args, &arg_i, "--head-planes")) |v| {
            head_planes = try std.fmt.parseInt(u8, v, 10);
        } else if (argValue(args, &arg_i, "--skip-first")) |v| {
            skip_first = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--skip-last")) |v| {
            skip_last = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &arg_i, "--save")) |v| {
            save_path = v;
        } else if (argValue(args, &arg_i, "--prompt")) |v| {
            prompt_text = v;
        } else if (argValue(args, &arg_i, "--max-new")) |v| {
            max_new = try std.fmt.parseInt(usize, v, 10);
        } else {
            try stdout.print("unknown argument: {s}\n", .{args[arg_i]});
            return error.UnknownArgument;
        }
    }
    if (planes > 3) return error.InvalidPlaneCount;

    const allocator = std.heap.smp_allocator;
    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = nowNs(io);
    var file = try fucina.gguf.File.loadMmap(allocator, init.io, args[1]);
    defer file.deinit();
    var model = try llm.qwen3.model.Model.loadGgufFromFile(&ctx, &file, try llm.qwen3.model.Config.fromGguf(&file));
    defer model.deinit();
    var tokenizer = try llm.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();
    try stdout.print("loaded {s}: {d} layers, hidden {d}, vocab {d} ({d:.2} s)\n", .{
        args[1],                     model.config.num_layers,
        model.config.hidden_size,    tokenizer.vocabSize(),
        seconds(nowNs(io) - load_start),
    });
    try stdout.flush();

    // Teacher-forced tokens for the before/after NLL comparison.
    var eval_tokens: ?[]usize = null;
    defer if (eval_tokens) |t| allocator.free(t);
    if (nll_path) |path| {
        const ids = try tokenizeTextFile(io, allocator, &tokenizer, path);
        if (ids.len < 2) {
            allocator.free(ids);
            return error.NllTextTooShort;
        }
        const keep = @min(ids.len, nll_tokens + 1); // n inputs supervise n-1 targets
        eval_tokens = try allocator.realloc(ids, keep);
    }

    if (eval_tokens) |tokens| {
        const before = try nllOverTokens(&ctx, &model, tokens);
        try stdout.print("nll before: {d:.4} (ppl {d:.2}) over {d} supervised tokens\n", .{
            before.nll, @exp(before.nll), before.count,
        });
        try stdout.flush();
    }

    if (head_planes) |hp| {
        // Decorating `model.output` replaces only the HEAD matmul:
        // on tied models (small Qwen3s share the embedding table) the
        // clone-view is swapped for planes while the embedding keeps
        // its own reference, so token lookups stay in source
        // precision. ISA trade-off (docs/PTQTP.md): the ternary GEMV
        // is ALU-bound, so a K=3 head can cost more than a bf16 AMX
        // head on ARM while winning on VNNI hardware and on memory.
        // Independent of --planes: `--planes 0 --head-planes N` is a
        // head-only decoration.
        var report_head = llm.weights.PtqtpReport{};
        try llm.weights.decoratePtqtpInto(&model.output, &ctx, .{ .planes = hp }, &report_head);
        try stdout.print("head decorated at {d} planes (rel err {d:.4})\n", .{ hp, report_head.rmsRelErr() });
        try stdout.flush();
    }

    if (planes > 0) {
        const q_start = nowNs(io);
        const report = try model.decoratePtqtp(&ctx, .{
            .solver = .{ .planes = @intCast(planes) },
            .skip_first_layers = skip_first,
            .skip_last_layers = skip_last,
            .down_planes = down_planes,
            .o_planes = o_planes,
        });
        const plane_mib = @as(f64, @floatFromInt(report.plane_weights)) * 2.0625 / 8.0 / (1024.0 * 1024.0);
        try stdout.print(
            "decorated {d} linears ({d} skipped, {d} layers left in source precision) in {d:.2} s: " ++
                "{d:.1}M weights -> {d:.1} MiB packed ({d:.2} planes/weight), " ++
                "rms rel err {d:.4} (worst {d:.4}), unconverged groups {d}/{d}\n",
            .{
                report.decorated,                report.skipped,
                report.skipped_layers,
                seconds(nowNs(io) - q_start),
                @as(f64, @floatFromInt(report.elements)) / 1e6,
                plane_mib,
                @as(f64, @floatFromInt(report.plane_weights)) / @as(f64, @floatFromInt(@max(report.elements, 1))),
                report.rmsRelErr(),              report.worst_rel_err,
                report.unconverged_groups,       report.group_count,
            },
        );
        try stdout.flush();
    }

    if (planes > 0 or head_planes != null) {
        if (eval_tokens) |tokens| {
            const after = try nllOverTokens(&ctx, &model, tokens);
            try stdout.print("nll after:  {d:.4} (ppl {d:.2}) over {d} supervised tokens\n", .{
                after.nll, @exp(after.nll), after.count,
            });
            try stdout.flush();
        }
    }

    if (save_path) |path| {
        const save_start = nowNs(io);
        const saved = try model.savePtqtpGguf(&ctx, io, &file, path);
        try stdout.print(
            "saved {s}: {d} decorated tensors -> {d} planes, {d} passed through, {d} appended ({d:.2} s)\n",
            .{ path, saved.decorated, saved.planes, saved.passthrough, saved.appended, seconds(nowNs(io) - save_start) },
        );
        try stdout.flush();
    }

    // Greedy completion + decode timing through the (possibly decorated) model.
    const prompt_ids32 = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(prompt_ids32);
    if (prompt_ids32.len == 0) return error.EmptyPrompt;
    const prompt_ids = try allocator.alloc(usize, prompt_ids32.len);
    defer allocator.free(prompt_ids);
    for (prompt_ids, prompt_ids32) |*d, s| d.* = s;

    var kv = try model.initKvCache(&ctx, prompt_ids.len + max_new);
    defer kv.deinit();

    const prefill_start = nowNs(io);
    var logits = try model.forwardStep(&ctx, &kv, prompt_ids, 0);
    const prefill_ns = nowNs(io) - prefill_start;

    var out_ids: std.ArrayList(u32) = .empty;
    defer out_ids.deinit(allocator);
    var last_id = try argmaxId(&ctx, &logits);
    logits.deinit();
    try out_ids.append(allocator, @intCast(last_id));

    const decode_start = nowNs(io);
    var produced: usize = 1;
    while (produced < max_new) : (produced += 1) {
        if (tokenizer.eosId()) |eos| if (last_id == eos) break;
        var fresh = try model.forwardStep(&ctx, &kv, &.{last_id}, kv.len);
        last_id = try argmaxId(&ctx, &fresh);
        fresh.deinit();
        try out_ids.append(allocator, @intCast(last_id));
    }
    const decode_ns = nowNs(io) - decode_start;

    const text = try tokenizer.decode(allocator, out_ids.items);
    defer allocator.free(text);
    try stdout.print("\nprompt: {s}\ncompletion: {s}\n", .{ prompt_text, text });
    try stdout.print("prefill {d:.1} ms ({d} tokens); decode {d:.2} tok/s ({d} steps)\n", .{
        millis(prefill_ns),
        prompt_ids.len,
        @as(f64, @floatFromInt(produced - 1)) / seconds(decode_ns),
        produced - 1,
    });
}

const NllResult = struct { nll: f64, count: usize };

/// Mean teacher-forced negative log-likelihood of tokens[1..] given their
/// prefixes, computed from full-position logits in 128-token chunks (one KV
/// pass over the sequence — prefill speed, deployed forward path).
fn nllOverTokens(ctx: *ExecContext, model: *llm.qwen3.model.Model, tokens: []const usize) !NllResult {
    var kv = try model.initKvCache(ctx, tokens.len);
    defer kv.deinit();
    const chunk_len: usize = 128;
    var total: f64 = 0;
    var count: usize = 0;
    var pos: usize = 0;
    while (pos < tokens.len) : (pos += chunk_len) {
        const chunk = @min(chunk_len, tokens.len - pos);
        var logits = try model.forwardStepAllLogits(ctx, &kv, tokens[pos..][0..chunk], pos);
        defer logits.deinit();
        const data = try logits.dataConst();
        const vocab = logits.dim(.vocab);
        for (0..chunk) |j| {
            if (pos + j + 1 >= tokens.len) break;
            const target = tokens[pos + j + 1];
            const row = data[j * vocab ..][0..vocab];
            var max_logit: f32 = row[0];
            for (row) |v| max_logit = @max(max_logit, v);
            var sum_exp: f64 = 0;
            for (row) |v| sum_exp += @exp(@as(f64, v - max_logit));
            total += @as(f64, max_logit) + @log(sum_exp) - @as(f64, row[target]);
            count += 1;
        }
    }
    return .{ .nll = total / @as(f64, @floatFromInt(count)), .count = count };
}

fn argmaxId(ctx: *ExecContext, logits: *const fucina.Tensor(.{ .seq, .vocab })) !usize {
    var pred = try logits.argmax(ctx, .vocab);
    defer pred.deinit();
    const values = try pred.dataConst();
    return @intFromFloat(values[values.len - 1]);
}

fn tokenizeTextFile(io: std.Io, allocator: std.mem.Allocator, tok: *const llm.tokenizer.Tokenizer, path: []const u8) ![]usize {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    const max_bytes = 16 * 1024 * 1024;
    if (stat.size > max_bytes) return error.FileTooLarge;
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    const ids32 = try tok.encodeRaw(allocator, bytes);
    defer allocator.free(ids32);
    const ids = try allocator.alloc(usize, ids32.len);
    errdefer allocator.free(ids);
    for (ids, ids32) |*d, s| d.* = s;
    return ids;
}

fn argValue(args: []const []const u8, arg_i: *usize, comptime flag: []const u8) ?[]const u8 {
    const arg = args[arg_i.*];
    if (std.mem.startsWith(u8, arg, flag ++ "=")) return arg[flag.len + 1 ..];
    if (std.mem.eql(u8, arg, flag)) {
        if (arg_i.* + 1 >= args.len) return null;
        arg_i.* += 1;
        return args[arg_i.*];
    }
    return null;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn millis(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}
