//! GLM-4.5 family runner: greedy completion with optional native MTP
//! (multi-token prediction) speculative decoding — the model's own nextn
//! layer drafts tokens, one batched trunk step verifies them, and only
//! greedy-matching prefixes commit (lossless).
//!   zig build glm4moe -- <model-part1.gguf> --prompt "..." --gen 64 \
//!     [--mtp[=depth]] [--moe-stream --moe-cache-mb=N]
const std = @import("std");
const fucina = @import("fucina");
const models = @import("fucina_models");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print("usage: zig build glm4moe -- <model.gguf> --prompt \"...\" [--gen N] [--mtp[=depth]] [--moe-stream --moe-cache-mb=N]\n", .{});
        return;
    }

    var prompt_text: []const u8 = "The capital of France is";
    var gen_count: usize = 32;
    var moe_cli: models.moe_stream_cli.MoeStreamCli = .{};
    var moe_cache_route = false;
    var moe_route_j: usize = 2;
    var moe_route_m: usize = 12;
    var mtp_depth: usize = 0; // 0 = plain decode
    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--prompt")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPrompt;
            prompt_text = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            prompt_text = arg["--prompt=".len..];
        } else if (std.mem.eql(u8, arg, "--gen")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingGenCount;
            gen_count = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--gen=")) {
            gen_count = try std.fmt.parseInt(usize, arg["--gen=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--mtp")) {
            mtp_depth = 2;
        } else if (std.mem.startsWith(u8, arg, "--mtp=")) {
            // Depth caps at 8: the verify runs kernel-pinned
            // (ExecContext.pinRowwiseKernels), so its batched quant
            // kernels reproduce the S=1 numerics bitwise, and 8 keeps the
            // verify batch (depth+1 rows) under the non-quant kernel
            // thresholds (f32/f16 fused-FFN at m >= 12, tiled attention
            // at seq >= 48) that bound losslessness.
            mtp_depth = @min(try std.fmt.parseInt(usize, arg["--mtp=".len..], 10), 8);
        } else if (try moe_cli.tryParse(arg)) {
            // Shared streamed-experts flags (models.moe_stream_cli.MoeStreamCli).
        } else if (std.mem.eql(u8, arg, "--moe-cache-route")) {
            // Cache-aware near-tie routing (QUALITY-AFFECTING, opt-in):
            // prefer already-resident experts among the top-M ranks.
            moe_cli.armed = true;
            moe_cache_route = true;
        } else if (std.mem.startsWith(u8, arg, "--moe-route-j=")) {
            // Sacred true-top ranks always taken (default 2).
            moe_route_j = try std.fmt.parseInt(usize, arg["--moe-route-j=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--moe-route-m=")) {
            // Max-rank window for the resident-preferring fill (default 12).
            moe_route_m = try std.fmt.parseInt(usize, arg["--moe-route-m=".len..], 10);
        } else {
            try stdout.print("unknown flag: {s}\n", .{arg});
            return error.UnknownArgument;
        }
    }

    const allocator = std.heap.smp_allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var file = try fucina.gguf.File.loadMmapAuto(allocator, init.io, args[1]);
    var tokenizer = try models.text.tokenizer.Tokenizer.initFromGguf(allocator, &file, .{});
    defer tokenizer.deinit();

    const capacity: usize = 2048;
    var moe_stream = try moe_cli.options(args[1]);
    if (moe_stream) |*m| {
        m.cache_route = moe_cache_route;
        m.route_sacred = moe_route_j;
        m.route_window = moe_route_m;
    }
    const load_options: models.glm4moe.model.Model.LoadOptions = if (moe_stream) |m| .{ .moe_stream = m } else .{};
    var model = try models.glm4moe.model.Model.loadGgufFromFileOptions(&ctx, &file, capacity, load_options);
    defer model.deinit();
    // The stats go through the SAME buffered stdout writer as everything
    // else: stdout's positional writes and stderr's offset-advancing writes
    // cannot safely share one redirected file (`cmd > f 2>&1` interleaves
    // destructively), so a std.debug stats line would get overwritten.
    defer if (model.expert_store) |store| models.moe_stream_cli.reportAndSaveMoeStream(store, true, stdout);
    const bos: ?u32 = tokenizer.bosId();
    const eos = tokenizer.eosId();
    file.deinit();
    try stdout.print("load: {d:.3} s\n", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - load_start)) / 1e9});
    if (mtp_depth > 0 and model.mtp == null) {
        try stdout.print("model has no nextn (MTP) layer; --mtp ignored\n", .{});
        mtp_depth = 0;
    }

    const ids32 = try tokenizer.encode(allocator, prompt_text);
    defer allocator.free(ids32);
    var tokens: std.ArrayList(usize) = .empty;
    defer tokens.deinit(allocator);
    if (bos) |b| try tokens.append(allocator, b);
    // GLM canonical opening: [gMASK]<sop> before content.
    if (bos != null and bos.? == 151331) try tokens.append(allocator, 151333);
    for (ids32) |id| try tokens.append(allocator, id);
    try stdout.print("prompt tokens: {d}\n", .{tokens.items.len});

    var cache = try model.initCache(&ctx, capacity);
    defer cache.deinit();

    const Model = models.glm4moe.model.Model;
    const vocab = model.config.vocab_size;
    // MTP self-speculation rides the shared verify loop: the family's
    // nextn head drafts through `MtpDraftSource` and
    // `SpeculativeDecoder` verifies/commits/rewinds.
    var draft_src: ?models.text.speculative.mtp.MtpDraftSource(Model) = null;
    defer if (draft_src) |*d| d.deinit();
    var spec: ?models.text.speculative.core.SpeculativeDecoder(Model) = null;
    defer if (spec) |*d| d.deinit();
    if (mtp_depth > 0) {
        draft_src = try models.text.speculative.mtp.MtpDraftSource(Model).init(&ctx, &model, capacity, mtp_depth);
        spec = try models.text.speculative.core.SpeculativeDecoder(Model).init(allocator, draft_src.?.source(), .{
            .max_draft = mtp_depth,
            .min_draft = 1,
            .stop_token = if (eos) |e| @as(usize, e) else null,
            // This runner always speculates at full depth (the cost gate
            // and the acceptance-adaptive budget stay out of the way).
            .min_speedup = 0,
            .adapt_budget = false,
        });
    }

    // Prefill (one batched step) and the first greedy token.
    const prefill_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    var next_token: usize = undefined;
    {
        var rows = try model.step(&ctx, &cache, tokens.items);
        defer rows.deinit();
        const flat = try rows.dataConst();
        next_token = argmax(flat[(tokens.items.len - 1) * vocab ..][0..vocab]);
    }
    if (draft_src) |*d| try d.observePrefill();
    try stdout.print("prefill: {d:.1} ms ({d} tokens, one batched step)\n", .{ @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - prefill_start)) / 1e6, tokens.items.len });

    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(allocator);
    var produced: usize = 0;
    var forwards: usize = 0;

    const decode_start = std.Io.Clock.awake.now(init.io).nanoseconds;
    if (mtp_depth == 0) {
        decode: while (produced < gen_count) {
            if (eos != null and next_token == eos.?) break :decode;
            try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
            try tokens.append(allocator, next_token);
            produced += 1;
            var one = [_]usize{next_token};
            var rows = try model.step(&ctx, &cache, &one);
            defer rows.deinit();
            forwards += 1;
            next_token = argmax((try rows.dataConst())[0..vocab]);
        }
    } else {
        // Decoder invariant: history holds every committed token and its
        // LAST element is not yet forwarded. Commit the prefill's first
        // greedy token by hand, then let the verify loop drive.
        var stopped = eos != null and next_token == eos.?;
        if (!stopped) {
            try tokenizer.decodeAppend(allocator, @intCast(next_token), &reply);
            try tokens.append(allocator, next_token);
            produced += 1;
        }
        var sampler = models.text.sampler.Sampler.init(.{}); // greedy
        var emit = ReplyEmit{ .allocator = allocator, .tokenizer = &tokenizer, .reply = &reply, .eos = eos };
        const sink = models.text.speculative.core.TokenSink{ .ptr = &emit, .func = ReplyEmit.emit };
        while (!stopped and produced < gen_count) {
            // Never draft past the remaining budget (the committed count
            // per step is at most max_draft + 1; a zero budget falls back
            // to a plain step).
            spec.?.options.max_draft = @min(mtp_depth, gen_count - produced -| 1);
            const committed = try spec.?.step(&ctx, &model, &cache, &sampler, &tokens, sink);
            produced += committed - @intFromBool(emit.saw_eos);
            stopped = emit.saw_eos;
        }
        // A committed stop token stays in history for the decoder; drop it
        // from the printout (the plain path never appends it).
        if (emit.saw_eos) _ = tokens.pop();
        forwards = spec.?.stats.spec_steps + spec.?.stats.fallback_steps + spec.?.stats.disabled_steps;
    }
    const decode_ns = std.Io.Clock.awake.now(init.io).nanoseconds - decode_start;
    try stdout.print("decode: {d} tokens in {d} forwards, {d:.1} ms, {d:.2} tok/s ({d:.2} tok/forward)\n", .{ produced, forwards, @as(f64, @floatFromInt(decode_ns)) / 1e6, @as(f64, @floatFromInt(produced)) * 1e9 / @as(f64, @floatFromInt(decode_ns)), @as(f64, @floatFromInt(produced)) / @as(f64, @floatFromInt(@max(forwards, 1))) });
    if (spec) |*d| {
        if (d.stats.drafted > 0) {
            try stdout.print("mtp: {d}/{d} drafts accepted ({d:.1}%)\n", .{ d.stats.accepted, d.stats.drafted, @as(f64, @floatFromInt(d.stats.accepted)) * 100.0 / @as(f64, @floatFromInt(d.stats.drafted)) });
        }
    }
    try stdout.print("generated ids:", .{});
    for (tokens.items[tokens.items.len - produced ..]) |t| try stdout.print(" {d}", .{t});
    try stdout.print("\nprompt: {s}\ntext:  {s}\n", .{ prompt_text, reply.items });
}

/// The decoder's token sink: decode committed tokens into the reply
/// buffer; a committed stop token is recorded, not decoded.
const ReplyEmit = struct {
    allocator: std.mem.Allocator,
    tokenizer: *const models.text.tokenizer.Tokenizer,
    reply: *std.ArrayList(u8),
    eos: ?u32,
    saw_eos: bool = false,

    fn emit(ptr: *anyopaque, token: usize) anyerror!void {
        const self: *ReplyEmit = @ptrCast(@alignCast(ptr));
        if (self.eos) |e| if (token == e) {
            self.saw_eos = true;
            return;
        };
        try self.tokenizer.decodeAppend(self.allocator, @intCast(token), self.reply);
    }
};

fn argmax(logits: []const f32) usize {
    var best: usize = 0;
    var best_v: f32 = -std.math.inf(f32);
    for (logits, 0..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    return best;
}
