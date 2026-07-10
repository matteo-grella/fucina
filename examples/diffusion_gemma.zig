//! DiffusionGemma (diffusion-gemma GGUF arch) CPU inference harness.
//!
//! Two jobs:
//!   1. Logit-parity gate vs llama.cpp PR #24423's `llama-diffusion-gemma-eval`:
//!      run ONE bidirectional canvas forward from raw token ids (prompt encoded
//!      causally into the KV cache first) and dump/compare the full
//!      `[canvas_length, vocab]` canvas logits as headerless little-endian f32
//!      (the oracle's exact output format). `--sc-logits P` feeds a previous
//!      step's raw logits file as the self-conditioning input with
//!      temp_inv = 1 (the oracle's optional 5th argument).
//!   2. End-to-end block-diffusion chat: entropy-bound denoising with the
//!      reference sampler defaults, Gemma 4's `<|turn>` chat format, and the
//!      block-autoregressive outer loop.
//!
//!   zig build diffusion-gemma -Doptimize=ReleaseFast -- <model.gguf> --eval 2,651,235 --canvas 0,0,...  [--logits-out P] [--compare-logits P] [--sc-logits P]
//!   zig build diffusion-gemma -Doptimize=ReleaseFast -- <model.gguf> --chat "Why is the sky blue?" [--steps N] [--seed N] [--max N] [--no-sc] [--visual]
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const dg = llm.diffusion_gemma.model;
const Model = dg.Model;
const Config = dg.Config;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) {
        try stdout.print(
            \\usage: zig build diffusion-gemma -- <model.gguf> [options]
            \\  eval:   --eval <prompt-ids> --canvas <canvas-ids> [--logits-out P] [--compare-logits P] [--sc-logits P]
            \\          (canvas length must equal diffusion.canvas_length; logits dumped as raw f32 [C, vocab])
            \\  chat:   --chat "message" [--system "..."] [--think] [--max N] [--steps N] [--seed N]
            \\          [--t-min F] [--t-max F] [--entropy-bound F] [--confidence F] [--stability N] [--no-sc]
            \\          (the reply denoises INLINE like streaming on a TTY; --no-visual disables,
            \\           --visual forces when piped, --visual-interval N redraws every Nth step)
            \\  repl:   --repl [--system "..."] [--think] [...]   multi-turn chat (full-history re-encode per turn)
            \\  gen:    --gen N <comma-separated-token-ids>   raw-token block generation (no template)
            \\  gpu:    --gpu-f16   dense weights resident as f16 -> big canvas GEMMs offload to the GPU (-Dgpu=metal builds; +~4.6 GB)
            \\  experts: --experts=borrow|pack   MoE expert load (CPU builds). borrow maps experts
            \\           zero-copy — near-instant load + ~half memory; pack (default) x4-packs for throughput.
            \\  other:  --info
            \\
        , .{});
        return;
    }

    var prompt_ids_arg: ?[]const u8 = null;
    var canvas_ids_arg: ?[]const u8 = null;
    var eval_flag = false;
    var logits_out: ?[]const u8 = null;
    var compare_logits: ?[]const u8 = null;
    var sc_logits_path: ?[]const u8 = null;
    var chat_text: ?[]const u8 = null;
    var system_text: ?[]const u8 = null;
    var think_flag = false;
    var info_flag = false;
    var visual_flag = false;
    var no_visual = false;
    var repl_flag = false;
    var visual_interval: usize = 1;
    var no_sc = false;
    var gpu_f16 = false;
    var experts_borrow = false; // --experts=borrow: zero-copy MoE load
    var max_resp: usize = 512;
    var gen_count: ?usize = null;
    var raw_tokens_arg: ?[]const u8 = null;
    var seed: u64 = 0;
    var steps_arg: ?usize = null;
    var t_min_arg: ?f32 = null;
    var t_max_arg: ?f32 = null;
    var entropy_bound_arg: ?f32 = null;
    var confidence_arg: ?f32 = null;
    var stability_arg: ?usize = null;

    var arg_i: usize = 2;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--eval")) {
            eval_flag = true;
            prompt_ids_arg = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--canvas")) {
            canvas_ids_arg = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--logits-out")) {
            logits_out = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--compare-logits")) {
            compare_logits = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--sc-logits")) {
            sc_logits_path = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--chat")) {
            chat_text = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--system")) {
            system_text = try argValue(args, &arg_i);
        } else if (std.mem.eql(u8, arg, "--think")) {
            think_flag = true;
        } else if (std.mem.eql(u8, arg, "--info")) {
            info_flag = true;
        } else if (std.mem.eql(u8, arg, "--repl")) {
            repl_flag = true;
        } else if (std.mem.eql(u8, arg, "--no-visual")) {
            no_visual = true;
        } else if (std.mem.eql(u8, arg, "--visual")) {
            visual_flag = true;
        } else if (std.mem.eql(u8, arg, "--visual-interval")) {
            visual_interval = @max(1, try std.fmt.parseInt(usize, try argValue(args, &arg_i), 10));
        } else if (std.mem.eql(u8, arg, "--gpu-f16")) {
            gpu_f16 = true;
        } else if (std.mem.startsWith(u8, arg, "--experts=")) {
            const v = arg["--experts=".len..];
            if (std.mem.eql(u8, v, "borrow")) {
                experts_borrow = true;
            } else if (std.mem.eql(u8, v, "pack")) {
                experts_borrow = false;
            } else return error.InvalidExpertsMode;
        } else if (std.mem.eql(u8, arg, "--no-sc")) {
            no_sc = true;
        } else if (std.mem.eql(u8, arg, "--max")) {
            max_resp = try std.fmt.parseInt(usize, try argValue(args, &arg_i), 10);
        } else if (std.mem.eql(u8, arg, "--gen")) {
            gen_count = try std.fmt.parseInt(usize, try argValue(args, &arg_i), 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            seed = try std.fmt.parseInt(u64, try argValue(args, &arg_i), 10);
        } else if (std.mem.eql(u8, arg, "--steps")) {
            steps_arg = try std.fmt.parseInt(usize, try argValue(args, &arg_i), 10);
        } else if (std.mem.eql(u8, arg, "--t-min")) {
            t_min_arg = try std.fmt.parseFloat(f32, try argValue(args, &arg_i));
        } else if (std.mem.eql(u8, arg, "--t-max")) {
            t_max_arg = try std.fmt.parseFloat(f32, try argValue(args, &arg_i));
        } else if (std.mem.eql(u8, arg, "--entropy-bound")) {
            entropy_bound_arg = try std.fmt.parseFloat(f32, try argValue(args, &arg_i));
        } else if (std.mem.eql(u8, arg, "--confidence")) {
            confidence_arg = try std.fmt.parseFloat(f32, try argValue(args, &arg_i));
        } else if (std.mem.eql(u8, arg, "--stability")) {
            stability_arg = try std.fmt.parseInt(usize, try argValue(args, &arg_i), 10);
        } else if (std.mem.startsWith(u8, arg, "--")) {
            try stdout.print("unknown flag: {s} (see usage)\n", .{arg});
            return error.UnknownArgument;
        } else {
            raw_tokens_arg = arg;
        }
    }
    if (raw_tokens_arg != null and gen_count == null) {
        // A bare token list is only meaningful as the --gen input.
        try stdout.print("token-list argument requires --gen N (see usage)\n", .{});
        return error.UnknownArgument;
    }

    const allocator = std.heap.smp_allocator;

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    const load_start = nowNs(init.io);
    var file = try fucina.gguf.File.loadMmap(allocator, init.io, args[1]);
    var config = try Config.fromGguf(&file);
    config.base.borrow_experts = experts_borrow;
    if (steps_arg) |v| config.eb.max_steps = v;
    if (t_min_arg) |v| config.eb.t_min = v;
    if (t_max_arg) |v| config.eb.t_max = v;
    if (entropy_bound_arg) |v| config.eb.entropy_bound = v;
    if (confidence_arg) |v| config.eb.confidence_threshold = v;
    if (stability_arg) |v| config.eb.stability_threshold = v;

    var spm: ?llm.spm_tokenizer.Tokenizer = llm.spm_tokenizer.Tokenizer.initFromGguf(allocator, &file, .{}) catch null;
    defer if (spm) |*t| t.deinit();
    const tok_ptr: ?*const llm.spm_tokenizer.Tokenizer = if (spm) |*t| t else null;

    if (info_flag) {
        try stdout.print("diffusion-gemma: canvas_length={d} eb: steps={d} t={d:.2}->{d:.2} bound={d:.3} conf={d:.4} stab={d}\n", .{
            config.canvas_length, config.eb.max_steps, config.eb.t_max, config.eb.t_min, config.eb.entropy_bound, config.eb.confidence_threshold, config.eb.stability_threshold,
        });
        try stdout.print("  base: vocab={d} hidden={d} layers={d} experts={d}/{d} window={d} softcap={d:.1}\n", .{
            config.base.vocab_size, config.base.hidden_size, config.base.num_layers, config.base.num_experts_used, config.base.num_experts, config.base.sliding_window, config.base.final_logit_softcapping,
        });
        if (tok_ptr) |t| try stdout.print("tokenizer: SPM, vocab {d}  bos {?d}  eos {?d}\n", .{ t.vocabSize(), t.bosId(), t.eosId() });
        file.deinit();
        return;
    }

    var model = try Model.loadGgufFromFile(&ctx, &file, config);
    defer model.deinit();
    file.deinit();
    if (gpu_f16) {
        // Dense weights -> resident f16 so the canvas GEMMs take the
        // -Dgpu=metal f16 offload (see Model.convertDenseWeightsToF16).
        const conv_start = nowNs(init.io);
        try model.convertDenseWeightsToF16(&ctx);
        try stdout.print("gpu-f16: dense weights resident as f16 ({d:.1} s)\n", .{seconds(nowNs(init.io) - conv_start)});
    }
    const load_ns = nowNs(init.io) - load_start;
    try stdout.print("load: {d:.3} s  canvas_length={d}  sc={s}\n", .{ seconds(load_ns), config.canvas_length, if (model.sc != null) "yes" else "no" });

    if (eval_flag) {
        try runEval(init.io, allocator, stdout, &ctx, &model, prompt_ids_arg.?, canvas_ids_arg orelse return error.MissingCanvasIds, logits_out, compare_logits, sc_logits_path);
        return;
    }

    // The inline denoising view is the chat-streaming equivalent: default ON
    // for chat/REPL on a TTY; --visual forces it elsewhere, --no-visual off.
    const tty = stdoutIsTty();
    const chat_visual = !no_visual and (visual_flag or tty);
    const gen_visual = !no_visual and visual_flag;

    if (repl_flag) {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        try runRepl(init.io, allocator, stdout, &ctx, &model, t, system_text, think_flag, max_resp, .{
            .seed = seed,
            .self_conditioning = !no_sc,
            .visual = chat_visual,
            .visual_interval = visual_interval,
            .verbose = false,
        });
        return;
    }

    if (chat_text) |msg| {
        const t = tok_ptr orelse return error.TokenizerUnavailable;
        try runChat(init.io, allocator, stdout, &ctx, &model, t, msg, system_text, think_flag, max_resp, .{
            .seed = seed,
            .self_conditioning = !no_sc,
            .visual = chat_visual,
            .visual_interval = visual_interval,
            .verbose = false,
        });
        return;
    }

    if (gen_count) |n| {
        const ids_text = raw_tokens_arg orelse return error.MissingTokens;
        var token_buf: [8192]usize = undefined;
        const tokens = try parseTokenList(ids_text, &token_buf);
        try runGen(init.io, allocator, stdout, &ctx, &model, tokens, n, tok_ptr, .{
            .seed = seed,
            .self_conditioning = !no_sc,
            .visual = gen_visual,
            .visual_interval = visual_interval,
            .verbose = true,
        });
        return;
    }

    try stdout.print("nothing to do (see usage)\n", .{});
}

/// Parity harness: encode the prompt causally, run one canvas forward
/// (zero-SC, or SC from a raw-logits file at temp_inv=1), dump/compare the
/// canvas logits. Mirrors `llama-diffusion-gemma-eval`.
fn runEval(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    prompt_arg: []const u8,
    canvas_arg: []const u8,
    logits_out: ?[]const u8,
    compare_path: ?[]const u8,
    sc_logits_path: ?[]const u8,
) !void {
    var prompt_buf: [8192]usize = undefined;
    const prompt = try parseTokenList(prompt_arg, &prompt_buf);
    var canvas_buf: [8192]usize = undefined;
    const canvas = try parseTokenList(canvas_arg, &canvas_buf);
    if (canvas.len != model.config.canvas_length) return error.CanvasLengthMismatch;

    var kv = try model.initKvCache(ctx, prompt.len + canvas.len);
    defer kv.deinit();

    const encode_start = nowNs(io);
    try model.encodeStep(ctx, &kv, prompt, 0);
    const encode_ns = nowNs(io) - encode_start;

    // Optional SC input: a raw [C, vocab] f32 logits file, applied with
    // temp_inv = 1 (the oracle's convention). The sparse candidate lists are
    // built by the sampler pass with a tiny p_min so the approximation error
    // is negligible for the parity check.
    var sc_signal: ?dg.ScSignal = null;
    defer if (sc_signal) |*s| s.deinit();
    if (sc_logits_path) |path| {
        const prev = try readF32File(io, allocator, path);
        defer allocator.free(prev);
        if (prev.len != canvas.len * model.config.base.vocab_size) return error.InvalidLogitsFile;
        var prev_tensor = try fucina.Tensor(.{ .seq, .vocab }).fromSlice(ctx, .{ canvas.len, model.config.base.vocab_size }, prev);
        defer prev_tensor.deinit();
        const u = try allocator.alloc(f32, canvas.len);
        defer allocator.free(u);
        @memset(u, 0);
        var pass = try dg.samplerPass(ctx, &prev_tensor, 1.0, u, .{ .sc_p_min = 1e-8, .sc_max_per_row = 2048 });
        defer pass.deinit(allocator);
        sc_signal = pass.sc;
        pass.sc = null;
    }

    const forward_start = nowNs(io);
    var logits = try model.canvasForward(ctx, &kv, canvas, if (sc_signal) |*s| s else null);
    defer logits.deinit();
    const forward_ns = nowNs(io) - forward_start;

    try stdout.print("encode: {d:.3} s ({d} tok)  canvas forward: {d:.3} s ({d} tok, sc={s})\n", .{
        seconds(encode_ns), prompt.len, seconds(forward_ns), canvas.len, if (sc_signal != null) "yes" else "no",
    });

    // Top tokens of the first and last canvas rows (a cheap sanity readout).
    const data = try logits.dataConst();
    const vocab = model.config.base.vocab_size;
    for ([_]usize{ 0, canvas.len - 1 }) |row| {
        var best: usize = 0;
        const r = data[row * vocab ..][0..vocab];
        for (r, 0..) |v, i| {
            if (v > r[best]) best = i;
        }
        try stdout.print("row {d}: top={d} ({d:.4})\n", .{ row, best, r[best] });
    }

    if (logits_out) |path| {
        try writeF32File(io, path, data);
        try stdout.print("logits: {s} ({d} x {d})\n", .{ path, canvas.len, vocab });
    }
    if (compare_path) |path| {
        try compareLogits(io, allocator, stdout, path, data, canvas.len, vocab);
    }
}

// ---------------------------------------------------------------------------
// Inline denoising view — the chat-streaming equivalent for block diffusion.
// The reply renders exactly where it belongs in the conversation transcript;
// each denoising step REPAINTS the in-progress region in place (cursor-up
// over the previous frame, every line cleared to EOL, leftovers wiped with
// ESC[J, the whole repaint a DEC-2026 synchronized update so it cannot
// tear). Tokens the entropy-bound sampler has not accepted this step render
// FAINT and crystallize as the canvas converges; a dim status line trails
// the text while denoising and disappears when the block finalizes.
// Finalized blocks are re-rendered ahead of the live canvas, so multi-block
// replies flow like ordinary streaming, and the last repaint leaves the
// clean reply as normal terminal output — scrollback stays a readable
// transcript, no full-screen viewport. If the region outgrows the window
// the tail stays visible (like scrolled streaming); the full text still
// lands in scrollback on finalize.
// ---------------------------------------------------------------------------

const Winsize = extern struct { row: u16, col: u16, xpixel: u16, ypixel: u16 };
extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;
extern "c" fn isatty(fd: c_int) c_int;

fn stdoutIsTty() bool {
    return isatty(1) == 1;
}

/// Terminal size via TIOCGWINSZ; 24x80 when stdout is not a TTY.
fn terminalSize() struct { rows: usize, cols: usize } {
    const TIOCGWINSZ: c_ulong = switch (@import("builtin").os.tag) {
        .macos, .ios => 0x40087468,
        .linux => 0x5413,
        else => return .{ .rows = 24, .cols = 80 },
    };
    var ws: Winsize = undefined;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 and ws.row > 1) {
        return .{ .rows = ws.row, .cols = if (ws.col > 0) ws.col else 80 };
    }
    return .{ .rows = 24, .cols = 80 };
}

/// Count display columns approximately: one per UTF-8 codepoint (continuation
/// bytes don't advance the cursor; CJK double-width is ignored).
fn visibleWidth(s: []const u8) usize {
    var n: usize = 0;
    for (s) |b| {
        if (b & 0xc0 != 0x80) n += 1;
    }
    return n;
}

/// Wraps (text, faint) segments into terminal lines at piece granularity, so
/// ANSI escapes and UTF-8 glyphs never split mid-sequence.
const LineBuilder = struct {
    a: std.mem.Allocator,
    cols: usize,
    lines: std.ArrayList([]u8) = .empty,
    cur: std.ArrayList(u8) = .empty,
    cur_width: usize = 0,

    fn deinit(self: *LineBuilder) void {
        for (self.lines.items) |l| self.a.free(l);
        self.lines.deinit(self.a);
        self.cur.deinit(self.a);
    }

    fn breakLine(self: *LineBuilder) !void {
        try self.lines.append(self.a, try self.cur.toOwnedSlice(self.a));
        self.cur_width = 0;
    }

    fn add(self: *LineBuilder, seg_text: []const u8, faint: bool) !void {
        var it = std.mem.splitScalar(u8, seg_text, '\n');
        var first = true;
        while (it.next()) |seg| {
            if (!first) try self.breakLine();
            first = false;
            if (seg.len == 0) continue;
            const w = visibleWidth(seg);
            if (self.cur_width + w > self.cols and self.cur_width > 0) try self.breakLine();
            if (faint) try self.cur.appendSlice(self.a, "\x1b[2m");
            try self.cur.appendSlice(self.a, seg);
            if (faint) try self.cur.appendSlice(self.a, "\x1b[22m");
            self.cur_width += w;
        }
    }

    /// Close the trailing partial line (always emits at least one line).
    fn finish(self: *LineBuilder) !void {
        if (self.cur.items.len > 0 or self.lines.items.len == 0) try self.breakLine();
    }
};

const InlineVis = struct {
    stdout: *std.Io.Writer,
    allocator: std.mem.Allocator,
    tok: ?*const llm.spm_tokenizer.Tokenizer,
    enabled: bool,
    interval: usize = 1,
    /// Finalized reply text (kept tokens of all completed blocks) — rendered
    /// ahead of the live canvas each frame; printed permanently at the end.
    final_text: std.ArrayList(u8) = .empty,
    /// Screen lines the live region currently occupies (for cursor-up).
    region_lines: usize = 0,
    failed: bool = false,

    fn deinit(self: *InlineVis) void {
        self.final_text.deinit(self.allocator);
    }
};

fn appendTokenText(vis: *InlineVis, out: *std.ArrayList(u8), id: usize) !void {
    if (vis.tok) |tok| {
        tok.decodeAppend(vis.allocator, @intCast(id), out) catch {};
    } else {
        try out.print(vis.allocator, "{d} ", .{id});
    }
}

fn onStepInline(user: ?*anyopaque, info: *const dg.StepInfo) void {
    const vis: *InlineVis = @ptrCast(@alignCast(user.?));
    if (!vis.enabled or vis.failed) return;
    // All steps are computed; only every Nth is drawn (always the first).
    if (vis.interval > 1 and info.step_index % vis.interval != 0) return;
    renderStepInline(vis, info) catch {
        vis.failed = true;
    };
}

fn renderStepInline(vis: *InlineVis, info: *const dg.StepInfo) !void {
    const a = vis.allocator;
    const term = terminalSize();
    var lb = LineBuilder{ .a = a, .cols = @max(20, term.cols) };
    defer lb.deinit();

    try lb.add(vis.final_text.items, false);
    var piece: std.ArrayList(u8) = .empty;
    defer piece.deinit(a);
    for (info.argmax_canvas, 0..) |id, i| {
        piece.clearRetainingCapacity();
        try appendTokenText(vis, &piece, id);
        try lb.add(piece.items, !info.accepted[i]);
    }
    try lb.finish();

    // Trailing dim status line while the block denoises.
    {
        var status: std.ArrayList(u8) = .empty;
        errdefer status.deinit(a);
        try status.print(a, "\x1b[2m… step {d}/{d} · accepted {d}/{d} · H\u{0304} {d:.3}\x1b[22m", .{
            info.step_index + 1, info.total_steps, info.n_accepted, info.argmax_canvas.len, info.mean_entropy,
        });
        try lb.lines.append(a, try status.toOwnedSlice(a));
    }

    // Tail view when the region outgrows the window (scrolled-streaming feel).
    const budget = @max(4, term.rows - 2);
    const all = lb.lines.items;
    const shown = if (all.len > budget) all[all.len - budget ..] else all;
    try paintRegion(vis, shown);
}

fn paintRegion(vis: *InlineVis, lines: []const []u8) !void {
    const a = vis.allocator;
    var frame: std.ArrayList(u8) = .empty;
    defer frame.deinit(a);
    try frame.appendSlice(a, "\x1b[?2026h");
    if (vis.region_lines > 1) try frame.print(a, "\x1b[{d}A", .{vis.region_lines - 1});
    if (vis.region_lines > 0) try frame.append(a, '\r');
    for (lines, 0..) |ln, i| {
        try frame.appendSlice(a, ln);
        try frame.appendSlice(a, "\x1b[K");
        if (i + 1 < lines.len) try frame.append(a, '\n');
    }
    // Wipe whatever a taller previous frame left below the new last line.
    try frame.appendSlice(a, "\x1b[J\x1b[?2026l");
    vis.region_lines = lines.len;
    try vis.stdout.writeAll(frame.items);
    try vis.stdout.flush();
}

fn wipeRegion(vis: *InlineVis) !void {
    if (vis.region_lines == 0) return;
    if (vis.region_lines > 1) try vis.stdout.print("\x1b[{d}A", .{vis.region_lines - 1});
    try vis.stdout.writeAll("\r\x1b[J");
    vis.region_lines = 0;
}

fn onBlockInline(user: ?*anyopaque, block_index: usize, kept: []const usize, finished: bool) void {
    _ = block_index;
    const vis: *InlineVis = @ptrCast(@alignCast(user.?));
    if (vis.failed) return;
    finalizeBlockInline(vis, kept, finished) catch {
        vis.failed = true;
    };
}

fn finalizeBlockInline(vis: *InlineVis, kept: []const usize, finished: bool) !void {
    // Accumulate the kept text regardless of the live view: non-visual
    // callers print it once at the end.
    for (kept) |id| try appendTokenText(vis, &vis.final_text, id);
    if (!vis.enabled or !finished) return;

    // Last block: replace the live region with the clean reply — it becomes
    // ordinary terminal output, exactly like a finished streaming response.
    const a = vis.allocator;
    const term = terminalSize();
    var lb = LineBuilder{ .a = a, .cols = @max(20, term.cols) };
    defer lb.deinit();
    try lb.add(vis.final_text.items, false);
    try lb.finish();
    try wipeRegion(vis);
    for (lb.lines.items, 0..) |ln, i| {
        try vis.stdout.writeAll(ln);
        if (i + 1 < lb.lines.items.len) try vis.stdout.writeAll("\n");
    }
    try vis.stdout.flush();
}

const TurnUx = struct {
    seed: u64,
    self_conditioning: bool,
    /// Live inline denoising view (chat streaming with diffusion).
    visual: bool,
    visual_interval: usize = 1,
    /// Harness-mode extras: the eb-config banner + the raw token-id dump.
    verbose: bool,
};

/// One generation over `prompt`: runs the block-diffusion loop with the
/// inline view wired in, prints the reply (live repaints when visual; once at
/// the end otherwise) + a dim stats trailer, and optionally appends the reply
/// TEXT to `reply_out` (the REPL's history accumulator).
fn runTurn(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    tok: ?*const llm.spm_tokenizer.Tokenizer,
    prompt: []const usize,
    max_new: usize,
    ux: TurnUx,
    reply_out: ?*std.ArrayList(u8),
) !void {
    const c_len = model.config.canvas_length;
    const blocks = (max_new + c_len - 1) / c_len;
    var kv = try model.initKvCache(ctx, prompt.len + (blocks + 1) * c_len);
    defer kv.deinit();

    const out = try allocator.alloc(usize, max_new);
    defer allocator.free(out);

    if (ux.verbose) {
        try stdout.print("prompt: {d} tok  eb: steps={d} t={d:.2}->{d:.2} bound={d:.3} seed={d} sc={}\n", .{
            prompt.len, model.config.eb.max_steps, model.config.eb.t_max, model.config.eb.t_min, model.config.eb.entropy_bound, ux.seed, ux.self_conditioning,
        });
    }
    try stdout.flush();

    var vis = InlineVis{
        .stdout = stdout,
        .allocator = allocator,
        .tok = tok,
        .enabled = ux.visual,
        .interval = ux.visual_interval,
    };
    defer vis.deinit();

    const start = nowNs(io);
    const result = try dg.generate(model, ctx, &kv, prompt, out, .{
        .denoise = .{
            .eb = model.config.eb,
            .seed = ux.seed,
            .self_conditioning = ux.self_conditioning,
            .on_step = if (ux.visual) onStepInline else null,
            .on_step_user = if (ux.visual) @ptrCast(&vis) else null,
        },
        .max_new_tokens = max_new,
        .on_block = onBlockInline,
        .on_block_user = @ptrCast(&vis),
    });
    const ns = nowNs(io) - start;

    if (ux.visual) {
        // The final repaint left the clean reply on screen; step off it.
        try stdout.writeAll("\n");
    } else {
        // No live view (piped / --no-visual): print the reply once.
        try stdout.writeAll(vis.final_text.items);
        try stdout.writeAll("\n");
    }

    const secs = seconds(ns);
    const per_forward = @as(f64, @floatFromInt(result.produced)) / @as(f64, @floatFromInt(@max(result.steps, 1)));
    try stdout.print("\x1b[2m[{d} tok · {d} block(s) · {d} steps · {d:.1} s · {d:.2} tok/fwd]\x1b[22m\n", .{
        result.produced, result.blocks, result.steps, secs, per_forward,
    });
    if (ux.verbose) {
        for (out[0..result.produced]) |t| try stdout.print("{d} ", .{t});
        try stdout.print("\n", .{});
    }
    try stdout.flush();

    if (reply_out) |r| try r.appendSlice(allocator, vis.final_text.items);
}

/// Append one user turn (+ the model-turn opener) in Gemma 4's chat format
/// (verified against this GGUF's embedded chat_template). `first` emits the
/// <bos> and the optional system turn; with thinking off the opener primes an
/// empty thought channel so the model answers directly.
fn appendUserTurn(
    a: std.mem.Allocator,
    history: *std.ArrayList(u8),
    system_text: ?[]const u8,
    user_msg: []const u8,
    first: bool,
    think: bool,
) !void {
    if (first) {
        try history.appendSlice(a, "<bos>");
        if (think or system_text != null) {
            try history.appendSlice(a, "<|turn>system\n");
            if (think) try history.appendSlice(a, "<|think|>\n");
            if (system_text) |s| try history.appendSlice(a, std.mem.trim(u8, s, " \t\r\n"));
            try history.appendSlice(a, "<turn|>\n");
        }
    }
    try history.appendSlice(a, "<|turn>user\n");
    try history.appendSlice(a, user_msg);
    try history.appendSlice(a, "<turn|>\n<|turn>model\n");
    if (!think) try history.appendSlice(a, "<|channel>thought\n<channel|>");
}

fn encodeHistory(allocator: std.mem.Allocator, tok: *const llm.spm_tokenizer.Tokenizer, text: []const u8) ![]usize {
    const ids32 = try tok.encodeRaw(allocator, text);
    defer allocator.free(ids32);
    const ids = try allocator.alloc(usize, ids32.len);
    for (ids, ids32) |*d, s| d.* = s;
    return ids;
}

fn runChat(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    tok: *const llm.spm_tokenizer.Tokenizer,
    user_msg: []const u8,
    system_text: ?[]const u8,
    think: bool,
    max_resp: usize,
    ux: TurnUx,
) !void {
    var history: std.ArrayList(u8) = .empty;
    defer history.deinit(allocator);
    try appendUserTurn(allocator, &history, system_text, user_msg, true, think);
    const prompt = try encodeHistory(allocator, tok, history.items);
    defer allocator.free(prompt);
    try runTurn(io, allocator, stdout, ctx, model, tok, prompt, max_resp, ux, null);
}

/// Multi-turn REPL. Like llama.cpp's diffusion -cnv, each turn re-templates
/// and re-encodes the FULL conversation (the diffusion KV flow has no
/// incremental turn state), so long histories pay growing prefill time.
fn runRepl(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    tok: *const llm.spm_tokenizer.Tokenizer,
    system_text: ?[]const u8,
    think: bool,
    max_resp: usize,
    ux_base: TurnUx,
) !void {
    var history: std.ArrayList(u8) = .empty;
    defer history.deinit(allocator);
    var turn: usize = 0;

    var stdin_buf: [8192]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_buf);
    const in = &stdin_reader.interface;
    try stdout.print("diffusion-gemma chat{s} — type a message, empty line or Ctrl-D to quit:\n", .{if (think) " (thinking)" else ""});
    while (true) {
        try stdout.writeAll("\n> ");
        try stdout.flush();
        const line = (try in.takeDelimiter('\n')) orelse break;
        const msg = std.mem.trim(u8, line, " \t\r");
        if (msg.len == 0) break;
        try stdout.writeAll("\n");
        try stdout.flush();

        try appendUserTurn(allocator, &history, system_text, msg, turn == 0, think);
        const prompt = try encodeHistory(allocator, tok, history.items);
        defer allocator.free(prompt);

        var reply: std.ArrayList(u8) = .empty;
        defer reply.deinit(allocator);
        var ux = ux_base;
        ux.seed = ux_base.seed +% turn;
        try runTurn(io, allocator, stdout, ctx, model, tok, prompt, max_resp, ux, &reply);

        // Close the model turn in the history so the next turn re-templates
        // the full conversation.
        try history.appendSlice(allocator, reply.items);
        try history.appendSlice(allocator, "<turn|>\n");
        turn += 1;
    }
}

fn runGen(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    ctx: *fucina.ExecContext,
    model: *const Model,
    prompt: []const usize,
    max_new: usize,
    tok: ?*const llm.spm_tokenizer.Tokenizer,
    ux: TurnUx,
) !void {
    return runTurn(io, allocator, stdout, ctx, model, tok, prompt, max_new, ux, null);
}

fn argValue(args: []const []const u8, i: *usize) ![]const u8 {
    i.* += 1;
    if (i.* >= args.len) return error.MissingArgValue;
    return args[i.*];
}

fn parseTokenList(input: []const u8, out: []usize) ![]const usize {
    var len: usize = 0;
    var it = std.mem.splitScalar(u8, input, ',');
    while (it.next()) |part| {
        if (part.len == 0) return error.InvalidTokenList;
        if (len == out.len) return error.TokenListTooLong;
        out[len] = try std.fmt.parseInt(usize, part, 10);
        len += 1;
    }
    if (len == 0) return error.InvalidTokenList;
    return out[0..len];
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
}

fn writeF32File(io: std.Io, path: []const u8, values: []const f32) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buffer);
    defer writer.interface.flush() catch {};
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
}

fn compareLogits(
    io: std.Io,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
    path: []const u8,
    values: []const f32,
    rows: usize,
    vocab: usize,
) !void {
    const reference = try readF32File(io, allocator, path);
    defer allocator.free(reference);
    if (reference.len != values.len) return error.CompareLogitsShapeMismatch;

    var max_abs: f64 = 0;
    var sum_abs: f64 = 0;
    var dot: f64 = 0;
    var norm_a: f64 = 0;
    var norm_b: f64 = 0;
    var top_match: usize = 0;
    for (0..rows) |r| {
        const a = values[r * vocab ..][0..vocab];
        const b = reference[r * vocab ..][0..vocab];
        var top_a: usize = 0;
        var top_b: usize = 0;
        for (a, b, 0..) |va, vb, i| {
            const diff: f64 = @floatCast(va - vb);
            max_abs = @max(max_abs, @abs(diff));
            sum_abs += @abs(diff);
            dot += @as(f64, va) * @as(f64, vb);
            norm_a += @as(f64, va) * @as(f64, va);
            norm_b += @as(f64, vb) * @as(f64, vb);
            if (va > a[top_a]) top_a = i;
            if (vb > b[top_b]) top_b = i;
        }
        if (top_a == top_b) top_match += 1;
    }
    const n: f64 = @floatFromInt(values.len);
    try stdout.print(
        "compare canvas logits: max_abs={d:.6} mean_abs={d:.6} cosine={d:.8} argmax_match={d}/{d}\n",
        .{ max_abs, sum_abs / n, dot / (@sqrt(norm_a) * @sqrt(norm_b)), top_match, rows },
    );
}

fn readF32File(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    if (stat.size % @sizeOf(f32) != 0) return error.InvalidLogitsFile;

    const byte_len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, byte_len);
    defer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const got = try file.readStreaming(io, &.{bytes[read_len..]});
        if (got == 0) return error.EndOfStream;
        read_len += got;
    }
    const out = try allocator.alloc(f32, byte_len / @sizeOf(f32));
    errdefer allocator.free(out);
    for (out, 0..) |*dst, i| {
        const bits = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
        dst.* = @bitCast(bits);
    }
    return out;
}
