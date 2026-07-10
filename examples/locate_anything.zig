//! LocateAnything-3B open-vocabulary detection CLI — Fucina port of
//! https://github.com/mudler/locate-anything.cpp (pinned 92c1682).
//!
//! Subcommands mirror the reference CLI plus the parity harness:
//!   detect  --model M --input img.png --prompt "..." [--mode hybrid|slow|fast]
//!           [--output out.json] [--annotated out.png] [--max-new N] [--no-early-stop]
//!   info    --model M
//!   compare --model M --dump fixture_dump.gguf --image img.png --prompt "..."
//!           [--stage NAME] [--max-new N] [--mtp-rounds N]
//!
//! `compare` gates every pipeline stage against a reference dump produced by
//! tools/ref-patches/la_dump.cpp: discrete outputs (token ids, streams) gate
//! on exact equality, f32 tensors on max-abs/relative tolerances, and the
//! process exits 1 on any gate failure.

const std = @import("std");
const fucina = @import("fucina");

const config_mod = @import("locate_anything/config.zig");
const tokenizer_mod = @import("locate_anything/tokenizer.zig");
const image_mod = @import("locate_anything/image.zig");
const preproc_mod = @import("locate_anything/preproc.zig");
const vit_mod = @import("locate_anything/vit.zig");
const lm_mod = @import("locate_anything/lm.zig");
const mtp = @import("locate_anything/mtp.zig");
const boxes_mod = @import("locate_anything/boxes.zig");
const engine_mod = @import("locate_anything/engine.zig");
const visualize_mod = @import("locate_anything/visualize.zig");

const Allocator = std.mem.Allocator;
const gguf = fucina.gguf;

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.smp_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
    var stderr_buffer: [4096]u8 = undefined;
    var stderr_writer = std.Io.File.stderr().writer(init.io, &stderr_buffer);
    const stderr = &stderr_writer.interface;
    defer stderr.flush() catch {};

    if (args.len < 2) {
        try printHelp(stdout);
        return;
    }
    const sub = args[1];
    if (std.mem.eql(u8, sub, "detect")) {
        return cmdDetect(init.io, allocator, args[2..], stdout, stderr);
    } else if (std.mem.eql(u8, sub, "info")) {
        return cmdInfo(init.io, allocator, args[2..], stdout, stderr);
    } else if (std.mem.eql(u8, sub, "compare")) {
        return cmdCompare(init.io, allocator, args[2..], stdout, stderr);
    } else if (std.mem.eql(u8, sub, "--help") or std.mem.eql(u8, sub, "help")) {
        return printHelp(stdout);
    }
    try stderr.print("error: unknown subcommand: {s}\n", .{sub});
    try printHelp(stdout);
    return error.InvalidArguments;
}

fn printHelp(out: anytype) !void {
    try out.print(
        \\fucina locate-anything — LocateAnything-3B open-vocabulary detection
        \\
        \\Usage:
        \\  locate-anything detect  --model <gguf> --input <image.png> --prompt <text>
        \\                          [--mode hybrid|slow|fast] [--output <json>]
        \\                          [--annotated <png>] [--max-new N] [--no-early-stop]
        \\  locate-anything info    --model <gguf>
        \\  locate-anything compare --model <gguf> --dump <dump.gguf> --image <image.png>
        \\                          --prompt <text> [--stage <name>] [--max-new N]
        \\
        \\Stages for compare: tokenizer preproc prompt vit projector lm slow hybrid fast all
        \\
    , .{});
}

fn argValue(args: []const []const u8, i: *usize, flag: []const u8) ?[]const u8 {
    if (!std.mem.eql(u8, args[i.*], flag)) return null;
    if (i.* + 1 >= args.len) return null;
    i.* += 1;
    return args[i.*];
}

// ---------------------------------------------------------------- detect ----

fn cmdDetect(io: std.Io, allocator: Allocator, args: []const []const u8, stdout: anytype, stderr: anytype) !void {
    var model_path: ?[]const u8 = null;
    var input_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;
    var annotated_path: ?[]const u8 = null;
    var mode: engine_mod.Mode = .hybrid;
    var max_new: usize = 256;
    var early_stop = true;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (argValue(args, &i, "--model")) |v| {
            model_path = v;
        } else if (argValue(args, &i, "--input")) |v| {
            input_path = v;
        } else if (argValue(args, &i, "--prompt")) |v| {
            prompt = v;
        } else if (argValue(args, &i, "--output")) |v| {
            output_path = v;
        } else if (argValue(args, &i, "--annotated")) |v| {
            annotated_path = v;
        } else if (argValue(args, &i, "--mode")) |v| {
            mode = std.meta.stringToEnum(engine_mod.Mode, v) orelse {
                try stderr.print("error: --mode must be hybrid|slow|fast (got {s})\n", .{v});
                return error.InvalidArguments;
            };
        } else if (argValue(args, &i, "--max-new")) |v| {
            max_new = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, args[i], "--no-early-stop")) {
            early_stop = false;
        } else {
            try stderr.print("error: unknown flag: {s}\n", .{args[i]});
            return error.InvalidArguments;
        }
    }
    const model = model_path orelse return missing(stderr, "--model");
    const input = input_path orelse return missing(stderr, "--input");
    const query = prompt orelse return missing(stderr, "--prompt");

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try gguf.File.loadMmap(allocator, io, model);
    defer file.deinit();
    var engine = try engine_mod.Engine.load(&ctx, &file);
    defer engine.deinit();

    var img = try image_mod.loadPng(allocator, io, input);
    defer img.deinit();

    var result = try engine.locate(img.rgb, img.w, img.h, query, mode, max_new, early_stop);
    defer result.deinit();

    const json = try boxes_mod.writeJson(allocator, result.boxes);
    defer allocator.free(json);
    if (output_path) |path| {
        var out_file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer out_file.close(io);
        try out_file.writeStreamingAll(io, json);
    } else {
        try stdout.print("{s}\n", .{json});
    }
    if (annotated_path) |path| {
        var annotated = try visualize_mod.renderBoxes(allocator, &img, result.boxes);
        defer annotated.deinit();
        try image_mod.writePng(allocator, io, path, annotated.w, annotated.h, annotated.rgb);
        try stderr.print("wrote {s}\n", .{path});
    }
    try stderr.print("{d} detections\n", .{result.boxes.len});
}

fn missing(stderr: anytype, flag: []const u8) !void {
    try stderr.print("error: {s} is required\n", .{flag});
    return error.InvalidArguments;
}

// ------------------------------------------------------------------ info ----

fn cmdInfo(io: std.Io, allocator: Allocator, args: []const []const u8, stdout: anytype, stderr: anytype) !void {
    var model_path: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (argValue(args, &i, "--model")) |v| model_path = v else {
            try stderr.print("error: unknown flag: {s}\n", .{args[i]});
            return error.InvalidArguments;
        }
    }
    const model = model_path orelse return missing(stderr, "--model");

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var file = try gguf.File.loadMmap(allocator, io, model);
    defer file.deinit();
    var engine = try engine_mod.Engine.load(&ctx, &file);
    defer engine.deinit();
    try stdout.print("model: {s}\nstatus: loaded ok\n", .{model});
}

// --------------------------------------------------------------- compare ----

const Gate = struct {
    stderr: *std.Io.Writer,
    failures: usize = 0,

    fn discreteU32(self: *Gate, name: []const u8, got: []const u32, want: []const i32) !void {
        var ok = got.len == want.len;
        var first_bad: ?usize = null;
        if (ok) {
            for (got, want, 0..) |g, w, i| {
                if (w < 0 or g != @as(u32, @intCast(w))) {
                    ok = false;
                    first_bad = i;
                    break;
                }
            }
        }
        if (ok) {
            try self.stderr.print("[gate] {s}: OK ({d} ids exact)\n", .{ name, got.len });
        } else {
            self.failures += 1;
            try self.stderr.print("[gate] {s}: FAIL got_len={d} want_len={d}", .{ name, got.len, want.len });
            if (first_bad) |i| {
                try self.stderr.print(" first_mismatch@{d}: got={d} want={d}", .{ i, got[i], want[i] });
            }
            try self.stderr.print("\n", .{});
        }
    }

    /// Gate on max-abs OR (when rel_tol > 0) max relative-to-magnitude error
    /// |g-w| / max(1, |w|). Deep pre-norm captures grow to O(1e3), where an
    /// absolute criterion measures activation magnitude rather than port
    /// error; the stages consumed downstream (post-norm, logits) keep tight
    /// absolute gates.
    fn tensorF32(self: *Gate, name: []const u8, got: []const f32, want: []const f32, max_abs_tol: f32, rel_tol: f32) !void {
        if (got.len != want.len) {
            self.failures += 1;
            try self.stderr.print("[gate] {s}: FAIL size got={d} want={d}\n", .{ name, got.len, want.len });
            return;
        }
        var max_abs: f64 = 0;
        var max_rel: f64 = 0;
        var mean_abs: f64 = 0;
        var worst: usize = 0;
        for (got, want, 0..) |g, w, i| {
            const d = @abs(@as(f64, g) - @as(f64, w));
            mean_abs += d;
            if (d > max_abs) {
                max_abs = d;
                worst = i;
            }
            max_rel = @max(max_rel, d / @max(1.0, @abs(@as(f64, w))));
        }
        mean_abs /= @floatFromInt(got.len);
        const ok = max_abs <= max_abs_tol or (rel_tol > 0 and max_rel <= rel_tol);
        if (!ok) self.failures += 1;
        try self.stderr.print("[gate] {s}: {s} max_abs={e:.3} max_rel={e:.3} mean_abs={e:.3} tol={e:.1}/{e:.1} worst@{d} got={e:.6} want={e:.6}\n", .{
            name,
            if (ok) "OK" else "FAIL",
            max_abs,
            max_rel,
            mean_abs,
            max_abs_tol,
            rel_tol,
            worst,
            got[worst],
            want[worst],
        });
    }
};

const Dump = struct {
    file: gguf.File,

    fn f32Tensor(self: *const Dump, allocator: Allocator, name: []const u8) ![]f32 {
        const info = try self.file.get(name);
        if (info.ggml_type != .f32) return error.WrongDumpType;
        var count: usize = 1;
        for (info.dims[0..info.n_dims]) |d| count *= d;
        const out = try allocator.alloc(f32, count);
        errdefer allocator.free(out);
        for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, info.data[i * 4 ..][0..4], .little));
        return out;
    }

    fn i32Tensor(self: *const Dump, allocator: Allocator, name: []const u8) ![]i32 {
        const info = try self.file.get(name);
        if (info.ggml_type != .i32) return error.WrongDumpType;
        var count: usize = 1;
        for (info.dims[0..info.n_dims]) |d| count *= d;
        // The dumper writes empty streams as a [-1] sentinel plus "<name>.empty".
        var key_buf: [128]u8 = undefined;
        const empty_key = try std.fmt.bufPrint(&key_buf, "{s}.empty", .{name});
        if (self.file.getBool(empty_key) orelse false) {
            return try allocator.alloc(i32, 0);
        }
        const out = try allocator.alloc(i32, count);
        errdefer allocator.free(out);
        for (out, 0..) |*v, i| v.* = std.mem.readInt(i32, info.data[i * 4 ..][0..4], .little);
        return out;
    }
};

fn wantStage(stage: []const u8, name: []const u8) bool {
    return std.mem.eql(u8, stage, "all") or std.mem.eql(u8, stage, name);
}

fn cmdCompare(io: std.Io, allocator: Allocator, args: []const []const u8, stdout: anytype, stderr: anytype) !void {
    var model_path: ?[]const u8 = null;
    var dump_path: ?[]const u8 = null;
    var image_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var stage: []const u8 = "all";
    var max_new: usize = 256;
    var mtp_rounds: usize = 12;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (argValue(args, &i, "--model")) |v| {
            model_path = v;
        } else if (argValue(args, &i, "--dump")) |v| {
            dump_path = v;
        } else if (argValue(args, &i, "--image")) |v| {
            image_path = v;
        } else if (argValue(args, &i, "--prompt")) |v| {
            prompt = v;
        } else if (argValue(args, &i, "--stage")) |v| {
            stage = v;
        } else if (argValue(args, &i, "--max-new")) |v| {
            max_new = try std.fmt.parseInt(usize, v, 10);
        } else if (argValue(args, &i, "--mtp-rounds")) |v| {
            mtp_rounds = try std.fmt.parseInt(usize, v, 10);
        } else {
            try stderr.print("error: unknown flag: {s}\n", .{args[i]});
            return error.InvalidArguments;
        }
    }
    const model = model_path orelse return missing(stderr, "--model");
    const dump_file_path = dump_path orelse return missing(stderr, "--dump");
    const img_path = image_path orelse return missing(stderr, "--image");
    const query = prompt orelse return missing(stderr, "--prompt");

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    var file = try gguf.File.loadMmap(allocator, io, model);
    defer file.deinit();
    var engine = try engine_mod.Engine.load(&ctx, &file);
    defer engine.deinit();

    var dump = Dump{ .file = try gguf.File.loadMmap(allocator, io, dump_file_path) };
    defer dump.file.deinit();

    var gate = Gate{ .stderr = stderr };

    // ---- tokenizer cases ----
    if (wantStage(stage, "tokenizer")) {
        var case_i: usize = 0;
        while (true) : (case_i += 1) {
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "tok_case_{d:0>2}", .{case_i});
            if (dump.file.maybeGet(name) == null) break;
            var key_buf: [80]u8 = undefined;
            const text_key = try std.fmt.bufPrint(&key_buf, "{s}.text", .{name});
            const text = dump.file.getString(text_key) orelse return error.MissingDumpText;
            const want = try dump.i32Tensor(allocator, name);
            defer allocator.free(want);
            const got = try engine.tok.encode(allocator, text);
            defer allocator.free(got);
            try gate.discreteU32(name, got, want);
        }
    }

    // ---- preprocess ----
    var pre = try blk: {
        var img = try image_mod.loadPng(allocator, io, img_path);
        defer img.deinit();
        break :blk preproc_mod.preprocess(allocator, img.rgb, img.w, img.h, engine.preprocLimits());
    };
    defer pre.deinit();

    if (wantStage(stage, "preproc")) {
        const want = try dump.f32Tensor(allocator, "pixel_values");
        defer allocator.free(want);
        try gate.tensorF32("pixel_values", pre.pixel_values, want, 0.0, 0.0);
        const gh_want = dump.file.getInt("la.gh") orelse 0;
        const gw_want = dump.file.getInt("la.gw") orelse 0;
        if (pre.gh != gh_want or pre.gw != gw_want) {
            gate.failures += 1;
            try stderr.print("[gate] grid: FAIL got=({d},{d}) want=({d},{d})\n", .{ pre.gh, pre.gw, gh_want, gw_want });
        } else {
            try stderr.print("[gate] grid: OK ({d},{d})\n", .{ pre.gh, pre.gw });
        }
    }

    // ---- prompt ----
    const n_image_tokens = (pre.gh / engine.config.vit_merge_h) * (pre.gw / engine.config.vit_merge_w);
    const prompt_ids = try tokenizer_mod.buildPrompt(allocator, &engine.tok, n_image_tokens, query);
    defer allocator.free(prompt_ids);
    if (wantStage(stage, "prompt")) {
        const want = try dump.i32Tensor(allocator, "prompt_ids");
        defer allocator.free(want);
        try gate.discreteU32("prompt_ids", prompt_ids, want);
    }

    // ---- vision tower ----
    if (wantStage(stage, "vit")) {
        {
            var patch_pos = try engine.vit.patchAndPos(&ctx, pre.pixel_values, pre.gh, pre.gw);
            defer patch_pos.deinit();
            const got = try tensorToSlice(allocator, &patch_pos);
            defer allocator.free(got);
            const want = try dump.f32Tensor(allocator, "vit_patch_pos");
            defer allocator.free(want);
            try gate.tensorF32("vit_patch_pos", got, want, 2e-4, 0.0);
        }
        {
            var block0 = try engine.vit.forward(&ctx, pre.pixel_values, pre.gh, pre.gw, 0);
            defer block0.deinit();
            const got = try tensorToSlice(allocator, &block0);
            defer allocator.free(got);
            const want = try dump.f32Tensor(allocator, "vit_block0");
            defer allocator.free(want);
            try gate.tensorF32("vit_block0", got, want, 5e-4, 0.0);
        }
        {
            var l26 = try engine.vit.forward(&ctx, pre.pixel_values, pre.gh, pre.gw, 26);
            defer l26.deinit();
            const got = try tensorToSlice(allocator, &l26);
            defer allocator.free(got);
            const want = try dump.f32Tensor(allocator, "vit_layer_26");
            defer allocator.free(want);
            // Deep pre-norm capture: 27 blocks of f32 with different GEMM
            // summation orders; measured max_rel 2.8e-3 on the fixture (the
            // localizing role only — the post-norm vit_final downstream stays
            // under a 2e-2 ABSOLUTE gate on O(10) values).
            try gate.tensorF32("vit_layer_26", got, want, 2e-2, 5e-3);
        }
    }

    // ---- vit_final + merge + projector (recomputed once, shared) ----
    var vfinal = try engine.vit.forward(&ctx, pre.pixel_values, pre.gh, pre.gw, null);
    defer vfinal.deinit();
    if (wantStage(stage, "vit")) {
        const got = try tensorToSlice(allocator, &vfinal);
        defer allocator.free(got);
        const want = try dump.f32Tensor(allocator, "vit_final");
        defer allocator.free(want);
        try gate.tensorF32("vit_final", got, want, 2e-2, 0.0);
    }

    var merged = try engine.vit.mergePatches(&ctx, &vfinal, pre.gh, pre.gw);
    defer merged.deinit();
    var projected_t = try engine.vit.project(&ctx, &merged);
    defer projected_t.deinit();
    const projected = try tensorToSlice(allocator, &projected_t);
    defer allocator.free(projected);

    if (wantStage(stage, "projector")) {
        {
            const got = try tensorToSlice(allocator, &merged);
            defer allocator.free(got);
            const want = try dump.f32Tensor(allocator, "merged");
            defer allocator.free(want);
            try gate.tensorF32("merged", got, want, 2e-2, 0.0);
        }
        const want = try dump.f32Tensor(allocator, "projected");
        defer allocator.free(want);
        try gate.tensorF32("projected", projected, want, 2e-2, 0.0);
    }

    // ---- LM prefill ----
    if (wantStage(stage, "lm")) {
        const spliced = try engine.embedAndSplice(prompt_ids, projected);
        defer allocator.free(spliced);
        {
            const want = try dump.f32Tensor(allocator, "embeds_spliced");
            defer allocator.free(want);
            try gate.tensorF32("embeds_spliced", spliced, want, 2e-2, 0.0);
        }
        const logits = try allocator.alloc(f32, engine.config.lm_vocab);
        defer allocator.free(logits);
        var cache = try engine.lm.initCache(&ctx, prompt_ids.len + 8);
        defer cache.deinit();
        try engine.lm.forwardCausal(&ctx, &cache, spliced, prompt_ids.len, 0, logits);
        const want = try dump.f32Tensor(allocator, "logits_step0");
        defer allocator.free(want);
        try gate.tensorF32("logits_step0", logits, want, 5e-2, 2e-3);
        const got_arg = mtp.argmaxRow(logits);
        const want_arg = mtp.argmaxRow(want);
        if (got_arg != want_arg) {
            gate.failures += 1;
            try stderr.print("[gate] logits_step0 argmax: FAIL got={d} want={d}\n", .{ got_arg, want_arg });
        } else {
            try stderr.print("[gate] logits_step0 argmax: OK ({d})\n", .{got_arg});
        }
    }

    // ---- decode streams ----
    if (wantStage(stage, "slow")) {
        const got = try engine.decodeSlow(prompt_ids, projected, max_new);
        defer allocator.free(got);
        const want = try dump.i32Tensor(allocator, "stream_slow");
        defer allocator.free(want);
        try gate.discreteU32("stream_slow", got, want);
    }
    if (wantStage(stage, "hybrid")) {
        var captured: std.ArrayList([]f32) = .empty;
        defer {
            for (captured.items) |c| allocator.free(c);
            captured.deinit(allocator);
        }
        const got = try engine.decodeHybrid(prompt_ids, projected, max_new, .{ .captured_logits = &captured });
        defer allocator.free(got);

        var round: usize = 0;
        while (round < mtp_rounds and round < captured.items.len) : (round += 1) {
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "mtp_logits6_r{d:0>3}", .{round});
            if (dump.file.maybeGet(name) == null) break;
            const want = try dump.f32Tensor(allocator, name);
            defer allocator.free(want);
            try gate.tensorF32(name, captured.items[round], want, 5e-2, 2e-3);
        }
        const want = try dump.i32Tensor(allocator, "stream_hybrid");
        defer allocator.free(want);
        try gate.discreteU32("stream_hybrid", got, want);
    }
    if (wantStage(stage, "fast")) {
        const got = try engine.decodeHybrid(prompt_ids, projected, max_new, .{ .fast = true });
        defer allocator.free(got);
        const want = try dump.i32Tensor(allocator, "stream_fast");
        defer allocator.free(want);
        try gate.discreteU32("stream_fast", got, want);
    }

    try stdout.print("compare: {d} gate failure(s)\n", .{gate.failures});
    try stdout.flush();
    try stderr.flush();
    if (gate.failures > 0) return error.ParityGateFailed;
}

fn tensorToSlice(allocator: Allocator, t: anytype) ![]f32 {
    var count: usize = 1;
    const raw = t.asRawTensor();
    for (0..raw.rank()) |i| count *= raw.shape.at(i);
    const out = try allocator.alloc(f32, count);
    errdefer allocator.free(out);
    try t.copyTo(out);
    return out;
}

test {
    _ = @import("locate_anything/tokenizer.zig");
    _ = @import("locate_anything/image.zig");
    _ = @import("locate_anything/preproc.zig");
    _ = @import("locate_anything/mtp.zig");
    _ = @import("locate_anything/boxes.zig");
    _ = @import("locate_anything/lm.zig");
}
