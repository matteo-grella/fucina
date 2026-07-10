const std = @import("std");
const fucina = @import("fucina");
const cli = @import("facedetect/cli.zig");
const pipeline = @import("facedetect/pipeline.zig");
const loader = @import("facedetect/loader.zig");
const recognizer = @import("facedetect/recognizer.zig");
const scrfd = @import("facedetect/scrfd.zig");
const genderage = @import("facedetect/genderage.zig");
const image = @import("facedetect/image.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

// Face detection / recognition — a Zig port of mudler/face-detect.cpp (buffalo_l
// pack: SCRFD det_10g detector + ArcFace R50 recognizer + genderage + anti-spoof;
// dense landmarks via a separate GGUF). This is the CLI entry point.

const usage =
    \\usage:
    \\  zig build facedetect -- info      <model.gguf>
    \\  zig build facedetect -- detect    --model <model.gguf> --input <img> [--json] [--threads N]
    \\  zig build facedetect -- embed     --model <model.gguf> --input <img> [--json] [--threads N]
    \\  zig build facedetect -- verify    --model <model.gguf> --a <imgA> --b <imgB> [--threshold T] [--anti-spoof] [--threads N]
    \\  zig build facedetect -- analyze   --model <model.gguf> --input <img> [--threads N]
    \\  zig build facedetect -- landmarks --model <landmarks.gguf> --input <img> [--3d] [--detector <det.gguf>] [--json] [--threads N]
    \\  zig build facedetect -- bench     --model <model.gguf> --input <img> [--mode pipeline|recognizer|detect|analyze] [--n N] [--threads N]
    \\
;

const Command = enum { info, detect, embed, verify, analyze, landmarks, bench };

fn flagVal(args: []const [:0]const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |a, i| if (std.mem.eql(u8, a, name) and i + 1 < args.len) return args[i + 1];
    return null;
}
fn hasFlag(args: []const [:0]const u8, name: []const u8) bool {
    for (args) |a| if (std.mem.eql(u8, a, name)) return true;
    return false;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    if (args.len < 2) return stdout.print("{s}", .{usage});
    const cmd = std.meta.stringToEnum(Command, args[1]) orelse
        return stdout.print("unknown command: {s}\n\n{s}", .{ args[1], usage });

    const model_path = flagVal(args, "--model") orelse (if (cmd == .info and args.len > 2) args[2] else null) orelse
        return stdout.print("error: --model required\n", .{});
    var file = gguf.File.loadMmap(allocator, init.io, model_path) catch
        return stdout.print("error: cannot load model {s}\n", .{model_path});
    defer file.deinit();

    if (cmd == .info) {
        const cov = loader.coverage(&file);
        return stdout.print("model: {s}\ntensors: {d} (det {d}, rec {d}, genderage {d}, anti-spoof {d})\ngenderage: {s}\nanti-spoof: {s}\n", .{
            model_path, cov.total, cov.det, cov.rec, cov.ga, cov.antispoof,
            if (cov.ga > 0) "present" else "absent",
            if (cov.antispoof > 0) "present" else "absent",
        });
    }

    // --threads N caps the worker team at runtime (equivalent to
    // FUCINA_MAX_THREADS; the reference's default is min(hw,8) like ours).
    if (flagVal(args, "--threads")) |t| {
        fucina.parallel.setMaxThreads(try std.fmt.parseInt(usize, t, 10));
    }

    var ctx: ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    switch (cmd) {
        .detect => {
            var img = try cli.readImage(allocator, init.io, flagVal(args, "--input").?);
            defer img.deinit();
            var det_model = scrfd.Model.init(allocator, &file);
            defer det_model.deinit();
            const dets = try pipeline.detect_allWith(&ctx, allocator, &det_model, &img);
            defer allocator.free(dets);
            const json = try cli.detectJson(allocator, dets);
            try stdout.print("{s}\n", .{json});
        },
        .embed => {
            var img = try cli.readImage(allocator, init.io, flagVal(args, "--input").?);
            defer img.deinit();
            var det_model = scrfd.Model.init(allocator, &file);
            defer det_model.deinit();
            var rec_model = try recognizer.Model.load(&ctx, allocator, &file);
            defer rec_model.deinit();
            const emb = try pipeline.embedWith(&ctx, allocator, &det_model, &rec_model, &img);
            defer allocator.free(emb);
            const json = try cli.embedJson(allocator, emb);
            try stdout.print("{s}\n", .{json});
        },
        .analyze => {
            var img = try cli.readImage(allocator, init.io, flagVal(args, "--input").?);
            defer img.deinit();
            var det_model = scrfd.Model.init(allocator, &file);
            defer det_model.deinit();
            var ga_model = genderage.Model.init(allocator, &file);
            defer ga_model.deinit();
            const primary = (try pipeline.primaryFaceWith(&ctx, allocator, &det_model, &img)) orelse
                return stdout.print("{{\"faces\":[]}}\n", .{});
            const r = try pipeline.analyzeWith(&ctx, allocator, &det_model, &ga_model, &img);
            const json = try cli.analyzeJson(allocator, primary, r);
            try stdout.print("{s}\n", .{json});
        },
        .verify => {
            var ia = try cli.readImage(allocator, init.io, flagVal(args, "--a").?);
            defer ia.deinit();
            var ib = try cli.readImage(allocator, init.io, flagVal(args, "--b").?);
            defer ib.deinit();
            var det_model = scrfd.Model.init(allocator, &file);
            defer det_model.deinit();
            var rec_model = try recognizer.Model.load(&ctx, allocator, &file);
            defer rec_model.deinit();
            const ea = try pipeline.embedWith(&ctx, allocator, &det_model, &rec_model, &ia);
            defer allocator.free(ea);
            const eb = try pipeline.embedWith(&ctx, allocator, &det_model, &rec_model, &ib);
            defer allocator.free(eb);
            const dist = 1.0 - pipeline.cosine(ea, eb);
            const thr: f64 = if (flagVal(args, "--threshold")) |t| try std.fmt.parseFloat(f64, t) else 0.35;
            const json = try cli.verifyJson(allocator, dist, dist <= thr);
            try stdout.print("{s}\n", .{json});
        },
        .landmarks => try stdout.print("landmarks: end-to-end wiring not implemented\n", .{}),
        .bench => {
            // Mirrors the reference bench protocol (examples/cli/main.cpp
            // cmd_bench): model load + image decode outside the loop, ONE
            // untimed warmup pass, arithmetic mean over N timed passes.
            var img = try cli.readImage(allocator, init.io, flagVal(args, "--input").?);
            defer img.deinit();
            const mode = flagVal(args, "--mode") orelse "recognizer";
            const n: usize = if (flagVal(args, "--n")) |s| try std.fmt.parseInt(usize, s, 10) else 20;

            var det_model = scrfd.Model.init(allocator, &file);
            defer det_model.deinit();
            var ga_model = genderage.Model.init(allocator, &file);
            defer ga_model.deinit();
            var rec_model: ?recognizer.Model = if (std.mem.eql(u8, mode, "detect") or std.mem.eql(u8, mode, "analyze"))
                null
            else
                try recognizer.Model.load(&ctx, allocator, &file);
            defer if (rec_model) |*m| m.deinit();

            const Once = struct {
                fn pass(mode_s: []const u8, ctx_p: *ExecContext, al: std.mem.Allocator, det: *scrfd.Model, ga: *genderage.Model, rec_m: ?*const recognizer.Model, im: *const image.Image) !void {
                    if (std.mem.eql(u8, mode_s, "recognizer")) {
                        al.free(try rec_m.?.embedImage(ctx_p, al, im));
                    } else if (std.mem.eql(u8, mode_s, "detect")) {
                        al.free(try pipeline.detect_allWith(ctx_p, al, det, im));
                    } else if (std.mem.eql(u8, mode_s, "analyze")) {
                        _ = try pipeline.analyzeWith(ctx_p, al, det, ga, im);
                    } else {
                        al.free(try pipeline.embedWith(ctx_p, al, det, rec_m.?, im));
                    }
                }
            };
            const rec_ptr: ?*const recognizer.Model = if (rec_model) |*m| m else null;
            try Once.pass(mode, &ctx, allocator, &det_model, &ga_model, rec_ptr, &img); // warmup (untimed)
            const t0 = std.Io.Clock.awake.now(init.io).nanoseconds;
            for (0..n) |_| try Once.pass(mode, &ctx, allocator, &det_model, &ga_model, rec_ptr, &img);
            const t1 = std.Io.Clock.awake.now(init.io).nanoseconds;
            const ms_per = @as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(n)) / 1e6;
            try stdout.print("bench {s}: {d:.2} ms/image ({d} runs)\n", .{ mode, ms_per, n });
        },
        .info => unreachable,
    }
}

test {
    _ = @import("facedetect/loader.zig");
    _ = @import("facedetect/loader_tests.zig");
    _ = @import("facedetect/image.zig");
    _ = @import("facedetect/image_tests.zig");
    _ = @import("facedetect/nn.zig");
    _ = @import("facedetect/nn_tests.zig");
    _ = @import("facedetect/recognizer.zig");
    _ = @import("facedetect/recognizer_tests.zig");
    _ = @import("facedetect/genderage.zig");
    _ = @import("facedetect/genderage_tests.zig");
    _ = @import("facedetect/graph.zig");
    _ = @import("facedetect/antispoof.zig");
    _ = @import("facedetect/antispoof_tests.zig");
    _ = @import("facedetect/landmarks.zig");
    _ = @import("facedetect/landmarks_tests.zig");
    _ = @import("facedetect/scrfd.zig");
    _ = @import("facedetect/scrfd_tests.zig");
    _ = @import("facedetect/detect.zig");
    _ = @import("facedetect/detect_tests.zig");
    _ = @import("facedetect/align.zig");
    _ = @import("facedetect/align_tests.zig");
    _ = @import("facedetect/preprocess.zig");
    _ = @import("facedetect/preprocess_tests.zig");
    _ = @import("facedetect/pipeline.zig");
    _ = @import("facedetect/pipeline_tests.zig");
    _ = @import("facedetect/cli.zig");
    _ = @import("facedetect/cli_tests.zig");
}
