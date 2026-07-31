//! MiniFASNet anti-spoof ensemble — drives the app-level graph.zig
//! replay over the GGUF-embedded member graphs (buffalo pack: as0 =
//! MiniFASNetV2 @ scale 2.7, as1 = V1SE @ scale 4.0). Each `size²` BGR crop →
//! replay → 3 logits → softmax; the averaged "real"-class (index 1)
//! probability is the liveness score. `Model` + `scoreWith` is the production
//! path (crop-from-bbox at the pack's scales, argmax-gated score — the
//! reference `antispoof_score`); `realProb` is the one-shot parity entry over
//! pre-captured crops.

const std = @import("std");
const fucina = @import("fucina");
const rec = @import("recognizer.zig");
const image = @import("image.zig");
const graph = @import("graph.zig");
const align_mod = @import("align.zig");

const gguf = fucina.gguf;
const ExecContext = fucina.ExecContext;

const as_bn_eps: f32 = 1e-5; // kAsBnEps

/// Whether the pack carries the MiniFASNet ensemble (converter-written key).
pub fn present(file: *const gguf.File) bool {
    return file.getBool("facedetect.antispoof.present") orelse false;
}

fn compileMember(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File, i: usize) !graph.Compiled {
    const gkey = try std.fmt.allocPrint(allocator, "facedetect.antispoof.{d}.graph", .{i});
    defer allocator.free(gkey);
    const okey = try std.fmt.allocPrint(allocator, "facedetect.antispoof.{d}.output", .{i});
    defer allocator.free(okey);
    const prefix = try std.fmt.allocPrint(allocator, "as{d}.", .{i});
    defer allocator.free(prefix);

    const arr = file.getArray(gkey) orelse return error.NoAntispoofGraph;
    const specs = try arr.stringSlices(allocator);
    defer allocator.free(specs);
    const out_name = file.getString(okey) orelse return error.NoAntispoofOutput;

    return graph.Compiled.compile(ctx, allocator, file, prefix, specs, out_name, "input", as_bn_eps);
}

/// One member forward: RGB crop pixels → BGR planes raw 0-255 (the net has no
/// in-graph normalize) → replay → 3-class softmax probabilities.
fn memberProbs(ctx: *ExecContext, allocator: std.mem.Allocator, compiled: *const graph.Compiled, pixels: []const u8, width: usize, height: usize) ![3]f64 {
    const npx = width * height;
    const buf = try allocator.alloc(f32, npx * 3);
    defer allocator.free(buf);
    for (0..npx) |p| {
        buf[p * 3 + 0] = @floatFromInt(pixels[p * 3 + 2]); // B
        buf[p * 3 + 1] = @floatFromInt(pixels[p * 3 + 1]); // G
        buf[p * 3 + 2] = @floatFromInt(pixels[p * 3 + 0]); // R
    }
    var input = try ctx.fromSlice(&.{ height, width, 3 }, buf);
    defer input.deinit();
    var logits = try compiled.run(ctx, allocator, &input);
    defer logits.deinit();
    const ld = logits.dataConst();
    std.debug.assert(ld.len == 3);

    const mx = @max(ld[0], @max(ld[1], ld[2]));
    var e: [3]f64 = undefined;
    var s: f64 = 0;
    for (0..3) |k| {
        e[k] = @exp(@as(f64, ld[k]) - mx);
        s += e[k];
    }
    for (&e) |*v| v.* /= s;
    return e;
}

/// Load-once ensemble: member count and crop scales from the pack's
/// `facedetect.antispoof.scales`, input size from `.input_size`; each member
/// graph compiled once (weights dequant + BN-folded). `file` must outlive the
/// model — compiled node names borrow its metadata bytes.
pub const Model = struct {
    allocator: std.mem.Allocator,
    members: []Member,
    input_size: usize,

    pub const Member = struct { compiled: graph.Compiled, scale: f32 };

    pub fn init(ctx: *ExecContext, allocator: std.mem.Allocator, file: *const gguf.File) !Model {
        const arr = file.getArray("facedetect.antispoof.scales") orelse return error.NoAntispoofScales;
        // f32 array (GGUF item type 6), one scale per ensemble member.
        if (arr.item_type != 6 or arr.len == 0 or arr.data.len != arr.len * 4) return error.NoAntispoofScales;
        const input_size: usize = @intCast(file.getInt("facedetect.antispoof.input_size") orelse 80);

        const members = try allocator.alloc(Member, arr.len);
        var built: usize = 0;
        errdefer {
            for (members[0..built]) |*m| m.compiled.deinit();
            allocator.free(members);
        }
        for (members, 0..) |*m, i| {
            const scale: f32 = @bitCast(std.mem.readInt(u32, arr.data[i * 4 ..][0..4], .little));
            m.* = .{ .compiled = try compileMember(ctx, allocator, file, i), .scale = scale };
            built += 1;
        }
        return .{ .allocator = allocator, .members = members, .input_size = input_size };
    }

    pub fn deinit(self: *Model) void {
        for (self.members) |*m| m.compiled.deinit();
        self.allocator.free(self.members);
        self.* = undefined;
    }
};

/// Ensemble liveness score from the source image + detected box — the
/// reference `antispoof_score`: per member, `_crop_face` at its scale →
/// replay → softmax; average the class probabilities, then argmax-gate:
/// the averaged real-class (index 1) probability when "real" wins, else 0.
pub fn scoreWith(ctx: *ExecContext, allocator: std.mem.Allocator, model: *const Model, src: *const image.Image, box: [4]f32) !f32 {
    var accum: [3]f64 = .{ 0, 0, 0 };
    for (model.members) |*m| {
        const crop = try align_mod.antispoofCrop(allocator, src, box, model.input_size, m.scale);
        defer allocator.free(crop);
        const probs = try memberProbs(ctx, allocator, &m.compiled, crop, model.input_size, model.input_size);
        for (0..3) |k| accum[k] += probs[k];
    }
    const n: f64 = @floatFromInt(model.members.len);
    for (&accum) |*a| a.* /= n;
    const arg: usize = if (accum[1] >= accum[0] and accum[1] >= accum[2]) 1 else if (accum[0] >= accum[2]) 0 else 2;
    return if (arg == 1) @floatCast(accum[1]) else 0.0;
}

/// Averaged ensemble "real" probability from the pre-captured member crops
/// (one FDR1 file per member, in member order). One-shot: compiles each
/// member graph per call — the parity tests' entry.
pub fn realProb(ctx: *ExecContext, allocator: std.mem.Allocator, io: std.Io, file: *const gguf.File, crop_paths: []const []const u8) !f32 {
    var accum1: f64 = 0;
    for (crop_paths, 0..) |path, i| {
        const bytes = try rec.readFile(io, allocator, path);
        defer allocator.free(bytes);
        var img = try image.fromRaw(allocator, bytes);
        defer img.deinit();

        var compiled = try compileMember(ctx, allocator, file, i);
        defer compiled.deinit();
        const probs = try memberProbs(ctx, allocator, &compiled, img.pixels, img.width, img.height);
        accum1 += probs[1];
    }
    return @floatCast(accum1 / @as(f64, @floatFromInt(crop_paths.len)));
}
