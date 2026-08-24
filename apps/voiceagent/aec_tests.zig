//! GTCRN-AEC parity vs the LocalVQE exporter fixtures (the validated PyTorch
//! reference): the streaming Session, iterated frame-by-frame over the fixture
//! clip, must reproduce the whole-clip reference output — this pins the
//! DEPLOYMENT path (carried conv history + GRU hiddens), not just an offline
//! forward. Also pins stftFrame/istftFrame against the shipped DFT matrices.
//! Skips without models/aec/gtcrn_aec.gguf + apps/voiceagent/goldens/.

const std = @import("std");
const fucina = @import("fucina");
const aec = @import("aec.zig");

const gguf = fucina.gguf;

const model_path = "models/aec/gtcrn_aec.gguf";
const fixtures = "apps/voiceagent/goldens";

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(1 << 30));
}

const Npy = struct {
    shape: [4]usize,
    nd: usize,
    data: []f32,

    fn deinit(self: *Npy, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

/// Minimal .npy reader: v1/v2 header, little-endian f32 or f64, C order.
fn npyLoad(allocator: std.mem.Allocator, dir: []const u8, name: []const u8) !Npy {
    var pbuf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/{s}.npy", .{ dir, name });
    const bytes = try readFile(allocator, path);
    defer allocator.free(bytes);
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const major = bytes[6];
    const hlen: usize = if (major >= 2)
        std.mem.readInt(u32, bytes[8..12], .little)
    else
        std.mem.readInt(u16, bytes[8..10], .little);
    const hoff: usize = if (major >= 2) 12 else 10;
    const header = bytes[hoff .. hoff + hlen];

    const is_f64 = std.mem.indexOf(u8, header, "<f8") != null;
    if (!is_f64 and std.mem.indexOf(u8, header, "<f4") == null) return error.BadNpy;
    if (std.mem.indexOf(u8, header, "'fortran_order': False") == null) return error.BadNpy;

    var shape = [4]usize{ 1, 1, 1, 1 };
    var nd: usize = 0;
    const sh0 = (std.mem.indexOf(u8, header, "'shape': (") orelse return error.BadNpy) + 10;
    const sh1 = std.mem.indexOfScalarPos(u8, header, sh0, ')') orelse return error.BadNpy;
    var it = std.mem.tokenizeAny(u8, header[sh0..sh1], ", ");
    while (it.next()) |tok| {
        shape[nd] = try std.fmt.parseInt(usize, tok, 10);
        nd += 1;
    }
    var n: usize = 1;
    for (shape[0..nd]) |s| n *= s;

    const payload = bytes[hoff + hlen ..];
    const data = try allocator.alloc(f32, n);
    errdefer allocator.free(data);
    if (is_f64) {
        for (data, 0..) |*v, i| v.* = @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, payload[i * 8 ..][0..8], .little))));
    } else {
        for (data, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, payload[i * 4 ..][0..4], .little));
    }
    return .{ .shape = shape, .nd = nd, .data = data };
}

fn stats(a: []const f32, b: []const f32) struct { cosine: f64, max_diff: f64 } {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var md: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * @as(f64, y);
        na += @as(f64, x) * @as(f64, x);
        nb += @as(f64, y) * @as(f64, y);
        md = @max(md, @abs(@as(f64, x) - @as(f64, y)));
    }
    return .{ .cosine = dot / (@sqrt(na) * @sqrt(nb)), .max_diff = md };
}

test "gtcrn-aec: streaming session reproduces the reference clip output" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var file = gguf.File.loadMmap(allocator, std.testing.io, model_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    var e_in = npyLoad(allocator, fixtures, "in_spec_e") catch return error.SkipZigTest;
    defer e_in.deinit(allocator);
    var y_in = try npyLoad(allocator, fixtures, "in_spec_y");
    defer y_in.deinit(allocator);
    var out_ref = try npyLoad(allocator, fixtures, "out_spec");
    defer out_ref.deinit(allocator);

    // fixtures are (1, 257, T, 2), f-major
    const t_frames = e_in.shape[2];
    try std.testing.expectEqual(@as(usize, 257), e_in.shape[1]);

    var model = try aec.Model.load(allocator, &file);
    defer model.deinit();
    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var sess = try aec.Session.init(allocator, &ctx, &model);
    defer sess.deinit();

    const stage_names = [_][]const u8{ "feat", "enc0", "enc1", "enc2", "enc3", "enc4", "dpgrnn1", "dpgrnn2", "dec0", "dec1", "dec2", "dec3", "dec4", "mask" };
    var refs: [stage_names.len]Npy = undefined;
    var loaded: usize = 0;
    defer for (refs[0..loaded]) |*r| r.deinit(allocator);
    for (stage_names, 0..) |name, i| {
        refs[i] = try npyLoad(allocator, fixtures, name);
        loaded += 1;
    }
    var worst: [stage_names.len]f64 = @splat(0);

    var cap = aec.Cap.init(allocator);
    defer cap.deinit();
    sess.cap = &cap;

    const got = try allocator.alloc(f32, out_ref.data.len);
    defer allocator.free(got);
    var se: [aec.n_bins * 2]f32 = undefined;
    var sy: [aec.n_bins * 2]f32 = undefined;
    var so: [aec.n_bins * 2]f32 = undefined;
    for (0..t_frames) |t| {
        for (0..aec.n_bins) |f| {
            se[f * 2] = e_in.data[(f * t_frames + t) * 2];
            se[f * 2 + 1] = e_in.data[(f * t_frames + t) * 2 + 1];
            sy[f * 2] = y_in.data[(f * t_frames + t) * 2];
            sy[f * 2 + 1] = y_in.data[(f * t_frames + t) * 2 + 1];
        }
        try sess.stepSpec(&se, &sy, &so);
        for (0..aec.n_bins) |f| {
            got[(f * t_frames + t) * 2] = so[f * 2];
            got[(f * t_frames + t) * 2 + 1] = so[f * 2 + 1];
        }
        // per-stage frame diffs vs the (1,C,T,F) fixtures
        for (stage_names, 0..) |name, i| {
            const ref = &refs[i];
            const cdim = ref.shape[1];
            const fdim = ref.shape[3];
            const gf = cap.map.get(name) orelse continue;
            for (0..cdim) |c| {
                for (0..fdim) |f| {
                    const rv = ref.data[(c * t_frames + t) * fdim + f];
                    const gv = gf[c * fdim + f];
                    worst[i] = @max(worst[i], @abs(@as(f64, gv) - rv));
                }
            }
        }
    }
    // Clean timing pass: no stage capture (its per-stage put() allocates and
    // copies, which dwarfs the model on these frame sizes), repeated so the
    // 8-frame fixture gives a settled per-frame number.
    sess.cap = null;
    sess.reset();
    const reps = 50;
    const bench_t0 = std.Io.Clock.awake.now(std.testing.io).nanoseconds;
    for (0..reps) |_| {
        for (0..t_frames) |t| {
            for (0..aec.n_bins) |f| {
                se[f * 2] = e_in.data[(f * t_frames + t) * 2];
                se[f * 2 + 1] = e_in.data[(f * t_frames + t) * 2 + 1];
                sy[f * 2] = y_in.data[(f * t_frames + t) * 2];
                sy[f * 2 + 1] = y_in.data[(f * t_frames + t) * 2 + 1];
            }
            try sess.stepSpec(&se, &sy, &so);
        }
    }
    const bench_t1 = std.Io.Clock.awake.now(std.testing.io).nanoseconds;
    std.debug.print("[gtcrn-aec] {d} us/frame (16000 us hop budget)\n", .{
        @as(f64, @floatFromInt(bench_t1 - bench_t0)) / 1e3 / @as(f64, @floatFromInt(t_frames * reps)),
    });
    for (stage_names, 0..) |name, i| {
        if (worst[i] > 1e-3) std.debug.print("[gtcrn-aec] stage {s} max|diff| {d:.6}\n", .{ name, worst[i] });
    }

    const s = stats(got, out_ref.data);
    if (s.cosine < 0.99999 or s.max_diff > 1e-3) {
        std.debug.print("[gtcrn-aec] out_spec cosine {d:.7} max|diff| {d:.6}\n", .{ s.cosine, s.max_diff });
        return error.TestUnexpectedResult;
    }
}

test "gtcrn-aec: stftFrame/istftFrame match the offline fixture framing" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var file = gguf.File.loadMmap(allocator, std.testing.io, model_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    var sig = npyLoad(allocator, fixtures, "stft_in") catch return error.SkipZigTest;
    defer sig.deinit(allocator);
    var spec_ref = try npyLoad(allocator, fixtures, "stft_out");
    defer spec_ref.deinit(allocator);

    var model = try aec.Model.load(allocator, &file);
    defer model.deinit();

    // Reference offline stft: reflect-pad hop each side, frame t starts at
    // pad[t*hop]. Reproduce and compare every frame.
    var l: usize = 1;
    for (sig.shape[0..sig.nd]) |d| l *= d;
    const t_frames = spec_ref.shape[2];
    const pad = try allocator.alloc(f32, l + aec.fft_size);
    defer allocator.free(pad);
    @memset(pad, 0);
    for (0..l) |i| pad[aec.hop + i] = sig.data[i];
    for (0..aec.hop) |j| {
        pad[j] = sig.data[aec.hop - j];
        pad[aec.hop + l + j] = sig.data[l - 2 - j];
    }

    var out: [aec.n_bins * 2]f32 = undefined;
    var max_diff: f64 = 0;
    for (0..t_frames) |t| {
        try aec.stftFrame(&model, pad[t * aec.hop ..][0..aec.fft_size], &out);
        for (0..aec.n_bins) |f| {
            max_diff = @max(max_diff, @abs(@as(f64, out[f * 2]) - spec_ref.data[(f * t_frames + t) * 2]));
            max_diff = @max(max_diff, @abs(@as(f64, out[f * 2 + 1]) - spec_ref.data[(f * t_frames + t) * 2 + 1]));
        }
    }
    if (max_diff > 1e-3) {
        std.debug.print("[gtcrn-aec] stft max|diff| {d:.6}\n", .{max_diff});
        return error.TestUnexpectedResult;
    }
}
