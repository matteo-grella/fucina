//! Pocket TTS parity vs the reference dumps (tools/pocket/pocket_dump.py,
//! text "Parity check.", voice alba, torch seed 42): tokenizer, text-pass
//! backbone, bos step, flow head (with the reference's sampled noise
//! INJECTED — torch RNG is not reproduced), and the first Mimi frame's four
//! stages. Skips without models/pocket-tts/pocket-tts-english-v2.gguf +
//! refs/pocket-tts-dumps/.
//!
//! Native backend only: these are real-model golden forwards — they pin
//! MODEL WIRING, not kernel math, so the scalar reference leg skips them
//! (its kernel coverage lives in the exec/backend suites).

const std = @import("std");
const fucina = @import("fucina");
const pocket = @import("model.zig");

const gguf = fucina.gguf;

const model_path = "models/pocket-tts/pocket-tts-english-v2.gguf";
const dumps = "refs/pocket-tts-dumps";

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

fn npyLoad(allocator: std.mem.Allocator, name: []const u8) !Npy {
    var pbuf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/{s}.npy", .{ dumps, name });
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
    const is_i64 = std.mem.indexOf(u8, header, "<i8") != null;
    if (!is_i64 and std.mem.indexOf(u8, header, "<f4") == null) return error.BadNpy;

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
    if (is_i64) {
        for (data, 0..) |*v, i| v.* = @floatFromInt(std.mem.readInt(i64, payload[i * 8 ..][0..8], .little));
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
    return .{ .cosine = dot / (@sqrt(na) * @sqrt(nb) + 1e-30), .max_diff = md };
}

fn expectClose(name: []const u8, got: []const f32, want: []const f32, max_diff: f64) !void {
    const s = stats(got, want);
    if (s.cosine < 0.99999 or s.max_diff > max_diff) {
        std.debug.print("[pocket] {s}: cosine {d:.7} max|diff| {d:.6}\n", .{ name, s.cosine, s.max_diff });
        return error.TestUnexpectedResult;
    }
}

test "pocket-tts: tokenizer, backbone, flow head, mimi frame vs reference dumps" {
    if (comptime @import("fucina").internal.backend_mod.active_kind != .native) return error.SkipZigTest; // real-model goldens: native only
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch @panic("leak");
    const allocator = gpa.allocator();

    var file = gguf.File.loadMmap(allocator, std.testing.io, model_path) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer file.deinit();
    var ref_tokens = npyLoad(allocator, "text_tokens") catch return error.SkipZigTest;
    defer ref_tokens.deinit(allocator);

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();
    var model = try pocket.Model.load(&ctx, &file);
    defer model.deinit();

    // 1) tokenizer: the dump was made with v1 flags (8 space-pad tokens
    // prepended); the run itself consumed the tail.
    const ids = try model.tokenize(allocator, "Parity check.");
    defer allocator.free(ids);
    const want_n = ref_tokens.data.len - 8;
    try std.testing.expectEqual(want_n, ids.len);
    for (ids, 0..) |id, i| {
        try std.testing.expectEqual(@as(u32, @intFromFloat(ref_tokens.data[8 + i])), id);
    }

    // 2) voice prefix + text pass
    var kv = try pocket.Kv.init(allocator, model.layers, 2048, model.heads, model.hd);
    defer kv.deinit();
    try pocket.loadVoice(&model, &file, &kv, "alba");

    var fl = try pocket.FlowLm.init(allocator, &model, 64);
    defer fl.deinit();

    var b_in = try npyLoad(allocator, "backbone_in_000");
    defer b_in.deinit(allocator);
    var b_out = try npyLoad(allocator, "backbone_out_000");
    defer b_out.deinit(allocator);
    const s_text = b_in.shape[1];
    try std.testing.expectEqual(ids.len, s_text);

    const rows = try allocator.alloc(f32, s_text * model.d);
    defer allocator.free(rows);
    for (ids, 0..) |id, i| {
        @memcpy(rows[i * model.d ..][0..model.d], model.embed[id * model.d ..][0..model.d]);
    }
    try expectClose("text_embeds", rows, b_in.data, 1e-4);

    const cond = try allocator.alloc(f32, model.d);
    defer allocator.free(cond);
    try fl.forward(&kv, rows, s_text, cond);
    try expectClose("backbone_out_000", fl.x[0 .. s_text * model.d], b_out.data, 2e-3);

    // 3) flow head on the text-pass cond (sample discarded by the reference
    // but dumped): inject x0_000 → x1_000
    var x0 = try npyLoad(allocator, "flow_noise_x0_000");
    defer x0.deinit(allocator);
    var x1_ref = try npyLoad(allocator, "flow_latent_x1_000");
    defer x1_ref.deinit(allocator);
    var x1: [pocket.latent_dim]f32 = undefined;
    try pocket.lsdStep(&model, allocator, cond, x0.data, &x1);
    try expectClose("flow_x1_000", &x1, x1_ref.data, 2e-3);

    // 4) bos step: input row must match backbone_in_001; out → x1_001
    var b_in1 = try npyLoad(allocator, "backbone_in_001");
    defer b_in1.deinit(allocator);
    const bos_row = try allocator.alloc(f32, model.d);
    defer allocator.free(bos_row);
    for (bos_row, 0..) |*v, i| v.* = 0 + blk: {
        var acc: f32 = 0;
        for (0..pocket.latent_dim) |j| acc += model.input_linear[i * pocket.latent_dim + j] * model.bos_emb[j];
        break :blk acc;
    };
    try expectClose("bos_input_row", bos_row, b_in1.data, 1e-4);
    try fl.forward(&kv, bos_row, 1, cond);

    var x0_1 = try npyLoad(allocator, "flow_noise_x0_001");
    defer x0_1.deinit(allocator);
    var x1_ref1 = try npyLoad(allocator, "flow_latent_x1_001");
    defer x1_ref1.deinit(allocator);
    try pocket.lsdStep(&model, allocator, cond, x0_1.data, &x1);
    try expectClose("flow_x1_001", &x1, x1_ref1.data, 2e-3);

    // 5) next input row = input_linear(x1_001) must match backbone_in_002
    var b_in2 = try npyLoad(allocator, "backbone_in_002");
    defer b_in2.deinit(allocator);
    const row2 = try allocator.alloc(f32, model.d);
    defer allocator.free(row2);
    for (row2, 0..) |*v, i| {
        var acc: f32 = 0;
        for (0..pocket.latent_dim) |j| acc += model.input_linear[i * pocket.latent_dim + j] * x1_ref1.data[j];
        v.* = acc;
    }
    try expectClose("latent_input_row", row2, b_in2.data, 2e-3);

    // 6) mimi frame 0 from the reference latent x1_001: four stages
    var mimi = try pocket.Mimi.init(allocator, &model, 64);
    defer mimi.deinit();
    var m_in = try npyLoad(allocator, "mimi_latent_in_000");
    defer m_in.deinit(allocator);
    var m_up = try npyLoad(allocator, "mimi_upsampled_000");
    defer m_up.deinit(allocator);
    var m_tf = try npyLoad(allocator, "mimi_dec_transformer_out_000");
    defer m_tf.deinit(allocator);
    var m_pcm = try npyLoad(allocator, "mimi_pcm_chunk_000");
    defer m_pcm.deinit(allocator);

    const cap_q = try allocator.alloc(f32, model.mimi_d);
    defer allocator.free(cap_q);
    const cap_up = try allocator.alloc(f32, model.mimi_d * 16);
    defer allocator.free(cap_up);
    const cap_tf = try allocator.alloc(f32, model.mimi_d * 16);
    defer allocator.free(cap_tf);
    mimi.cap_q = cap_q;
    mimi.cap_up = cap_up;
    mimi.cap_tf = cap_tf;

    var pcm: [pocket.frame_samples]f32 = undefined;
    try mimi.decodeFrame(x1_ref1.data, &pcm);
    try expectClose("mimi_latent_in", cap_q, m_in.data, 2e-3);
    try expectClose("mimi_upsampled", cap_up, m_up.data, 2e-3);
    try expectClose("mimi_dec_tf", cap_tf, m_tf.data, 5e-3);
    try expectClose("mimi_pcm", &pcm, m_pcm.data, 5e-3);
    std.debug.print("[pocket] parity: tokenizer + backbone + flow + mimi all within gates\n", .{});
}
