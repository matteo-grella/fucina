//! DeepSeek-V4 fp4-expert → tied-PTQTP GGUF converter.
//!
//!   zig build convert-ds4-fp4 -Doptimize=ReleaseFast -- \
//!       --src-dir /path/to/deepseek-v4-fp8 --trunk-gguf TRUNK.gguf --out OUT.gguf \
//!       [--planes K] [--no-tie] [--dry-run] [--smoke N]
//!
//! The DeepSeek-V4-Flash checkpoint stores its routed experts in MXFP4:
//! fp4 e2m1 codes packed two per byte into I8 safetensors (`layers.N.ffn.
//! experts.E.{w1,w2,w3}.weight`, stored `[out, in/2]`, logical `[out, in]`)
//! with one e8m0 power-of-two scale per 32 elements along K (`.scale`,
//! `[out, in/32]`, per the checkpoint's own inference/convert.py: low
//! nibble = element 2i, value = FP4_TABLE[code] * 2^(scale-127)). Both
//! factors are powers-of-two-scaled dyadics, so the f32 dequant here is
//! EXACT — quantizing from it is a single-hop requant of the released
//! expert weights, unlike requantizing an intermediate GGUF.
//!
//! Output: the trunk GGUF's metadata and every non-expert tensor pass
//! through byte-verbatim (identical attention/router/shared-expert bytes —
//! a clean A/B against that file), while each `blk.N.ffn_*_exps.weight`
//! Q4_K stack is replaced by K plane-major TQ2_0 sibling tensors
//! (`<name>.ptqtpK`) solved per expert from the fp4 source, following the
//! src/ptqtp_gguf.zig persistence conventions (`fucina.ptqtp.version`
//! stamp; `fucina.ptqtp.tie_scales` when tie-fitted), so the output loads
//! through the deepseek4 pair-detection and streams through the
//! ExpertStore multi-plane path.
//!
//! Memory discipline: the writer's streaming path (`declareTensor` +
//! `beginStream`) plus K reusable plane stacks and one expert's f32 slice
//! — the whole conversion holds ~1.2 GiB at K=2 regardless of model size.
//! Source shard pages are `release`d expert-by-expert.
//!
//! `--smoke N` converts nothing: it dequantizes two experts of layer N
//! from the fp4 source AND from the trunk GGUF's Q4_K stacks and reports
//! cosine/relative error between the two (both derive from the same
//! released weights, so agreement ~cos>0.99 validates the name map, the
//! nibble order, the scale layout, and the expert-major slicing in one
//! shot), then times one tied solve to project the full run.

const builtin = @import("builtin");
const std = @import("std");
const fucina = @import("fucina");

const gguf = fucina.gguf;
const ptqtp = fucina.ptqtp;
const ptqtp_gguf = fucina.ptqtp_gguf;
const safetensors = fucina.safetensors;

const usage =
    "usage: zig build convert-ds4-fp4 -Doptimize=ReleaseFast -- --src-dir DIR --trunk-gguf TRUNK.gguf --out OUT.gguf [--planes K] [--no-tie] [--dry-run] [--smoke N] [--verify N]\n" ++
    "       zig build convert-ds4-fp4 -- --repack-native PTQTP.gguf --out OUT.gguf\n" ++
    "       zig build convert-ds4-fp4 -- --src-dir DIR --trunk-gguf TRUNK.gguf --out OUT.gguf --repack-mxfp4   (lossless native-fp4 experts)\n" ++
    "  --smoke N   compare layer N fp4-dequant vs the trunk GGUF's Q4_K experts (no output)\n" ++
    "  --verify N  re-solve sample experts of layer N and byte-compare against the planes stored in --out (post-conversion integrity check)\n" ++
    "  --repack-native  transform a tied-K2 sibling-plane PTQTP GGUF into the native pre-folded\n" ++
    "                   expert format (tq2_0_fx4): each blk.N.ffn_*_exps plane pair folds into one\n" ++
    "                   base-named 4-bit pack tensor; no re-solve, pure I/O + pack (~minutes)\n";

/// e2m1 code values, index = 4-bit code, bit 3 = sign — the checkpoint's
/// own FP4_TABLE (inference/convert.py).
const fp4_table = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

const fp4_group = 32; // elements per e8m0 scale along K

/// e8m0 (power-of-two exponent byte, bias 127) → f32. Exact for every
/// non-NaN byte; 0 maps to 2^-127 (subnormal). 0xFF is NaN in the OCP
/// spec and hard-errors in `dequantExpertFp4` before reaching here — a
/// NaN marker must abort the conversion, not quantize to a zero group.
/// (The 0731 checkpoint's e8m0 scale bytes lie in [114, 126]; 0xFF and
/// 0x00 do not occur in its 35,718 scale tensors.)
fn e8m0ToF32(b: u8) f32 {
    return std.math.scalbn(@as(f32, 1.0), @as(i32, b) - 127);
}

/// Dequantize one expert matrix (row-major [out, in] f32) from fp4-x2
/// packed bytes + per-32 e8m0 scales. `in` must be a multiple of 32
/// (4096/2048 here). Errors on a 0xFF (NaN) scale byte.
fn dequantExpertFp4(w_packed: []const u8, scales: []const u8, out_dim: usize, in_dim: usize, dst: []f32) !void {
    const in_half = in_dim / 2;
    const in_groups = in_dim / fp4_group;
    for (0..out_dim) |r| {
        const row_packed = w_packed[r * in_half ..][0..in_half];
        const row_scales = scales[r * in_groups ..][0..in_groups];
        const row_dst = dst[r * in_dim ..][0..in_dim];
        for (0..in_groups) |g| {
            if (row_scales[g] == 0xFF) return error.NanE8m0Scale;
            const s = e8m0ToF32(row_scales[g]);
            const packed_bytes = row_packed[g * (fp4_group / 2) ..][0 .. fp4_group / 2];
            for (packed_bytes, 0..) |b, i| {
                row_dst[g * fp4_group + 2 * i] = fp4_table[b & 0xF] * s;
                row_dst[g * fp4_group + 2 * i + 1] = fp4_table[b >> 4] * s;
            }
        }
    }
}

/// Always-on invocation guard: hand-computed dequant cases pin the nibble
/// order (low = element 2i), the value table, and the e8m0 bias.
fn selftest() !void {
    if (e8m0ToF32(127) != 1.0 or e8m0ToF32(126) != 0.5 or e8m0ToF32(130) != 8.0) return error.SelftestFailed;
    var w: [64]f32 = undefined;
    const packed_bytes = [32]u8{
        0x21, 0x9C, 0x08, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x67, 0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    const scales = [2]u8{ 127, 126 };
    try dequantExpertFp4(&packed_bytes, &scales, 1, 64, &w);
    const expect = [8]f32{ 0.5, 1.0, -2.0, -0.5, 0.0, 0.0, -6.0, -6.0 };
    for (expect, w[0..8]) |e, got| {
        if (got != e) return error.SelftestFailed;
    }
    // second group: scale 0.5 — 0x67: low 0x7 → 6.0*0.5, high 0x6 → 4.0*0.5
    if (w[32] != 3.0 or w[33] != 2.0 or w[34] != 0.0) return error.SelftestFailed;
    // a NaN e8m0 scale byte must abort, never quantize as a zero group
    const nan_scales = [2]u8{ 127, 0xFF };
    if (dequantExpertFp4(&packed_bytes, &nan_scales, 1, 64, &w)) |_| {
        return error.SelftestFailed;
    } else |err| if (err != error.NanE8m0Scale) return error.SelftestFailed;

    // Recode leg: the same hand case through the mxfp4 recode must
    // dequantize (ggml split-halves layout, doubled table x halved scale)
    // to the BITWISE identical f32 row — every product is an exact dyadic.
    var fresh: [64]f32 = undefined;
    try dequantExpertFp4(&packed_bytes, &scales, 1, 64, &fresh);
    var blocks: [2]BlockMXFP4 = undefined;
    recodeGroupMxfp4(packed_bytes[0..16], scales[0], &blocks[0]);
    recodeGroupMxfp4(packed_bytes[16..32], scales[1], &blocks[1]);
    var via_ggml: [64]f32 = undefined;
    try fucina.internal.backend_mod.quantized_matmul.dequantizeRowMXFP4Into(&via_ggml, &blocks);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(&fresh), std.mem.sliceAsBytes(&via_ggml))) return error.SelftestFailed;
}

/// All safetensors shards of the source dir, with one global name index.
const ShardSet = struct {
    allocator: std.mem.Allocator,
    files: std.ArrayList(safetensors.File),
    index: std.StringHashMap(*const safetensors.TensorInfo),

    fn load(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8, stdout: *std.Io.Writer) !ShardSet {
        var self = ShardSet{
            .allocator = allocator,
            .files = .empty,
            .index = std.StringHashMap(*const safetensors.TensorInfo).init(allocator),
        };
        errdefer self.deinit();

        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| allocator.free(n);
            names.deinit(allocator);
        }
        {
            var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
            defer dir.close(io);
            var walker = try dir.walk(allocator);
            defer walker.deinit();
            while (try walker.next(io)) |entry| {
                if (entry.kind != .file) continue;
                if (std.mem.indexOfScalar(u8, entry.path, '/') != null) continue; // top level only
                if (!std.mem.startsWith(u8, entry.path, "model-")) continue; // skips "._" AppleDouble twins
                if (!std.mem.endsWith(u8, entry.path, ".safetensors")) continue;
                try names.append(allocator, try allocator.dupe(u8, entry.path));
            }
        }
        if (names.items.len == 0) return error.NoShardsFound;
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);

        try self.files.ensureTotalCapacity(allocator, names.items.len);
        for (names.items) |shard_name| {
            const path = try std.fs.path.join(allocator, &.{ dir_path, shard_name });
            defer allocator.free(path);
            const file = try safetensors.File.loadMmap(allocator, io, path);
            self.files.appendAssumeCapacity(file);
        }
        // Index AFTER all loads: ArrayList growth is over (capacity fixed),
        // so &files.items[i] stays stable; TensorInfo names live in each File.
        var tensor_count: usize = 0;
        for (self.files.items) |*file| {
            for (file.tensors) |*info| {
                try self.index.put(info.name, info);
                tensor_count += 1;
            }
        }
        try stdout.print("source: {d} shards, {d} tensors\n", .{ self.files.items.len, tensor_count });
        return self;
    }

    fn get(self: *const ShardSet, name: []const u8) !*const safetensors.TensorInfo {
        return self.index.get(name) orelse error.SourceTensorNotFound;
    }

    fn deinit(self: *ShardSet) void {
        self.index.deinit();
        for (self.files.items) |*file| file.deinit();
        self.files.deinit(self.allocator);
    }
};

/// blk.N.ffn_{gate,up,down}_exps.weight → layer + safetensors projection.
const ExpsName = struct {
    layer: usize,
    proj: []const u8, // "w1" gate / "w3" up / "w2" down
};

fn parseExpsName(tensor_name: []const u8) ?ExpsName {
    if (!std.mem.startsWith(u8, tensor_name, "blk.")) return null;
    const rest = tensor_name["blk.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    const suffix = rest[dot + 1 ..];
    const proj: []const u8 = if (std.mem.eql(u8, suffix, "ffn_gate_exps.weight"))
        "w1"
    else if (std.mem.eql(u8, suffix, "ffn_up_exps.weight"))
        "w3"
    else if (std.mem.eql(u8, suffix, "ffn_down_exps.weight"))
        "w2"
    else
        return null;
    return .{ .layer = layer, .proj = proj };
}

const BlockMXFP4 = fucina.internal.backend_mod.quantized_matmul.BlockMXFP4;

/// One 32-element group, source pair-packed nibbles (elements 2i, 2i+1 per
/// byte) → ggml split-halves (qs[j] holds elements j and j+16). Codes and
/// the e8m0 scale byte pass through UNCHANGED: the value tables are
/// identical (ggml's doubled decode table folds its /2 into the halved
/// block scale), so the recode is lossless by construction.
fn recodeGroupMxfp4(packed_bytes: *const [fp4_group / 2]u8, scale: u8, dst: *BlockMXFP4) void {
    dst.e = scale;
    for (0..fp4_group / 2) |j| {
        const lo_t = j; // element j
        const hi_t = j + fp4_group / 2; // element j + 16
        const lo: u8 = if (lo_t % 2 == 0) packed_bytes[lo_t / 2] & 0x0f else packed_bytes[lo_t / 2] >> 4;
        const hi: u8 = if (hi_t % 2 == 0) packed_bytes[hi_t / 2] & 0x0f else packed_bytes[hi_t / 2] >> 4;
        dst.qs[j] = lo | (hi << 4);
    }
}

/// Fetch + validate one expert's fp4 weight/scale pair and recode it into
/// ggml MXFP4 blocks (row-major, in_dim/32 blocks per row). Same source
/// contract as `dequantSourceExpert`, including the NaN-scale abort.
fn recodeExpertMxfp4(
    shards: *const ShardSet,
    layer: usize,
    expert: usize,
    proj: []const u8,
    out_dim: usize,
    in_dim: usize,
    dst: []BlockMXFP4,
) !void {
    var name_buf: [96]u8 = undefined;
    const w_name = try std.fmt.bufPrint(&name_buf, "layers.{d}.ffn.experts.{d}.{s}.weight", .{ layer, expert, proj });
    const w_info = try shards.get(w_name);
    if (w_info.dtype != .I8) return error.UnexpectedSourceDtype;
    if (w_info.shape.len != 2 or w_info.shape[0] != out_dim or w_info.shape[1] * 2 != in_dim) return error.UnexpectedSourceShape;

    var scale_buf: [96]u8 = undefined;
    const s_name = try std.fmt.bufPrint(&scale_buf, "layers.{d}.ffn.experts.{d}.{s}.scale", .{ layer, expert, proj });
    const s_info = try shards.get(s_name);
    if (s_info.dtype != .F8_E8M0) return error.UnexpectedSourceDtype;
    if (s_info.shape.len != 2 or s_info.shape[0] != out_dim or s_info.shape[1] * fp4_group != in_dim) return error.UnexpectedSourceShape;

    const in_groups = in_dim / fp4_group;
    const in_half = in_dim / 2;
    if (dst.len != out_dim * in_groups) return error.UnexpectedSourceShape;
    for (0..out_dim) |r| {
        const row_packed = w_info.data[r * in_half ..][0..in_half];
        const row_scales = s_info.data[r * in_groups ..][0..in_groups];
        for (0..in_groups) |g| {
            if (row_scales[g] == 0xFF) return error.NanE8m0Scale;
            recodeGroupMxfp4(row_packed[g * (fp4_group / 2) ..][0 .. fp4_group / 2], row_scales[g], &dst[r * in_groups + g]);
        }
    }
    gguf.release(w_info.data);
    gguf.release(s_info.data);
}

/// Fetch + validate one expert's fp4 weight/scale pair and dequantize it
/// into `dst` (row-major [out, in] f32). Releases the shard pages after.
fn dequantSourceExpert(
    shards: *const ShardSet,
    layer: usize,
    expert: usize,
    proj: []const u8,
    out_dim: usize,
    in_dim: usize,
    dst: []f32,
) !void {
    var name_buf: [96]u8 = undefined;
    const w_name = try std.fmt.bufPrint(&name_buf, "layers.{d}.ffn.experts.{d}.{s}.weight", .{ layer, expert, proj });
    const w_info = try shards.get(w_name);
    if (w_info.dtype != .I8) return error.UnexpectedSourceDtype;
    if (w_info.shape.len != 2 or w_info.shape[0] != out_dim or w_info.shape[1] * 2 != in_dim) return error.UnexpectedSourceShape;

    var scale_buf: [96]u8 = undefined;
    const s_name = try std.fmt.bufPrint(&scale_buf, "layers.{d}.ffn.experts.{d}.{s}.scale", .{ layer, expert, proj });
    const s_info = try shards.get(s_name);
    if (s_info.dtype != .F8_E8M0) return error.UnexpectedSourceDtype;
    if (s_info.shape.len != 2 or s_info.shape[0] != out_dim or s_info.shape[1] * fp4_group != in_dim) return error.UnexpectedSourceShape;

    try dequantExpertFp4(w_info.data, s_info.data, out_dim, in_dim, dst);
    gguf.release(w_info.data);
    gguf.release(s_info.data);
}

const StackStats = struct {
    rel_sum: f64 = 0,
    rel_max: f64 = 0,
    unconverged: usize = 0,
    experts: usize = 0,

    fn add(self: *StackStats, stats: ptqtp.MatrixStats) void {
        self.rel_sum += stats.rel_frob_err;
        self.rel_max = @max(self.rel_max, stats.rel_frob_err);
        self.unconverged += stats.unconverged_groups;
        self.experts += 1;
    }
};

fn mib(bytes: anytype) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn cosineAndRelErr(a: []const f32, b: []const f32) struct { cos: f64, rel: f64 } {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    var diff: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
        const d = @as(f64, x) - y;
        diff += d * d;
    }
    const denom = @sqrt(na) * @sqrt(nb);
    return .{
        .cos = if (denom == 0) 0 else dot / denom,
        .rel = if (na == 0) 0 else @sqrt(diff / na),
    };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;
    const allocator = std.heap.smp_allocator;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    try selftest();

    var src_dir: ?[]const u8 = null;
    var trunk_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var planes: u8 = 2;
    var tie = true;
    var dry_run = false;
    var smoke_layer: ?usize = null;
    var verify_layer: ?usize = null;
    var repack_path: ?[]const u8 = null;
    var repack_slab_path: ?[]const u8 = null;
    var repack_mxfp4 = false;
    var trunk_only = false;

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--src-dir")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSrcDir;
            src_dir = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--src-dir=")) {
            src_dir = arg["--src-dir=".len..];
        } else if (std.mem.eql(u8, arg, "--trunk-gguf")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingTrunkPath;
            trunk_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--trunk-gguf=")) {
            trunk_path = arg["--trunk-gguf=".len..];
        } else if (std.mem.eql(u8, arg, "--out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingOutPath;
            out_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--out=")) {
            out_path = arg["--out=".len..];
        } else if (std.mem.startsWith(u8, arg, "--planes=")) {
            planes = try std.fmt.parseInt(u8, arg["--planes=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--planes")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPlaneCount;
            planes = try std.fmt.parseInt(u8, args[arg_i], 10);
        } else if (std.mem.eql(u8, arg, "--no-tie")) {
            tie = false;
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            dry_run = true;
        } else if (std.mem.startsWith(u8, arg, "--smoke=")) {
            smoke_layer = try std.fmt.parseInt(usize, arg["--smoke=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--smoke")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingSmokeLayer;
            smoke_layer = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--verify=")) {
            verify_layer = try std.fmt.parseInt(usize, arg["--verify=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--verify")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingVerifyLayer;
            verify_layer = try std.fmt.parseInt(usize, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--repack-native=")) {
            repack_path = arg["--repack-native=".len..];
        } else if (std.mem.eql(u8, arg, "--repack-native")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepackPath;
            repack_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--repack-slab=")) {
            repack_slab_path = arg["--repack-slab=".len..];
        } else if (std.mem.eql(u8, arg, "--repack-slab")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingRepackPath;
            repack_slab_path = args[arg_i];
        } else if (std.mem.eql(u8, arg, "--repack-mxfp4")) {
            repack_mxfp4 = true;
        } else if (std.mem.eql(u8, arg, "--trunk-only")) {
            trunk_only = true;
        } else {
            try stdout.print("unknown argument: {s}\n{s}", .{ arg, usage });
            return error.UnknownArgument;
        }
    }
    if (repack_path) |src_path| {
        const dst_path = out_path orelse {
            try stdout.print("{s}", .{usage});
            return error.MissingOutPath;
        };
        if (std.mem.eql(u8, dst_path, src_path)) {
            try stdout.print("--out must differ from --repack-native: passthrough bytes stream from the source mmap\n", .{});
            return error.UnsupportedArgumentCombination;
        }
        return repackNative(allocator, io, src_path, dst_path, stdout);
    }
    if (repack_slab_path) |src_path| {
        const dst_path = out_path orelse {
            try stdout.print("{s}", .{usage});
            return error.MissingOutPath;
        };
        if (std.mem.eql(u8, dst_path, src_path)) {
            try stdout.print("--out must differ from --repack-slab: passthrough bytes stream from the source mmap\n", .{});
            return error.UnsupportedArgumentCombination;
        }
        return repackSlab(allocator, io, src_path, dst_path, stdout);
    }
    if (trunk_only) {
        const trunk_file_path = trunk_path orelse {
            try stdout.print("{s}", .{usage});
            return error.MissingTrunkPath;
        };
        const dst_path = out_path orelse {
            try stdout.print("{s}", .{usage});
            return error.MissingOutPath;
        };
        if (std.mem.eql(u8, dst_path, trunk_file_path)) {
            try stdout.print("--out must differ from --trunk-gguf: passthrough bytes stream from the trunk mmap\n", .{});
            return error.UnsupportedArgumentCombination;
        }
        var trunk_file = try gguf.File.loadMmapAuto(allocator, io, trunk_file_path);
        defer trunk_file.deinit();
        return trunkOnly(allocator, io, &trunk_file, dst_path, stdout);
    }

    const src = src_dir orelse {
        try stdout.print("{s}", .{usage});
        return error.MissingSrcDir;
    };
    const trunk_file_path = trunk_path orelse {
        try stdout.print("{s}", .{usage});
        return error.MissingTrunkPath;
    };
    if (planes < 1 or planes > 3) return error.InvalidPlaneCount;
    if (tie and planes < 2) return error.InvalidPlaneCount;

    var trunk = try gguf.File.loadMmapAuto(allocator, io, trunk_file_path);
    defer trunk.deinit();

    var shards = try ShardSet.load(allocator, io, src, stdout);
    defer shards.deinit();

    const options = ptqtp.Options{ .planes = planes, .tie_scales = tie };

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    if (repack_mxfp4) {
        const dst_path = out_path orelse {
            try stdout.print("{s}", .{usage});
            return error.MissingOutPath;
        };
        if (std.mem.eql(u8, dst_path, trunk_file_path)) {
            try stdout.print("--out must differ from --trunk-gguf: passthrough bytes stream from the trunk mmap\n", .{});
            return error.UnsupportedArgumentCombination;
        }
        return repackMxfp4(allocator, io, &trunk, &shards, dst_path, stdout);
    }
    if (smoke_layer) |layer| {
        return smoke(allocator, io, &ctx, &trunk, &shards, layer, options, stdout);
    }
    if (verify_layer) |layer| {
        const converted_path = out_path orelse {
            try stdout.print("{s}", .{usage});
            return error.MissingOutPath;
        };
        return verify(allocator, io, &ctx, &shards, converted_path, layer, options, stdout);
    }
    const dst_path = out_path orelse {
        try stdout.print("{s}", .{usage});
        return error.MissingOutPath;
    };
    if (std.mem.eql(u8, dst_path, trunk_file_path)) {
        try stdout.print("--out must differ from --trunk-gguf: passthrough bytes stream from the trunk mmap\n", .{});
        return error.UnsupportedArgumentCombination;
    }

    // The contract is an UNDECORATED trunk: pre-existing plane tensors
    // would pass through verbatim under this run's fresh global stamps and
    // mislabel their fit (free planes folded as tied = wrong logits). The
    // stamps are also skip-listed below so trunk-copied ones never survive.
    if (trunk.getInt(ptqtp_gguf.version_key) != null) return error.TrunkAlreadyDecorated;

    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.copyAllMetadata(&trunk, &.{ "split.no", "split.count", "split.tensors.count", ptqtp_gguf.version_key, ptqtp_gguf.tie_key });
    try writer.addMetaInt(ptqtp_gguf.version_key, u32, ptqtp_gguf.format_version);
    if (tie) try writer.addMetaInt(ptqtp_gguf.tie_key, u32, 1);

    // Plan pass: verbatim trunk, K plane tensors per expert stack.
    var stack_count: usize = 0;
    var passthrough_count: usize = 0;
    var total_src_bytes: u64 = 0;
    var total_dst_bytes: u64 = 0;
    var max_stack_blocks: usize = 0;
    var max_expert_f32: usize = 0;
    var name_buf: [192]u8 = undefined;
    for (trunk.tensors) |*info| {
        total_src_bytes += info.data.len;
        const dims = info.dims[0..info.n_dims];
        if (parseExpsName(info.name)) |exps| {
            _ = exps;
            if (info.n_dims != 3) return error.UnexpectedTrunkShape;
            const in_dim = info.dims[0];
            const out_dim = info.dims[1];
            const n_expert = info.dims[2];
            if (in_dim % ptqtp.block_len != 0) return error.UnexpectedTrunkShape;
            const plane_bytes = try gguf.tensorByteLen(.tq2_0, dims);
            total_dst_bytes += plane_bytes * planes;
            stack_count += 1;
            max_stack_blocks = @max(max_stack_blocks, n_expert * out_dim * (in_dim / ptqtp.block_len));
            max_expert_f32 = @max(max_expert_f32, out_dim * in_dim);
            if (dry_run) {
                try stdout.print("ptqtp {s} [{d} x {d} x {d}] {s} -> tq2_0 x{d}  ({d:.1} -> {d:.1} MiB)\n", .{
                    info.name, in_dim, out_dim, n_expert, @tagName(info.ggml_type), planes, mib(info.data.len), mib(plane_bytes * planes),
                });
                continue;
            }
            for (0..planes) |plane_i| {
                const plane_name = try ptqtp_gguf.planeName(&name_buf, info.name, plane_i);
                try writer.declareTensor(plane_name, .tq2_0, dims);
            }
        } else {
            if (std.mem.indexOf(u8, info.name, ".ptqtp") != null) return error.TrunkAlreadyDecorated;
            passthrough_count += 1;
            total_dst_bytes += info.data.len;
            if (dry_run) continue;
            try writer.declareTensor(info.name, info.ggml_type, dims);
        }
    }
    if (stack_count == 0) return error.NoExpertStacks;

    try stdout.print("plan: {d} expert stacks -> {d} plane tensors (K={d}{s}), {d} passthrough; bytes {d:.1} -> {d:.1} MiB\n", .{
        stack_count,             stack_count * planes, planes, if (tie) " tied" else "", passthrough_count,
        mib(total_src_bytes), mib(total_dst_bytes),
    });
    try stdout.flush();
    if (dry_run) return;

    // Reusable buffers: K plane stacks + one expert f32 slice.
    var plane_stacks: [3][]fucina.ptqtp.BlockTQ2_0 = .{ &.{}, &.{}, &.{} };
    for (0..planes) |k| plane_stacks[k] = try allocator.alloc(fucina.ptqtp.BlockTQ2_0, max_stack_blocks);
    defer for (0..planes) |k| allocator.free(plane_stacks[k]);
    const expert_f32 = try allocator.alloc(f32, max_expert_f32);
    defer allocator.free(expert_f32);

    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    var streamer = try writer.beginStream(&out_writer.interface);

    const total_start = nowNs(io);
    var solved_stacks: usize = 0;
    for (trunk.tensors) |*info| {
        if (parseExpsName(info.name)) |exps| {
            const in_dim = info.dims[0];
            const out_dim = info.dims[1];
            const n_expert = info.dims[2];
            const blocks_per_expert = out_dim * (in_dim / ptqtp.block_len);

            const stack_start = nowNs(io);
            var stats = StackStats{};
            for (0..n_expert) |e| {
                try dequantSourceExpert(&shards, exps.layer, e, exps.proj, out_dim, in_dim, expert_f32[0 .. out_dim * in_dim]);
                var pair = try ptqtp.quantizeMatrix(&ctx, expert_f32[0 .. out_dim * in_dim], out_dim, in_dim, options);
                defer pair.deinit(ctx.allocator);
                if (pair.planeCount() != planes) return error.PlaneCountMismatch;
                const src_planes = [3][]const fucina.ptqtp.BlockTQ2_0{ pair.plane1, pair.plane2, pair.plane3 };
                for (0..planes) |k| {
                    @memcpy(plane_stacks[k][e * blocks_per_expert ..][0..blocks_per_expert], src_planes[k]);
                }
                stats.add(pair.stats);
            }
            for (0..planes) |k| {
                try streamer.writeTensorData(std.mem.sliceAsBytes(plane_stacks[k][0 .. n_expert * blocks_per_expert]));
            }
            solved_stacks += 1;
            const elapsed_s = @as(f64, @floatFromInt(nowNs(io) - total_start)) / 1e9;
            const eta_s = elapsed_s / @as(f64, @floatFromInt(solved_stacks)) * @as(f64, @floatFromInt(stack_count - solved_stacks));
            try stdout.print("[{d}/{d}] {s}: mean_rel {d:.4} max_rel {d:.4} unconverged {d}  {d:.1}s  elapsed {d:.1}m eta {d:.1}m\n", .{
                solved_stacks,                            stack_count,      info.name,
                stats.rel_sum / @as(f64, @floatFromInt(stats.experts)), stats.rel_max, stats.unconverged,
                @as(f64, @floatFromInt(nowNs(io) - stack_start)) / 1e9,  elapsed_s / 60.0, eta_s / 60.0,
            });
            try stdout.flush();
        } else {
            gguf.prefetch(info.data);
            try streamer.writeTensorData(info.data);
            gguf.release(info.data);
        }
    }
    try streamer.finish();
    try out_writer.interface.flush();
    try stdout.print("done: {s} ({d:.1} GiB) in {d:.1} min\n", .{
        dst_path,
        @as(f64, @floatFromInt(total_dst_bytes)) / (1024.0 * 1024.0 * 1024.0),
        @as(f64, @floatFromInt(nowNs(io) - total_start)) / 6e10,
    });
}

/// Transform a tied-K2 sibling-plane PTQTP GGUF into the native pre-folded
/// expert format: every `blk.N.ffn_*_exps.weight.ptqtp0/1` pair becomes one
/// base-named `.tq2_0_fx4` tensor (the exact bytes fill-time folding
/// produces — `packMatmulRhsTQ2_0Foldedx4Into` per expert), everything else
/// passes through byte-verbatim minus the `fucina.ptqtp.*` stamps (the
/// native type needs no pair-detection). Pure I/O + pack, no re-solve —
/// decode output must be bit-identical to folded serving of the source.
/// Lossless native-fp4 output: every expert stack recoded byte-for-byte
/// into ggml MXFP4 (same codes, same scales, ggml block layout), trunk
/// passthrough verbatim. No quantization step anywhere — the output serves
/// the original released expert weights.
/// One expert-stack derivation from the SOURCE checkpoint (used when the
/// trunk donor carries no expert tensors — a `--trunk-only` donor): layers
/// and expert counts come from the shard name index, dims from the shard
/// shapes. Byte-for-byte the same stacks a full trunk would declare; only
/// the tensor ORDER in the output differs (experts appended after the
/// passthrough trunk), which loaders resolve by name.
const DerivedStack = struct {
    layer: usize,
    proj: []const u8, // "w1" / "w3" / "w2"
    name_suffix: []const u8,
    in_dim: usize,
    out_dim: usize,
    n_expert: usize,
};

fn deriveStacks(allocator: std.mem.Allocator, shards: *const ShardSet) ![]DerivedStack {
    var max_layer: usize = 0;
    var max_expert: usize = 0;
    var any = false;
    var it = shards.index.keyIterator();
    while (it.next()) |key| {
        const name = key.*;
        if (!std.mem.startsWith(u8, name, "layers.")) continue;
        const rest = name["layers.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
        const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch continue;
        const suffix = rest[dot + 1 ..];
        if (!std.mem.startsWith(u8, suffix, "ffn.experts.")) continue;
        const erest = suffix["ffn.experts.".len..];
        const edot = std.mem.indexOfScalar(u8, erest, '.') orelse continue;
        const expert = std.fmt.parseInt(usize, erest[0..edot], 10) catch continue;
        if (!std.mem.endsWith(u8, erest, ".w1.weight")) continue;
        any = true;
        max_layer = @max(max_layer, layer);
        max_expert = @max(max_expert, expert);
    }
    if (!any) return error.NoExpertStacks;

    var list: std.ArrayList(DerivedStack) = .empty;
    errdefer list.deinit(allocator);
    var name_buf: [96]u8 = undefined;
    for (0..max_layer + 1) |layer| {
        const probe = try std.fmt.bufPrint(&name_buf, "layers.{d}.ffn.experts.0.w1.weight", .{layer});
        const w1 = shards.index.get(probe) orelse continue; // dense layer
        const probe2 = try std.fmt.bufPrint(&name_buf, "layers.{d}.ffn.experts.0.w2.weight", .{layer});
        const w2 = shards.index.get(probe2) orelse return error.SourceTensorNotFound;
        if (w1.shape.len != 2 or w2.shape.len != 2) return error.UnexpectedSourceShape;
        const gu_in = w1.shape[1] * 2;
        const gu_out = w1.shape[0];
        const down_in = w2.shape[1] * 2;
        const down_out = w2.shape[0];
        const projs = [3]DerivedStack{
            .{ .layer = layer, .proj = "w1", .name_suffix = "ffn_gate_exps.weight", .in_dim = gu_in, .out_dim = gu_out, .n_expert = max_expert + 1 },
            .{ .layer = layer, .proj = "w3", .name_suffix = "ffn_up_exps.weight", .in_dim = gu_in, .out_dim = gu_out, .n_expert = max_expert + 1 },
            .{ .layer = layer, .proj = "w2", .name_suffix = "ffn_down_exps.weight", .in_dim = down_in, .out_dim = down_out, .n_expert = max_expert + 1 },
        };
        for (projs) |p| try list.append(allocator, p);
    }
    return list.toOwnedSlice(allocator);
}

/// Trunk-only donor: every tensor of the source EXCEPT the expert stacks,
/// metadata verbatim — the ~9 GB file that makes expert conversion
/// self-sufficient anywhere the fp4 checkpoint is available.
fn trunkOnly(
    allocator: std.mem.Allocator,
    io: std.Io,
    trunk: *const gguf.File,
    dst_path: []const u8,
    stdout: *std.Io.Writer,
) !void {
    if (trunk.getInt(ptqtp_gguf.version_key) != null) return error.TrunkAlreadyDecorated;
    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.copyAllMetadata(trunk, &.{ "split.no", "split.count", "split.tensors.count", ptqtp_gguf.version_key, ptqtp_gguf.tie_key });
    var kept: usize = 0;
    var dropped: usize = 0;
    var total_bytes: u64 = 0;
    for (trunk.tensors) |*info| {
        if (parseExpsName(info.name)) |_| {
            dropped += 1;
            continue;
        }
        if (std.mem.indexOf(u8, info.name, ".ptqtp") != null) return error.TrunkAlreadyDecorated;
        kept += 1;
        total_bytes += info.data.len;
        try writer.declareTensor(info.name, info.ggml_type, info.dims[0..info.n_dims]);
    }
    if (dropped == 0) return error.NoExpertStacks;
    try stdout.print("trunk-only: {d} tensors kept, {d} expert stacks dropped, {d:.1} MiB\n", .{ kept, dropped, mib(total_bytes) });
    try stdout.flush();

    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    var streamer = try writer.beginStream(&out_writer.interface);
    for (trunk.tensors) |*info| {
        if (parseExpsName(info.name)) |_| continue;
        gguf.prefetch(info.data);
        try streamer.writeTensorData(info.data);
        gguf.release(info.data);
    }
    try streamer.finish();
    try out_writer.interface.flush();
    try stdout.print("done: {s}\n", .{dst_path});
}

fn repackMxfp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    trunk: *const gguf.File,
    shards: *const ShardSet,
    dst_path: []const u8,
    stdout: *std.Io.Writer,
) !void {
    if (trunk.getInt(ptqtp_gguf.version_key) != null) return error.TrunkAlreadyDecorated;

    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.copyAllMetadata(trunk, &.{ "split.no", "split.count", "split.tensors.count", ptqtp_gguf.version_key, ptqtp_gguf.tie_key });

    var stack_count: usize = 0;
    var passthrough_count: usize = 0;
    var total_src_bytes: u64 = 0;
    var total_dst_bytes: u64 = 0;
    var max_stack_blocks: usize = 0;
    for (trunk.tensors) |*info| {
        total_src_bytes += info.data.len;
        const dims = info.dims[0..info.n_dims];
        if (parseExpsName(info.name)) |_| {
            if (info.n_dims != 3) return error.UnexpectedTrunkShape;
            const in_dim = dims[0];
            if (in_dim % fp4_group != 0) return error.UnexpectedTrunkShape;
            total_dst_bytes += try gguf.tensorByteLen(.mxfp4, dims);
            stack_count += 1;
            max_stack_blocks = @max(max_stack_blocks, dims[2] * dims[1] * (in_dim / 32));
            try writer.declareTensor(info.name, .mxfp4, dims);
        } else {
            if (std.mem.indexOf(u8, info.name, ".ptqtp") != null) return error.TrunkAlreadyDecorated;
            passthrough_count += 1;
            total_dst_bytes += info.data.len;
            try writer.declareTensor(info.name, info.ggml_type, dims);
        }
    }
    // Expert-less donor (--trunk-only output): derive the stacks from the
    // checkpoint itself and append their declarations after the trunk.
    var derived: []DerivedStack = &.{};
    defer if (derived.len > 0) allocator.free(derived);
    if (stack_count == 0) {
        derived = try deriveStacks(allocator, shards);
        var name_buf: [96]u8 = undefined;
        for (derived) |d| {
            if (d.in_dim % fp4_group != 0) return error.UnexpectedSourceShape;
            const dims = [3]u64{ d.in_dim, d.out_dim, d.n_expert };
            total_dst_bytes += try gguf.tensorByteLen(.mxfp4, &dims);
            stack_count += 1;
            max_stack_blocks = @max(max_stack_blocks, d.n_expert * d.out_dim * (d.in_dim / 32));
            const tname = try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ d.layer, d.name_suffix });
            try writer.declareTensor(tname, .mxfp4, &dims);
        }
    }
    if (stack_count == 0) return error.NoExpertStacks;
    try stdout.print("plan: {d} expert stacks -> native mxfp4 (lossless recode{s}), {d} passthrough; bytes {d:.1} -> {d:.1} MiB\n", .{
        stack_count, if (derived.len > 0) ", checkpoint-derived geometry" else "", passthrough_count, mib(total_src_bytes), mib(total_dst_bytes),
    });
    try stdout.flush();

    const stack = try allocator.alloc(BlockMXFP4, max_stack_blocks);
    defer allocator.free(stack);

    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    var streamer = try writer.beginStream(&out_writer.interface);

    const total_start = nowNs(io);
    var done: usize = 0;
    for (trunk.tensors) |*info| {
        if (parseExpsName(info.name)) |exps| {
            const in_dim = info.dims[0];
            const out_dim = info.dims[1];
            const n_expert = info.dims[2];
            const blocks_per_expert = out_dim * (in_dim / 32);
            for (0..n_expert) |e| {
                try recodeExpertMxfp4(shards, exps.layer, e, exps.proj, out_dim, in_dim, stack[e * blocks_per_expert ..][0..blocks_per_expert]);
            }
            try streamer.writeTensorData(std.mem.sliceAsBytes(stack[0 .. n_expert * blocks_per_expert]));
            done += 1;
            try stdout.print("[{d}/{d}] {s} -> mxfp4 ({d:.1} MiB)  {d:.1}s\n", .{
                done, stack_count, info.name, mib(n_expert * blocks_per_expert * @sizeOf(BlockMXFP4)),
                @as(f64, @floatFromInt(nowNs(io) - total_start)) / 1e9,
            });
            try stdout.flush();
        } else {
            gguf.prefetch(info.data);
            try streamer.writeTensorData(info.data);
            gguf.release(info.data);
        }
    }
    // Checkpoint-derived stacks stream after the trunk, in declaration order.
    for (derived) |d| {
        const blocks_per_expert = d.out_dim * (d.in_dim / 32);
        for (0..d.n_expert) |e| {
            try recodeExpertMxfp4(shards, d.layer, e, d.proj, d.out_dim, d.in_dim, stack[e * blocks_per_expert ..][0..blocks_per_expert]);
        }
        try streamer.writeTensorData(std.mem.sliceAsBytes(stack[0 .. d.n_expert * blocks_per_expert]));
        done += 1;
        try stdout.print("[{d}/{d}] blk.{d}.{s} -> mxfp4 ({d:.1} MiB)  {d:.1}s\n", .{
            done, stack_count, d.layer, d.name_suffix, mib(d.n_expert * blocks_per_expert * @sizeOf(BlockMXFP4)),
            @as(f64, @floatFromInt(nowNs(io) - total_start)) / 1e9,
        });
        try stdout.flush();
    }
    try streamer.finish();
    try out_writer.interface.flush();
    try stdout.print("done: {s} ({d:.1} GiB) in {d:.1} min\n", .{
        dst_path,
        @as(f64, @floatFromInt(total_dst_bytes)) / (1024.0 * 1024.0 * 1024.0),
        @as(f64, @floatFromInt(nowNs(io) - total_start)) / 6e10,
    });
}

fn repackNative(
    allocator: std.mem.Allocator,
    io: std.Io,
    src_path: []const u8,
    dst_path: []const u8,
    stdout: *std.Io.Writer,
) !void {
    const bq = fucina.internal.backend_mod.quantized_matmul;

    var src = try gguf.File.loadMmapAuto(allocator, io, src_path);
    defer src.deinit();

    if ((src.getInt(ptqtp_gguf.version_key) orelse 0) != ptqtp_gguf.format_version) return error.SourceNotPtqtp;
    if ((src.getInt(ptqtp_gguf.tie_key) orelse 0) != 1) return error.SourceNotTied;

    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.copyAllMetadata(&src, &.{ "split.no", "split.count", "split.tensors.count", ptqtp_gguf.version_key, ptqtp_gguf.tie_key });

    // Plan pass. A `.ptqtp0` expert-stack tensor declares the base-named
    // fx4 pack; its `.ptqtp1` sibling is consumed silently; any other
    // decorated tensor would make the output claim stamps it dropped.
    var stack_count: usize = 0;
    var total_dst_bytes: u64 = 0;
    var max_pack_bytes: usize = 0;
    for (src.tensors) |*info| {
        const dims = info.dims[0..info.n_dims];
        if (std.mem.endsWith(u8, info.name, ".ptqtp0") or std.mem.endsWith(u8, info.name, ".ptqtp1")) {
            const base = info.name[0 .. info.name.len - ".ptqtp0".len];
            if (parseExpsName(base) == null) return error.UnsupportedRepackTensor;
            if (info.n_dims != 3 or info.ggml_type != .tq2_0) return error.UnsupportedRepackTensor;
            if (std.mem.endsWith(u8, info.name, ".ptqtp1")) continue;
            const pack_bytes = try gguf.tensorByteLen(.tq2_0_fx4, dims);
            try writer.declareTensor(base, .tq2_0_fx4, dims);
            stack_count += 1;
            total_dst_bytes += pack_bytes;
            max_pack_bytes = @max(max_pack_bytes, pack_bytes);
        } else if (std.mem.endsWith(u8, info.name, ".ptqtp2")) {
            return error.UnsupportedRepackTensor; // K=3 cannot fold into a nibble
        } else {
            try writer.declareTensor(info.name, info.ggml_type, dims);
            total_dst_bytes += info.data.len;
        }
    }
    if (stack_count == 0) return error.NoExpertStacks;

    const pack_buf = try allocator.alloc(bq.BlockTQ2_0Foldedx4, max_pack_bytes / @sizeOf(bq.BlockTQ2_0Foldedx4));
    defer allocator.free(pack_buf);

    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    var streamer = try writer.beginStream(&out_writer.interface);

    const total_start = nowNs(io);
    var name_buf: [192]u8 = undefined;
    var done: usize = 0;
    for (src.tensors) |*info| {
        if (std.mem.endsWith(u8, info.name, ".ptqtp1")) continue;
        if (std.mem.endsWith(u8, info.name, ".ptqtp0")) {
            const base = info.name[0 .. info.name.len - ".ptqtp0".len];
            const in_dim = info.dims[0];
            const out_dim = info.dims[1];
            const n_expert = info.dims[2];
            const bpc = in_dim / 256;
            const plane_blocks = out_dim * bpc; // per expert, per plane
            const fg = (out_dim / 4) * bpc; // folded blocks per expert

            const p1_info = try src.get(try ptqtp_gguf.planeName(&name_buf, base, 1));
            gguf.prefetch(info.data);
            gguf.prefetch(p1_info.data);
            const p0_blocks: []const fucina.ptqtp.BlockTQ2_0 = @alignCast(std.mem.bytesAsSlice(fucina.ptqtp.BlockTQ2_0, info.data));
            const p1_blocks: []const fucina.ptqtp.BlockTQ2_0 = @alignCast(std.mem.bytesAsSlice(fucina.ptqtp.BlockTQ2_0, p1_info.data));

            for (0..n_expert) |e| {
                // The rows container carries mutable blocks; the pack never
                // writes them, so the @constCast borrow of mmap bytes is
                // sound (same note as exec/moe.zig tq2_0View).
                var v0 = bq.QuantizedMatmulRhsTQ2_0{ .rows = .{ .allocator = null, .blocks = @constCast(p0_blocks[e * plane_blocks ..][0..plane_blocks]), .rows = out_dim, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = out_dim };
                var v1 = bq.QuantizedMatmulRhsTQ2_0{ .rows = .{ .allocator = null, .blocks = @constCast(p1_blocks[e * plane_blocks ..][0..plane_blocks]), .rows = out_dim, .cols = in_dim, .blocks_per_row = bpc }, .k = in_dim, .n = out_dim };
                try bq.packMatmulRhsTQ2_0Foldedx4Into(pack_buf[e * fg ..][0..fg], &v0, &v1);
            }
            try streamer.writeTensorData(std.mem.sliceAsBytes(pack_buf[0 .. n_expert * fg]));
            gguf.release(info.data);
            gguf.release(p1_info.data);
            done += 1;
            try stdout.print("[{d}/{d}] {s} -> tq2_0_fx4\n", .{ done, stack_count, base });
            try stdout.flush();
        } else {
            gguf.prefetch(info.data);
            try streamer.writeTensorData(info.data);
            gguf.release(info.data);
        }
    }
    try streamer.finish();
    try out_writer.interface.flush();
    try stdout.print("repacked {s} -> {s}: {d} stacks native-folded, {d:.1} GiB, {d:.1} min\n", .{
        src_path,
        dst_path,
        stack_count,
        @as(f64, @floatFromInt(total_dst_bytes)) / (1024.0 * 1024.0 * 1024.0),
        @as(f64, @floatFromInt(nowNs(io) - total_start)) / 6e10,
    });
}

/// Transform an fx4-experts GGUF into the slab-native container: each
/// layer's three `blk.N.ffn_*_exps.weight` tq2_0_fx4 stacks become ONE
/// `blk.N.ffn_exps_slab.weight` record tensor (.i8, dims [record_bytes,
/// n_expert]) whose per-expert bytes are the expert store's RAM slab
/// verbatim — gate|up|down pack sections at 16 KiB-aligned offsets, tail
/// padding zeroed — so a streamed miss is ONE contiguous pread. Everything
/// else (metadata, trunk, any dense fx4 tensors) passes through verbatim.
fn repackSlab(
    allocator: std.mem.Allocator,
    io: std.Io,
    src_path: []const u8,
    dst_path: []const u8,
    stdout: *std.Io.Writer,
) !void {
    const slab_align: usize = 16384;
    var src = try gguf.File.loadMmapAuto(allocator, io, src_path);
    defer src.deinit();

    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.copyAllMetadata(&src, &.{ "split.no", "split.count", "split.tensors.count" });

    const Rec = struct {
        // Section offsets within one expert record, gate|up|down order —
        // MUST mirror expert_store.LayerState.expertSlabOffsets (validated
        // at load by addLayerSlab's record_bytes check).
        off: [3]usize,
        total: usize,
        expert_bytes: [3]usize,
        fn init(sizes: [3]usize) @This() {
            var off: [3]usize = undefined;
            var at: usize = 0;
            for (sizes, 0..) |sz, p| {
                at = std.mem.alignForward(usize, at, slab_align);
                off[p] = at;
                at += sz;
            }
            return .{ .off = off, .total = std.mem.alignForward(usize, at, slab_align), .expert_bytes = sizes };
        }
    };

    // Plan pass: one slab tensor replaces each gate/up/down fx4 trio.
    var stack_count: usize = 0;
    var max_record: usize = 0;
    var max_ne: usize = 0;
    var name_buf: [192]u8 = undefined;
    for (src.tensors) |*info| {
        const dims = info.dims[0..info.n_dims];
        if (parseExpsName(info.name)) |exps| {
            if (info.ggml_type != .tq2_0_fx4 or info.n_dims != 3) return error.UnsupportedRepackTensor;
            if (std.mem.eql(u8, exps.proj, "w3") or std.mem.eql(u8, exps.proj, "w2")) continue; // consumed with the gate tensor
            const n_expert = info.dims[2];
            const gu_bytes = info.data.len / n_expert;
            const down_info = try src.get(try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down_exps.weight", .{exps.layer}));
            const rec = Rec.init(.{ gu_bytes, gu_bytes, down_info.data.len / n_expert });
            const slab_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_exps_slab.weight", .{exps.layer});
            try writer.declareTensor(slab_name, .i8, &.{ rec.total, n_expert });
            stack_count += 1;
            max_record = @max(max_record, rec.total);
            max_ne = @max(max_ne, n_expert);
        } else {
            try writer.declareTensor(info.name, info.ggml_type, dims);
        }
    }
    if (stack_count == 0) return error.NoExpertStacks;

    const record_buf = try allocator.alloc(u8, max_record * max_ne);
    defer allocator.free(record_buf);

    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    var streamer = try writer.beginStream(&out_writer.interface);

    const total_start = nowNs(io);
    var done: usize = 0;
    for (src.tensors) |*info| {
        if (parseExpsName(info.name)) |exps| {
            if (std.mem.eql(u8, exps.proj, "w3") or std.mem.eql(u8, exps.proj, "w2")) continue;
            const n_expert = info.dims[2];
            const up_info = try src.get(try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up_exps.weight", .{exps.layer}));
            const down_info = try src.get(try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down_exps.weight", .{exps.layer}));
            const gu_bytes = info.data.len / n_expert;
            const down_bytes = down_info.data.len / n_expert;
            const rec = Rec.init(.{ gu_bytes, gu_bytes, down_bytes });
            const stack = record_buf[0 .. rec.total * n_expert];
            @memset(stack, 0);
            const sources = [3]*const gguf.TensorInfo{ info, up_info, down_info };
            for (sources) |s| gguf.prefetch(s.data);
            for (0..n_expert) |e| {
                const dst = stack[e * rec.total ..][0..rec.total];
                for (sources, 0..) |s, p| {
                    const bytes = rec.expert_bytes[p];
                    @memcpy(dst[rec.off[p]..][0..bytes], s.data[e * bytes ..][0..bytes]);
                }
            }
            for (sources) |s| gguf.release(s.data);
            try streamer.writeTensorData(stack);
            done += 1;
            try stdout.print("[{d}/{d}] blk.{d} -> ffn_exps_slab ({d:.1} MiB/expert)\n", .{ done, stack_count, exps.layer, mib(rec.total) });
            try stdout.flush();
        } else {
            gguf.prefetch(info.data);
            try streamer.writeTensorData(info.data);
            gguf.release(info.data);
        }
    }
    try streamer.finish();
    try out_writer.interface.flush();
    try stdout.print("slab-repacked {s} -> {s}: {d} layers in {d:.1} min\n", .{
        src_path, dst_path, stack_count, @as(f64, @floatFromInt(nowNs(io) - total_start)) / 6e10,
    });
}

/// Post-conversion integrity check: re-solve sample experts of one layer
/// from the fp4 source and byte-compare against the plane row-blocks the
/// converted GGUF stores. The solver is bitwise-deterministic for any
/// thread count, so equality is exact — a stack-offset or write-order bug
/// reads as a byte mismatch here, never as a silent quality loss at the
/// fixture gate. Also checks the metadata stamps and that the base expert
/// tensors were actually replaced.
fn verify(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *fucina.ExecContext,
    shards: *const ShardSet,
    converted_path: []const u8,
    layer: usize,
    options: ptqtp.Options,
    stdout: *std.Io.Writer,
) !void {
    var out = try gguf.File.loadMmapAuto(allocator, io, converted_path);
    defer out.deinit();

    const stored_version = out.getInt(ptqtp_gguf.version_key) orelse return error.VerifyMissingVersionStamp;
    if (stored_version != ptqtp_gguf.format_version) return error.VerifyVersionMismatch;
    const stored_tie = (out.getInt(ptqtp_gguf.tie_key) orelse 0) == 1;
    if (stored_tie != options.tie_scales) return error.VerifyTieStampMismatch;

    const projections = [3]struct { gguf_name: []const u8, proj: []const u8 }{
        .{ .gguf_name = "ffn_gate_exps.weight", .proj = "w1" },
        .{ .gguf_name = "ffn_up_exps.weight", .proj = "w3" },
        .{ .gguf_name = "ffn_down_exps.weight", .proj = "w2" },
    };
    const experts = [3]usize{ 0, 137, 255 };

    var name_buf: [96]u8 = undefined;
    var plane_buf: [192]u8 = undefined;
    var failures: usize = 0;
    for (projections) |p| {
        const base_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, p.gguf_name });
        if (out.maybeGet(base_name) != null) {
            try stdout.print("verify {s}: FAIL — base tensor still present (planes must replace it)\n", .{base_name});
            failures += 1;
            continue;
        }
        var plane_infos: [3]*const gguf.TensorInfo = undefined;
        for (0..options.planes) |k| {
            plane_infos[k] = try out.get(try ptqtp_gguf.planeName(&plane_buf, base_name, k));
            if (plane_infos[k].ggml_type != .tq2_0 or plane_infos[k].n_dims != 3) return error.VerifyBadPlaneTensor;
        }
        const in_dim = plane_infos[0].dims[0];
        const out_dim = plane_infos[0].dims[1];
        const n_expert = plane_infos[0].dims[2];
        const expert_bytes = out_dim * (in_dim / ptqtp.block_len) * @sizeOf(fucina.ptqtp.BlockTQ2_0);

        const expert_f32 = try allocator.alloc(f32, out_dim * in_dim);
        defer allocator.free(expert_f32);
        for (experts) |e| {
            if (e >= n_expert) continue;
            try dequantSourceExpert(shards, layer, e, p.proj, out_dim, in_dim, expert_f32);
            var pair = try ptqtp.quantizeMatrix(ctx, expert_f32, out_dim, in_dim, options);
            defer pair.deinit(ctx.allocator);
            const solved = [3][]const fucina.ptqtp.BlockTQ2_0{ pair.plane1, pair.plane2, pair.plane3 };
            for (0..options.planes) |k| {
                const stored = plane_infos[k].data[e * expert_bytes ..][0..expert_bytes];
                const ok = std.mem.eql(u8, stored, std.mem.sliceAsBytes(solved[k]));
                if (!ok) failures += 1;
                try stdout.print("verify {s}.ptqtp{d} expert {d}: {s}\n", .{ base_name, k, e, if (ok) "byte-identical" else "MISMATCH" });
            }
        }
        try stdout.flush();
    }
    if (failures != 0) {
        try stdout.print("verify: {d} FAILURES\n", .{failures});
        return error.VerifyFailed;
    }
    try stdout.print("verify: all checks byte-identical (layer {d}, {d} experts x {d} planes x 3 projections + stamps)\n", .{ layer, experts.len, options.planes });
}

/// Cross-source oracle + solve timing, no output written. The trunk GGUF's
/// Q4_K expert stacks and the fp4 shards both derive from the same released
/// weights, so high agreement (~cos > 0.99, rel err ~ the Q4_K quant error)
/// proves the fp4 name map, nibble order, scale layout, and expert-major
/// slicing all at once; a layout mistake reads as noise (cos ~ 0).
fn smoke(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *fucina.ExecContext,
    trunk: *const gguf.File,
    shards: *const ShardSet,
    layer: usize,
    options: ptqtp.Options,
    stdout: *std.Io.Writer,
) !void {
    const projections = [3]struct { gguf_name: []const u8, proj: []const u8 }{
        .{ .gguf_name = "ffn_gate_exps.weight", .proj = "w1" },
        .{ .gguf_name = "ffn_up_exps.weight", .proj = "w3" },
        .{ .gguf_name = "ffn_down_exps.weight", .proj = "w2" },
    };
    const experts = [2]usize{ 0, 137 };

    var name_buf: [96]u8 = undefined;
    for (projections) |p| {
        const tensor_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, p.gguf_name });
        const info = try trunk.get(tensor_name);
        if (info.n_dims != 3) return error.UnexpectedTrunkShape;
        const in_dim = info.dims[0];
        const out_dim = info.dims[1];
        const n_expert = info.dims[2];

        const fp4_f32 = try allocator.alloc(f32, out_dim * in_dim);
        defer allocator.free(fp4_f32);
        const q4k_f32 = try allocator.alloc(f32, out_dim * in_dim);
        defer allocator.free(q4k_f32);

        for (experts) |e| {
            if (e >= n_expert) continue;
            try dequantSourceExpert(shards, layer, e, p.proj, out_dim, in_dim, fp4_f32);

            const expert_src_bytes = info.data.len / n_expert;
            const slice = info.data[e * expert_src_bytes ..][0..expert_src_bytes];
            try gguf.decodeF32(info.ggml_type, slice, q4k_f32);

            const cmp = cosineAndRelErr(fp4_f32, q4k_f32);
            try stdout.print("smoke blk.{d}.{s} expert {d}: cos {d:.6} rel_err {d:.4}  fp4[0..4] {d:.4} {d:.4} {d:.4} {d:.4}  q4k[0..4] {d:.4} {d:.4} {d:.4} {d:.4}\n", .{
                layer,      p.gguf_name, e,          cmp.cos,    cmp.rel,
                fp4_f32[0], fp4_f32[1],  fp4_f32[2], fp4_f32[3], q4k_f32[0],
                q4k_f32[1], q4k_f32[2],  q4k_f32[3],
            });
        }

        // Solve timing on expert 0 (also exercises the full pipeline).
        const solve_start = nowNs(io);
        try dequantSourceExpert(shards, layer, 0, p.proj, out_dim, in_dim, fp4_f32);
        var pair = try ptqtp.quantizeMatrix(ctx, fp4_f32, out_dim, in_dim, options);
        defer pair.deinit(ctx.allocator);
        const solve_ms = @as(f64, @floatFromInt(nowNs(io) - solve_start)) / 1e6;
        try stdout.print("smoke solve blk.{d}.{s} expert 0: K={d}{s} rel_err {d:.4} unconverged {d}/{d}  {d:.1} ms/expert -> est {d:.0} min for 129 stacks x 256 experts\n", .{
            layer,                      p.gguf_name,               options.planes, if (options.tie_scales) " tied" else "",
            pair.stats.rel_frob_err, pair.stats.unconverged_groups, pair.stats.group_count,
            solve_ms,                   solve_ms * 129.0 * 256.0 / 60000.0,
        });
        try stdout.flush();
    }
}
