//! GGUF export tool — closes the train→export→serve-anywhere loop.
//!
//! Modes:
//!   (a) re-emit / transcode:
//!       zig build export-gguf -- --from-gguf in.gguf --out out.gguf [--dtype f16|bf16|f32|q8_0|q4_k|q5_k|q6_k|tq2_0|verbatim]
//!   (b) merge Fucina LoRA adapters (safetensors, as saved by `zig build finetune`)
//!       into dense f32/f16/bf16 base weights and re-emit:
//!       zig build export-gguf -- --from-gguf in.gguf --adapters lora-dir --out out.gguf --alpha F
//!   (c) shard-streaming PTQTP quantization (docs/PTQTP.md) — models far
//!       bigger than RAM quantize tensor-at-a-time from the source mmap:
//!       zig build export-gguf -- --from-gguf in.gguf --out out.gguf --ptqtp[=K]
//!           [--ptqtp-planes K] [--ptqtp-include SUB[,SUB]] [--ptqtp-exclude SUB[,SUB]] [--dry-run]
//!
//! Transcode policy (documented choice, llama.cpp-convention): only matrix
//! weights transcode — n_dims >= 2, name ending ".weight", name not
//! containing "norm" — including token_embd/output, like llama.cpp's
//! quantize tool. Norms and 1D tensors keep their stored type (f32 in the
//! Qwen3/Gemma GGUFs). Sources must be f32/f16/bf16: transcoding an
//! already-quantized source would chain-requantize, so it errors — re-emit
//! those verbatim instead. Quantized targets whose innermost dim is not
//! divisible by the block size keep the source dtype (see transcodeTarget
//! for how this differs from llama-quantize).
//!
//! `--experts-dtype` (the MiMo-report experts-only-quantization insight)
//! overrides the target for tensors whose name contains "_exps.weight" —
//! everything else follows `--dtype` (default verbatim). Unlike the global
//! policy, the experts override IS allowed to requantize an already-quantized
//! source (dequant → re-encode via gguf.decodeF32/encodeF32): the shipped
//! MoE GGUFs store experts pre-quantized (Q6_K/Q8_0), and experts are exactly
//! where shrinking bytes pays in decode bandwidth while the redundancy keeps
//! quality risk lowest. Block-divisibility still rules: rows that don't
//! divide the target block size keep the source dtype (gemma's 704-wide
//! ffn_down_exps stays Q8_0 under `--experts-dtype q4_k`).
//!
//! PTQTP policy (mode c): every eligible 2D matrix — name ending ".weight",
//! not containing "norm", contract dim divisible by 256 (the plane block
//! width), source dtype decodable to f32 — is replaced by K byte-valid
//! standalone TQ2_0 plane tensors named `<name>.ptqtp0..K-1`, following the
//! src/llm/ptqtp_gguf.zig persistence conventions exactly (same names, same
//! `fucina.ptqtp.version` metadata stamp), so the output loads through the
//! existing family pair-detection. By default embeddings (`token_embd`) and
//! the output head stay in source precision (docs/PTQTP.md guidance);
//! `--ptqtp-include` replaces that default name policy with substring
//! matches, `--ptqtp-exclude` always subtracts. 3D MoE expert stacks
//! (`*_exps` names, expert-major contiguous `[in, out, n_expert]`) quantize
//! per expert slice: each expert's [out x in] matrix runs through
//! `ptqtp.quantizeMatrix` independently and the K plane tensors keep the
//! base 3D shape, plane-major (`ptqtp_gguf.quantizeMoeStack` — the MoE
//! convention the qwen3 loaders pair-detect). Unlike the --dtype policy,
//! PTQTP deliberately accepts quantized sources (q8_0/K-quants dequantize
//! through `gguf.decodeF32` first): the paper-validated from-quantized path
//! degrades gracefully, and hundreds-of-GB models often only ship quantized.
//!
//! Memory discipline (the point of mode c): the output is written with the
//! writer's streaming path (`declareTensor` + `beginStream`), so at any
//! moment the tool holds ONE source tensor's f32 buffer plus its quantized
//! planes — never the whole output. Source bytes arrive through the mmap
//! (prefetched, then `gguf.release`d per tensor), so residency stays
//! bounded no matter the model size; the run ends with a peak-RSS report.
//! Expert stacks are the one deliberate exception: their K plane stacks
//! accumulate in RAM for the stack's duration (source pages still release
//! expert-by-expert), so the peak is K plane stacks + one expert f32 slice
//! — ~550 MiB per plane for a 4096 x 2048 x 256-expert stack, ~1.7 GiB at
//! K=3 on the largest stacks. Both the dry-run plan and the final summary
//! report that figure.
//!
//! Merge policy: adapters named "layers.<i>.<q|k|v|o|gate|up|down>.lora_a/b"
//! (the qwen3_train.Trainer naming) merge into the matching
//! "blk.<i>.attn_*/ffn_*.weight" tensors via `lora.Adapter.mergeInto`
//! (f32 base) / `mergeF16` (f16 base) / widen→merge→re-encode (bf16 base).
//! Quantized bases error: merging into a quantized base would need
//! dequant→merge→requant, which compounds quantization error — the supported
//! path is merge on an f32/f16/bf16 base, then `--dtype` transcode in a
//! second pass. `--adapters` accepts either a checkpoint directory containing
//! adapters.safetensors or a direct safetensors file. The adapter tensor file
//! stores A/B but not alpha, so `--alpha` (the training-time value; finetune
//! default: 16) is REQUIRED with `--adapters`.

const builtin = @import("builtin");
const std = @import("std");
const fucina = @import("fucina");
const llm = @import("fucina_llm");

const gguf = fucina.gguf;
const lora = fucina.lora;
const optim = fucina.optim;
const ptqtp = fucina.ptqtp;
const ptqtp_gguf = llm.ptqtp_gguf;
const safetensors = fucina.safetensors;

const DtypeMode = enum { verbatim, f32, f16, bf16, q8_0, q4_k, q5_k, q6_k, tq2_0 };

const usage =
    "usage: zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf IN.gguf --out OUT.gguf [--dtype f16|bf16|f32|q8_0|q4_k|q5_k|q6_k|tq2_0|verbatim] [--experts-dtype DTYPE (only tensors named *_exps.weight; may requantize)] [--adapters DIR_OR_SAFETENSORS --alpha F (required together)] [--ptqtp[=K] --ptqtp-planes K --ptqtp-include SUB[,SUB] --ptqtp-exclude SUB[,SUB] --dry-run (shard-streaming PTQTP quantization; docs/PTQTP.md)]\n";

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var from_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var adapters_path: ?[]const u8 = null;
    var dtype_mode: DtypeMode = .verbatim;
    var experts_dtype_mode: ?DtypeMode = null; // null = experts follow --dtype
    var alpha: ?f32 = null;
    var ptqtp_mode = false;
    var ptqtp_planes: u8 = 2;
    var dry_run = false;
    const arena = init.arena.allocator();
    var includes: std.ArrayList([]const u8) = .empty;
    var excludes: std.ArrayList([]const u8) = .empty;

    var arg_i: usize = 1;
    while (arg_i < args.len) : (arg_i += 1) {
        const arg = args[arg_i];
        if (std.mem.eql(u8, arg, "--from-gguf")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingFromPath;
            from_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--from-gguf=")) {
            from_path = arg["--from-gguf=".len..];
        } else if (std.mem.eql(u8, arg, "--out")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingOutPath;
            out_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--out=")) {
            out_path = arg["--out=".len..];
        } else if (std.mem.eql(u8, arg, "--adapters")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingAdaptersPath;
            adapters_path = args[arg_i];
        } else if (std.mem.startsWith(u8, arg, "--adapters=")) {
            adapters_path = arg["--adapters=".len..];
        } else if (std.mem.eql(u8, arg, "--dtype")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingDtype;
            dtype_mode = std.meta.stringToEnum(DtypeMode, args[arg_i]) orelse return error.UnknownDtype;
        } else if (std.mem.startsWith(u8, arg, "--dtype=")) {
            dtype_mode = std.meta.stringToEnum(DtypeMode, arg["--dtype=".len..]) orelse return error.UnknownDtype;
        } else if (std.mem.eql(u8, arg, "--experts-dtype")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingDtype;
            experts_dtype_mode = std.meta.stringToEnum(DtypeMode, args[arg_i]) orelse return error.UnknownDtype;
        } else if (std.mem.startsWith(u8, arg, "--experts-dtype=")) {
            experts_dtype_mode = std.meta.stringToEnum(DtypeMode, arg["--experts-dtype=".len..]) orelse return error.UnknownDtype;
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingAlpha;
            alpha = try std.fmt.parseFloat(f32, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--alpha=")) {
            alpha = try std.fmt.parseFloat(f32, arg["--alpha=".len..]);
        } else if (std.mem.eql(u8, arg, "--ptqtp")) {
            ptqtp_mode = true;
        } else if (std.mem.startsWith(u8, arg, "--ptqtp=")) {
            ptqtp_mode = true;
            ptqtp_planes = try std.fmt.parseInt(u8, arg["--ptqtp=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--ptqtp-planes")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingPlaneCount;
            ptqtp_mode = true;
            ptqtp_planes = try std.fmt.parseInt(u8, args[arg_i], 10);
        } else if (std.mem.startsWith(u8, arg, "--ptqtp-planes=")) {
            ptqtp_mode = true;
            ptqtp_planes = try std.fmt.parseInt(u8, arg["--ptqtp-planes=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--ptqtp-include")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingFilter;
            ptqtp_mode = true;
            try appendFilters(arena, &includes, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--ptqtp-include=")) {
            ptqtp_mode = true;
            try appendFilters(arena, &includes, arg["--ptqtp-include=".len..]);
        } else if (std.mem.eql(u8, arg, "--ptqtp-exclude")) {
            arg_i += 1;
            if (arg_i >= args.len) return error.MissingFilter;
            ptqtp_mode = true;
            try appendFilters(arena, &excludes, args[arg_i]);
        } else if (std.mem.startsWith(u8, arg, "--ptqtp-exclude=")) {
            ptqtp_mode = true;
            try appendFilters(arena, &excludes, arg["--ptqtp-exclude=".len..]);
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            dry_run = true;
        } else {
            try stdout.print(usage, .{});
            return error.UnknownArgument;
        }
    }
    const in_path = from_path orelse {
        try stdout.print(usage, .{});
        return error.MissingFromPath;
    };
    if (adapters_path != null and (dtype_mode != .verbatim or experts_dtype_mode != null)) {
        try stdout.print("--adapters merges into the base dtype; combining it with --dtype/--experts-dtype transcoding is not supported\n", .{});
        return error.UnsupportedArgumentCombination;
    }
    if (adapters_path != null and alpha == null) {
        try stdout.print("--adapters requires --alpha: the safetensors checkpoint stores the adapter A/B matrices but not alpha, so pass the training-time value (finetune default: 16)\n", .{});
        return error.MissingAlpha;
    }
    if (ptqtp_mode and (adapters_path != null or dtype_mode != .verbatim or experts_dtype_mode != null)) {
        try stdout.print("--ptqtp is its own streaming mode; combining it with --dtype/--experts-dtype/--adapters is not supported (quantize in a separate pass)\n", .{});
        return error.UnsupportedArgumentCombination;
    }
    if (dry_run and !ptqtp_mode) {
        try stdout.print("--dry-run belongs to the --ptqtp mode (it prints the per-tensor quantization plan)\n", .{});
        return error.UnsupportedArgumentCombination;
    }
    if (ptqtp_mode and (ptqtp_planes < 1 or ptqtp_planes > 3)) {
        try stdout.print("--ptqtp plane count must be 1, 2, or 3 (got {d})\n", .{ptqtp_planes});
        return error.InvalidPlaneCount;
    }
    if (out_path == null and !dry_run) {
        try stdout.print(usage, .{});
        return error.MissingOutPath;
    }

    const allocator = std.heap.smp_allocator;

    if (ptqtp_mode) {
        return runPtqtp(allocator, io, in_path, out_path, .{
            .planes = ptqtp_planes,
            .includes = includes.items,
            .excludes = excludes.items,
            .dry_run = dry_run,
        }, stdout);
    }
    const dst_path = out_path.?;

    var file = try gguf.File.loadMmap(allocator, io, in_path);
    defer file.deinit();

    // Owned transcode/merge buffers; the GGUF writer borrows them until
    // finish() returns. Declared BEFORE the writer so the LIFO defers run
    // writer.deinit (dropping the borrowed data pointers) before the buffers
    // are freed.
    var owned: std.ArrayList([]u8) = .empty;
    defer {
        for (owned.items) |buf| allocator.free(buf);
        owned.deinit(allocator);
    }
    // GGUF tensor name (borrowed from `file`) → merged replacement bytes.
    var merged = std.StringHashMap([]const u8).init(allocator);
    defer merged.deinit();

    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    try writer.copyAllMetadata(&file, &.{});

    if (adapters_path) |ckpt_path| {
        var ctx: fucina.ExecContext = undefined;
        ctx.init(allocator);
        defer ctx.deinit();
        try mergeAdapters(allocator, &ctx, io, &file, ckpt_path, alpha.?, &owned, &merged, stdout);
    }

    var verbatim_count: usize = 0;
    var transcoded_count: usize = 0;
    for (file.tensors) |*info| {
        const dims = info.dims[0..info.n_dims];
        if (merged.get(info.name)) |bytes| {
            try writer.addTensor(info.name, info.ggml_type, dims, bytes);
            continue;
        }
        const target = try transcodeTarget(info, dtype_mode, experts_dtype_mode);
        if (target == info.ggml_type) {
            try writer.addTensor(info.name, info.ggml_type, dims, info.data);
            verbatim_count += 1;
        } else {
            const bytes = transcodeTensor(allocator, info, target) catch |err| {
                if (err == gguf.Error.NonFiniteValue) {
                    try stdout.print("tensor {s} contains NaN/inf values; refusing to quantize it to {s}\n", .{ info.name, @tagName(target) });
                }
                return err;
            };
            {
                // Scoped: once `owned` holds the buffer its deferred sweep
                // frees it, so the errdefer must not stay armed past here.
                errdefer allocator.free(bytes);
                try owned.append(allocator, bytes);
            }
            try writer.addTensor(info.name, target, dims, bytes);
            transcoded_count += 1;
        }
    }
    if (dtype_mode != .verbatim) {
        try writer.addMetaInt("general.file_type", u32, fileTypeFor(dtype_mode));
    }

    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    try writer.finish(&out_writer.interface);
    try out_writer.interface.flush();

    try stdout.print("exported {s} -> {s}\n", .{ in_path, dst_path });
    try stdout.print("tensors: {d} verbatim, {d} transcoded ({s}), {d} merged\n", .{
        verbatim_count, transcoded_count, @tagName(dtype_mode), merged.count(),
    });
    if (experts_dtype_mode) |mode| {
        try stdout.print("experts (*_exps.weight): targeted {s}\n", .{@tagName(mode)});
    }
}

// ---------------------------------------------------------------------------
// Transcoding.
// ---------------------------------------------------------------------------

/// llama.cpp `llama_ftype` codes for `general.file_type`
/// (refs/llama.cpp/include/llama.h). Fucina quantizes every eligible matrix
/// to ONE type, so the uniform K-quant exports are the _S ("small") variants
/// — the _M mixes upgrade attn_v/ffn_down to a bigger quant per layer.
fn fileTypeFor(mode: DtypeMode) u32 {
    return switch (mode) {
        .verbatim => unreachable,
        .f32 => 0, // LLAMA_FTYPE_ALL_F32
        .f16 => 1, // LLAMA_FTYPE_MOSTLY_F16
        .q8_0 => 7, // LLAMA_FTYPE_MOSTLY_Q8_0
        .q4_k => 14, // LLAMA_FTYPE_MOSTLY_Q4_K_S
        .q5_k => 16, // LLAMA_FTYPE_MOSTLY_Q5_K_S
        .q6_k => 18, // LLAMA_FTYPE_MOSTLY_Q6_K
        .bf16 => 32, // LLAMA_FTYPE_MOSTLY_BF16
        .tq2_0 => 37, // LLAMA_FTYPE_MOSTLY_TQ2_0
    };
}

/// See the module doc for the policy. Returning the source type means
/// "re-emit verbatim".
fn transcodeTarget(info: *const gguf.TensorInfo, base_mode: DtypeMode, experts_mode: ?DtypeMode) !gguf.GgmlType {
    const experts_override = experts_mode != null and std.mem.indexOf(u8, info.name, "_exps.weight") != null;
    const mode = if (experts_override) experts_mode.? else base_mode;
    const target: gguf.GgmlType = switch (mode) {
        .verbatim => return info.ggml_type,
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        .tq2_0 => .tq2_0,
    };
    if (info.n_dims < 2) return info.ggml_type;
    if (!std.mem.endsWith(u8, info.name, ".weight")) return info.ggml_type;
    if (std.mem.indexOf(u8, info.name, "norm") != null) return info.ggml_type;
    if (target == info.ggml_type) return info.ggml_type;
    switch (info.ggml_type) {
        .f32, .f16, .bf16 => {},
        // The experts override accepts quantized sources (dequant →
        // re-encode; see the module doc) — the global --dtype keeps
        // refusing chain-requantization.
        else => if (!experts_override) return error.QuantizedSourceUnsupported, // no chain-requantize; use --dtype verbatim
    }
    if (target == .q8_0 and info.dims[0] % 32 != 0) return info.ggml_type;
    // K-quant super-blocks and TQ2_0 blocks span 256 elements: rows that
    // don't divide keep the SOURCE dtype. llama-quantize instead falls back
    // to a smaller-block QUANT (tensor_type_fallback in refs/llama.cpp/src/llama-quant.cpp:
    // Q4_K→Q5_0, Q5_K→Q5_1, Q6_K→Q8_0); keeping the f32/f16/bf16 source is
    // the more conservative choice — no extra quantization loss on those
    // tensors, at a small size cost.
    switch (target) {
        .q4_k, .q5_k, .q6_k, .tq2_0 => if (info.dims[0] % 256 != 0) return info.ggml_type,
        else => {},
    }
    return target;
}

/// Row-wise widen/dequantize-to-f32 → encode; returns the owned wire bytes
/// for `target`. Scalar sources widen; quantized sources (the experts
/// override) dequantize through the same `gguf.decodeF32` seam.
fn transcodeTensor(allocator: std.mem.Allocator, info: *const gguf.TensorInfo, target: gguf.GgmlType) ![]u8 {
    const dims = info.dims[0..info.n_dims];
    var elems: usize = 1;
    for (dims) |dim| elems = try std.math.mul(usize, elems, dim);
    const row_len = info.dims[0];
    const rows = elems / row_len;

    const dst = try allocator.alloc(u8, try gguf.tensorByteLen(target, dims));
    errdefer allocator.free(dst);
    const row_dst_bytes = try gguf.tensorByteLen(target, &.{row_len});

    const row_f32 = try allocator.alloc(f32, row_len);
    defer allocator.free(row_f32);
    const row_src_bytes = try gguf.tensorByteLen(info.ggml_type, &.{row_len});
    for (0..rows) |row_i| {
        const src_row = info.data[row_i * row_src_bytes ..][0..row_src_bytes];
        try gguf.decodeF32(info.ggml_type, src_row, row_f32);
        try gguf.encodeF32(target, row_f32, dst[row_i * row_dst_bytes ..][0..row_dst_bytes]);
    }
    return dst;
}

fn widenRow(src_type: gguf.GgmlType, src: []const u8, dst: []f32) void {
    switch (src_type) {
        .f32 => for (dst, 0..) |*value, i| {
            value.* = @bitCast(std.mem.readInt(u32, src[i * 4 ..][0..4], .little));
        },
        .f16 => for (dst, 0..) |*value, i| {
            value.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[i * 2 ..][0..2], .little))));
        },
        .bf16 => for (dst, 0..) |*value, i| {
            value.* = bf16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
        },
        else => unreachable,
    }
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

// ---------------------------------------------------------------------------
// PTQTP shard-streaming quantization (mode c). See the module doc for the
// policy; docs/PTQTP.md for the method and the loader-side conventions.
// ---------------------------------------------------------------------------

const PtqtpArgs = struct {
    planes: u8,
    includes: []const []const u8,
    excludes: []const []const u8,
    dry_run: bool,
};

/// Split a (repeatable) comma-separated filter argument into the list. The
/// substrings borrow the argv arena, which outlives the run.
fn appendFilters(allocator: std.mem.Allocator, list: *std.ArrayList([]const u8), value: []const u8) !void {
    var it = std.mem.splitScalar(u8, value, ',');
    while (it.next()) |part| {
        if (part.len == 0) continue;
        try list.append(allocator, part);
    }
}

fn matchesAny(name: []const u8, needles: []const []const u8) bool {
    for (needles) |needle| {
        if (std.mem.indexOf(u8, name, needle) != null) return true;
    }
    return false;
}

/// Whether mode (c) quantizes this tensor — the GGUF-level mirror of
/// `LinearWeight.ptqtpEligible` plus the matrix name policy. Must be pure:
/// the plan pass (declarations) and the stream pass (data) both call it and
/// have to agree tensor-for-tensor.
fn ptqtpQuantizes(info: *const gguf.TensorInfo, args: *const PtqtpArgs) bool {
    switch (info.n_dims) {
        2 => {},
        // Stacked MoE experts ([in, out, n_expert]) quantize per expert
        // slice; any other 3D tensor passes through.
        3 => if (std.mem.indexOf(u8, info.name, "_exps") == null) return false,
        else => return false,
    }
    if (!std.mem.endsWith(u8, info.name, ".weight")) return false;
    if (std.mem.indexOf(u8, info.name, "norm") != null) return false;
    if (info.dims[0] % ptqtp.block_len != 0) return false; // 256-block contract
    switch (info.ggml_type) {
        // Everything gguf.decodeF32 handles except tq2_0 (already ternary).
        .f32, .f16, .bf16, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q4_k, .q5_k, .q6_k => {},
        else => return false,
    }
    if (args.includes.len != 0) {
        if (!matchesAny(info.name, args.includes)) return false;
    } else {
        // Default name policy (docs/PTQTP.md): embeddings and the output
        // head stay in source precision. --ptqtp-include replaces this.
        if (std.mem.indexOf(u8, info.name, "token_embd") != null) return false;
        if (std.mem.eql(u8, info.name, "output.weight")) return false;
    }
    if (matchesAny(info.name, args.excludes)) return false;
    return true;
}

fn runPtqtp(
    allocator: std.mem.Allocator,
    io: std.Io,
    in_path: []const u8,
    out_path: ?[]const u8,
    args: PtqtpArgs,
    stdout: *std.Io.Writer,
) !void {
    if (out_path) |dst| {
        if (std.mem.eql(u8, dst, in_path)) {
            try stdout.print("--out must differ from --from-gguf: the streaming writer reads source tensors from the mmap while writing the destination\n", .{});
            return error.UnsupportedArgumentCombination;
        }
    }

    // Split-aware mmap load: pages are file-backed and evictable, so walking
    // one tensor at a time never needs the model to fit in RAM.
    var file = try gguf.File.loadMmapAuto(allocator, io, in_path);
    defer file.deinit();

    const options = ptqtp.Options{ .planes = args.planes };

    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    // The output is always a single file: drop the llama.cpp split markers
    // a multi-part source carries.
    try writer.copyAllMetadata(&file, &.{ "split.no", "split.count", "split.tensors.count" });

    // Plan pass: decide per tensor; print the plan (--dry-run) or declare
    // the output tensor set (names, dims, offsets — no data yet).
    var quantized_count: usize = 0;
    var quant_stacks: usize = 0;
    var passthrough_count: usize = 0;
    var kept_stacks: usize = 0;
    var quant_src_bytes: u64 = 0;
    var quant_dst_bytes: u64 = 0;
    var total_src_bytes: u64 = 0;
    var total_dst_bytes: u64 = 0;
    var peak_stack_buffer: u64 = 0;
    var name_buf: [256]u8 = undefined;
    for (file.tensors) |*info| {
        total_src_bytes += info.data.len;
        const dims = info.dims[0..info.n_dims];
        if (ptqtpQuantizes(info, &args)) {
            const out_bytes = try gguf.tensorByteLen(.tq2_0, dims) * args.planes;
            quantized_count += 1;
            quant_src_bytes += info.data.len;
            quant_dst_bytes += out_bytes;
            total_dst_bytes += out_bytes;
            if (info.n_dims == 3) {
                quant_stacks += 1;
                // K accumulating plane stacks + one expert's transient
                // planes + one expert's f32 slice (see the module doc).
                const workset = out_bytes + out_bytes / info.dims[2] +
                    @as(u64, info.dims[0] * info.dims[1] * @sizeOf(f32));
                peak_stack_buffer = @max(peak_stack_buffer, workset);
            }
            if (args.dry_run) {
                try stdout.print("ptqtp {s} ", .{info.name});
                try printShape(stdout, info);
                if (info.n_dims == 3) {
                    try stdout.print(" {s} -> tq2_0 x{d} per expert ({d} experts)  ({d:.1} -> {d:.1} MiB)\n", .{
                        @tagName(info.ggml_type), args.planes, info.dims[2], mib(info.data.len), mib(out_bytes),
                    });
                } else {
                    try stdout.print(" {s} -> tq2_0 x{d}  ({d:.1} -> {d:.1} MiB)\n", .{
                        @tagName(info.ggml_type), args.planes, mib(info.data.len), mib(out_bytes),
                    });
                }
                continue;
            }
            for (0..args.planes) |plane_i| {
                const plane_name = try ptqtp_gguf.planeName(&name_buf, info.name, plane_i);
                try writer.declareTensor(plane_name, .tq2_0, dims);
            }
        } else {
            passthrough_count += 1;
            if (info.n_dims == 3 and std.mem.indexOf(u8, info.name, "_exps") != null) kept_stacks += 1;
            total_dst_bytes += info.data.len;
            if (args.dry_run) {
                try stdout.print("pass  {s} ", .{info.name});
                try printShape(stdout, info);
                try stdout.print(" {s} (kept)\n", .{@tagName(info.ggml_type)});
                continue;
            }
            try writer.declareTensor(info.name, info.ggml_type, dims);
        }
    }

    if (args.dry_run) {
        try stdout.print(
            "plan: {d} tensors; {d} ptqtp-quantized (K={d}, {d} plane tensors, {d} expert stacks), {d} passthrough ({d} expert stacks kept)\n",
            .{ file.tensors.len, quantized_count, args.planes, quantized_count * args.planes, quant_stacks, passthrough_count, kept_stacks },
        );
        try stdout.print("bytes: {d:.1} -> {d:.1} MiB total; quantized linears {d:.1} -> {d:.1} MiB\n", .{
            mib(total_src_bytes), mib(total_dst_bytes), mib(quant_src_bytes), mib(quant_dst_bytes),
        });
        if (quant_stacks != 0) {
            try stdout.print("peak expert-stack buffering: {d:.1} MiB (largest stack: K plane stacks + one expert f32 slice)\n", .{mib(peak_stack_buffer)});
        }
        return;
    }

    // Same stamp as ptqtp_gguf.build: only a file that actually carries
    // planes claims the PTQTP format (gates loader pair-detection).
    if (quantized_count != 0) {
        try writer.addMetaInt(ptqtp_gguf.version_key, u32, ptqtp_gguf.format_version);
    }

    var ctx: fucina.ExecContext = undefined;
    ctx.init(allocator);
    defer ctx.deinit();

    // Stream pass: header first, then one tensor at a time — decode, solve,
    // write, release. Peak memory = one source tensor (f32) + its planes.
    const dst_path = out_path.?;
    var out_file = try std.Io.Dir.cwd().createFile(io, dst_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buffer);
    var streamer = try writer.beginStream(&out_writer.interface);

    var peak_workset: u64 = 0;
    for (file.tensors) |*info| {
        if (ptqtpQuantizes(info, &args)) {
            const workset = if (info.n_dims == 3)
                try quantizeExpertStackStream(&ctx, &streamer, info, options, file.is_mmap, stdout)
            else
                try quantizeTensorStream(allocator, &ctx, &streamer, info, options, file.is_mmap, stdout);
            peak_workset = @max(peak_workset, workset);
        } else {
            gguf.prefetch(info.data);
            try streamer.writeTensorData(info.data);
            if (file.is_mmap) gguf.release(info.data);
        }
    }
    try streamer.finish();
    try out_writer.interface.flush();

    try stdout.print("exported {s} -> {s} (PTQTP K={d})\n", .{ in_path, dst_path, args.planes });
    try stdout.print("tensors: {d} quantized -> {d} plane tensors ({d} expert stacks), {d} passthrough ({d} expert stacks kept)\n", .{
        quantized_count, quantized_count * args.planes, quant_stacks, passthrough_count, kept_stacks,
    });
    try stdout.print("bytes: {d:.1} -> {d:.1} MiB total; quantized linears {d:.1} -> {d:.1} MiB\n", .{
        mib(total_src_bytes), mib(total_dst_bytes), mib(quant_src_bytes), mib(quant_dst_bytes),
    });
    // The one-tensor-at-a-time witness: the largest simultaneous heap hold
    // (one tensor's f32 decode buffer + its packed planes; for an expert
    // stack, the K resident plane stacks + one expert slice).
    try stdout.print("peak tensor working set: {d:.1} MiB\n", .{mib(peak_workset)});
    if (peakRssBytes()) |rss| {
        // Clean source-mmap pages count toward RSS until the OS drops them:
        // Linux honors the per-tensor MADV.DONTNEED release immediately;
        // Darwin ignores it for file-backed maps and only evicts under
        // memory pressure, so the figure there approaches min(file size,
        // free RAM) while the heap stays at the working set above.
        const note = if (builtin.os.tag == .macos) " (macOS: includes clean evictable mmap pages)" else "";
        try stdout.print("peak RSS: {d:.1} MiB{s}\n", .{ mib(rss), note });
    }
}

/// Decode one source tensor to f32 (then release its mapped pages), solve
/// the trit-planes, stream them out, print the reconstruction stats the
/// solver measured (against the exact fp16-rounded scales inference will
/// use). Never holds more than this tensor's f32 buffer plus its planes —
/// the buffer is freed before the planes are written. Returns that largest
/// simultaneous hold in bytes (the summary's working-set witness).
fn quantizeTensorStream(
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    streamer: *gguf.Writer.DataStreamer,
    info: *const gguf.TensorInfo,
    options: fucina.ptqtp.Options,
    release_pages: bool,
    stdout: *std.Io.Writer,
) !u64 {
    const shape = try info.logicalMatrixShape(); // [out, in]
    const rows = shape[0];
    const cols = shape[1];

    var pair = blk: {
        const values = try allocator.alloc(f32, rows * cols);
        defer allocator.free(values);
        gguf.prefetch(info.data);
        try gguf.decodeF32(info.ggml_type, info.data, values);
        if (release_pages) gguf.release(info.data);
        break :blk try ptqtp.quantizeMatrix(ctx, values, rows, cols, options);
    };
    defer pair.deinit(ctx.allocator);

    var out_bytes: usize = 0;
    const planes = [3][]const fucina.ptqtp.BlockTQ2_0{ pair.plane1, pair.plane2, pair.plane3 };
    for (planes) |plane| {
        if (plane.len == 0) break;
        const bytes = std.mem.sliceAsBytes(plane);
        try streamer.writeTensorData(bytes);
        out_bytes += bytes.len;
    }

    const stats = pair.stats;
    try stdout.print("ptqtp {s} [{d} x {d}] {s} -> tq2_0 x{d}  rel_err {d:.4}  iters {d:.1}  unconverged {d}/{d}  ({d:.1} -> {d:.1} MiB)\n", .{
        info.name,          rows,                  cols,                     @tagName(info.ggml_type), pair.planeCount(),
        stats.rel_frob_err, stats.mean_iterations, stats.unconverged_groups, stats.group_count,        mib(info.data.len),
        mib(out_bytes),
    });
    try stdout.flush();
    // f32 decode buffer + all planes coexisted inside quantizeMatrix.
    return @as(u64, rows * cols * @sizeOf(f32)) + out_bytes;
}

/// Expert-stack counterpart of `quantizeTensorStream`: quantize each expert
/// slice independently (`ptqtp_gguf.quantizeMoeStack` — expert-major
/// source, plane-major output), then stream the K plane stacks in
/// declaration order. The K accumulating stacks stay resident for the whole
/// tensor — the mode's one deliberate exception to one-tensor residency
/// (~550 MiB per plane for a 4096 x 2048 x 256-expert stack, ~1.7 GiB at
/// K=3) — while source pages still release expert-by-expert. Solver stats
/// print aggregated across experts (mean + max), not one line per expert.
/// Returns the largest simultaneous heap hold in bytes.
fn quantizeExpertStackStream(
    ctx: *fucina.ExecContext,
    streamer: *gguf.Writer.DataStreamer,
    info: *const gguf.TensorInfo,
    options: fucina.ptqtp.Options,
    release_pages: bool,
    stdout: *std.Io.Writer,
) !u64 {
    const in_dim = info.dims[0];
    const out_dim = info.dims[1];
    const n_expert = info.dims[2];

    var quant = try ptqtp_gguf.quantizeMoeStack(ctx, info.ggml_type, info.data, in_dim, out_dim, n_expert, options, release_pages);
    defer quant.deinit(ctx.allocator);

    var out_bytes: usize = 0;
    for (quant.planes[0..quant.plane_count]) |plane| {
        const bytes = std.mem.sliceAsBytes(plane);
        try streamer.writeTensorData(bytes);
        out_bytes += bytes.len;
    }

    const stats = quant.stats;
    try stdout.print("ptqtp {s} [{d} x {d} x {d}] {s} -> tq2_0 x{d}  rel_err mean {d:.4} max {d:.4}  iters {d:.1}  unconverged {d}/{d}  ({d:.1} -> {d:.1} MiB)\n", .{
        info.name,         n_expert,           out_dim,           in_dim,                @tagName(info.ggml_type),
        quant.plane_count, stats.mean_rel_err, stats.max_rel_err, stats.mean_iterations, stats.unconverged_groups,
        stats.group_count, mib(info.data.len), mib(out_bytes),
    });
    try stdout.flush();
    // K resident plane stacks + one expert's transient planes + one
    // expert's f32 decode slice coexisted at each expert's solve.
    return @as(u64, out_bytes) + out_bytes / n_expert + @as(u64, out_dim * in_dim * @sizeOf(f32));
}

/// Logical (row-major) shape — reversed ne dims, outermost first.
fn printShape(stdout: *std.Io.Writer, info: *const gguf.TensorInfo) !void {
    try stdout.print("[", .{});
    var i: usize = info.n_dims;
    while (i > 0) {
        i -= 1;
        try stdout.print("{d}", .{info.dims[i]});
        if (i != 0) try stdout.print(" x ", .{});
    }
    try stdout.print("]", .{});
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

/// Peak resident set size — the memory-discipline witness for the streaming
/// path (`ru_maxrss` is bytes on Darwin, KiB on Linux).
fn peakRssBytes() ?u64 {
    switch (builtin.os.tag) {
        .macos, .ios, .linux => {},
        else => return null,
    }
    const ru = std.posix.getrusage(std.posix.rusage.SELF);
    const maxrss: u64 = @intCast(@max(ru.maxrss, 0));
    return if (builtin.os.tag == .linux) maxrss * 1024 else maxrss;
}

// ---------------------------------------------------------------------------
// LoRA adapter merging.
// ---------------------------------------------------------------------------

/// The qwen3_train.Trainer target order and its checkpoint/GGUF names.
const lora_target_count = 7;
const lora_ckpt_names = [lora_target_count][]const u8{ "q", "k", "v", "o", "gate", "up", "down" };
const lora_gguf_suffix = [lora_target_count][]const u8{
    "attn_q.weight",   "attn_k.weight", "attn_v.weight",   "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
};

fn LoraAdapterFor(comptime t: usize) type {
    return switch (t) {
        0 => lora.Adapter(.embed, .q),
        1 => lora.Adapter(.embed, .k),
        2 => lora.Adapter(.embed, .v),
        3 => lora.Adapter(.attn, .embed),
        4, 5 => lora.Adapter(.embed, .ffn),
        6 => lora.Adapter(.ffn, .embed),
        else => unreachable,
    };
}

/// The frozen weight's facade type, tags { out_tag, in_tag } (mergeInto's
/// contract).
fn WeightF32For(comptime t: usize) type {
    return switch (t) {
        0 => fucina.Tensor(.{ .q, .embed }),
        1 => fucina.Tensor(.{ .k, .embed }),
        2 => fucina.Tensor(.{ .v, .embed }),
        3 => fucina.Tensor(.{ .embed, .attn }),
        4, 5 => fucina.Tensor(.{ .ffn, .embed }),
        6 => fucina.Tensor(.{ .embed, .ffn }),
        else => unreachable,
    };
}

fn WeightF16For(comptime t: usize) type {
    return switch (t) {
        0 => fucina.Tensor(.{ .dtype = .f16, .tags = .{ .q, .embed } }),
        1 => fucina.Tensor(.{ .dtype = .f16, .tags = .{ .k, .embed } }),
        2 => fucina.Tensor(.{ .dtype = .f16, .tags = .{ .v, .embed } }),
        3 => fucina.Tensor(.{ .dtype = .f16, .tags = .{ .embed, .attn } }),
        4, 5 => fucina.Tensor(.{ .dtype = .f16, .tags = .{ .ffn, .embed } }),
        6 => fucina.Tensor(.{ .dtype = .f16, .tags = .{ .embed, .ffn } }),
        else => unreachable,
    };
}

fn mergeAdapters(
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    io: std.Io,
    file: *const gguf.File,
    ckpt_path: []const u8,
    alpha: f32,
    owned: *std.ArrayList([]u8),
    merged: *std.StringHashMap([]const u8),
    stdout: *std.Io.Writer,
) !void {
    const adapter_path = try resolveAdaptersPath(allocator, io, ckpt_path);
    defer allocator.free(adapter_path);
    const ckpt_bytes = try readFileAlloc(allocator, io, adapter_path);
    defer allocator.free(ckpt_bytes);

    var reader = std.Io.Reader.fixed(ckpt_bytes);
    var state = try safetensors.readPrefix(allocator, &reader);
    defer state.deinit();

    var names = std.StringHashMap(void).init(allocator);
    defer names.deinit();
    for (state.tensors) |*entry| try names.put(entry.name, {});

    var merged_count: usize = 0;
    for (state.tensors) |*entry| {
        if (!std.mem.endsWith(u8, entry.name, ".lora_a")) continue;
        const prefix = entry.name[0 .. entry.name.len - ".lora_a".len];
        if (!std.mem.startsWith(u8, prefix, "layers.")) return error.UnknownAdapterName;
        const rest = prefix["layers.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnknownAdapterName;
        const layer = try std.fmt.parseInt(usize, rest[0..dot], 10);
        const target_name = rest[dot + 1 ..];

        var b_name_buf: [96]u8 = undefined;
        const b_name = try std.fmt.bufPrint(&b_name_buf, "{s}.lora_b", .{prefix});
        if (!names.contains(b_name)) return error.MissingLoraB;
        if (entry.shape.len != 2 or entry.dtype != .F32) return error.InvalidStateDict;

        var matched = false;
        inline for (0..lora_target_count) |t| {
            if (std.mem.eql(u8, target_name, lora_ckpt_names[t])) {
                try mergeOne(t, allocator, ctx, file, ckpt_bytes, layer, entry.shape[0], alpha, owned, merged, stdout);
                matched = true;
            }
        }
        if (!matched) return error.UnknownAdapterName;
        merged_count += 1;
    }
    if (merged_count == 0) return error.NoAdaptersFound;

    // Every .lora_b must have been consumed by a matching .lora_a above
    // (each .lora_a requires its .lora_b, so a leftover .lora_b means a
    // half-present adapter that would otherwise be merged as zero).
    var orphan_count: usize = 0;
    for (state.tensors) |*entry| {
        if (!std.mem.endsWith(u8, entry.name, ".lora_b")) continue;
        var a_name_buf: [96]u8 = undefined;
        const a_name = std.fmt.bufPrint(&a_name_buf, "{s}.lora_a", .{entry.name[0 .. entry.name.len - ".lora_b".len]}) catch return error.UnknownAdapterName;
        if (!names.contains(a_name)) {
            try stdout.print("orphan adapter tensor in {s}: {s} has no matching .lora_a\n", .{ adapter_path, entry.name });
            orphan_count += 1;
        }
    }
    if (orphan_count != 0) return error.OrphanLoraB;

    try stdout.print("merged {d} LoRA adapters from {s} (alpha {d})\n", .{ merged_count, adapter_path, alpha });
}

fn mergeOne(
    comptime t: usize,
    allocator: std.mem.Allocator,
    ctx: *fucina.ExecContext,
    file: *const gguf.File,
    ckpt_bytes: []const u8,
    layer: usize,
    rank: usize,
    alpha: f32,
    owned: *std.ArrayList([]u8),
    merged: *std.StringHashMap([]const u8),
    stdout: *std.Io.Writer,
) !void {
    var name_buf: [64]u8 = undefined;
    const gguf_name = try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, lora_gguf_suffix[t] });
    const info = try file.get(gguf_name);
    const shape = try info.logicalMatrixShape(); // [out, in]
    const out_dim = shape[0];
    const in_dim = shape[1];

    switch (info.ggml_type) {
        .f32, .f16, .bf16 => {},
        else => {
            try stdout.print(
                "cannot merge into {s}: base is {s}; merging into a quantized base would need dequant->merge->requant, which compounds quantization error. Merge into an f32/f16/bf16 GGUF, then quantize the merged file with --dtype in a second pass\n",
                .{ gguf_name, @tagName(info.ggml_type) },
            );
            return error.QuantizedBaseUnsupported;
        },
    }

    var adapter = try LoraAdapterFor(t).init(ctx, in_dim, out_dim, .{ .rank = rank, .alpha = alpha }, 0);
    defer adapter.deinit();

    var a_name_buf: [96]u8 = undefined;
    const a_name = try std.fmt.bufPrint(&a_name_buf, "layers.{d}.{s}.lora_a", .{ layer, lora_ckpt_names[t] });
    var b_name_buf: [96]u8 = undefined;
    const b_name = try std.fmt.bufPrint(&b_name_buf, "layers.{d}.{s}.lora_b", .{ layer, lora_ckpt_names[t] });
    var entries = [_]optim.NamedTensorMut{
        try optim.NamedTensorMut.of(a_name, &adapter.a),
        try optim.NamedTensorMut.of(b_name, &adapter.b),
    };
    var reader = std.Io.Reader.fixed(ckpt_bytes);
    try optim.loadStateDict(allocator, &reader, &entries, .{ .strict = false });

    const merged_bytes = switch (info.ggml_type) {
        .f32 => blk: {
            const values = try allocator.alloc(f32, out_dim * in_dim);
            defer allocator.free(values);
            widenRow(.f32, info.data, values);
            var weight = inner: {
                var raw = try ctx.fromSliceRank(2, .{ out_dim, in_dim }, values);
                errdefer raw.deinit();
                break :inner try WeightF32For(t).constant(ctx, raw);
            };
            defer weight.deinit();
            try adapter.mergeInto(ctx, &weight);
            break :blk try allocator.dupe(u8, std.mem.sliceAsBytes(try weight.dataConst()));
        },
        .bf16 => blk: {
            // bf16 is scalar: widen rows to f32, run the same merge math as
            // the f32 path, then re-encode through gguf.encodeF32(.bf16).
            const values = try allocator.alloc(f32, out_dim * in_dim);
            defer allocator.free(values);
            widenRow(.bf16, info.data, values);
            var weight = inner: {
                var raw = try ctx.fromSliceRank(2, .{ out_dim, in_dim }, values);
                errdefer raw.deinit();
                break :inner try WeightF32For(t).constant(ctx, raw);
            };
            defer weight.deinit();
            try adapter.mergeInto(ctx, &weight);
            const merged_f32 = try weight.dataConst();
            const bytes = try allocator.alloc(u8, merged_f32.len * 2);
            errdefer allocator.free(bytes);
            try gguf.encodeF32(.bf16, merged_f32, bytes);
            break :blk bytes;
        },
        .f16 => blk: {
            const halves = try allocator.alloc(f16, out_dim * in_dim);
            defer allocator.free(halves);
            for (halves, 0..) |*half, i| {
                half.* = @bitCast(std.mem.readInt(u16, info.data[i * 2 ..][0..2], .little));
            }
            var weight = inner: {
                var raw = try ctx.fromSliceRankTyped(.f16, 2, .{ out_dim, in_dim }, halves);
                errdefer raw.deinit();
                break :inner try WeightF16For(t).constant(ctx, raw);
            };
            defer weight.deinit();
            var result = try adapter.mergeF16(ctx, &weight);
            defer result.deinit();
            break :blk try allocator.dupe(u8, std.mem.sliceAsBytes(try result.dataConst()));
        },
        else => unreachable,
    };
    errdefer allocator.free(merged_bytes);
    // Reserve the map slot BEFORE ownership moves into `owned`: after the
    // append, `owned`'s deferred sweep frees merged_bytes, so a failing
    // `merged.put` here would have double-freed it via the errdefer.
    try merged.ensureUnusedCapacity(1);
    try owned.append(allocator, merged_bytes);
    merged.putAssumeCapacity(info.name, merged_bytes);
}

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var handle = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer handle.close(io);
    const stat = try handle.stat(io);
    if (stat.kind != .file) return error.IsDir;
    const len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, len);
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try handle.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

fn resolveAdaptersPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| switch (err) {
        error.NotDir, error.FileNotFound => return allocator.dupe(u8, path),
        else => |e| return e,
    };
    defer dir.close(io);
    return fucina.training_checkpoint.pathJoin(allocator, path, fucina.training_checkpoint.adapters_state_file);
}
