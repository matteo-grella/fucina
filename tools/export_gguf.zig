//! GGUF export tool — closes the train→export→serve-anywhere loop.
//!
//! Modes:
//!   (a) re-emit / transcode:
//!       zig build export-gguf -- --from-gguf in.gguf --out out.gguf [--dtype f16|bf16|f32|q8_0|q4_k|q5_k|q6_k|tq2_0|verbatim]
//!   (b) merge Fucina LoRA adapters (safetensors, as saved by `zig build finetune`)
//!       into dense f32/f16/bf16 base weights and re-emit:
//!       zig build export-gguf -- --from-gguf in.gguf --adapters lora-dir --out out.gguf --alpha F
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

const std = @import("std");
const fucina = @import("fucina");

const gguf = fucina.gguf;
const lora = fucina.lora;
const optim = fucina.optim;
const safetensors = fucina.safetensors;

const DtypeMode = enum { verbatim, f32, f16, bf16, q8_0, q4_k, q5_k, q6_k, tq2_0 };

const usage =
    "usage: zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf IN.gguf --out OUT.gguf [--dtype f16|bf16|f32|q8_0|q4_k|q5_k|q6_k|tq2_0|verbatim] [--experts-dtype DTYPE (only tensors named *_exps.weight; may requantize)] [--adapters DIR_OR_SAFETENSORS --alpha F (required together)]\n";

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
        } else {
            try stdout.print(usage, .{});
            return error.UnknownArgument;
        }
    }
    const in_path = from_path orelse {
        try stdout.print(usage, .{});
        return error.MissingFromPath;
    };
    const dst_path = out_path orelse {
        try stdout.print(usage, .{});
        return error.MissingOutPath;
    };
    if (adapters_path != null and (dtype_mode != .verbatim or experts_dtype_mode != null)) {
        try stdout.print("--adapters merges into the base dtype; combining it with --dtype/--experts-dtype transcoding is not supported\n", .{});
        return error.UnsupportedArgumentCombination;
    }
    if (adapters_path != null and alpha == null) {
        try stdout.print("--adapters requires --alpha: the safetensors checkpoint stores the adapter A/B matrices but not alpha, so pass the training-time value (finetune default: 16)\n", .{});
        return error.MissingAlpha;
    }

    const allocator = std.heap.smp_allocator;

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
