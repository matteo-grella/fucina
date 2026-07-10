//! nanochat port (karpathy/nanochat → Fucina): a from-scratch CPU pipeline for a
//! small GPT — BPE tokenizer training, base pretraining, supervised fine-tuning,
//! bits-per-byte evaluation, and an interactive chat CLI with a calculator tool.
//!
//! Parity target: the Python reference (nanochat @ 92d63d4) running on CPU in
//! fp32. Interchange file formats are documented at their read/write sites in
//! tokenizer.zig and data.zig.
//!
//! Subcommands (dispatched in `main`):
//!   tok-train   train the BPE tokenizer (rustbpe-equivalent) → tokenizer.bin
//!   base-train  pretrain the GPT on framed pretraining docs
//!   sft         supervised fine-tune a base checkpoint on the task mixture
//!   eval-bpb    bits-per-byte on a val split
//!   chat        interactive chat / single-prompt generation
//!
//! Run with `zig build nanochat -- <subcommand> [args]`.
const std = @import("std");

pub const tokenizer = @import("nanochat/tokenizer.zig");
pub const model = @import("nanochat/model.zig");
pub const train = @import("nanochat/train.zig");
pub const chat = @import("nanochat/chat.zig");

const Subcommand = enum { @"tok-train", @"base-train", sft, @"eval-bpb", chat, help };

const usage_text =
    \\nanochat (Fucina port)
    \\usage: nanochat <subcommand> [args]
    \\  tok-train   train the BPE tokenizer
    \\              --input <NCTXT_01 file> [--vocab N (default 32768)] --out <dir>
    \\  base-train  pretrain the GPT
    \\              --data <NCDOC> --tokenizer <tokenizer.bin> --out <dir>
    \\              [--val-data <NCDOC>] [--init-from <safetensors> | --resume <dir>]
    \\              [--total-batch-size N (0 = auto Power-Lines batch)]
    \\              [--num-iterations N (0 = auto target_tokens/batch)]
    \\  sft         supervised fine-tune a base checkpoint on the task mixture
    \\              --init-from <base model.safetensors> --mixture <SFT JSONL>
    \\              --tokenizer <tokenizer.bin> --out <dir> [--num-iterations N]
    \\              [--device-batch-size B] [--max-seq-len T] [--total-batch-size N]
    \\              [--init-lr-frac 0.8] [--warmup-ratio 0.0] [--warmdown-ratio 0.5]
    \\              [--load-optimizer 1] [--val-mixture <JSONL>] [--resume <dir>]
    \\  eval-bpb    bits-per-byte eval
    \\  chat        chat / single-prompt generation
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    const sub: Subcommand = if (args.len >= 2)
        std.meta.stringToEnum(Subcommand, args[1]) orelse .help
    else
        .help;

    switch (sub) {
        .help => try stdout.writeAll(usage_text),
        .@"tok-train" => try runTokTrain(init.io, stdout, args[2..]),
        .@"base-train" => try train.runBaseTrain(init.io, stdout, args[2..]),
        .sft => try train.runSft(init.io, stdout, args[2..]),
        .@"eval-bpb" => try train.runEvalBpb(init.io, stdout, args[2..]),
        .chat => try chat.runChat(init.io, stdout, args[2..]),
    }
}

/// `tok-train --input <NCTXT_01> [--vocab N] --out <dir>`: train the
/// rustbpe-equivalent tokenizer and write tokenizer.bin + token_bytes.bin.
fn runTokTrain(io: std.Io, stdout: *std.Io.Writer, args: []const []const u8) !void {
    const allocator = std.heap.smp_allocator;

    var input_path: ?[]const u8 = null;
    var out_dir: ?[]const u8 = null;
    var vocab_size: u32 = 32768;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--input") and i + 1 < args.len) {
            i += 1;
            input_path = args[i];
        } else if (std.mem.eql(u8, arg, "--vocab") and i + 1 < args.len) {
            i += 1;
            vocab_size = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--out") and i + 1 < args.len) {
            i += 1;
            out_dir = args[i];
        } else {
            try stdout.print("unknown tok-train argument: {s}\n\n{s}", .{ arg, usage_text });
            return error.InvalidArgument;
        }
    }
    if (input_path == null or out_dir == null) {
        try stdout.writeAll(usage_text);
        return error.InvalidArgument;
    }

    var docs = try tokenizer.readDocsFile(allocator, io, input_path.?);
    defer docs.deinit(allocator);
    try stdout.print("tok-train: {d} docs from {s}, vocab {d}\n", .{ docs.docs.len, input_path.?, vocab_size });
    try stdout.flush();

    const t0 = std.Io.Clock.awake.now(io).nanoseconds;
    var tok = try tokenizer.Tokenizer.trainFromDocs(allocator, docs.docs, vocab_size);
    defer tok.deinit();
    const train_ns: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - t0);

    try std.Io.Dir.cwd().createDirPath(io, out_dir.?);
    const tok_path = try std.fs.path.join(allocator, &.{ out_dir.?, "tokenizer.bin" });
    defer allocator.free(tok_path);
    const tb_path = try std.fs.path.join(allocator, &.{ out_dir.?, "token_bytes.bin" });
    defer allocator.free(tb_path);
    try tok.saveBin(io, tok_path);
    try tok.saveTokenBytes(allocator, io, tb_path);

    try stdout.print(
        "trained {d} merges (n_vocab {d}) in {d:.1}s\nwrote {s}\nwrote {s}\n",
        .{ tok.nMerges(), tok.n_vocab, @as(f64, @floatFromInt(train_ns)) / 1e9, tok_path, tb_path },
    );
}

test {
    // refAllDecls covers the root pub decls (tokenizer/model/train/chat);
    // optim and data are not re-exported, so reference them explicitly.
    std.testing.refAllDecls(@This());
    _ = @import("nanochat/optim.zig");
    _ = @import("nanochat/data.zig");
}
