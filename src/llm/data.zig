//! Minimal SFT dataset/dataloader helpers.
//!
//! Three small pieces, generic across model families:
//! - `SftText`: (instruction, response) pair source — a JSONL file (owned
//!   copy) or a zero-copy borrow of caller-owned pairs.
//! - `encodePair`/`encodePrompt`: chat-template render + tokenize + next-token
//!   shift + prompt masking, generic over the tokenizer via `anytype`
//!   (byte-BPE and SPM share the `encodeRaw` surface). The trainer's
//!   `ignore_index` sentinel is passed in via `EncodeOptions` — this module
//!   never imports a trainer.
//! - `Loader`: a deterministic sample-order iterator with resumable `State`
//!   (the shuffled epoch permutation is a pure function of (seed, epoch) —
//!   a checkpoint contract, see `Loader`).
//!
//! Deliberately deferred: a padded `[batch, seq]` Batcher. Both LLM trainers
//! take one flat token sequence per loss call — there is no batched-CE
//! consumer — and gradient accumulation is the honest batch mechanism on this
//! runtime today. Land it together with a batched loss when one exists.

const std = @import("std");
const builtin = @import("builtin");
const fucina = @import("fucina");
const chat = @import("chat.zig");

const rng = fucina.rng;
const Allocator = std.mem.Allocator;

pub const Error = error{
    /// A JSONL line failed to parse or lacked the configured string keys
    /// (the offending line number is logged).
    MalformedJsonl,
    /// The rendered prompt leaves no room for a supervised token within
    /// `seq_max` (or the pair encodes to fewer than two tokens).
    SampleTooLong,
    /// `Loader.init` over zero samples.
    EmptyDataset,
    /// `Loader.restore` with an index outside the dataset.
    InvalidLoaderState,
};

/// One supervised-fine-tuning example: a user instruction and the assistant
/// response text to supervise.
pub const Pair = struct { instruction: []const u8, response: []const u8 };

/// A pair source: either an owned copy loaded from a JSONL file or a borrow
/// of caller-owned pairs (e.g. a const in-source dataset).
pub const SftText = struct {
    /// The (instruction, response) pairs. Slices into `blob` when loaded via
    /// `fromJsonl` (owned); borrows of the caller's storage after `fromPairs`.
    pairs: []const Pair,
    /// Owned backing bytes for JSONL-loaded pair strings (null = borrowed).
    blob: ?[]u8 = null,

    pub const JsonlOptions = struct {
        instruction_key: []const u8 = "instruction",
        response_key: []const u8 = "response",
    };

    /// Zero-copy borrow of caller-owned pairs; `deinit` frees nothing.
    pub fn fromPairs(pairs: []const Pair) SftText {
        return .{ .pairs = pairs, .blob = null };
    }

    /// Load pairs from a JSONL file: one JSON object per line, with the
    /// instruction/response under `opts` keys (both must be strings). Blank
    /// lines are skipped; a malformed line is a loud error with its line
    /// number. Strings are duped out of the per-line `std.json` arena into
    /// one owned blob, so the result outlives the file bytes.
    pub fn fromJsonl(allocator: Allocator, io: std.Io, path: []const u8, opts: JsonlOptions) !SftText {
        const bytes = try readFileAlloc(allocator, io, path);
        defer allocator.free(bytes);

        // Byte spans into `blob`, resolved to slices only once the blob is
        // final (ArrayList growth may move its storage).
        const Span = struct { off: usize, len: usize };
        var spans: std.ArrayList([2]Span) = .empty;
        defer spans.deinit(allocator);
        var blob: std.ArrayList(u8) = .empty;
        defer blob.deinit(allocator);

        var line_no: usize = 0;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |raw_line| {
            line_no += 1;
            const line = std.mem.trim(u8, raw_line, " \t\r");
            if (line.len == 0) continue;
            var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch {
                jsonlError(path, line_no, "not valid JSON");
                return Error.MalformedJsonl;
            };
            defer parsed.deinit();
            if (parsed.value != .object) {
                jsonlError(path, line_no, "line is not a JSON object");
                return Error.MalformedJsonl;
            }
            const object = parsed.value.object;
            const instruction = stringField(object, opts.instruction_key) orelse {
                jsonlError(path, line_no, "missing/non-string instruction key");
                return Error.MalformedJsonl;
            };
            const response = stringField(object, opts.response_key) orelse {
                jsonlError(path, line_no, "missing/non-string response key");
                return Error.MalformedJsonl;
            };
            const ioff = blob.items.len;
            try blob.appendSlice(allocator, instruction);
            const roff = blob.items.len;
            try blob.appendSlice(allocator, response);
            try spans.append(allocator, .{
                .{ .off = ioff, .len = instruction.len },
                .{ .off = roff, .len = response.len },
            });
        }

        const blob_owned = try blob.toOwnedSlice(allocator);
        errdefer allocator.free(blob_owned);
        const pairs = try allocator.alloc(Pair, spans.items.len);
        for (pairs, spans.items) |*pair, span| {
            pair.* = .{
                .instruction = blob_owned[span[0].off..][0..span[0].len],
                .response = blob_owned[span[1].off..][0..span[1].len],
            };
        }
        return .{ .pairs = pairs, .blob = blob_owned };
    }

    pub fn deinit(self: *SftText, allocator: Allocator) void {
        if (self.blob) |blob| {
            allocator.free(self.pairs);
            allocator.free(blob);
        }
        self.* = undefined;
    }
};

fn stringField(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const value = object.get(key) orelse return null;
    return if (value == .string) value.string else null;
}

/// Loud malformed-JSONL diagnostic. Err-level so it survives ReleaseFast's
/// default log filter; skipped under the test runner, which counts err-level
/// logs as failures (the malformed-line tests exercise this path on purpose).
fn jsonlError(path: []const u8, line_no: usize, what: []const u8) void {
    if (!builtin.is_test) {
        std.log.err("data: {s}:{d}: malformed JSONL line ({s})", .{ path, line_no, what });
    }
}

/// One encoded training example.
pub const Sample = struct {
    /// Model inputs: full sequence minus the final token.
    inputs: []usize,
    /// Next-token labels; prompt positions masked with the ignore sentinel.
    labels: []usize,

    pub fn deinit(self: *Sample, allocator: Allocator) void {
        allocator.free(self.labels);
        allocator.free(self.inputs);
        self.* = undefined;
    }
};

pub const EncodeOptions = struct {
    /// Maximum input length: the sample is truncated to `seq_max` input
    /// positions (`seq_max + 1` sequence tokens).
    seq_max: usize = 256,
    /// The consuming trainer's label-mask sentinel (e.g.
    /// `llm.qwen3.train.ignore_index`), passed in so this module never
    /// imports a trainer.
    ignore_index: usize = std.math.maxInt(usize),
    /// Mask prompt positions in `labels` (supervise the response only).
    mask_prompt: bool = true,
    /// Optional system prompt for the rendered turn.
    system: ?[]const u8 = null,
    /// Suppress the model's reasoning channel in the rendered turn.
    think_off: bool = true,
};

/// Render one user turn (system per `opts`, first turn, think per `opts`) and
/// encode it. `tokenizer` is duck-typed over `encodeRaw` (byte-BPE and SPM).
pub fn encodePrompt(
    allocator: Allocator,
    tokenizer: anytype,
    template: chat.Template,
    instruction: []const u8,
    opts: EncodeOptions,
) ![]usize {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    try template.renderTurn(allocator, &text, opts.system, instruction, true, opts.think_off);
    return encodeUsize(allocator, tokenizer, text.items);
}

/// prompt tokens ++ response tokens (with the turn-end marker); inputs are
/// the sequence minus its last token, labels the next-token shift with all
/// prompt positions masked (unless `opts.mask_prompt` is off). Truncated to
/// `opts.seq_max` inputs. The response (+ stop marker) is encoded SEPARATELY
/// from the prompt: concatenating the text before encoding would move BPE
/// chunk boundaries across the join and change token IDs.
pub fn encodePair(
    allocator: Allocator,
    tokenizer: anytype,
    template: chat.Template,
    pair: Pair,
    opts: EncodeOptions,
) !Sample {
    const prompt = try encodePrompt(allocator, tokenizer, template, pair.instruction, opts);
    defer allocator.free(prompt);

    var response_text: std.ArrayList(u8) = .empty;
    defer response_text.deinit(allocator);
    try response_text.appendSlice(allocator, pair.response);
    try response_text.appendSlice(allocator, template.stopMarker());
    const response = try encodeUsize(allocator, tokenizer, response_text.items);
    defer allocator.free(response);

    const total = @min(prompt.len + response.len, opts.seq_max + 1);
    // The prompt must leave room for at least one supervised token.
    if (total < 2 or prompt.len + 1 > total) return Error.SampleTooLong;

    const inputs = try allocator.alloc(usize, total - 1);
    errdefer allocator.free(inputs);
    const labels = try allocator.alloc(usize, total - 1);
    errdefer allocator.free(labels);
    for (0..total) |i| {
        const token = if (i < prompt.len) prompt[i] else response[i - prompt.len];
        if (i < total - 1) inputs[i] = token;
        // Position i-1 predicts token i; supervise response tokens only.
        if (i > 0) labels[i - 1] = if (opts.mask_prompt and i < prompt.len) opts.ignore_index else token;
    }
    return .{ .inputs = inputs, .labels = labels };
}

/// `tokenizer.encodeRaw` (no BOS/EOS policy — the template controls all
/// structural tokens) widened to the `usize` ids the trainers consume.
fn encodeUsize(allocator: Allocator, tokenizer: anytype, text: []const u8) ![]usize {
    const ids32 = try tokenizer.encodeRaw(allocator, text);
    defer allocator.free(ids32);
    const ids = try allocator.alloc(usize, ids32.len);
    for (ids, ids32) |*dst, src| dst.* = src;
    return ids;
}

/// Deterministic sample-order iterator over `n` dataset indices.
///
/// `.sequential` is the plain round-robin `0, 1, …, n-1, 0, …` (bit-identical
/// to indexing with `step % n`). `.shuffled` draws each epoch as a fresh
/// permutation that is a pure function of `(seed, epoch)`, so a saved `State`
/// fully reconstructs the stream position.
///
/// CHECKPOINT CONTRACT (same rationale as `src/rng.zig`): the shuffled epoch
/// permutation starts from the identity permutation of `0..n-1` and applies a
/// Fisher–Yates pass driven by a splitmix64 stream seeded with
/// `rng.at(seed, epoch)` — for `i` from `n-1` down to `1`:
/// `j = rng.splitmix64(&stream) % (i + 1)`, then `swap(perm[i], perm[j])`.
/// `restore` regenerates the exact order from a saved `State`, so this
/// formula may never change once checkpoints exist against it. Golden-pinned
/// in `data_tests.zig`.
pub const Loader = struct {
    pub const Order = enum { sequential, shuffled };
    /// Everything needed to reconstruct the stream position (the epoch
    /// permutation regenerates from (seed, epoch); `index` is the offset of
    /// the NEXT draw within it). u64 fields so it round-trips through
    /// `trainer_state.json` unchanged.
    pub const State = struct { seed: u64, epoch: u64, index: u64 };

    order: Order,
    seed: u64,
    epoch: u64,
    /// Position of the next draw within `perm`; always < `perm.len`.
    index: usize,
    /// The current epoch's index order (identity for `.sequential`).
    perm: []usize,

    pub fn init(allocator: Allocator, n: usize, order: Order, seed: u64) !Loader {
        if (n == 0) return Error.EmptyDataset;
        const perm = try allocator.alloc(usize, n);
        var self = Loader{ .order = order, .seed = seed, .epoch = 0, .index = 0, .perm = perm };
        self.fillPerm();
        return self;
    }

    pub fn deinit(self: *Loader, allocator: Allocator) void {
        allocator.free(self.perm);
        self.* = undefined;
    }

    /// The next dataset index; consuming the epoch's last element wraps into
    /// a fresh epoch (re-permuted under `.shuffled`).
    pub fn next(self: *Loader) usize {
        const idx = self.perm[self.index];
        self.index += 1;
        if (self.index == self.perm.len) {
            self.epoch += 1;
            self.index = 0;
            self.fillPerm();
        }
        return idx;
    }

    pub fn state(self: *const Loader) State {
        return .{ .seed = self.seed, .epoch = self.epoch, .index = self.index };
    }

    /// Rebuild the exact stream position from a saved `State`: the epoch
    /// permutation is regenerated from (seed, epoch) and the next draw
    /// resumes at `index`. Fails if `index` does not fit this dataset (e.g.
    /// the checkpoint was written against a different dataset size).
    pub fn restore(self: *Loader, s: State) !void {
        if (s.index >= self.perm.len) return Error.InvalidLoaderState;
        self.seed = s.seed;
        self.epoch = s.epoch;
        self.index = @intCast(s.index);
        self.fillPerm();
    }

    fn fillPerm(self: *Loader) void {
        for (self.perm, 0..) |*p, i| p.* = i;
        if (self.order == .shuffled) {
            // The contract formula documented on `Loader` — do not change.
            var stream = rng.at(self.seed, self.epoch);
            var i: usize = self.perm.len - 1;
            while (i >= 1) : (i -= 1) {
                const j: usize = @intCast(rng.splitmix64(&stream) % (i + 1));
                std.mem.swap(usize, &self.perm[i], &self.perm[j]);
            }
        }
    }
};

/// Whole-file read via the event-loop-friendly streaming API (mirrors
/// `training_checkpoint.readFileAlloc`).
fn readFileAlloc(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.IsDir;
    const len: usize = @intCast(stat.size);
    const bytes = try allocator.alloc(u8, len);
    errdefer allocator.free(bytes);
    var read_len: usize = 0;
    while (read_len < bytes.len) {
        const n = try file.readStreaming(io, &.{bytes[read_len..]});
        if (n == 0) return error.EndOfStream;
        read_len += n;
    }
    return bytes;
}

test {
    _ = @import("data_tests.zig");
}
