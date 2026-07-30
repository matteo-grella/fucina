//! Directory checkpoint helpers for resumable training.
//!
//! Portable tensor payloads stay in standalone safetensors files. Runtime
//! resume state stays in sidecars: native optimizer frames plus a small JSON
//! trainer-state sentinel written last.
const std = @import("std");

const Allocator = std.mem.Allocator;

pub const model_state_file = "model.safetensors";
pub const adapters_state_file = "adapters.safetensors";
pub const optimizer_state_file = "optimizer.fucina";
pub const trainer_state_file = "trainer_state.json";

pub const Error = error{
    InvalidTrainerState,
    UnsupportedTrainerStateVersion,
};

pub const TrainerState = struct {
    version: u32 = 1,
    step: u64 = 0,
    seed: u64 = 0,
    lora_rank: ?u64 = null,
    lora_alpha: ?f64 = null,
    lora_dropout_p: ?f64 = null,
    learning_rate: ?f64 = null,
    /// Gradient-accumulation window size the loop trained with (optional,
    /// like `lora_rank`). Checkpoints must be written at window boundaries —
    /// accumulated gradients are never serialized — so on resume `step`
    /// (micro-batch count) satisfies `step % accum_steps == 0`.
    accum_steps: ?u64 = null,
    /// Dataloader stream position (`llm.data.Loader.State`), optional as a
    /// triple: the epoch permutation is a pure function of
    /// (data_seed, data_epoch), so these three fields fully reconstruct the
    /// sample order on resume.
    data_seed: ?u64 = null,
    data_epoch: ?u64 = null,
    data_index: ?u64 = null,
    /// Evolution-strategies trainer state (`fucina.es`), optional like
    /// `lora_rank`: sigma/alpha/population pin the run configuration
    /// (validate them on resume), `es_noise` pins the noise scheme (STABLE
    /// on-disk mapping — 0 = iid, 1 = correlated; never `@intFromEnum`), and
    /// `es_iteration` restores the member-seed stream position — (seed,
    /// iteration, population, scheme) fully regenerate the population, so
    /// nothing else needs serializing. Flags that do NOT affect the noise
    /// contract (restore mode, reward) are re-passed on the CLI like
    /// `--shuffle`.
    es_sigma: ?f64 = null,
    es_alpha: ?f64 = null,
    es_population: ?u64 = null,
    es_noise: ?u64 = null,
    /// 1 = mirrored (antithetic) pairs, 0/absent = independent members.
    /// Part of the noise contract like `es_noise`.
    es_antithetic: ?u64 = null,
    /// Anchored weight decay: 0/absent = none, 1 = l1, 2 = l2 (stable
    /// mapping), with `es_anchor_lambda` the configured lambda. The anchor
    /// ITSELF is not serialized — it is reconstructable (reload the
    /// pretrained weights / re-init adapters from the seed) and must be
    /// re-captured BEFORE loading the checkpointed parameters on resume.
    es_anchor_decay: ?u64 = null,
    es_anchor_lambda: ?f64 = null,
    /// Ternary genome knobs (es.Config.ternary_*): flip rate and update
    /// fraction/decay are part of the flip-stream and top-K contracts like
    /// `es_antithetic` — (seed, iteration, population, these rates) fully
    /// regenerate every member's flips and the update schedule.
    es_ternary_flip_rate: ?f64 = null,
    es_ternary_update_fraction: ?f64 = null,
    es_ternary_update_decay: ?f64 = null,
    es_iteration: ?u64 = null,
};

pub fn pathJoin(allocator: Allocator, dir_path: []const u8, leaf: []const u8) ![]u8 {
    return std.fs.path.join(allocator, &.{ dir_path, leaf });
}

/// Prepare a checkpoint directory for a new save. The trainer-state file is
/// removed first because it is the commit sentinel.
pub fn beginSave(allocator: Allocator, io: std.Io, dir_path: []const u8) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir_path);
    const state_path = try pathJoin(allocator, dir_path, trainer_state_file);
    defer allocator.free(state_path);
    std.Io.Dir.cwd().deleteFile(io, state_path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => |e| return e,
    };
}

pub fn writeFileAtomic(
    io: std.Io,
    path: []const u8,
    context: anytype,
    comptime writeFn: fn (@TypeOf(context), *std.Io.Writer) anyerror!void,
) !void {
    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, path, .{ .make_path = true, .replace = true });
    defer atomic.deinit(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = atomic.file.writer(io, &buffer);
    try writeFn(context, &writer.interface);
    try writer.interface.flush();
    try atomic.replace(io);
}

pub fn saveTrainerState(allocator: Allocator, io: std.Io, dir_path: []const u8, state: TrainerState) !void {
    const path = try pathJoin(allocator, dir_path, trainer_state_file);
    defer allocator.free(path);
    try writeFileAtomic(io, path, state, writeTrainerStateJson);
}

pub fn loadTrainerState(allocator: Allocator, io: std.Io, dir_path: []const u8) !TrainerState {
    const path = try pathJoin(allocator, dir_path, trainer_state_file);
    defer allocator.free(path);
    const bytes = try readFileAlloc(allocator, io, path);
    defer allocator.free(bytes);
    return parseTrainerState(allocator, bytes);
}

/// The hand-written required header: the `format` sentinel plus these
/// three fields. Every OTHER `TrainerState` field must be optional — the
/// writer and parser bodies are comptime-generated from the struct, so
/// adding a field is ONE struct line (JSON key = field name, emission
/// order = declaration order) and a writer/parser mismatch is impossible
/// by construction.
fn trainerStateFieldIsHeader(comptime name: []const u8) bool {
    return std.mem.eql(u8, name, "version") or std.mem.eql(u8, name, "step") or std.mem.eql(u8, name, "seed");
}

fn writeTrainerStateJson(state: TrainerState, writer: *std.Io.Writer) !void {
    try writer.print(
        "{{\n  \"format\": \"fucina.training_checkpoint\",\n  \"version\": {d},\n  \"step\": {d},\n  \"seed\": {d}",
        .{ state.version, state.step, state.seed },
    );
    inline for (@typeInfo(TrainerState).@"struct".fields) |field| {
        if (comptime trainerStateFieldIsHeader(field.name)) continue;
        if (@field(state, field.name)) |value| {
            try writer.print(",\n  \"" ++ field.name ++ "\": {d}", .{value});
        }
    }
    try writer.writeAll("\n}\n");
}

fn parseTrainerState(allocator: Allocator, bytes: []const u8) !TrainerState {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return Error.InvalidTrainerState;
    defer parsed.deinit();
    if (parsed.value != .object) return Error.InvalidTrainerState;
    const object = parsed.value.object;

    const format = object.get("format") orelse return Error.InvalidTrainerState;
    if (format != .string or !std.mem.eql(u8, format.string, "fucina.training_checkpoint")) return Error.InvalidTrainerState;

    const version = try jsonU32(object.get("version") orelse return Error.InvalidTrainerState);
    if (version != 1) return Error.UnsupportedTrainerStateVersion;

    var state: TrainerState = .{
        .version = version,
        .step = try jsonU64(object.get("step") orelse return Error.InvalidTrainerState),
        .seed = try jsonU64(object.get("seed") orelse return Error.InvalidTrainerState),
    };
    inline for (@typeInfo(TrainerState).@"struct".fields) |field| {
        if (comptime trainerStateFieldIsHeader(field.name)) continue;
        if (object.get(field.name)) |value| {
            @field(state, field.name) = switch (comptime @typeInfo(field.type).optional.child) {
                u64 => try jsonU64(value),
                f64 => try jsonF64(value),
                else => @compileError("TrainerState optional fields must be ?u64 or ?f64: " ++ field.name),
            };
        }
    }
    return state;
}

fn jsonU32(value: std.json.Value) !u32 {
    const v = try jsonU64(value);
    return std.math.cast(u32, v) orelse Error.InvalidTrainerState;
}

fn jsonU64(value: std.json.Value) !u64 {
    return switch (value) {
        .integer => |v| if (v < 0) Error.InvalidTrainerState else @intCast(v),
        .number_string => |v| std.fmt.parseInt(u64, v, 10) catch Error.InvalidTrainerState,
        else => Error.InvalidTrainerState,
    };
}

fn jsonF64(value: std.json.Value) !f64 {
    return switch (value) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        .number_string => |v| std.fmt.parseFloat(f64, v) catch Error.InvalidTrainerState,
        else => Error.InvalidTrainerState,
    };
}

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

test "trainer state serializes with the pinned json shape" {
    // Byte-format pin: the writer/parser bodies are comptime-generated, so
    // this is what keeps them from co-drifting to a DIFFERENT format that
    // still roundtrips (key spelling, emission order, layout).
    var aw = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer aw.deinit();
    try writeTrainerStateJson(.{ .step = 12, .seed = 34, .lora_rank = 8, .learning_rate = 1e-3, .es_iteration = 17 }, &aw.writer);
    try std.testing.expectEqualStrings(
        "{\n" ++
            "  \"format\": \"fucina.training_checkpoint\",\n" ++
            "  \"version\": 1,\n" ++
            "  \"step\": 12,\n" ++
            "  \"seed\": 34,\n" ++
            "  \"lora_rank\": 8,\n" ++
            "  \"learning_rate\": 0.001,\n" ++
            "  \"es_iteration\": 17\n" ++
            "}\n",
        aw.written(),
    );
}

test "trainer state roundtrips through directory sentinel" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var path_buf: [128]u8 = undefined;
    const dir_path = try std.fmt.bufPrint(&path_buf, "training_checkpoint_test_{d}", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteTree(io, dir_path) catch {};

    try beginSave(allocator, io, dir_path);
    try saveTrainerState(allocator, io, dir_path, .{
        .step = 12,
        .seed = 34,
        .lora_rank = 8,
        .lora_alpha = 16,
        .lora_dropout_p = 0.125,
        .learning_rate = 1e-3,
        .accum_steps = 4,
        .data_seed = 42,
        .data_epoch = 2,
        .data_index = 3,
        .es_sigma = 0.001,
        .es_alpha = 0.0005,
        .es_population = 30,
        .es_noise = 1,
        .es_antithetic = 1,
        .es_anchor_decay = 2,
        .es_anchor_lambda = 10.0,
        .es_ternary_flip_rate = 0.001,
        .es_ternary_update_fraction = 0.005,
        .es_ternary_update_decay = 0.015,
        .es_iteration = 17,
    });
    const loaded = try loadTrainerState(allocator, io, dir_path);
    try std.testing.expectEqual(@as(u32, 1), loaded.version);
    try std.testing.expectEqual(@as(u64, 12), loaded.step);
    try std.testing.expectEqual(@as(u64, 34), loaded.seed);
    try std.testing.expectEqual(@as(?u64, 8), loaded.lora_rank);
    try std.testing.expectEqual(@as(?f64, 16), loaded.lora_alpha);
    try std.testing.expectEqual(@as(?f64, 0.125), loaded.lora_dropout_p);
    try std.testing.expectEqual(@as(?f64, 1e-3), loaded.learning_rate);
    try std.testing.expectEqual(@as(?u64, 4), loaded.accum_steps);
    try std.testing.expectEqual(@as(?u64, 42), loaded.data_seed);
    try std.testing.expectEqual(@as(?u64, 2), loaded.data_epoch);
    try std.testing.expectEqual(@as(?u64, 3), loaded.data_index);
    try std.testing.expectEqual(@as(?f64, 0.001), loaded.es_sigma);
    try std.testing.expectEqual(@as(?f64, 0.0005), loaded.es_alpha);
    try std.testing.expectEqual(@as(?u64, 30), loaded.es_population);
    try std.testing.expectEqual(@as(?u64, 1), loaded.es_noise);
    try std.testing.expectEqual(@as(?u64, 1), loaded.es_antithetic);
    try std.testing.expectEqual(@as(?u64, 2), loaded.es_anchor_decay);
    try std.testing.expectEqual(@as(?f64, 10.0), loaded.es_anchor_lambda);
    try std.testing.expectEqual(@as(?f64, 0.001), loaded.es_ternary_flip_rate);
    try std.testing.expectEqual(@as(?f64, 0.005), loaded.es_ternary_update_fraction);
    try std.testing.expectEqual(@as(?f64, 0.015), loaded.es_ternary_update_decay);
    try std.testing.expectEqual(@as(?u64, 17), loaded.es_iteration);

    // Absent optionals stay null (older checkpoints without accum_steps or
    // dataloader state).
    try beginSave(allocator, io, dir_path);
    try saveTrainerState(allocator, io, dir_path, .{ .step = 3, .seed = 7 });
    const bare = try loadTrainerState(allocator, io, dir_path);
    try std.testing.expectEqual(@as(?u64, null), bare.accum_steps);
    try std.testing.expectEqual(@as(?u64, null), bare.lora_rank);
    try std.testing.expectEqual(@as(?u64, null), bare.data_seed);
    try std.testing.expectEqual(@as(?u64, null), bare.data_epoch);
    try std.testing.expectEqual(@as(?u64, null), bare.data_index);
    try std.testing.expectEqual(@as(?f64, null), bare.es_sigma);
    try std.testing.expectEqual(@as(?f64, null), bare.es_ternary_flip_rate);
    try std.testing.expectEqual(@as(?u64, null), bare.es_iteration);
}
