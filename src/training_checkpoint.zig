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

fn writeTrainerStateJson(state: TrainerState, writer: *std.Io.Writer) !void {
    try writer.print(
        "{{\n  \"format\": \"fucina.training_checkpoint\",\n  \"version\": {d},\n  \"step\": {d},\n  \"seed\": {d}",
        .{ state.version, state.step, state.seed },
    );
    if (state.lora_rank) |rank| try writer.print(",\n  \"lora_rank\": {d}", .{rank});
    if (state.lora_alpha) |alpha| try writer.print(",\n  \"lora_alpha\": {d}", .{alpha});
    if (state.lora_dropout_p) |dropout_p| try writer.print(",\n  \"lora_dropout_p\": {d}", .{dropout_p});
    if (state.learning_rate) |lr| try writer.print(",\n  \"learning_rate\": {d}", .{lr});
    if (state.accum_steps) |accum| try writer.print(",\n  \"accum_steps\": {d}", .{accum});
    if (state.data_seed) |seed| try writer.print(",\n  \"data_seed\": {d}", .{seed});
    if (state.data_epoch) |epoch| try writer.print(",\n  \"data_epoch\": {d}", .{epoch});
    if (state.data_index) |index| try writer.print(",\n  \"data_index\": {d}", .{index});
    if (state.es_sigma) |sigma| try writer.print(",\n  \"es_sigma\": {d}", .{sigma});
    if (state.es_alpha) |alpha| try writer.print(",\n  \"es_alpha\": {d}", .{alpha});
    if (state.es_population) |population| try writer.print(",\n  \"es_population\": {d}", .{population});
    if (state.es_noise) |noise| try writer.print(",\n  \"es_noise\": {d}", .{noise});
    if (state.es_antithetic) |antithetic| try writer.print(",\n  \"es_antithetic\": {d}", .{antithetic});
    if (state.es_anchor_decay) |decay| try writer.print(",\n  \"es_anchor_decay\": {d}", .{decay});
    if (state.es_anchor_lambda) |lambda| try writer.print(",\n  \"es_anchor_lambda\": {d}", .{lambda});
    if (state.es_ternary_flip_rate) |rate| try writer.print(",\n  \"es_ternary_flip_rate\": {d}", .{rate});
    if (state.es_ternary_update_fraction) |fraction| try writer.print(",\n  \"es_ternary_update_fraction\": {d}", .{fraction});
    if (state.es_ternary_update_decay) |decay| try writer.print(",\n  \"es_ternary_update_decay\": {d}", .{decay});
    if (state.es_iteration) |iteration| try writer.print(",\n  \"es_iteration\": {d}", .{iteration});
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

    return .{
        .version = version,
        .step = try jsonU64(object.get("step") orelse return Error.InvalidTrainerState),
        .seed = try jsonU64(object.get("seed") orelse return Error.InvalidTrainerState),
        .lora_rank = if (object.get("lora_rank")) |value| try jsonU64(value) else null,
        .lora_alpha = if (object.get("lora_alpha")) |value| try jsonF64(value) else null,
        .lora_dropout_p = if (object.get("lora_dropout_p")) |value| try jsonF64(value) else null,
        .learning_rate = if (object.get("learning_rate")) |value| try jsonF64(value) else null,
        .accum_steps = if (object.get("accum_steps")) |value| try jsonU64(value) else null,
        .data_seed = if (object.get("data_seed")) |value| try jsonU64(value) else null,
        .data_epoch = if (object.get("data_epoch")) |value| try jsonU64(value) else null,
        .data_index = if (object.get("data_index")) |value| try jsonU64(value) else null,
        .es_sigma = if (object.get("es_sigma")) |value| try jsonF64(value) else null,
        .es_alpha = if (object.get("es_alpha")) |value| try jsonF64(value) else null,
        .es_population = if (object.get("es_population")) |value| try jsonU64(value) else null,
        .es_noise = if (object.get("es_noise")) |value| try jsonU64(value) else null,
        .es_antithetic = if (object.get("es_antithetic")) |value| try jsonU64(value) else null,
        .es_anchor_decay = if (object.get("es_anchor_decay")) |value| try jsonU64(value) else null,
        .es_anchor_lambda = if (object.get("es_anchor_lambda")) |value| try jsonF64(value) else null,
        .es_ternary_flip_rate = if (object.get("es_ternary_flip_rate")) |value| try jsonF64(value) else null,
        .es_ternary_update_fraction = if (object.get("es_ternary_update_fraction")) |value| try jsonF64(value) else null,
        .es_ternary_update_decay = if (object.get("es_ternary_update_decay")) |value| try jsonF64(value) else null,
        .es_iteration = if (object.get("es_iteration")) |value| try jsonU64(value) else null,
    };
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
