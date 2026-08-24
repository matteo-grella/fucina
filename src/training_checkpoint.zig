//! Directory checkpoint helpers for resumable training.
//!
//! Portable tensor payloads stay in standalone safetensors files. Runtime
//! resume state stays in sidecars: native optimizer frames plus a small JSON
//! trainer-state sentinel written last. The trainer-state codec is generic
//! over the caller's state struct (a version/step/seed header plus optional
//! `?u64`/`?f64` fields); the LLM trainers' concrete struct is
//! `fucina_models.train.trainer_state.TrainerState`.
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

pub fn saveTrainerState(allocator: Allocator, io: std.Io, dir_path: []const u8, state: anytype) !void {
    const path = try pathJoin(allocator, dir_path, trainer_state_file);
    defer allocator.free(path);
    const write = struct {
        fn write(s: @TypeOf(state), writer: *std.Io.Writer) anyerror!void {
            try writeTrainerStateJson(s, writer);
        }
    }.write;
    try writeFileAtomic(io, path, state, write);
}

pub fn loadTrainerState(comptime State: type, allocator: Allocator, io: std.Io, dir_path: []const u8) !State {
    const path = try pathJoin(allocator, dir_path, trainer_state_file);
    defer allocator.free(path);
    const bytes = try readFileAlloc(allocator, io, path);
    defer allocator.free(bytes);
    return parseTrainerState(State, allocator, bytes);
}

/// The hand-written required header: the `format` sentinel plus these
/// three fields. Every OTHER state field must be optional — the writer
/// and parser bodies are comptime-generated from the struct, so adding a
/// field is ONE struct line (JSON key = field name, emission order =
/// declaration order) and a writer/parser mismatch is impossible by
/// construction.
fn trainerStateFieldIsHeader(comptime name: []const u8) bool {
    return std.mem.eql(u8, name, "version") or std.mem.eql(u8, name, "step") or std.mem.eql(u8, name, "seed");
}

/// Serialize `state` in the pinned trainer-state JSON shape: the `format`
/// sentinel, the version/step/seed header, then every non-null optional
/// field in declaration order. Generic over the state struct; this frame
/// knows only the header.
pub fn writeTrainerStateJson(state: anytype, writer: *std.Io.Writer) !void {
    try writer.print(
        "{{\n  \"format\": \"fucina.training_checkpoint\",\n  \"version\": {d},\n  \"step\": {d},\n  \"seed\": {d}",
        .{ state.version, state.step, state.seed },
    );
    inline for (@typeInfo(@TypeOf(state)).@"struct".fields) |field| {
        if (comptime trainerStateFieldIsHeader(field.name)) continue;
        if (@field(state, field.name)) |value| {
            try writer.print(",\n  \"" ++ field.name ++ "\": {d}", .{value});
        }
    }
    try writer.writeAll("\n}\n");
}

/// Parse trainer-state JSON bytes into `State`. Unknown keys are ignored;
/// absent optional fields stay null.
pub fn parseTrainerState(comptime State: type, allocator: Allocator, bytes: []const u8) !State {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return Error.InvalidTrainerState;
    defer parsed.deinit();
    if (parsed.value != .object) return Error.InvalidTrainerState;
    const object = parsed.value.object;

    const format = object.get("format") orelse return Error.InvalidTrainerState;
    if (format != .string or !std.mem.eql(u8, format.string, "fucina.training_checkpoint")) return Error.InvalidTrainerState;

    const version = try jsonU32(object.get("version") orelse return Error.InvalidTrainerState);
    if (version != 1) return Error.UnsupportedTrainerStateVersion;

    var state: State = .{
        .version = version,
        .step = try jsonU64(object.get("step") orelse return Error.InvalidTrainerState),
        .seed = try jsonU64(object.get("seed") orelse return Error.InvalidTrainerState),
    };
    inline for (@typeInfo(State).@"struct".fields) |field| {
        if (comptime trainerStateFieldIsHeader(field.name)) continue;
        if (object.get(field.name)) |value| {
            @field(state, field.name) = switch (comptime @typeInfo(field.type).optional.child) {
                u64 => try jsonU64(value),
                f64 => try jsonF64(value),
                else => @compileError("trainer-state optional fields must be ?u64 or ?f64: " ++ field.name),
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
