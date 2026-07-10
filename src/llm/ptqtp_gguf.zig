//! PTQTP GGUF persistence (docs/PTQTP.md): a decorated model round-trips
//! through GGUF as one byte-valid standalone TQ2_0 tensor per trit-plane —
//! `<name>.ptqtp0/1/2` replaces `<name>` — plus a `fucina.ptqtp.version`
//! metadata key that marks the file and gates loader pair-detection. Every
//! other tensor and metadata entry passes through byte-verbatim, so
//! decoration happens once (`--save` in examples/ptqtp_qwen3.zig) and the
//! saved file serves through the ordinary family loaders forever after.
//!
//! Fused in-memory weights persist under their SOURCE tensor names: the
//! solver treats every 256-column group independently (src/ptqtp.zig), so
//! the plane rows of a fused q/k/v or gate/up matrix are byte-identical to
//! the planes of each part decorated alone, and a fused weight's planes
//! slice row-wise into per-part tensors. Loading re-fuses them through the
//! `fuseLinear` ptqtp arm, so save→load reproduces the exact in-memory
//! decoration — plane bytes and serving path alike.

const std = @import("std");
const fucina = @import("fucina");
const weights = @import("weights.zig");

const Allocator = std.mem.Allocator;
const ExecContext = fucina.ExecContext;
const gguf = fucina.gguf;
const LinearWeight = weights.LinearWeight;

pub const Error = error{
    /// Two save entries claim the same tensor name.
    DuplicateSaveEntry,
    /// A save entry's row window falls outside its weight's out dim.
    InvalidRowRange,
    /// A base tensor exists in the source file but its dims disagree with
    /// the planes about to replace it (row-range bookkeeping error).
    PlaneShapeMismatch,
    /// A `.ptqtpK` tensor is not TQ2_0.
    PlaneTypeMismatch,
    /// `.ptqtp2` present without `.ptqtp1` — planes fill in order.
    InvalidPlaneSet,
    /// The file's `fucina.ptqtp.version` is newer than this build reads.
    UnsupportedPtqtpVersion,
};

/// Bumped only for layout-incompatible changes; the loader refuses newer.
pub const format_version: u32 = 1;
pub const version_key = "fucina.ptqtp.version";

/// Longest tensor name this module reads or writes (base + `.ptqtp0`).
const max_name_len = 160;

/// `<base>.ptqtpK` — one byte-valid TQ2_0 tensor per plane.
pub fn planeName(buf: []u8, base: []const u8, plane: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.ptqtp{d}", .{ base, plane });
}

/// Whether `name` is a plane tensor name; returns its base name and plane.
fn parsePlaneName(name: []const u8) ?struct { base: []const u8, plane: usize } {
    const suffix = ".ptqtp";
    if (name.len < suffix.len + 2) return null;
    const digit = name[name.len - 1];
    if (digit < '0' or digit > '2') return null;
    if (!std.mem.eql(u8, name[name.len - suffix.len - 1 .. name.len - 1], suffix)) return null;
    return .{ .base = name[0 .. name.len - suffix.len - 1], .plane = digit - '0' };
}

/// One weight to persist under a GGUF tensor name. For fused in-memory
/// weights, several entries share one `weight` with disjoint `[row0,
/// row0+rows)` windows over its out dim — one entry per source tensor.
/// `rows == null` means the whole out dim. Entries whose weight is not the
/// `.ptqtp` arm are ignored (their base tensor passes through verbatim), so
/// a family walk may list every projection unconditionally.
pub const SaveEntry = struct {
    name: []const u8,
    weight: *const LinearWeight,
    row0: usize = 0,
    rows: ?usize = null,
};

pub const SaveOptions = struct {
    /// Raw source-file bytes for the byte-verbatim metadata copy when the
    /// `File`'s mapping was transferred away (`gguf.File.takeMapping` —
    /// MoE loads move it into the model); null uses `src.bytes`.
    header_bytes: ?[]const u8 = null,
};

pub const SaveReport = struct {
    /// Source-name tensors replaced (or appended) as plane sets.
    decorated: usize = 0,
    /// Total plane tensors written.
    planes: usize = 0,
    /// Tensors copied through byte-verbatim.
    passthrough: usize = 0,
    /// Decorated entries whose base tensor was absent in the source (e.g.
    /// a decorated head on a tied-embedding model), appended at the end.
    appended: usize = 0,
};

/// Fill `writer` with the persisted model: source metadata byte-verbatim
/// plus the version key, each decorated entry's planes at its base tensor's
/// position, everything else copied through. Re-saving a model loaded FROM
/// a decorated file emits the (byte-identical) planes at the positions of
/// the source plane tensors, so save→load→save is byte-stable.
///
/// The writer BORROWS plane bytes and source tensor data until `finish`
/// returns: `src` and every entry's weight must outlive the final write.
pub fn build(allocator: Allocator, src: *const gguf.File, entries: []const SaveEntry, options: SaveOptions, writer: *gguf.Writer) !SaveReport {
    var decorated = std.StringHashMap(usize).init(allocator);
    defer decorated.deinit();
    for (entries, 0..) |*entry, entry_i| {
        switch (entry.weight.*) {
            .ptqtp => {},
            else => continue,
        }
        const slot = try decorated.getOrPut(entry.name);
        if (slot.found_existing) return Error.DuplicateSaveEntry;
        slot.value_ptr.* = entry_i;
    }
    const emitted = try allocator.alloc(bool, entries.len);
    defer allocator.free(emitted);
    @memset(emitted, false);

    try writer.copyAllMetadataRaw(options.header_bytes orelse src.bytes, &.{});
    // Stamped only when planes are actually written: a save with nothing
    // decorated is a pure re-emit, not a file claiming the PTQTP format
    // (a claim a future format_version bump would refuse to load).
    if (decorated.count() != 0) {
        try writer.addMetaInt(version_key, u32, format_version);
    }

    var report = SaveReport{};
    for (src.tensors) |*info| {
        if (decorated.get(info.name)) |entry_i| {
            report.planes += try addPlanes(writer, &entries[entry_i], info);
            report.decorated += 1;
            emitted[entry_i] = true;
            continue;
        }
        if (parsePlaneName(info.name)) |parsed| {
            if (decorated.get(parsed.base)) |entry_i| {
                // A source plane set being re-saved from the live weight:
                // emit all planes at the `.ptqtp0` position, drop the rest.
                if (parsed.plane == 0) {
                    report.planes += try addPlanes(writer, &entries[entry_i], null);
                    report.decorated += 1;
                    emitted[entry_i] = true;
                }
                continue;
            }
        }
        try writer.addTensor(info.name, info.ggml_type, info.dims[0..info.n_dims], info.data);
        report.passthrough += 1;
    }

    for (entries, 0..) |*entry, entry_i| {
        switch (entry.weight.*) {
            .ptqtp => {},
            else => continue,
        }
        if (emitted[entry_i]) continue;
        report.planes += try addPlanes(writer, entry, null);
        report.decorated += 1;
        report.appended += 1;
    }
    return report;
}

/// `build` + write to `out_path`. Same lifetime rule: `src` and the entry
/// weights stay alive until this returns. The write is temp-file + rename
/// (`createFileAtomic`, the training_checkpoint pattern): `finish` streams
/// passthrough bytes it borrows from the source mmap, so the destination
/// must not be truncated until every byte is out — saving over the source
/// path replaces it safely instead of truncating the mapping mid-read.
pub fn saveFile(allocator: Allocator, io: std.Io, src: *const gguf.File, entries: []const SaveEntry, options: SaveOptions, out_path: []const u8) !SaveReport {
    var writer = gguf.Writer.init(allocator);
    defer writer.deinit();
    const report = try build(allocator, src, entries, options, &writer);

    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, out_path, .{ .replace = true });
    defer atomic.deinit(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var out_writer = atomic.file.writer(io, &write_buffer);
    try writer.finish(&out_writer.interface);
    try out_writer.interface.flush();
    try atomic.replace(io);
    return report;
}

fn addPlanes(writer: *gguf.Writer, entry: *const SaveEntry, base_info: ?*const gguf.TensorInfo) !usize {
    const weight = &entry.weight.ptqtp;
    const in_dim = weight.p1.dim(.in);
    const out_dim = weight.p1.dim(.out);
    const rows = entry.rows orelse out_dim;
    if (rows == 0 or entry.row0 + rows > out_dim) return Error.InvalidRowRange;
    if (base_info) |info| {
        const shape = try info.logicalMatrixShape();
        if (shape[0] != rows or shape[1] != in_dim) return Error.PlaneShapeMismatch;
    }

    var planes: [3]?*const weights.QuantWeight(.tq2_0) = .{ &weight.p1, null, null };
    if (weight.p2) |*plane| planes[1] = plane;
    if (weight.p3) |*plane| planes[2] = plane;

    // The ptqtp arm's contract dim is 256-aligned, blocks are row-major.
    const blocks_per_row = in_dim / fucina.ptqtp.block_len;
    var name_buf: [max_name_len]u8 = undefined;
    var written: usize = 0;
    for (planes, 0..) |maybe_plane, plane_i| {
        const plane = maybe_plane orelse break;
        const blocks = try plane.dataConst();
        const slice = blocks[entry.row0 * blocks_per_row ..][0 .. rows * blocks_per_row];
        const name = try planeName(&name_buf, entry.name, plane_i);
        try writer.addTensor(name, .tq2_0, &.{ in_dim, rows }, std.mem.sliceAsBytes(slice));
        written += 1;
    }
    return written;
}

/// Loader pair-detection: when `file` is decorated and carries
/// `<base_name>.ptqtp0`, load the plane set as a ready `.ptqtp` weight;
/// null otherwise (caller falls back to the base tensor — which is how
/// skip-layer tensors inside a decorated file load). Plane payloads are
/// copied, so the weight outlives the file.
pub fn maybeLoadPlanes(ctx: *ExecContext, file: *const gguf.File, base_name: []const u8, expected_rows: usize, expected_cols: usize) !?LinearWeight {
    const stored = file.getInt(version_key) orelse return null;
    if (stored != format_version) return Error.UnsupportedPtqtpVersion;

    var name_buf: [max_name_len]u8 = undefined;
    const info0 = file.maybeGet(try planeName(&name_buf, base_name, 0)) orelse return null;

    var p1 = try loadPlane(ctx, info0, expected_rows, expected_cols);
    errdefer p1.deinit();
    var p2: ?weights.QuantWeight(.tq2_0) = null;
    errdefer if (p2) |*plane| plane.deinit();
    if (file.maybeGet(try planeName(&name_buf, base_name, 1))) |info| {
        p2 = try loadPlane(ctx, info, expected_rows, expected_cols);
    }
    var p3: ?weights.QuantWeight(.tq2_0) = null;
    if (file.maybeGet(try planeName(&name_buf, base_name, 2))) |info| {
        if (p2 == null) return Error.InvalidPlaneSet;
        p3 = try loadPlane(ctx, info, expected_rows, expected_cols);
    }
    return .{ .ptqtp = .{ .p1 = p1, .p2 = p2, .p3 = p3 } };
}

fn loadPlane(ctx: *ExecContext, info: *const gguf.TensorInfo, expected_rows: usize, expected_cols: usize) !weights.QuantWeight(.tq2_0) {
    if (info.ggml_type != .tq2_0) return Error.PlaneTypeMismatch;
    const loaded = try LinearWeight.load(ctx, info, expected_rows, expected_cols);
    return switch (loaded) {
        .tq2_0 => |plane| plane,
        else => unreachable, // load() maps .tq2_0 infos to the .tq2_0 arm.
    };
}

test {
    _ = @import("ptqtp_gguf_tests.zig");
}
