//! Streamed-expert geometry: the streamable quant formats
//! (`StreamedQuant`), the projection identity (`Proj`), the registration
//! spec (`ProjSpec`), and the derived per-projection layout math
//! (`ProjGeometry`). Pure data and arithmetic -- no file descriptors, no
//! store state.
const std = @import("std");
const dtype_mod = @import("../dtype.zig");
const backend_mod = @import("../backend.zig");
const io = @import("io.zig");

const DType = dtype_mod.DType;
const qm = backend_mod.quant;
const Error = io.Error;

/// Quantized formats an expert stack may stream in: the K-quant family
/// every real MoE GGUF uses (matching `MoeRhs`'s resident arms). Every
/// member but `tq2_0_fx4` is spelled exactly like its `DType` tag and
/// derives its block geometry from `dtype.block_formats`; this enum is the
/// one place the streamable subset is listed (`fromDType`, `streamable`).
pub const StreamedQuant = enum {
    q4_k,
    q5_k,
    q6_k,
    q8_0,
    tq2_0,
    q2_k,
    iq2_xxs,
    iq3_xxs,
    iq2_s,
    iq4_xs,
    q3_k,
    mxfp4,
    /// Tie-fitted K=2 PTQTP pre-folded on disk (`gguf.GgmlType.tq2_0_fx4`):
    /// one contiguous pack per expert projection (single pread per miss,
    /// slab bytes == file bytes, L2-stripeable), served by the one-pass
    /// folded kernel. Resident MoE and dense loaders carry the same format
    /// (docs/PTQTP.md, "Native folded expert format"). Not a `DType`: its
    /// 520-byte block spans four columns, outside the one-row block
    /// contract of `dtype.block_formats`.
    tq2_0_fx4,

    comptime {
        for (@typeInfo(StreamedQuant).@"enum".fields) |field| {
            if (std.mem.eql(u8, field.name, "tq2_0_fx4")) continue;
            if (!@hasField(DType, field.name))
                @compileError("StreamedQuant." ++ field.name ++ " must be spelled like its DType tag");
        }
    }

    /// The streamed format for a block dtype, or null when expert stacks
    /// of that dtype are not streamable.
    pub fn fromDType(dt: DType) ?StreamedQuant {
        switch (dt) {
            inline else => |tag| {
                if (comptime @hasField(StreamedQuant, @tagName(tag))) return @field(StreamedQuant, @tagName(tag));
                return null;
            },
        }
    }

    pub fn streamable(dt: DType) bool {
        return fromDType(dt) != null;
    }

    /// The `DType` this format stores; null for the folded pack.
    pub fn dtype(self: StreamedQuant) ?DType {
        return switch (self) {
            .tq2_0_fx4 => null,
            inline else => |tag| @field(DType, @tagName(tag)),
        };
    }

    /// Bytes per weight block as the slab accounts for them.
    pub fn blockSize(self: StreamedQuant) usize {
        return switch (self) {
            // Amortized per-column bytes: the physical 520-byte
            // `BlockTQ2_0Foldedx4` spans FOUR columns' 256-element blocks,
            // so per column it costs 130; geometry math (rows x bpc x
            // blockSize) then yields exact byte counts (out_dim % 4 == 0
            // enforced in ProjGeometry.init).
            .tq2_0_fx4 => @sizeOf(qm.types.BlockTQ2_0Foldedx4) / 4,
            inline else => |tag| dtype_mod.blockByteSize(@field(DType, @tagName(tag))),
        };
    }

    /// Weight blocks per row for a row of `in_dim` inputs.
    pub fn blocksPerColumn(self: StreamedQuant, in_dim: usize) Error!usize {
        const block_len: usize = switch (self) {
            // The folded pack keeps TQ2_0's 256-element block along k.
            .tq2_0_fx4 => dtype_mod.blockSize(.tq2_0),
            inline else => |tag| dtype_mod.blockSize(@field(DType, @tagName(tag))),
        };
        if (in_dim == 0 or in_dim % block_len != 0) return Error.InvalidExpertGeometry;
        return in_dim / block_len;
    }
};

/// The three projections of one MoE FFN layer, in slab order.
pub const Proj = enum(u2) { gate = 0, up = 1, down = 2 };

/// One projection's stacked expert tensor as it sits in the GGUF file:
/// expert-major contiguous, so expert `e` occupies the byte range
/// `[file_offset + e*expert_bytes, +expert_bytes)`. PTQTP expert stacks
/// (docs/PTQTP.md) are `plane_count` such tensors — the `<name>.ptqtpK`
/// siblings, each plane-major on disk exactly like the dense plane
/// convention; the store gathers one expert's K plane row-blocks into a
/// contiguous slab section (expert-major planes in RAM), staying pure byte
/// plumbing either way.
pub const ProjSpec = struct {
    quant: StreamedQuant,
    /// Which split part (file) holds the tensor; 0 for single-file GGUFs.
    part: u16 = 0,
    /// Absolute offset of the tensor's data within its part on disk
    /// (`gguf.File.partDataOffset(part) + TensorInfo.offset`).
    file_offset: u64,
    /// One plane tensor's bytes — all planes are identically sized
    /// (validated against `n_expert * plane_bytes` per plane).
    byte_len: usize,
    in_dim: usize,
    out_dim: usize,
    /// Trit-plane count: 1 for every ordinary single-tensor projection;
    /// 2 or 3 for PTQTP stacks (tq2_0 only), whose plane 0 sits at
    /// `file_offset` and planes 1..2 at `plane_offsets`.
    plane_count: u8 = 1,
    /// Absolute on-disk offsets of planes 1 and 2 within `part`; read only
    /// when `plane_count` exceeds 1.
    plane_offsets: [2]u64 = .{ 0, 0 },
    /// Tie-fitted K=2 PTQTP only (docs/PTQTP.md): the fill folds the two
    /// plane row-blocks into the 4-bit column-interleaved pack, so the slab
    /// section serves the one-pass folded kernel instead of two plane
    /// passes. Disk layout is unchanged (still two plane reads per expert);
    /// the fold happens in memory on the way into the slab. Requires
    /// `plane_count == 2`, `quant == .tq2_0`, and `out_dim % 4 == 0`.
    fold: bool = false,
};

pub const ProjGeometry = struct {
    quant: StreamedQuant,
    part: u16,
    /// Per-plane tensor base offsets; entries past `plane_count` unused.
    plane_offsets: [3]u64,
    in_dim: usize,
    out_dim: usize,
    blocks_per_column: usize,
    plane_count: usize,
    /// One plane's bytes for one expert.
    plane_bytes: usize,
    /// One expert's slab section: `plane_count * plane_bytes`.
    expert_bytes: usize,
    /// Fill-time fold into the 4-bit pack (ProjSpec.fold): the folded
    /// blocks fit inside the same section (520 vs 528 bytes per 4-column
    /// group), so slab geometry is unchanged.
    fold: bool,

    pub fn init(spec: ProjSpec, n_expert: usize) Error!ProjGeometry {
        if (spec.plane_count == 0 or spec.plane_count > 3) return Error.InvalidExpertGeometry;
        // Multi-plane is the PTQTP tq2_0 container only — the MoE dispatch
        // interprets a >1 plane slab as summed ternary planes.
        if (spec.plane_count > 1 and spec.quant != .tq2_0) return Error.InvalidExpertGeometry;
        if (spec.fold and (spec.plane_count != 2 or spec.quant != .tq2_0 or spec.out_dim % 4 != 0))
            return Error.InvalidExpertGeometry;
        // The native pre-folded pack: one plane by definition (the fold is
        // on disk), never fill-folded, 4-column blocks need out_dim % 4.
        if (spec.quant == .tq2_0_fx4 and (spec.plane_count != 1 or spec.fold or spec.out_dim % 4 != 0))
            return Error.InvalidExpertGeometry;
        const bpc = try spec.quant.blocksPerColumn(spec.in_dim);
        const row_bytes = std.math.mul(usize, bpc, spec.quant.blockSize()) catch return Error.InvalidExpertGeometry;
        const plane_bytes = std.math.mul(usize, spec.out_dim, row_bytes) catch return Error.InvalidExpertGeometry;
        const expert_bytes = std.math.mul(usize, plane_bytes, spec.plane_count) catch return Error.InvalidExpertGeometry;
        const total = std.math.mul(usize, plane_bytes, n_expert) catch return Error.InvalidExpertGeometry;
        if (total != spec.byte_len or plane_bytes == 0) return Error.InvalidExpertGeometry;
        return .{
            .quant = spec.quant,
            .part = spec.part,
            .plane_offsets = .{ spec.file_offset, spec.plane_offsets[0], spec.plane_offsets[1] },
            .in_dim = spec.in_dim,
            .out_dim = spec.out_dim,
            .blocks_per_column = bpc,
            .plane_count = spec.plane_count,
            .plane_bytes = plane_bytes,
            .expert_bytes = expert_bytes,
            .fold = spec.fold,
        };
    }

    pub fn planeFileOffset(self: *const ProjGeometry, eid: usize, plane: usize) u64 {
        return self.plane_offsets[plane] + @as(u64, eid) * self.plane_bytes;
    }
};
