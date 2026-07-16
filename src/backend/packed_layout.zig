//! Layout discriminants shared by dense and quantized load-time RHS packs.
//!
//! This leaf exists so `packed.zig` and `quant/types.zig` can attach the same
//! comptime layout type to their containers without importing each other.

pub const PackedRhsLayout = enum {
    /// f32 output-row panels, with the logical RHS stored as [n, k]. f16 and
    /// bf16 sources are widened exactly once while the panel is built.
    dense_f32,
    q8_0x4,
    q6_kx4,
    q4_kx4,
    q4_kx8,
    q4_kx2mmla,
    q5_kx8,
};
