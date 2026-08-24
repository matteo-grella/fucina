//! GGUF container I/O, public as `fucina.gguf`: `File` parses/mmaps a GGUF
//! (metadata, tensor directory, split files, madvise prefetch/release),
//! `Writer` emits one (the export/transcode path), and the codec layer
//! (`encodeF32`/`decodeF32`, `tensorByteLen`, `GgmlType` <-> `DType`
//! mapping) round-trips tensor payloads. `RowTable` is the mmap-backed
//! row-lookup view quantized embedding tables serve from.
//!
//! Layout: this file is the facade; the bodies live in `gguf/` (`wire`
//! types + byte lengths, `codec` encode/decode, `reader` File, `writer`
//! Writer).

const codec = @import("gguf/codec.zig");
const reader = @import("gguf/reader.zig");
const wire = @import("gguf/wire.zig");
const writer = @import("gguf/writer.zig");

pub const prefetch = reader.prefetch;
pub const release = reader.release;
pub const Error = wire.Error;
pub const GgmlType = wire.GgmlType;
pub const dtypeForGgmlType = wire.dtypeForGgmlType;
pub const Value = wire.Value;
pub const Array = wire.Array;
pub const TensorInfo = wire.TensorInfo;
pub const File = reader.File;
pub const tensorByteLen = wire.tensorByteLen;
pub const MetaType = wire.MetaType;
pub const Writer = writer.Writer;
pub const encodeF32 = codec.encodeF32;
pub const decodeF32 = codec.decodeF32;
pub const decodeAllocF32 = codec.decodeAllocF32;
pub const RowTable = codec.RowTable;

test {
    _ = codec;
    _ = reader;
    _ = wire;
    _ = writer;
    _ = @import("gguf_tests.zig");
}
