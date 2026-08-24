# 12. Model I/O: GGUF and safetensors

Fucina speaks two interchange formats and two native sidecar formats:

| Format | Module (facade export) | Role |
|---|---|---|
| GGUF v2/v3 | `fucina.gguf` (`src/gguf.zig`) | llama.cpp-ecosystem model interop: read quantized weights, re-emit/transcode/export |
| safetensors | `fucina.safetensors` (`src/safetensors.zig`) | Hugging Face tensor container: neutral named-tensor payloads |
| state-dict stream | `fucina.state_dict` (`src/state_dict.zig`) | named, dtype-aware checkpoint entries; wire format IS safetensors |
| checkpoint directory | `fucina.training_checkpoint` (`src/training_checkpoint.zig`) | resumable-training layout: safetensors payloads + native `optimizer.fucina` frames + JSON sentinel |

One floor above the container parsers, the executable-weight band is also
core: `fucina.weights` (GGUF tensor → typed quantized weight containers,
MoE streaming glue), `fucina.ptqtp_gguf` (PTQTP plane sidecars), and
`fucina.gguf_meta` (metadata readers, parallel layer loading) — documented
with their main consumers in [§13.2](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig)-13.3.

All integers in both binary formats are little-endian. None of the types in
this section have internal locking: a parsed `File` is safe to read from many
threads once construction returns, but `deinit`, `takeMapping`, and every
writer mutation require external serialization.

## 12.1 GGUF reader (`src/gguf.zig`)

### 12.1.1 Opening and lifetime

```zig
pub const File = struct {
    allocator: Allocator,
    bytes: []u8,                          // entire file (heap copy or mmap)
    tensors: []TensorInfo,
    index: std.StringHashMap(usize),      // tensor name -> tensors[i]
    metadata: std.StringHashMap(Value),
    alignment: usize,                     // data-section alignment (default 32)
    data_offset: usize,                   // file offset of the tensor data section
    is_mmap: bool = false,
    extra_bytes: [][]u8 = &.{},           // mappings of split parts 2..N (part 1 = bytes)
    part_data_offsets: []u64 = &.{},      // each part's data-section offset

    pub fn load(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn loadMmap(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn loadMmapAuto(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn splitPartPaths(allocator: Allocator, path: []const u8) !?[][]u8
    pub fn parseOwned(allocator: Allocator, bytes: []u8) !File
    pub fn deinit(self: *File) void
    pub fn takeMapping(self: *File) ?MappedRegion
    pub fn isSplit(self: *const File) bool
    pub fn partDataOffset(self: *const File, part: u16) u64
};
```

Four constructors, one parse core:

- `File.load` reads the whole file into a heap buffer, then parses. Errors:
  `error.IsDir` for non-regular files, `error.EndOfStream` on a short read,
  plus the parse errors below. Prefer for small files (tests, tools).
- `File.loadMmap` maps the file read-only (`PROT_READ`, `MAP_PRIVATE`) and
  parses in place; the fd is closed immediately (POSIX keeps the mapping
  valid). An empty file is `Error.InvalidMagic`. This is the path for
  multi-GB models: pages are file-backed and evictable, and no heap copy
  coexists with the materialized weights.
- `File.loadMmapAuto` is `loadMmap` that transparently follows llama.cpp
  split GGUFs: when the path names the first `-00001-of-0000N` part
  (`splitPartPaths` enumerates all part paths; null for any other path),
  every part is mapped and parsed into one merged `File` — part 1's
  metadata (splits carry the full metadata there), the union of all parts'
  tensors (each tagged with its `TensorInfo.part`), one index over all of
  them. The mappings of parts 2..N live in `extra_bytes`; `isSplit`
  reports a split load, and `partDataOffset(part)` is each part's
  data-section offset within its own file. For a non-split path it behaves
  exactly like `loadMmap`.
- `File.parseOwned` takes ownership of caller-provided bytes; on parse
  failure the bytes are freed (exactly once — callers must not also free).

Parsing accepts GGUF versions 2 and 3 (`Error.UnsupportedVersion` otherwise),
magic `"GGUF"` (`Error.InvalidMagic` otherwise). Header counts larger than
the file length are rejected before any allocation (`Error.InvalidTensorInfo`).
A tensor whose declared extent runs past EOF logs a self-diagnosing message
(name, shortfall in bytes/GB — the signature of a truncated download) and
returns `Error.InvalidTensorInfo`.

**Ownership.** Everything a `File` hands out — metadata strings, array
payloads, `TensorInfo.name`, `TensorInfo.data` — is a zero-copy slice into
`File.bytes`. All of it dies at `deinit`, which frees the hash maps and the
`tensors` slice, then frees (heap) or `munmap`s (mmap) the bytes.

```zig
// nested inside File — the symbol is gguf.File.MappedRegion
pub const MappedRegion = struct {
    bytes: []const u8,
    pub fn deinit(self: *MappedRegion) void   // munmap
};
```

`takeMapping` transfers ownership of the underlying mmap to the caller and
returns `null` when the file was heap-read or split-loaded (a split's
tensors point into all part mappings, but a `MappedRegion` can carry only
one). After a successful `takeMapping`,
`File.deinit` no longer unmaps; every previously parsed slice (metadata,
`TensorInfo.data`) stays valid for as long as the returned `MappedRegion`
lives. This is how a model borrows quantized weight blocks straight from the
mapping instead of copying them (the `fucina_models` loaders do this for large
expert tensors — [§13](13-the-model-stack-fucina_models.md)). The absolute addresses of tensor payloads are aligned
(page-aligned mapping base plus `alignment`-multiple offsets), so borrowed
block slices satisfy the block-struct alignment that [§10](10-quantization.md)'s kernels assume.

```zig
pub fn prefetch(data: []const u8) void
```

`prefetch` issues `madvise(WILLNEED)` over a mapped region about to be read
in full, letting OS readahead run ahead of a sequential copy/pack loop — the
dominant cold-load cost. It is a silent no-op on heap buffers and wherever
the advice call fails. Deliberately not called for borrowed (zero-copy)
blocks, which stay lazily paged.

```zig
pub fn release(data: []const u8) void
```

`release` is the counterpart hint for a mapped region the caller is DONE
reading (the tensor-at-a-time streaming paths): it issues
`madvise(DONTNEED)` so the pages drop from residency immediately instead
of waiting for memory pressure. Only meaningful on read-only file-backed
mappings (`loadMmap*`) — the pages are clean, so a later touch simply
refaults from the file. Best-effort like `prefetch`, and like it the
range is rounded to page boundaries (shared first/last pages refault
harmlessly).

### 12.1.2 Metadata

```zig
pub const Value = union(enum) {
    int: i64, float: f64, boolean: bool,
    string: []const u8, array: Array,
};
pub const Array = struct {
    item_type: u32,        // wire value-type code of the elements
    len: usize,
    data: []const u8,      // raw bytes spanning all elements
    pub fn stringSlices(self: Array, allocator: Allocator) ![][]const u8
};
```

The parser widens scalars: every integer wire type (u8/i8/u16/i16/u32/i32/
u64/i64) becomes `i64`, every float becomes `f64`. A wire `uint64 >= 2^63`
cannot be represented and fails the whole parse with
`Error.MetadataValueOutOfRange`. Strings and arrays are zero-copy. Wire
value-type codes (ggml's `GGUF_TYPE_*`, also the writer-side `MetaType`
enum): 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string,
9 array, 10 u64, 11 i64, 12 f64. Nested arrays (array-of-array) are
`Error.UnsupportedValueType`.

`Array.stringSlices` decodes a string array (item_type 8, otherwise
`Error.UnsupportedValueType`) into a caller-freed outer slice; the inner
strings still borrow the file bytes.

Typed lookups (all `null` when the key is absent or the wrong kind):

```zig
pub fn meta(self: *const File, key: []const u8) ?Value
pub fn getString(self: *const File, key: []const u8) ?[]const u8
pub fn getInt(self: *const File, key: []const u8) ?i64
pub fn getFloat(self: *const File, key: []const u8) ?f64   // also widens int
pub fn getBool(self: *const File, key: []const u8) ?bool   // also accepts int != 0
pub fn getArray(self: *const File, key: []const u8) ?Array
```

`general.alignment` is special-cased: it is validated straight from the wire
value, before the lossy widening — a non-integer, negative, zero,
non-power-of-two, or out-of-range (`>= 2^63` or `> 2^20`) alignment returns
`Error.InvalidAlignment` instead of reaching undefined behavior at a cast or
`alignForward`. The validated value replaces the default alignment of 32.

### 12.1.3 Tensor directory

```zig
pub const TensorInfo = struct {
    name: []const u8,
    dims: [4]usize,        // GGUF ne[] order: innermost/fastest-varying FIRST
    n_dims: usize,         // 1..4
    ggml_type: GgmlType,
    offset: usize,         // relative to the data section (data_offset)
    data: []const u8,      // exact wire bytes, borrowed from File.bytes
    part: u16 = 0,         // split part holding this tensor (0 = single-file)

    pub fn dim(self: TensorInfo, index: usize) !usize            // InvalidTensorInfo past n_dims
    pub fn logicalMatrixShape(self: TensorInfo) ![2]usize        // 2-D only: {dims[1], dims[0]}
};

pub fn get(self: *const File, name: []const u8) !*const TensorInfo   // Error.TensorNotFound
pub fn maybeGet(self: *const File, name: []const u8) ?*const TensorInfo
```

`dims` follow ggml's `ne[]` convention — the innermost axis first — so a
Fucina row-major logical `[out, in]` matrix appears as `dims = { in, out }`.
`logicalMatrixShape` performs that swap for rank-2 tensors and errors with
`Error.InvalidTensorInfo` for any other rank.

```zig
pub const GgmlType = enum(u32) { f32 = 0, f16 = 1, q4_0 = 2, ... };
pub fn dtypeForGgmlType(value: GgmlType) ?DType
pub fn tensorByteLen(ggml_type: GgmlType, dims: []const usize) !usize
```

`GgmlType` carries the ggml wire codes for: `f32`, `f16`, `bf16`, `f64`,
`i8`, `i16`, `i32`, `i64`, the block quants `q1_0 q2_0 q4_0 q4_1 q5_0 q5_1 q8_0
q8_1 q2_k q3_k q4_k q5_k q6_k q8_k`, the i-quants `iq1_s iq1_m iq2_xxs
iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs`, the ternaries `tq1_0 tq2_0`, and
the microscaling floats `mxfp4 nvfp4`. An unknown wire code fails parsing
with `Error.UnsupportedGgmlType`. `dtypeForGgmlType` maps every quantized
and float format to the corresponding core `DType` ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)); only the integer
scalars (`i8 i16 i32 i64`) and `f64` return `null` (they have no core dtype
— their bytes are still readable via `TensorInfo.data`).

`tensorByteLen` computes exact wire size: scalar formats multiply the element
count by the element size; block formats require the innermost dim to be a
whole number of blocks (`dims[0] % blockSize == 0`, like `ggml_row_size` —
`Error.InvalidTensorInfo` otherwise) and multiply block count by
`blockByteSize` ([§10](10-quantization.md)). A zero-length dimension is a legitimate empty ggml
tensor and yields 0 bytes.

**Layout rules.** The tensor data section starts at
`data_offset = alignForward(header_end, alignment)`; each `TensorInfo.offset`
is relative to that. The parser bounds-checks `data_offset + offset +
tensorByteLen` against the file and slices `data` accordingly.

Reading metadata and the directory of a real model file:

```zig
fn snippetInspectGguf(alloc: std.mem.Allocator, io: std.Io) !void {
    var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3-0.6B-Q4_K_S.gguf");
    defer file.deinit(); // unmaps; every borrowed slice dies here

    const arch = file.getString("general.architecture").?;
    const n_layers = file.getInt("qwen3.block_count").?;
    const tokens = file.getArray("tokenizer.ggml.tokens").?;
    const vocab = try tokens.stringSlices(alloc); // slices point into the mapping
    defer alloc.free(vocab);

    const embd = try file.get("token_embd.weight");
    fucina.gguf.prefetch(embd.data); // OS readahead before a sequential copy
    const dt = fucina.gguf.dtypeForGgmlType(embd.ggml_type); // core DType (§8)
    _ = .{ arch, n_layers, dt };
} // requires model assets to run
```

## 12.2 GGUF writer (`src/gguf.zig`)

```zig
pub const Writer = struct {
    pub fn init(allocator: Allocator) Writer
    pub fn deinit(self: *Writer) void

    pub fn addMetaString(self: *Writer, key: []const u8, value: []const u8) !void
    pub fn addMetaInt(self: *Writer, key: []const u8, comptime Int: type, value: Int) !void
    pub fn addMetaFloat(self: *Writer, key: []const u8, comptime Float: type, value: Float) !void
    pub fn addMetaBool(self: *Writer, key: []const u8, value: bool) !void
    pub fn addMetaArray(self: *Writer, key: []const u8, comptime Elem: type, values: []const Elem) !void
    pub fn addMetaStringArray(self: *Writer, key: []const u8, values: []const []const u8) !void
    pub fn addMetaCopy(self: *Writer, from: *const File, key: []const u8) !void
    pub fn copyAllMetadata(self: *Writer, from: *const File, skip_keys: []const []const u8) !void
    pub fn copyAllMetadataRaw(self: *Writer, file_bytes: []const u8, skip_keys: []const []const u8) !void

    pub fn addTensor(self: *Writer, name: []const u8, ggml_type: GgmlType,
                     dims: []const usize, data: []const u8) !void
    pub fn declareTensor(self: *Writer, name: []const u8, ggml_type: GgmlType,
                         dims: []const usize) !void
    pub fn finish(self: *const Writer, out: *std.Io.Writer) !void
    pub fn beginStream(self: *const Writer, out: *std.Io.Writer) !DataStreamer

    pub const DataStreamer = struct {
        pub fn nextTensorName(self: *const DataStreamer) ?[]const u8
        pub fn writeTensorData(self: *DataStreamer, data: []const u8) !void
        pub fn finish(self: *const DataStreamer) !void
    };
};
pub const MetaType = enum(u32) { uint8 = 0, int8, uint16, int16, uint32, int32,
                                 float32, boolean, string, array, uint64, int64, float64 };
```

The writer buffers metadata KVs and tensor declarations, then `finish`
serializes everything in one pass: `"GGUF"` magic, version 3, tensor count,
KV count, the KV section, the tensor-info section, zero padding to
`alignment`, then each tensor's data padded to `alignment` — including the
last tensor, matching ggml's writer byte-for-byte.

**llama.cpp-exact offsets.** Tensor offsets are precomputed as the running
padded total, relative to the data-section start — llama.cpp's reader
rejects files whose offsets are not exactly that value, so nothing here is
optional. Re-parsing a `finish` output and re-emitting it reproduces the
file byte-identically, and a real-model re-emit preserves every KV and
tensor payload verbatim (both asserted in `src/gguf_tests.zig`).

**Ownership.** Metadata keys/payloads and tensor names are duplicated into
the writer (`deinit` frees them). Tensor `data` is **borrowed** and must stay
alive until `finish` returns. `finish` is `*const` and repeatable.

**Streaming serialization.** For outputs too large to hold every tensor
buffer at once (the `export-gguf --ptqtp` path), `declareTensor` records
name/type/dims without bytes (same validation; offsets come from
`tensorByteLen`), and `beginStream` writes the complete header immediately,
returning a `DataStreamer` that feeds each tensor's bytes **in declaration
order** (`writeTensorData` writes data + padding straight through, so each
buffer can be freed before producing the next; wrong length is
`Error.InvalidTensorInfo`, past-the-end is `Error.TensorDataMissing`).
`DataStreamer.finish` errors with `Error.TensorDataMissing` unless every
declared tensor was streamed, as does `Writer.finish` if it meets a
data-less declaration. Tensors added *with* data stream too — pass their
bytes at their turn. The streamed output is byte-identical to the `finish`
path (pinned in `src/gguf_tests.zig`).

**Metadata semantics.**

- `addMetaInt`/`addMetaFloat`/`addMetaArray` select the exact wire type from
  the comptime scalar type (u8/i8/u16/i16/u32/i32/u64/i64, f32/f64; anything
  else is a compile error). This matters: llama.cpp type-checks many keys, so
  passthrough-adjacent metadata must keep its original width.
- Re-adding an existing key replaces its value **in place** — file order is
  preserved and GGUF keys stay unique.
- `addMetaCopy` copies one KV **byte-verbatim** from a parsed `File`,
  preserving the exact wire type that the parser's widened `Value` map drops
  (it re-reads the raw file bytes). Absent key: `Error.KeyNotFound`.
  `copyAllMetadata` does the same for every KV except `skip_keys`, in the
  source file's order. Both require `from` to still own its bytes — call
  them before `from.deinit()`/`takeMapping()`. `copyAllMetadataRaw` is
  `copyAllMetadata` over a raw GGUF byte region, for callers whose `File`
  transferred its mapping away via `takeMapping` while the region is still
  alive.
- `general.alignment` is tracked no matter how it is added: it must be wire
  type uint32, a power of two, and `<= 2^20`, else `Error.InvalidAlignment`;
  it changes the padding rule used by `finish` (mirroring the parser).

**Tensor declaration.** `addTensor` dims are ne-order (innermost first),
exactly as the parser surfaces them — re-emitting a parsed tensor is
`addTensor(info.name, info.ggml_type, info.dims[0..info.n_dims], info.data)`.
Validation: non-empty name, 1–4 dims, `data.len == tensorByteLen(...)`
(`Error.InvalidTensorInfo` otherwise), unique name
(`Error.DuplicateTensorName`).

Write-then-read round-trip:

```zig
test "gguf: write, reopen, verify" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;

    var w = fucina.gguf.Writer.init(alloc);
    defer w.deinit();
    try w.addMetaString("general.name", "demo");
    try w.addMetaInt("demo.heads", u32, 8);

    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var wire: [24]u8 = undefined;
    try fucina.gguf.encodeF32(.f32, &values, &wire);
    try w.addTensor("w", .f32, &.{ 3, 2 }, &wire); // ne order: logical [2, 3]

    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "doc_demo_{d}.gguf", .{std.Io.Clock.real.now(io).nanoseconds});
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};
    {
        var out = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer out.close(io);
        var buf: [4096]u8 = undefined;
        var writer = out.writer(io, &buf);
        try w.finish(&writer.interface);
        try writer.interface.flush();
    }

    var file = try fucina.gguf.File.load(alloc, io, path);
    defer file.deinit();
    try std.testing.expectEqualStrings("demo", file.getString("general.name").?);
    try std.testing.expectEqual(@as(i64, 8), file.getInt("demo.heads").?);
    const info = try file.get("w");
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, &(try info.logicalMatrixShape()));
    try std.testing.expectEqualSlices(u8, &wire, info.data);
}
```

## 12.3 The f32 transcode seam: `encodeF32` / `decodeF32` (`src/gguf.zig`)

```zig
pub fn encodeF32(ggml_type: GgmlType, src: []const f32, dst: []u8) !void
pub fn decodeF32(ggml_type: GgmlType, src: []const u8, dst: []f32) !void
```

`encodeF32` is the writer-side quantize seam: it encodes f32 values as
`ggml_type` wire bytes. `decodeF32` is its exact mirror. Both enforce the
length contract — the byte slice must equal
`tensorByteLen(ggml_type, &.{float_slice.len})`, else
`Error.InvalidTensorInfo`.

| Format group | `encodeF32` | `decodeF32` | Path |
|---|---|---|---|
| `f32`, `f16`, `bf16` | yes | yes | element-wise cast (f16 may overflow to inf on out-of-range values, matching ggml's scalar conversion) |
| `q2_0 q4_0 q4_1 q5_0 q5_1 q8_0 q4_k q5_k q6_k tq2_0` | yes | yes | byte-exact ggml-parity block codecs: `quantizeRowForDType` / `dequantizeRowForDType` ([§10](10-quantization.md)) |
| everything else (`q2_k`, `q3_k`, i-quants, `mxfp4`, ...) | `Error.EncoderUnavailable` | `Error.DecoderUnavailable` | — |

Additional block-format contracts:

- **Finite input only:** the block encoders assume finite input, so
  `encodeF32` rejects any NaN/inf in `src` with `Error.NonFiniteValue`
  (release builds included) — the same seam llama.cpp guards with
  `ggml_validate_row_data`. Scalar casts stay unguarded.
- **Alignment:** for block formats, `dst`/`src` must be aligned to the block
  struct (`@alignOf(Storage(dt))`, [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)), else `Error.InvalidTensorInfo`.
  Little-endian targets only (the same assumption as the parser's zero-copy
  blocks).

```zig
test "gguf: q8_0 quantize seam round-trip" {
    var src: [64]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 3.2;

    // tensorByteLen(.q8_0, &.{64}) = 2 blocks x 34 bytes.
    var wire: [68]u8 align(2) = undefined;
    try fucina.gguf.encodeF32(.q8_0, &src, &wire);

    var back: [64]f32 = undefined;
    try fucina.gguf.decodeF32(.q8_0, &wire, &back);
    for (src, back) |a, b| try std.testing.expectApproxEqAbs(a, b, 0.05);
}
```

## 12.4 The export-gguf tool (`tools/export_gguf.zig`)

`zig build export-gguf` builds and runs the exporter (installed as
`fucina-export-gguf`), which closes the train → export → serve-anywhere
loop on top of `Writer` + `encodeF32`/`decodeF32`:

```sh
# (a) re-emit / transcode
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf Qwen3-0.6B-f16.gguf --out Qwen3-0.6B-Q4_K_S.gguf --dtype q4_k

# (b) merge Fucina LoRA adapters (safetensors, as saved by `zig build finetune`)
#     into dense f32/f16/bf16 base weights and re-emit
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf base-f16.gguf --adapters ckpt-dir --alpha 16 --out merged.gguf

# (c) shard-streaming PTQTP quantization (docs/PTQTP.md): one tensor at a
#     time, so models far bigger than RAM quantize on a small machine
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf big-BF16.gguf --out big-ptqtp3.gguf --ptqtp=3
```

| Flag | Meaning |
|---|---|
| `--from-gguf PATH` | input model (mmap-loaded); required |
| `--out PATH` | output path; required (except `--dry-run`) |
| `--dtype MODE` | global transcode target: `verbatim` (default), `f32`, `f16`, `bf16`, `q8_0`, `q4_k`, `q5_k`, `q6_k`, `tq2_0` |
| `--experts-dtype MODE` | override for tensors named `*_exps.weight` only; may requantize a quantized source |
| `--adapters DIR_OR_FILE` | checkpoint directory containing `adapters.safetensors`, or a safetensors file directly |
| `--alpha F` | LoRA scaling; **required** with `--adapters` (the safetensors checkpoint stores A/B but not alpha; finetune default 16) |
| `--ptqtp[=K]` | mode (c): replace eligible matrices with `K` (1–3, default 2) `<name>.ptqtp0..K-1` TQ2_0 plane tensors, streamed tensor-at-a-time; exclusive with the other modes |
| `--ptqtp-planes K` | plane count as a separate knob (implies `--ptqtp`) |
| `--ptqtp-tie` | scale-tied fit (`ptqtp.Options.tie_scales`, implies `--ptqtp`): plane scales locked to the exact ratio 3 — one uniform 3ᴷ-level quantizer per group — so loaders can fold tied K=2 planes into a single one-pass pack; needs at least 2 planes ([§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)) |
| `--ptqtp-include SUB[,SUB]` | only quantize names containing a substring (replaces the default embeddings/head-stay policy); repeatable |
| `--ptqtp-exclude SUB[,SUB]` | never quantize matching names; repeatable |
| `--dry-run` | (with `--ptqtp`) print the per-tensor plan — name, shape, source dtype → target, bytes before/after — and exit without writing |

All metadata passes through byte-verbatim (`copyAllMetadata`); a non-verbatim
`--dtype` additionally sets `general.file_type` to the matching llama.cpp
`llama_ftype` code (uniform K-quant exports report the `_S` variants).

**Transcode policy** (llama.cpp-convention): only matrix weights transcode —
`n_dims >= 2`, name ends `.weight`, name does not contain `norm` — including
`token_embd`/`output`. Norms and 1-D tensors keep their stored type. Sources
must be f32/f16/bf16: transcoding an already-quantized source would
chain-requantize, so the global `--dtype` errors
(`error.QuantizedSourceUnsupported`); re-emit those verbatim instead.
Block-divisibility rules: a `q8_0` target with `dims[0] % 32 != 0`, or a
K-quant/`tq2_0` target with `dims[0] % 256 != 0`, keeps the **source** dtype
(more conservative than llama-quantize's smaller-quant fallback: no extra
quantization loss, small size cost). A tensor containing NaN/inf is refused
at the `encodeF32` seam with a named diagnostic.

The `--experts-dtype` override (experts-only quantization) IS allowed to
requantize pre-quantized expert tensors (dequant → re-encode through
`decodeF32`/`encodeF32`): shipped MoE GGUFs store experts pre-quantized, and
experts are where shrinking bytes pays most in decode bandwidth at lowest
quality risk. Block divisibility still rules.

**PTQTP policy** (mode c — full treatment in docs/PTQTP.md): eligible = 2D
matrix or 3D `*_exps` expert stack, name ends `.weight`, no `norm`,
`dims[0] % 256 == 0`, source dtype decodable to f32 and not already ternary
(quantized sources ARE
accepted — they dequantize first, the paper-validated
graceful-degradation path). Embeddings and `output.weight` stay in source
precision unless `--ptqtp-include` says otherwise. Expert stacks quantize
per expert slice (`ptqtp_gguf.quantizeMoeStack`): each expert's
`[out x in]` matrix solves independently, and the K plane tensors keep the
base 3D `[in, out, n_expert]` shape, plane-major — the MoE convention the
qwen3 loaders pair-detect. The output carries the `fucina.ptqtp.version`
stamp and the `src/ptqtp_gguf.zig` plane names, so it loads through
the existing pair-detection; a `--ptqtp-tie` run additionally stamps
`fucina.ptqtp.tie_scales` — all-or-nothing by construction, since one
`Options` covers every quantized tensor — which the loaders read to
rebuild the folded one-pass serving form. Split sources load via `loadMmapAuto` (the
single-file output drops `split.*` keys), and the data section is produced
with the streaming writer — the tool holds one source tensor's f32 buffer
plus its planes, never the whole model (expert stacks are the exception:
their K plane stacks stay resident per stack, ~1.7 GiB peak at K=3 on a
4096 x 2048 x 256 stack, reported in the plan and summary), and reports
the peak working set and peak RSS at the end.

**Merge policy**: adapters named `layers.<i>.<q|k|v|o|gate|up|down>.lora_a/b`
merge into the matching `blk.<i>.attn_*/ffn_*.weight` tensors via
`lora.Adapter.mergeInto`/`mergeF16` ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)). Quantized bases error
(`error.QuantizedBaseUnsupported`): merge on an f32/f16/bf16 base, then
`--dtype`-transcode in a second pass. `--adapters` cannot be combined with
`--dtype`/`--experts-dtype` in one run. A `.lora_b` without its `.lora_a`
(or vice versa) is a hard error, as is an empty adapter set.

## 12.5 safetensors (`src/safetensors.zig`)

### 12.5.1 Format and dtypes

Layout (current upstream contract): a u64 little-endian JSON header length,
the UTF-8 JSON header, then one contiguous tensor data buffer. Tensor
`data_offsets` are relative to the start of that buffer and must cover it
exactly, in ascending order, with no holes or overlap. When writing, the
header is padded to an 8-byte multiple with spaces so the first data byte is
naturally aligned for scalar dtypes. `pub const max_header_size` is
100,000,000 bytes, enforced both ways.

```zig
pub const DType = enum {
    BOOL, F4, F6_E2M3, F6_E3M2, U8, I8,
    F8_E5M2, F8_E4M3, F8_E8M0, F8_E4M3FNUZ, F8_E5M2FNUZ,
    I16, U16, F16, BF16, I32, U32, F32, C64, F64, I64, U64,
    pub fn bitsize(self: DType) usize
    pub fn string(self: DType) []const u8
};
pub fn dtypeFromFucina(dtype: fucina.DType) !DType   // f32/f16/bf16/f8_e4m3/f8_e5m2/i64 only
pub fn dtypeToFucina(dtype: DType) !fucina.DType     // F32/F16/BF16/F8_E4M3/F8_E5M2/I64 only
```

Every upstream dtype tag round-trips as raw bytes, including the sub-byte
`F4` (4-bit) and `F6_*` (6-bit) types — for those, the total bit count of a
tensor must land on a byte boundary (`Error.MisalignedSlice` otherwise).
Only `F32`/`F16`/`BF16`/`F8_E4M3`/`F8_E5M2`/`I64` map to core `DType`s
(the FP8 tags to the `.f8_e4m3`/`.f8_e5m2` storage floats, [§8.1](08-data-types-storage-and-the-raw-tensor-layer-internal.md#81-the-dtype-enum-srcdtypezig)); both
direction functions return `Error.UnsupportedDtype` for the rest.

### 12.5.2 Reading

```zig
pub const File = struct {
    allocator: Allocator,
    bytes: []const u8,
    tensors: []TensorInfo,                       // sorted by data offset
    metadata: std.StringHashMap([]const u8),     // __metadata__ entries (owned copies)
    index: std.StringHashMap(usize),
    ownership: Ownership = .borrowed,            // borrowed | owned | mmap

    pub fn parse(allocator: Allocator, bytes: []const u8) !File        // borrows bytes
    pub fn parseOwned(allocator: Allocator, bytes: []u8) !File         // takes ownership
    pub fn load(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn loadMmap(allocator: Allocator, io: std.Io, path: []const u8) !File
    pub fn deinit(self: *File) void
    pub fn tensor(self: *const File, name: []const u8) !*const TensorInfo   // Error.TensorNotFound
    pub fn maybeTensor(self: *const File, name: []const u8) ?*const TensorInfo
    pub fn tensorNames(self: *const File, allocator: Allocator) ![][]const u8
    pub fn len(self: *const File) usize
    pub fn isEmpty(self: *const File) bool
};
pub fn readPrefix(allocator: Allocator, reader: *std.Io.Reader) !File
```

Ownership mirrors GGUF with one extra mode: `parse` **borrows** the input
bytes (the caller keeps them alive until `deinit` and frees them itself),
`parseOwned`/`load` own a heap buffer, `loadMmap` owns a read-only mapping.
`deinit` always frees the per-tensor `name`/`shape` copies, the metadata
copies, and the maps, then releases the bytes according to the mode.
`readPrefix` consumes exactly one safetensors frame from a stream reader —
header length, header, then precisely the data the header describes —
leaving the reader positioned after it (multiple frames can share a stream);
the result is an `owned` `File`.

Unlike GGUF, tensor **names and shapes are allocated copies** (they survive
into error paths cleanly); only `TensorInfo.data` borrows `File.bytes`.

Sharded checkpoints (the Hugging Face big-model layout: an
`*.safetensors.index.json` whose `weight_map` object maps tensor names to
shard files next to the index) load through `Sharded`:

```zig
pub fn isIndexPath(path: []const u8) bool   // ends with ".safetensors.index.json"

pub const Sharded = struct {
    total_size: ?u64,   // metadata.total_size from the index, when present

    pub fn load(allocator: Allocator, io: std.Io, index_path: []const u8) !Sharded
    pub fn loadMmap(allocator: Allocator, io: std.Io, index_path: []const u8) !Sharded
    pub fn deinit(self: *Sharded) void
    pub fn tensor(self: *const Sharded, name: []const u8) !*const TensorInfo
    pub fn maybeTensor(self: *const Sharded, name: []const u8) ?*const TensorInfo
    pub fn tensorNames(self: *const Sharded, allocator: Allocator) ![][]const u8  // weight_map order
    pub fn len(self: *const Sharded) usize
    pub fn isEmpty(self: *const Sharded) bool
};
```

Every shard opens eagerly (`loadMmap` maps each shard exactly like
`File.loadMmap`; `load` reads each into memory), and every `weight_map`
entry is verified to exist in its mapped shard at open — a truncated or
mismatched download fails with `Error.InvalidIndex` before any lookup.
Malformed index JSON, a missing `weight_map`, a non-string mapping, and a
shard filename that is not a plain basename (any path separator or `..`)
all reject with `Error.InvalidIndex`; a missing shard file surfaces the
underlying open error. Lookups resolve to the owning shard's `TensorInfo`,
so `data` borrows that shard's bytes with the same lifetime rules as
`File`.

```zig
pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []usize,
    data_offsets: [2]usize,      // [begin, end) relative to the data buffer
    data: []const u8,            // borrowed from File.bytes

    pub const Slice = struct { start: usize = 0, end: ?usize = null };
    pub fn sliceBytesAlloc(self: *const TensorInfo, allocator: Allocator,
                           ranges: []const Slice) ![]u8
};
```

`sliceBytesAlloc` gathers a row-major sub-block into a caller-owned byte
buffer: one `Slice` per leading axis (missing trailing axes take the full
extent; `ranges.len > rank` is `Error.InvalidSlice`, `start > end` or
`end > dim` likewise). Byte-aligned dtypes only (`Error.MisalignedSlice`
for `F4`/`F6_*`); a rank-0 tensor returns a copy of its whole payload.

Validation on parse (all before any tensor is exposed): UTF-8 header
(`InvalidHeader`), JSON well-formedness and schema
(`InvalidHeaderDeserialization` — including duplicate JSON keys), header
length sanity (`HeaderTooSmall`/`HeaderTooLarge`/`InvalidHeaderLength`),
offsets contiguous-ascending and matching `dtype x shape` byte length
(`InvalidOffset`/`TensorInvalidInfo`), buffer covered exactly — trailing
polyglot bytes or missing data are `MetadataIncompleteBuffer` — duplicate
tensor names (`DuplicateTensorName`), string-only `__metadata__`
(`InvalidMetadata`), and checked arithmetic throughout
(`ValidationOverflow`). Zero-sized tensors (a 0 in the shape) are legal.

### 12.5.3 Writing

```zig
pub const Tensor = struct { name: []const u8, dtype: DType,
                            shape: []const usize, data: []const u8 };
pub const MetadataEntry = struct { key: []const u8, value: []const u8 };

pub fn serialize(allocator: Allocator, writer: *std.Io.Writer,
                 tensors: []const Tensor, metadata: ?[]const MetadataEntry) !void
pub fn serializeAlloc(allocator: Allocator, tensors: []const Tensor,
                      metadata: ?[]const MetadataEntry) ![]u8
pub fn saveFileAtomic(allocator: Allocator, io: std.Io, path: []const u8,
                      tensors: []const Tensor, metadata: ?[]const MetadataEntry) !void
```

All three build the same bytes (golden-pinned against upstream safetensors
output in `src/safetensors_tests.zig`): everything is validated before the
first byte is written. Tensors are sorted by descending `DType` declaration
order, then ascending name — input order does not matter and is not
preserved. (Declaration order is not bit width: `BOOL` is declared first,
so under the descending sort its tensors land last, after the narrower
F4/F6 types.) Names must be non-empty UTF-8 and not `__metadata__`
(`InvalidTensorName`); duplicate names are `DuplicateTensorName`; each
tensor's `data.len` must equal its `dtype x shape` byte length
(`TensorInvalidInfo`). Metadata is an optional flat string map: unique
UTF-8 keys/values (`InvalidMetadata`), emitted as `__metadata__`.

`saveFileAtomic` writes to `PATH.tmp.<nanotimestamp>` (preallocated via
`setLength`, `F_NOCACHE` on macOS to skip the page cache on a one-shot
sequential write) and renames over `path`; on rename failure the temp file
is removed. Readers never observe a half-written file.

```zig
test "safetensors: serialize and parse back" {
    const alloc = std.testing.allocator;
    const st = fucina.safetensors;

    const values = [_]f32{ 1.5, -2.0, 0.25, 8.0 };
    const tensors = [_]st.Tensor{.{
        .name = "layer.weight",
        .dtype = .F32,
        .shape = &.{ 2, 2 },
        .data = std.mem.sliceAsBytes(&values),
    }};
    const meta = [_]st.MetadataEntry{.{ .key = "format", .value = "pt" }};

    const bytes = try st.serializeAlloc(alloc, &tensors, &meta);
    defer alloc.free(bytes);

    var file = try st.File.parse(alloc, bytes); // borrows `bytes`
    defer file.deinit();
    const t = try file.tensor("layer.weight");
    try std.testing.expectEqual(st.DType.F32, t.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, t.shape);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), t.data);
    try std.testing.expectEqualStrings("pt", file.metadata.get("format").?);
}
```

## 12.6 Named state dicts (`src/state_dict.zig`)

The API — `NamedTensor`/`NamedTensorMut`, `Alias`/`LoadOptions`,
`saveStateDict`/`loadStateDict`, name validation, alias remapping, strict
one-to-one matching, and the two-pass transactional load — is [§11.7](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#117-state-dicts-srcstate_dictzig); the
consumers (`fucina.ParamRegistry`, the `fucina.optim` re-exports) and the
schema-stability contract for registered names are [§11.5](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#115-optimizer-state-persistence-fzt1-snapshots-vs-named-state-dicts-srcoptimzig)–[§11.6](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#116-paramregistry-srcparam_registryzig). What
belongs here is the wire format: a state dict is exactly one safetensors
frame of [§12.5](12-model-io-gguf-and-safetensors.md#125-safetensors-srcsafetensorszig) with no `__metadata__` entry — there is no bespoke stream
format for state dicts. Entry names become the safetensors header keys
unchanged (hence the [§11.7](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#117-state-dicts-srcstate_dictzig) name rules: non-empty, NUL-free, unique UTF-8,
not `"__metadata__"`); dtypes map through `dtypeFromFucina`/`dtypeToFucina`
([§12.5](12-model-io-gguf-and-safetensors.md#125-safetensors-srcsafetensorszig)), with only F32/F16/BF16/I64 produced or accepted; tensor payloads are
raw little-endian storage bytes, no conversion in either direction.
`loadStateDict` consumes one frame from a stream via `readPrefix` ([§11.8](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#118-safetensors-readwrite-surface-srcsafetensorszig)),
so a state dict can be embedded in a longer stream. Any safetensors
consumer can read the file; GGUF remains a separate interop/export codec.

## 12.7 Training-checkpoint directory and native optimizer frames (`src/training_checkpoint.zig`, `src/optim.zig`)

The directory layout, the four file-name constants, the save/load API
(`pathJoin`, `beginSave`, `writeFileAtomic`, `saveTrainerState`,
`loadTrainerState`), the sentinel commit protocol, and the `TrainerState`
fields are [§11.9](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#119-checkpoint-directories-srctraining_checkpointzig); the frame-magic table and all load-time validation and
resume semantics are [§11.5](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#115-optimizer-state-persistence-fzt1-snapshots-vs-named-state-dicts-srcoptimzig). This subsection documents the on-disk formats.

**`trainer_state.json`.** A flat JSON object with the fixed marker
`"format": "fucina.training_checkpoint"` and `"version": 1` (anything else
is `Error.InvalidTrainerState` / `Error.UnsupportedTrainerStateVersion`).
`step` and `seed` (u64) are always present; every optional field is simply
omitted when null and parses to `null` when absent. Enum-like fields
(`es_noise`, `es_anchor_decay`) serialize through stable on-disk integer
mappings — never `@intFromEnum` of an in-memory enum.

**`optimizer.fucina`.** The raw byte stream produced by one optimizer's
`saveState` (or `OptimizerSet.saveState`) from `src/optim.zig`; the
magic ↔ optimizer table is [§11.5](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#115-optimizer-state-persistence-fzt1-snapshots-vs-named-state-dicts-srcoptimzig). Common frame shape: 4-byte magic,
optional optimizer-config scalars (Muon, Apollo, SGD write theirs right
after the magic), a u32 slot count, then per-slot records:

- **name** — u16 length + bytes; the explicit `addParamNamed` name or the
  auto-name `param<i>`.
- **dims** — u64 rows, u64 cols.
- **scalars** — u64 step (Adam/AdamW/Apollo/SGD); Apollo main slots add a
  u64 projection seed and a u32 f32-bit `prev_norm`.
- **state buffers** — in v3 frames, raw f32 bytes; in v4 and v5 frames each
  buffer is prefixed by one u8 `StateDType` tag (0 = f32, 1 = bf16;
  wire-stable) followed by the raw storage bytes. Apollo state buffers stay
  raw f32 in every version.
- **master record** — v5 frames only: each slot ends with a u8 presence
  flag, then the slot's raw f32 master weights when the flag is set.

A Muon frame is immediately followed by its embedded AdamW fallback frame;
an Apollo frame carries a second u32 count + slot list for its fallback
slots inside the same `FZP3`/`FZP5` frame; an `FZO3` container is the magic,
a u32 member count, then each member's frame in registration order.

**`FZT1`** (`optim.saveTensors`/`loadTensors`) is the minimal positional
format for parameter values only: magic, u32 tensor count, then per tensor a
u32 rank, rank u64 dims, and the raw f32 little-endian data. It carries no
names and no dtypes (usage rules and the named alternative are [§11.5](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#115-optimizer-state-persistence-fzt1-snapshots-vs-named-state-dicts-srcoptimzig)).
