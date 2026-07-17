# Chapter 11 — Model files and quantization

*Part V — Language models*

Everything so far has been self-contained: we built tensors, ops, autograd, training — and in [Chapter 10](10-the-guitar-amp.md) shipped a neural guitar amp whose weights we trained ourselves. Language models change the economics. A useful LLM has hundreds of millions to hundreds of billions of parameters, was trained by someone else on hardware you do not own, and arrives as a *file*. Before [Chapter 12](12-a-transformer-from-scratch.md) can run a transformer, this chapter has to answer two unglamorous questions that decide whether local LLM inference is possible at all:

1. **How do weights get into memory?** — the file-format question.
2. **How do they fit, and how fast can they be read?** — the quantization question.

The first has a one-line answer that demystifies every model file you will ever download: **a model is tensors plus metadata**. The second has a one-line answer too: **on a CPU, generating a token is a memory-bandwidth problem, so store fewer bytes per weight**. The rest of the chapter is the engineering that makes both answers real — mmap and zero-copy ownership, block codecs that match a reference implementation byte-for-byte, integer dot products over dynamically quantized activations, and an export tool that quantizes models far bigger than RAM one tensor at a time.

## 11.1 A model is tensors + metadata

Open a `.gguf` file in a hex editor and there is no magic in it — literally almost none; the magic number is the four ASCII bytes `GGUF`. The whole format is:

```
"GGUF"                      4 bytes, the magic
version                     u32 (Fucina parses v2 and v3, writes v3)
tensor_count                u64
metadata_kv_count           u64
metadata                    kv_count × (string key, typed value)
tensor directory            tensor_count × (name, n_dims, dims, type, offset)
padding                     zeros up to the next multiple of `alignment` (default 32)
tensor data                 raw bytes, each tensor padded to `alignment`
```

That is the entire container (parsed in `src/gguf.zig:358–443`; layout rules in `docs/REFERENCE.md` §12.1). The metadata is a flat key/value list — architecture name, layer count, RoPE base, the entire tokenizer vocabulary as a string array — and the tensor directory is a table of contents: for each named tensor, its shape, its wire type, and its offset into the data section. All integers are little-endian. Nothing else. Once you internalize "it's a table of contents plus bytes", every loader, exporter, and quantizer in this chapter is bookkeeping.

GGUF is the interchange format of the ggml/llama.cpp ecosystem, which means the thousands of quantized models published on Hugging Face in `.gguf` form are directly Fucina's input. That is a deliberate choice you will see justified throughout: rather than inventing a format, Fucina treats the ecosystem's wire formats as first-class and earns the right to interoperate by matching them exactly.

Fucina speaks two interchange formats and builds two native sidecar layers on top of them (`docs/REFERENCE.md` §12):

| Format | Module | Role |
|---|---|---|
| GGUF v2/v3 | `fucina.gguf` (`src/gguf.zig`) | llama.cpp-ecosystem interop: read quantized weights, re-emit/transcode/export |
| safetensors | `fucina.safetensors` (`src/safetensors.zig`) | Hugging Face tensor container: neutral named-tensor payloads |
| state-dict stream | `fucina.state_dict` | named checkpoint entries — the wire format *is* safetensors ([Chapter 8](08-training.md)) |
| checkpoint directory | `fucina.training_checkpoint` | resumable training: safetensors payloads + native optimizer frames |

safetensors is even simpler than GGUF: a u64 length, a JSON header describing named tensors, then one contiguous data buffer (§11.5). The lesson of the table is the same one the state-dict row makes explicit: *there is no bespoke stream format* — when Fucina needed a checkpoint container it reused safetensors rather than inventing one (`docs/REFERENCE.md` §12.6).

> **ML note** — Why two formats? History and division of labor. safetensors (from Hugging Face) stores tensors and nothing else — no architecture info, no tokenizer — because the Python ecosystem keeps those in separate config files. GGUF (from ggml) is self-contained by design: one file carries weights, hyperparameters, *and* tokenizer, because a C inference engine has no `config.json` ecosystem to lean on. Self-containment is why a single `.gguf` path is all Fucina's LLM runners need.

## 11.2 Reading the container: a cursor over bytes

Parsing a binary format needs exactly three primitives: read `n` bytes, read a little-endian integer, read a length-prefixed string. Fucina's GGUF parser builds them as a `Cursor` (`src/gguf.zig:641–740`):

```zig
const Cursor = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn readBytes(self: *Cursor, len: usize) ![]const u8 {
        const end = try std.math.add(usize, self.offset, len);
        if (end > self.bytes.len) return error.EndOfStream;
        const out = self.bytes[self.offset..end];
        self.offset = end;
        return out;
    }

    fn readInt(self: *Cursor, comptime Int: type) !Int {
        return std.mem.readInt(Int, (try self.readBytes(@sizeOf(Int)))[0..@sizeOf(Int)], .little);
    }

    fn readString(self: *Cursor) ![]const u8 {
        const len: usize = @intCast(try self.readInt(u64));
        return self.readBytes(len);
    }
    // ... readValue, readArray, readAlignment
};
```

Note what `readBytes` *returns*: a slice into the file's bytes, not a copy. Every metadata string, every vocabulary entry, every tensor payload the parser hands out is a zero-copy view into `File.bytes`. That is why loading a 4 GB model's metadata costs almost nothing — and it is the chapter's first ownership lesson: **everything a `File` hands out dies at `file.deinit()`** (`docs/REFERENCE.md` §12.1.1). Cheap reads mean lifetime discipline.

Here is a complete, minimal header parser in the same style — course code, not repo code (compile-checked with `zig test`):

```zig
// Course code — a minimal GGUF-style header parser.
const std = @import("std");

const Cursor = struct {
    bytes: []const u8,
    offset: usize = 0,

    fn readBytes(c: *Cursor, len: usize) ![]const u8 {
        const end = try std.math.add(usize, c.offset, len); // checked: len is untrusted
        if (end > c.bytes.len) return error.EndOfStream;
        const out = c.bytes[c.offset..end];
        c.offset = end;
        return out;
    }

    fn readInt(c: *Cursor, comptime Int: type) !Int {
        const raw = try c.readBytes(@sizeOf(Int));
        return std.mem.readInt(Int, raw[0..@sizeOf(Int)], .little);
    }
};

const Header = struct { version: u32, tensor_count: u64, kv_count: u64 };

fn parseHeader(c: *Cursor) !Header {
    const magic = try c.readBytes(4);
    if (!std.mem.eql(u8, magic, "GGUF")) return error.InvalidMagic;
    const version = try c.readInt(u32);
    if (version != 2 and version != 3) return error.UnsupportedVersion;
    return .{
        .version = version,
        .tensor_count = try c.readInt(u64),
        .kv_count = try c.readInt(u64),
    };
}

test "parse a hand-built GGUF header" {
    var bytes: [4 + 4 + 8 + 8]u8 = undefined;
    bytes[0..4].* = "GGUF".*;
    std.mem.writeInt(u32, bytes[4..8], 3, .little);
    std.mem.writeInt(u64, bytes[8..16], 291, .little); // tensor count
    std.mem.writeInt(u64, bytes[16..24], 42, .little); // metadata KV count

    var cursor: Cursor = .{ .bytes = &bytes };
    const h = try parseHeader(&cursor);
    try std.testing.expectEqual(@as(u32, 3), h.version);
    try std.testing.expectEqual(@as(u64, 291), h.tensor_count);
    try std.testing.expectEqual(@as(u64, 42), h.kv_count);
}

test "hostile input: a truncated header is an error, not a crash" {
    const bytes = [_]u8{ 'G', 'G', 'U', 'F', 3, 0 };
    var cursor: Cursor = .{ .bytes = &bytes };
    try std.testing.expectError(error.EndOfStream, parseHeader(&cursor));
}
```

### Typed values from untyped bytes

A GGUF metadata value can be any of thirteen wire types — eight integer widths, two floats, bool, string, array. The parser surfaces them as a *tagged union* (`src/gguf.zig:138–144`):

```zig
pub const Value = union(enum) {
    int: i64, float: f64, boolean: bool,
    string: []const u8, array: Array,
};
```

Note the deliberate collapse: thirteen wire types become five cases, because every integer widens to `i64` and every float to `f64` — pleasant for consumers (`file.getInt("qwen3.block_count")` does not care whether the file said u32 or u64), lossy for writers (§11.4 shows the raw path that recovers the exact widths). Arrays stay cheap by staying raw: an `Array` records the element type, the count, and the *undecoded byte span* — a tokenizer vocabulary of a hundred thousand strings is not parsed into a hundred thousand slices unless someone asks (`Array.stringSlices`).

> **Zig note** — `union(enum)` is a sum type with a runtime tag: a `Value` is exactly one of its cases, `switch` over it must handle all of them, and accessing the wrong case is a checked panic in safe builds rather than a reinterpretation. It is the natural target for any "one of several wire types" decode — compare `readValue` (`src/gguf.zig:667–686`), which maps each wire code to a union case in one `switch`, including the checked `std.math.cast` that rejects a `uint64 >= 2^63` instead of corrupting it into a negative `i64`.

### Parsing hostile input

A model file is untrusted input — usually a multi-gigabyte download from the internet — and the parser treats it that way at every arithmetic step. Three patterns from `src/gguf.zig` are worth stealing for any binary parser you ever write.

**Checked arithmetic everywhere.** `std.math.add`, `std.math.mul`, and `std.math.cast` replace bare `+`/`*`/`@intCast` wherever a wire value participates. A header that declares a tensor of 2⁶³ elements must fail with an ordinary error — `error.Overflow` from the checked multiply, or `error.InvalidTensorInfo` at the EOF bounds check — not an integer overflow that wraps into a small "valid" length (`src/gguf.zig:592–599` shows the checked block-count multiply; the same pattern guards the safetensors parser at `src/safetensors.zig:623–627`).

> **Zig note** — In Debug and ReleaseSafe builds, Zig traps integer overflow with a panic — already better than silent wraparound. But release binaries of a library ship as ReleaseFast, where overflow is undefined behavior. `std.math.add(usize, a, b)` returns `error.Overflow` in *all* build modes, turning "hostile header" from a safety bug into an ordinary error-union branch. For untrusted input, make overflow a value, not a trap.

**Validate before the lossy widening.** The parser widens every metadata integer to `i64` for convenience — but `general.alignment` controls padding arithmetic, so a hostile value must be rejected *before* the widening and before any cast can bite (`src/gguf.zig:377–390`):

```zig
if (std.mem.eql(u8, key, "general.alignment")) {
    // Validate directly from the wire value (before readValue's lossy
    // uint64->i64 narrowing and before the unchecked i64->usize cast),
    // so a hostile alignment can't reach UB at the cast or alignForward.
    alignment = try cursor.readAlignment(value_type);
    metadata.putAssumeCapacity(key, .{ .int = @intCast(alignment) });
    continue;
}
```

`readAlignment` (`src/gguf.zig:693–713`) accepts only positive powers of two up to 2²⁰, reading the exact wire type. Validation happens at the one point where the raw value is still visible.

**Bounds checks that double as diagnostics.** The most common real-world failure is not an attack — it is a truncated download. The parser names it (`src/gguf.zig:419–429`):

```zig
if (end > bytes.len) {
    // The header describes a tensor that runs past EOF — almost
    // always a truncated/incomplete download or a botched export ...
    std.log.err("gguf: '{s}' ends at {d} but file is only {d} bytes — short by {d} ({d:.2} GB); the GGUF is truncated/incomplete (re-download or re-export)", .{
        info.name, end, bytes.len, end - bytes.len, @as(f64, @floatFromInt(end - bytes.len)) / 1e9,
    });
    return Error.InvalidTensorInfo;
}
```

A bare `InvalidTensorInfo` would be correct; a message naming the first offending tensor and the shortfall in gigabytes is correct *and* self-diagnosing. Error ergonomics are part of the format contract.

## 11.3 mmap: the file is the memory

`File.load` reads the whole file into a heap buffer — fine for tests and tools. For a 20 GB model it would be a disaster twice over: the copy takes time, and the heap buffer would coexist with everything materialized from it. `File.loadMmap` (`src/gguf.zig:244–260`) instead maps the file read-only (`PROT_READ`, `MAP_PRIVATE`, fd closed immediately — POSIX keeps the mapping valid) and parses in place. Now the pages are *file-backed and evictable*: the OS pages tensor bytes in on first touch and can drop clean pages under memory pressure, because it knows where to get them back. The file, quite literally, is the memory.

Reading a real model's metadata and directory looks like this (from `docs/REFERENCE.md` §12.1.3 — a machine-verified snippet, like every named `test` in that file, via `zig build snippet-check`; comments lightly adapted for the course):

```zig
var file = try fucina.gguf.File.loadMmap(alloc, io, "models/Qwen3-0.6B-Q4_K_S.gguf");
defer file.deinit(); // unmaps; every borrowed slice dies here

const arch = file.getString("general.architecture").?;
const n_layers = file.getInt("qwen3.block_count").?;
const tokens = file.getArray("tokenizer.ggml.tokens").?;
const vocab = try tokens.stringSlices(alloc); // slices point into the mapping
defer alloc.free(vocab);

const embd = try file.get("token_embd.weight");
fucina.gguf.prefetch(embd.data); // OS readahead before a sequential copy
const dt = fucina.gguf.dtypeForGgmlType(embd.ggml_type); // core DType (Chapter 3)
```

Three details of this surface repay attention.

**`prefetch`/`release` steer the OS.** `prefetch(data)` issues `madvise(WILLNEED)` over a mapped region about to be read in full, letting readahead run ahead of a sequential pack loop — the dominant cold-load cost; `release` is the eviction-side hint (`src/gguf.zig:17–43`). `prefetch` is a silent no-op on heap-read files (already resident), and both degrade to no-ops when the advice call itself is unsupported or fails — but `release` is only valid on read-only file-backed mappings (`loadMmap*`): those pages are clean, so `MADV.DONTNEED` merely drops them to refault from the file later, whereas on a heap buffer the same call can discard live data. The export tool in §11.11 uses the pair to keep a hundreds-of-GB quantization run inside a bounded working set.

**`takeMapping` is an explicit ownership handoff.** A model wants to borrow weight bytes for its whole lifetime, which is longer than the `File` object's. Rather than keeping the `File` alive forever, the loader takes the mapping out of it (`src/gguf.zig:477–490`):

```zig
pub fn takeMapping(self: *File) ?MappedRegion {
    if (!self.is_mmap) return null;
    // Split files: tensors point into ALL part mappings, but a
    // MappedRegion can carry only one — borrowing across a split load
    // is not supported (stream the experts instead).
    if (self.isSplit()) return null;
    self.is_mmap = false;
    const bytes = self.bytes;
    // Leave an empty slice so deinit's heap branch is a no-op; previously
    // parsed metadata/TensorInfo slices keep pointing into the (still
    // mapped) region now owned by the caller.
    self.bytes = &.{};
    return .{ .bytes = bytes };
}
```

No borrow checker, no reference counting: a flag flip and an empty-slice sentinel transfer the `munmap` responsibility to the caller, while every already-parsed slice stays valid because the mapping itself never moved. This is Zig's ownership style at its most explicit — the transfer is a visible line of code, and `deinit` afterwards is a no-op by construction. (`loadMmapAuto` extends the same machinery to llama.cpp *split* GGUFs — `-00001-of-00003` part files mapped and merged into one `File` — which is why `takeMapping` refuses split loads: one region cannot carry three mappings.)

**Dims are in `ne` order — innermost first.** GGUF inherits ggml's convention: `dims[0]` is the *fastest-varying* axis. A row-major logical `[out, in]` matrix appears in the directory as `dims = { in, out }`, and `TensorInfo.logicalMatrixShape()` swaps it back (`src/gguf.zig:181–184`). Forgetting this does not error — it silently transposes every shape you print. Conventions are part of a format; the reader that "mostly works" is the one that has this bug.

> **Zig note** — `std.posix.mmap`, `munmap`, and `madvise` are right there in the standard library — no libc binding layer, no wrapper crate. Systems programming in Zig means the OS interface is a function call away, and the GGUF module uses exactly three of them. When Fucina needs macOS-specific behavior (like `F_NOCACHE` in §11.5), it gates on `builtin.os.tag == .macos` at comptime — the foreign branch is not compiled into other targets' binaries.

## 11.4 Writing GGUF: byte-identical or wrong

Reading a format is half the discipline. The `gguf.Writer` (`src/gguf.zig:841–1156`) closes the loop: it buffers metadata KVs and tensor declarations, then `finish` serializes magic, version 3, counts, KV section, tensor directory, padding, and each tensor's data padded to `alignment` — *matching ggml's writer byte-for-byte* (`docs/REFERENCE.md` §12.2). The parser accepts versions 2 and 3; the writer emits version 3 only.

The load-bearing detail is offsets. llama.cpp's reader rejects any file whose tensor offsets are not exactly the running padded total, so the writer computes them rather than discovering them (`src/gguf.zig:1083–1095`, inside `writeHeader`):

```zig
var data_offset: usize = 0;
for (self.tensors.items) |t| {
    try out.writeInt(u64, @intCast(t.name.len), .little);
    try out.writeAll(t.name);
    try out.writeInt(u32, @intCast(t.n_dims), .little);
    for (t.dims[0..t.n_dims]) |dim| try out.writeInt(u64, @intCast(dim), .little);
    try out.writeInt(u32, @intFromEnum(t.ggml_type), .little);
    try out.writeInt(u64, @intCast(data_offset), .little);
    header_len += 8 + t.name.len + 4 + t.n_dims * 8 + 4 + 8;
    data_offset = try std.math.add(usize, data_offset, std.mem.alignForward(usize, t.byte_len, self.alignment));
}
```

Every offset is computable at *declaration* time because a tensor's wire size follows from its type and dims (`tensorByteLen`). Hold that thought — it is exactly what makes the streaming writer of §11.11 possible.

The discipline this buys is stated as a test invariant, not a hope: **re-parsing a `finish` output and re-emitting it reproduces the file byte-identically, and a real-model re-emit preserves every KV and tensor payload verbatim** — both asserted in `src/gguf_tests.zig` (`docs/REFERENCE.md` §12.2). "Byte-identical re-emit" is a brutal test: any sloppiness in padding, any lossy metadata round-trip, any off-by-one in an offset fails it. Interop is a byte contract, and the only way to know you honor a byte contract is to check bytes.

One subtlety makes the verbatim claim harder than it sounds. The parser deliberately widens metadata scalars (every integer wire type becomes `i64`) — convenient for lookups, but lossy: the original wire width is gone, and llama.cpp *type-checks* many keys. The writer therefore has two metadata paths: the typed `addMetaInt(key, u32, value)` family, where the caller chooses the exact wire width at comptime, and `addMetaCopy`/`copyAllMetadata`, which **re-read the original file bytes** to copy KVs byte-verbatim, preserving widths the parsed view dropped (`src/gguf.zig:938–1016`). A lossy convenience view plus a lossless raw path — a pattern worth remembering whenever you find yourself normalizing someone else's data.

> **Zig note** — `addMetaInt(self, key, comptime Int: type, value: Int)` is a small API-design gem. The wire format has eight integer types; instead of eight methods or a runtime enum parameter, the *type itself* is the parameter. A comptime `switch` on `Int` maps `u32` to wire code 4, `i64` to 11, and so on — and hits `@compileError` for anything unsupported (`src/gguf.zig:764–789`), so `addMetaInt("k", usize, x)` fails at build time with a message, not at runtime with a corrupt file. The caller cannot avoid deciding the wire width, which is exactly the point.

The whole arc — write, reopen, verify — fits in one machine-verified snippet (`docs/REFERENCE.md` §12.2, abridged; note the `ne`-order dims):

```zig
var w = fucina.gguf.Writer.init(alloc);
defer w.deinit();
try w.addMetaString("general.name", "demo");
try w.addMetaInt("demo.heads", u32, 8);

const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
var wire: [24]u8 = undefined;
try fucina.gguf.encodeF32(.f32, &values, &wire);
try w.addTensor("w", .f32, &.{ 3, 2 }, &wire); // ne order: logical [2, 3]

// ... finish() into a file, reopen with File.load ...

const info = try file.get("w");
try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, &(try info.logicalMatrixShape()));
try std.testing.expectEqualSlices(u8, &wire, info.data);
```

Ownership on the write side mirrors the read side inverted: metadata keys and tensor *names* are duplicated into the writer, but tensor *data* is borrowed and must stay alive until `finish` returns (`docs/REFERENCE.md` §12.2). The writer holds a table of contents; the bytes remain yours.

## 11.5 safetensors alongside

The safetensors module (`src/safetensors.zig`) is the same story with a JSON accent: a u64 little-endian header length, a UTF-8 JSON header mapping tensor names to `{dtype, shape, data_offsets}`, then one contiguous data buffer. The header length is capped at 100 MB in both directions (`max_header_size`, `src/safetensors.zig:16`), and on write the header is padded to an 8-byte multiple with spaces so the first data byte lands naturally aligned for scalar dtypes (`docs/REFERENCE.md` §12.5.1) — the same align-the-data-section instinct as GGUF, expressed in JSON whitespace.

The parser validates *everything* before exposing a single tensor — UTF-8, JSON schema, duplicate keys, offsets contiguous-ascending and covering the buffer exactly, checked arithmetic throughout (`docs/REFERENCE.md` §12.5.2). Trailing bytes after the described data are an error too (`MetadataIncompleteBuffer`): a file that parses is a file with no polyglot smuggling room. One reader deserves a call-out for what it enables: `readPrefix` consumes *exactly one* safetensors frame from a stream — header length, header, precisely the data the header describes — leaving the reader positioned after it. Multiple frames can therefore share one stream, which is how [Chapter 8](08-training.md)'s state dicts embed into longer checkpoint streams without any framing format of their own.

Three facts to keep straight, because each contradicts a plausible assumption:

- **The dtype bridge is narrow by design.** safetensors *tags* cover everything from `BOOL` to sub-byte `F4`, and all of them round-trip through Fucina as raw bytes — but only `F32`/`F16`/`BF16` map to core `DType`s (`dtypeFromFucina`/`dtypeToFucina`, `src/safetensors.zig:108–124`; anything else is `error.UnsupportedDtype`). Quantized weights live in GGUF; safetensors is Fucina's checkpoint and adapter format, and checkpoints are float ([Chapter 8](08-training.md)).
- **The writer does not preserve input order.** `serialize` sorts tensors by descending `DType` declaration order, then ascending name (`tensorLessThan`, `src/safetensors.zig:478–483`) — and since `BOOL` is declared first, its tensors land *last*. Output bytes are golden-pinned against upstream safetensors output in `src/safetensors_tests.zig` (`docs/REFERENCE.md` §12.5.3): the parity religion again, applied to a format Fucina writes more often than it reads.
- **`parse` borrows.** Unlike GGUF's constructors, plain `parse(allocator, bytes)` leaves the bytes yours — keep them alive, free them yourself. `parseOwned`/`load`/`loadMmap` own. Read the constructor name, not your habit.

The write path ends with a small masterpiece of boring reliability, `saveFileAtomic` (`src/safetensors.zig:366–398`): serialize to `PATH.tmp.<nanotimestamp>` (preallocated with `setLength`; `F_NOCACHE` on macOS so a one-shot sequential write skips the page cache), then `rename` over the destination — removing the temp file on failure. Readers never observe a half-written checkpoint. If your training run dies mid-save at step 40,000, the previous checkpoint is still intact; that property is worth more than any speedup in this chapter.

> **Zig note** — The write path shows the Zig 0.16 I/O style you will see everywhere in the repo: a `file.writer(io, &buffer)` wraps an OS file in a buffered `std.Io.Writer`, calls go through `writer.interface.writeInt/writeAll`, and nothing hits the disk until `flush`. `errdefer std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {}` is the cleanup idiom: on any error after temp-file creation, the partial file is removed — and the `catch {}` acknowledges that cleanup failure is not worth masking the original error for.

## 11.6 Why quantize: the bandwidth wall

Now the second question: why does every practical CPU LLM store weights in 4–8 bits?

The answer is not "to fit in RAM" — though that helps — it is *speed*. Recall [Chapter 6](06-going-fast-on-cpus.md)'s `m` parameter: a matmul with `m` activation rows does `m·n·k` multiply-adds over `n·k` weights, so each weight loaded from memory is reused `m` times. LLM inference has two phases with opposite characters ([Chapter 12](12-a-transformer-from-scratch.md) develops them): *prefill* processes the whole prompt at once — `m` is large, weights are reused, compute dominates, and Chapter 6's cache blocking earns its keep. *Decode* generates one token at a time — `m = 1`. Each weight byte is read once from memory, used for exactly one multiply-add, and never reused. There is no cache blocking that saves you, because there is no reuse to block for. The arithmetic is trivial; the *reading* is the work. Fucina's benchmark doc states it as the design fact it is: "Decode is weight-bandwidth-bound, so batching N streams into one m=N pass reads the weights once instead of N times" (`docs/BENCHMARK.md`, batch-decode section — that observation is Chapter 13's multi-stream trick).

Back-of-envelope, explicitly arithmetic and not a measurement: a 7-billion-parameter model in f32 is 28 GB of weights. At a nominal 100 GB/s of sustained memory bandwidth, touching every weight once costs ~0.28 s — under 4 tokens/second as a *hard floor*, before any compute, no matter how many cores you have. The same model at 4.5 bits per weight is ~3.9 GB — the identical machine's floor becomes ~25 tokens/second. Nothing about the ALUs changed. **Bytes per weight is the speed knob**, which is why the units below are always bits per weight (bpw):

| Storage | bits/weight | 7B model |
|---|---:|---:|
| f32 | 32 | 28 GB |
| f16 / bf16 | 16 | 14 GB |
| Q8_0 (34 B / 32 weights) | 8.5 | 7.4 GB |
| Q6_K (210 B / 256) | ~6.56 | 5.7 GB |
| Q4_K (144 B / 256) | 4.5 | 3.9 GB |

(The bpw figures are arithmetic on the block struct sizes you will meet in §11.7; the gigabytes are `7e9 × bpw / 8`.)

Honesty note: bandwidth is the first-order story, not the whole story. `docs/BENCHMARK.md` records a measured case where Q6_K (fewer bytes per weight) read *slower* hot than Q8_0 — "bandwidth would predict the opposite" — because at hot decode sizes kernel arithmetic cost matters too. The model "fewer bytes = faster" is the right default and the wrong absolute.

> **ML note** — Why does throwing away 27.5 of 32 bits not destroy the model? Two reasons. First, trained weights are redundant: language models are heavily over-parameterized, and the function they compute is far smoother than any individual weight. Second — and this is the part people miss — quantization here is applied to *inference only*, weights frozen after training. The errors act like small fixed noise on each matrix, and the network's own nonlinear redundancy absorbs it. Accuracy loss is real but small for ≥4-bit formats on most models; measuring it (perplexity deltas) rather than asserting it is [Chapter 14](14-the-low-bit-frontier.md)'s topic.

## 11.7 Block quantization from first principles: Q8_0

Storing a weight in fewer bits means storing a small integer plus enough information to undo the scaling. One scale for a whole tensor cannot work — a tensor with one large outlier would crush everything else to zero. One scale per weight defeats the purpose. The compromise that won: one scale per **block** of consecutive weights.

Q8_0 is the teachable atom. Take 32 consecutive f32 values; find the largest magnitude `amax`; set the scale `d = amax / 127`; store each value as `round(x / d)`, an i8 in [−127, 127]; store `d` itself as an f16. The entire encoder, from `src/backend/quant/q8k.zig:57–68` (this is the real production code for non-aarch64 targets):

```zig
var block_index: usize = 0;
while (block_index < block_count) : (block_index += 1) {
    const row = src[block_index * q8_0_block_size ..][0..q8_0_block_size];
    var amax: f32 = 0;
    for (row) |v| amax = @max(amax, @abs(v));

    const d = amax / 127.0;
    const inv_d: f32 = if (d == 0) 0 else 1.0 / d;

    dst[block_index].d = f32ToF16Bits(d);
    for (&dst[block_index].qs, row) |*q, v| q.* = quantizeToI8(v * inv_d);
}
```

Eleven lines. The size math: 32 weights that cost 128 bytes in f32 now cost 32 bytes of i8 plus 2 bytes of f16 scale = **34 bytes** — 8.5 bits per weight, with a worst-case error of half a quantization step (`d/2`) per value.

And here is the type those 34 bytes decode *as* — this is the heart of Fucina's I/O design (`src/dtype.zig:74–77`, field comments added here, and the comptime asserts at `:221–248`):

```zig
pub const BlockQ8_0 = extern struct {
    d: u16,                        // f16 bits of the scale
    qs: [q8_0_block_size]i8,       // 32 quantized values
};

comptime {
    std.debug.assert(@sizeOf(BlockQ8_0) == 34);
    // ... one assert per block struct, 27 in total
}
```

`extern struct` gives C-compatible, declaration-order layout — the struct *is* the wire format. Loading a Q8_0 tensor from an mmap'd GGUF is reinterpreting mapped bytes as `[]const BlockQ8_0` and borrowing them zero-copy; no deserialization step exists (`docs/REFERENCE.md` §10). The `comptime` asserts pin the ABI: if any Zig version, target, or careless edit ever changed a block's size, the build fails — at compile time, not at 3 a.m. in a corrupted export. (The alignment story completes the trick: mmap bases are page-aligned and tensor offsets are multiples of `alignment`, so borrowed block slices really do satisfy `@alignOf(BlockQ8_0)` — the file format's alignment rule and the kernel tier's pointer contract are the same fact, noted at `docs/REFERENCE.md` §12.1.1.)

> **Zig note** — A plain Zig `struct` gives the compiler freedom to reorder fields; `extern struct` forbids it, guaranteeing C ABI layout. That plus the size assert is the whole "serialization framework": Fucina has no encode/decode step for quantized weights because the in-memory type and the on-disk type are the same bytes. When the file *is* the memory (§11.3), the type system reaches all the way down to the disk.

To make the round-trip concrete, here is a self-contained version with an f32 scale for clarity — course code, not repo code (compile-checked; the real format's f16 scale is the only simplification):

```zig
// Course code — block quantization from first principles.
const std = @import("std");

const block_len = 32;

const BlockQ8 = extern struct {
    d: f32,               // course simplification: the real BlockQ8_0 stores f16 bits
    qs: [block_len]i8,
};

comptime {
    std.debug.assert(@sizeOf(BlockQ8) == 36);
}

fn quantizeBlock(values: *const [block_len]f32) BlockQ8 {
    var amax: f32 = 0;
    for (values) |v| amax = @max(amax, @abs(v));

    const d = amax / 127.0;
    const inv_d: f32 = if (d == 0) 0 else 1.0 / d;

    var block: BlockQ8 = .{ .d = d, .qs = undefined };
    for (&block.qs, values) |*q, v| {
        q.* = @intFromFloat(std.math.clamp(@round(v * inv_d), -127.0, 127.0));
    }
    return block;
}

fn dequantizeBlock(block: *const BlockQ8, out: *[block_len]f32) void {
    for (block.qs, out) |q, *v| v.* = block.d * @as(f32, @floatFromInt(q));
}

test "quantize -> dequantize round-trips within half a step" {
    var rng = std.Random.DefaultPrng.init(7);
    var values: [block_len]f32 = undefined;
    for (&values) |*v| v.* = rng.random().float(f32) * 4.0 - 2.0;

    const block = quantizeBlock(&values);
    var back: [block_len]f32 = undefined;
    dequantizeBlock(&block, &back);

    for (values, back) |want, got| {
        try std.testing.expect(@abs(want - got) <= block.d / 2 + 1e-6);
    }
}
```

In the repo, the same round-trip runs through the public seam — `fucina.gguf.encodeF32(.q8_0, &src, &wire)` then `decodeF32` back, verified within 0.05 in the machine-checked snippet at `docs/REFERENCE.md` §12.3. The production decoder, incidentally, is explicitly vectorized with 8-lane `@Vector`s, and its comment quantifies why: the Q8_0 KV-cache attention path dequantizes every K/V row it streams, and the scalar loop costs "~2.4x decode attention cost vs f16" (`src/backend/quant/q8k.zig:98–113`) — bit-for-bit the same values, in vector chunks. Even a dequantizer is a hot loop somewhere.

## 11.8 K-quants, conceptually — and who can encode what

Push below 8 bits and one scale per 32 values stops being enough: a 4-bit code has only 16 levels, so the scale must adapt to finer-grained structure, but per-32 f16 scales would cost 0.5 bpw of pure overhead. The K-quants answer with *hierarchy* — a 256-element **super-block** carrying two f16 super-scales and eight packed 6-bit sub-scales/sub-mins, one pair per 32-element sub-block (`BlockQ4_K`, `src/dtype.zig:119–123`; field comments added):

```zig
pub const BlockQ4_K = extern struct {
    dm: [2]u16,                       // f16 super-scale d and super-min dmin
    scales: [k_scale_size]u8,         // 8 six-bit sub-scales + 8 six-bit sub-mins, bit-packed in 12 bytes
    qs: [qk_k_block_size / 2]u8,      // 256 four-bit codes, two per byte
};
```

144 bytes for 256 weights — 4.5 bpw, of which only 0.5 bpw is scale metadata. A sub-block's effective scale is `f16(d) × 6-bit sub-scale`: the hierarchy amortizes the expensive f16s across eight sub-scales that cost six bits each. Reconstruction is `x ≈ d·sc·q − dmin·m` — note the *minimum* term: Q4_K is an asymmetric (scale + offset) format, which buys precision when a block's values do not straddle zero (common in real weight matrices). Every K-quant variant (Q2_K through Q6_K) is a different point on the same design: how many bits per code, per sub-scale, symmetric or asymmetric.

A representative slice of the 27-format inventory (full table with kernel tiers in `docs/REFERENCE.md` §10.1; bpw = bytes/block × 8 ÷ elems/block):

| DType | Block struct | Elems/block | Bytes/block | bpw | Design point |
|---|---|---:|---:|---:|---|
| `.q8_0` | `BlockQ8_0` | 32 | 34 | 8.5 | flat: one f16 scale per 32 |
| `.q4_1` | `BlockQ4_1` | 32 | 20 | 5.0 | flat, asymmetric (scale + min) |
| `.q6_k` | `BlockQ6_K` | 256 | 210 | ~6.56 | super-block, symmetric, 8-bit-ish quality |
| `.q4_k` | `BlockQ4_K` | 256 | 144 | 4.5 | super-block, asymmetric — the workhorse |
| `.q2_k` | `BlockQ2_K` | 256 | 84 | 2.625 | super-block, 2-bit codes (decode-only) |
| `.tq2_0` | `BlockTQ2_0` | 256 | 66 | 2.0625 | ternary {−1, 0, +1} — [Chapter 14](14-the-low-bit-frontier.md) |

That is all the conceptual machinery you need; the bit-packing details are in `src/backend/quant/` when you want them. Each row of the table is one `extern struct` in `src/dtype.zig:69–219` with its own comptime size assert — the §11.7 pattern, 27 times over.

**The encode/decode asymmetry.** The full inventory's "f32 encoder" column (`docs/REFERENCE.md` §10.1) teaches a design stance. Every one of the 27 block formats *decodes* to f32 at the kernel tier (`dequantizeRowForDType` covers all of them — a loaded weight is always usable). But the public `gguf.encodeF32` seam *produces* exactly ten block formats — `q2_0 q4_0 q4_1 q5_0 q5_1 q8_0 q4_k q5_k q6_k tq2_0` (plus f32/f16/bf16 scalar casts); everything else returns `error.EncoderUnavailable` (`src/gguf.zig:1168–1204`, table in `docs/REFERENCE.md` §12.3). Q2_K, Q3_K, the entire IQ family, MXFP4, NVFP4, TQ1_0, Q1_0: **decode-only** — Fucina reads models shipped in those formats but never writes them. (The two activation formats Q8_1/Q8_K sit in between: the kernel tier encodes them on the fly for activations — §11.9 — but `encodeF32` does not emit them as tensor data.) The repo states the split as fact, and it is consistent with the verification bar of §11.10: an encoder ships when it can be proven byte-exact against ggml's reference — and in the llama.cpp ecosystem some of the exotic encoders require importance-matrix calibration, which raises that bar further. Half-supported writing is worse than honest `EncoderUnavailable`.

Five formats are *first-class end-to-end* — encoder, tuned hot kernel, GGUF export: `q8_0`, `q4_k`, `q5_k`, `q6_k`, and the ternary `tq2_0` (`docs/REFERENCE.md` §10.1). The ternary story — 2.06 bits per weight, multiplication-free kernels, and PTQTP's multi-plane decomposition — deserves its own chapter: [Chapter 14](14-the-low-bit-frontier.md).

> **ML note** — You will also meet Q4_K_S / Q4_K_M suffixes on downloaded models. The letter is a *mix* policy, not a format: which tensors of the model get Q4_K versus a higher-precision format (attention/output layers often get more bits). The per-tensor freedom is native to GGUF — every directory entry carries its own type, so "a Q4_K_S model" is really a policy over a directory of independently typed tensors.

## 11.9 The compute path: multiplying by weights you never dequantize

Storing weights small is pointless if the matmul dequantizes them to f32 first — that would pay the bandwidth *and* the decode cost. The point of the block formats is that the kernels consume them directly. Three layers make that work in Fucina.

### Quantized tensors are constants

A block-quantized tensor enters the [Chapter 4](04-axes-with-names.md) type system with its dtype in the spec — `fucina.Tensor(.{ .dtype = .q4_k, .tags = .{ .out, .in } })` — and the facade "deliberately exposes **only** quantized operations — no autograd (`requiresGrad()` is always `false` ...), no float math (`add`, `softmax`, ... are absent at comptime)" (`docs/REFERENCE.md` §10.2). Calling `softmax` on a Q4_K tensor is not a runtime error; it is a compile error, because the method does not exist on that type. What does exist: `to(.f32)` (full dequantize), `getRows` (gather + dequantize only the looked-up rows — the embedding path: a vocab-sized table stays quantized and only the tokens in flight are widened), `concat`, `packRhs`, and construction from blocks:

- `fromBlocks` — copies blocks into context-owned storage;
- `fromBorrowedBlocks` — zero-copy borrow: this is the mmap path, where the loader reinterprets mapped GGUF bytes as a block slice, and the §11.3 `MappedRegion` keeps them alive.

Shapes are *logical* element counts; a `[n, k]` tensor holds `n × (k / block_size)` blocks, and `k` must be a whole number of blocks — a multiple of the block size, 256 for every K-quant. Weight tensors follow the ggml convention `[out, in]`: block row `r` is output column `r`.

### The dot: integer arithmetic inside, f32 outside

The f32 tensor's `dot` from [Chapter 5](05-the-operation-library.md) dispatches on the RHS dtype at comptime: a quantized RHS routes the contraction to the quantized matmul instead of dense GEMM. The requirements are comptime-checked — the RHS must be stored `[free, contract]` (weight layout, never transposed at runtime) with exactly one free axis (`docs/REFERENCE.md` §10.2). From the machine-verified snippet (comments annotated):

```zig
const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
var w = try W.fromBlocks(&ctx, .{ 2, fucina.q8_0_block_size }, &blocks);
defer w.deinit();

var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, fucina.q8_0_block_size }, &x_values);
defer x.deinit();

var y = try x.dot(&ctx, &w, .in); // y: .{ .batch, .out } — f32 out, int8 dots inside
defer y.deinit();

var row = try w.getRows(&ctx, .out, &.{1}, .seq); // f32 gather of weight row 1
defer row.deinit();
```

(The `getRows` line is the embedding pattern in miniature: pull out and widen exactly the rows you need — for a token-embedding table, the tokens in flight — while the table itself stays quantized.)

What happens inside is the part worth understanding. The weights are int8-with-scales; multiplying them against f32 activations lane-by-lane would waste the integer format. So the runtime **quantizes the activations too, on the fly**, into the block format that matches the weight family (`docs/REFERENCE.md` §10.5): Q8_0 (per-32 absmax) for the 32-element weight formats, Q8_K (per-256, f32 scale) for all 256-element formats, Q8_1 for the offset formats. Then the inner loop is a pure integer dot per block — exactly the shape of our course-code `dotBlocks` (compile-checked, and its test bounds the error analytically):

```zig
/// The core of every quantized matmul: an integer dot product,
/// rescaled once per block by the product of the two scales.
fn dotBlocks(w: *const BlockQ8, a: *const BlockQ8) f32 {
    var acc: i32 = 0;
    for (w.qs, a.qs) |wq, aq| acc += @as(i32, wq) * @as(i32, aq);
    return w.d * a.d * @as(f32, @floatFromInt(acc));
}
```

This is why "weights are static, activations are dynamic" is the standard vocabulary: weights were quantized once at export; activations are re-quantized per row, per forward pass, because their values change every call. The activation formats even carry extra fields shaped by the weight algebra. Look at what rides along in the Q8_K activation block (`src/dtype.zig:139–143`; field comments added):

```zig
pub const BlockQ8_K = extern struct {
    d: f32,                                 // per-256 scale (f32 — activations get the precision)
    qs: [qk_k_block_size]i8,                // 256 quantized activations
    bsums: [qk_k_block_size / 16]i16,       // 16 per-16-element partial sums
};
```

Those `bsums` are precomputed partial sums of the activations. Why store them? Because the *weight* formats need them: an asymmetric K-quant reconstructs `x ≈ d·sc·q − dmin·m`, and when you expand the dot product, the minimum term multiplies `Σa` over each sub-block — a value that depends only on the activations, computable once at quantization time instead of inside every weight row's inner loop. (`BlockQ8_1` plays the same trick for the 32-element offset formats, carrying `d·Σq` as an f16.) The activation format is shaped by the weight format's algebra — and the same `bsums` field is what makes [Chapter 14](14-the-low-bit-frontier.md)'s multiplication-free ternary kernel nearly free. In production the per-block loop is `sdot`/`vpdpbusd` SIMD instructions ([Chapter 6](06-going-fast-on-cpus.md)'s dual-arm pattern), but the algebra is these six lines.

**And gradients?** A quantized weight is a constant — it never receives one; there is no encoder in the backward direction. But the *LHS* gradient is fully supported: the backward node holds a view of the block data and dequantizes it transiently (`ConstRhsDotBackward`, `src/ag/backward.zig`; `docs/REFERENCE.md` §10.2). So you can backpropagate *through* a frozen quantized layer into trainable f32 parameters — precisely what [Chapter 15](15-training-llms-on-cpu.md)'s LoRA fine-tuning of a quantized GGUF does. What you cannot do is train the quantized weights themselves through this path (the straight-through-estimator op that *does* train ternary weights is Chapter 14's business).

### Below the facade: containers that borrow or own

Between the tensor facade and the kernels sits a small container layer (`src/backend/quant/types.zig`) that is worth a look purely as Zig craft. A weight-side container is just blocks plus dimensions — and since there are 27 formats, the containers are generated by a type-returning function (`src/backend/quant/types.zig:145–163`):

```zig
pub fn QuantizedRowsFor(comptime dtype: DType) type {
    return struct {
        /// Owning allocator, or null when `blocks` borrows external storage
        /// kept alive by the caller (e.g. packed ES genome blocks); deinit
        /// then frees nothing.
        allocator: ?Allocator,
        blocks: []dtype_mod.Storage(dtype),
        rows: usize,
        cols: usize,
        blocks_per_row: usize,

        const Self = @This();
        pub const format = formatForDType(dtype);
        pub const traits = matmulTraits(format);

        pub fn deinit(self: *Self) void {
            if (self.allocator) |allocator| allocator.free(self.blocks);
            self.* = undefined;
        }
        // ...
    };
}
```

Two idioms here carry the whole chapter's ownership story into the kernel tier. The `?Allocator` field is the borrow/own discriminant: `null` means the blocks belong to someone else — an mmap'd GGUF region, typically — and `deinit` frees nothing; a non-null allocator means the container owns its copy. One type serves both the zero-copy load path and owned buffers, and the difference is a single optional. And `self.* = undefined` after deinit poisons the container so use-after-free becomes loud in Debug builds instead of quietly reading stale data. Note also the comptime constants baked into each instantiation: `format` and `traits` are resolved per-dtype at compile time, so a kernel asking `Self.traits.block_size` pays nothing at runtime.

> **Zig note** — `QuantizedRowsFor(comptime dtype: DType) type` is Zig's entire generics story ([Chapter 4](04-axes-with-names.md)): a function that runs at compile time and returns a struct type. Twenty-seven formats get container types from one definition, each with its own `Storage(dtype)` element type — and formats that need bespoke handling simply don't use the generic (`QuantizedRowsQ4_0` is hand-written with a *non-optional* allocator, because that path never borrows). The related `PackedRhsFor(layout)` dispatcher is an exhaustive `switch` over the `PackedRhsLayout` enum (`src/backend/quant/types.zig:283–292`) — adding a layout member breaks every dispatch site until handled, the same "registry as enum" pattern as Chapter 6's backend switch.

### Packed RHS and the `RhsLifetime` promise

Two performance seams finish the picture. First, hot formats get **packed RHS layouts**: column-interleaved copies (`q8_0x4`, `q4_kx8`, `q4_kx2mmla`, `q5_kx8`, `q6_kx4`) arranged so the inner loop feeds the CPU's int8 dot instruction directly with 4 or 8 output columns per pass. The intended pattern for model code is pack once at load, `w.packRhs(ctx)` → `x.dotPacked(ctx, &packed, .in, .out)` per step (`docs/REFERENCE.md` §10.3) — the LLM weight wrappers of Chapter 12 do exactly this. `dotPacked` has **no gradient support** at all (`error.GradientQuantizedMatmulUnsupported` when the LHS requires grad): it is an inference fast path, and it says so rather than silently doing something slow.

Second, lifetime as an API type (`src/exec/quant_matmul.zig:22–36`):

```zig
pub const RhsLifetime = enum {
    /// Ordinary tensor/temporary storage. The backend may still use the GPU,
    /// but it must not cache an address-keyed wrap beyond this dispatch.
    transient,
    /// Caller guarantees the RHS bytes stay mapped at the same address for the
    /// process lifetime, or are registered device-resident storage ... A
    /// backend may cache address-keyed wraps.
    stable_process,
};
```

The problem this solves: a GPU backend would like to cache its device-side wrapper for a weight it sees every single token, keyed by the weight's address. That is only sound if the bytes at that address never move or change — true for mmap'd model weights, false for a pooled scratch buffer that gets recycled. So the *caller* declares it: the facade `dot` defaults to `.transient`; the LLM weight wrappers thread `.stable_process` for weights that live in the §11.3 mapping for the process lifetime. The doc comment is emphatic that this is "about storage stability, not whether the operand is a model weight" (`src/exec/quant_matmul.zig:43–45`) — a promise about addresses, expressible only by whoever owns the allocation. Ownership discipline, again, as an enum.

## 11.10 Byte-exact parity: the verification religion, applied to codecs

Fucina's encoders are operation-for-operation ports of ggml's `ggml-quants.c` — the same `nearestInt` rounding via the 1.5·2²³ magic constant, the same `makeQkx2Quants` scale/min grid search for Q4_K/Q5_K (`docs/REFERENCE.md` §10.6).

That magic constant is worth a second of appreciation for what "operation-for-operation" means: adding `12582912.0` (= 1.5·2²³) to a small f32 forces its fraction bits to hold the rounded integer directly — round-to-nearest-even, courtesy of IEEE 754's default rounding — after which a `@bitCast` and a mask extract it (`src/backend/quant/q8k.zig:650–655`). That is how ggml rounds, so Fucina rounds that way too, *even though* `@round` would be more idiomatic Zig — `@round` rounds halves away from zero and would differ on exactly the ties. Porting for byte parity means porting the arithmetic, not the intent. A port can still silently drift, so the claim is pinned, not trusted (all in-tree, run by `zig build test`):

- `src/backend/quant/encode_golden_test.zig` — goldens generated once by a C harness linking *ggml's own reference encoders*, over 8 adversarial input vectors (ramp, alternating, near-zero, wide-range, all-equal, denormals, random, zeros); the Zig encoders for Q4_K/Q5_K/Q6_K/Q4_1/Q5_0/Q5_1 match **byte-for-byte**, and the oracle was verified stable across three compiler/FP-contraction configurations.
- `src/backend/quant/cold_tests.zig` — ggml-golden dequantize fixtures reproduced **bit-for-bit** for every cold decode format (Q2_K/Q3_K, the IQ family, TQ*, MXFP4/NVFP4).
- The per-format hot-kernel tests — SIMD kernels checked against the scalar reference, the same referee pattern as [Chapter 6](06-going-fast-on-cpus.md).

Why insist on *byte*-exact rather than "close enough"? Because closeness cannot compose. If your Q4_K encoder rounds one code differently than llama.cpp's, a model you export will score slightly differently there, and every downstream comparison between the two runtimes becomes noise. Byte-exact makes "Fucina and llama.cpp agree on this file" a checkable fact — and it turns "we ported the encoder" from a claim into a falsifiable test. The same stance produced §11.4's byte-identical re-emit and §11.5's golden-pinned safetensors output. When [Chapter 16](16-the-craft.md) collects the library's verification practices, this is one of the load-bearing examples.

One guard completes the encoder contract: block encoders assume finite input, so `encodeF32` rejects any NaN/inf with `error.NonFiniteValue` — in release builds too, the same seam llama.cpp guards with `ggml_validate_row_data`. The check itself is a nice bit of Zig (`src/gguf.zig:1207–1221`):

```zig
/// One vectorizable pass: finite iff |x| < inf (NaN compares false).
fn allFinite(values: []const f32) bool {
    const lanes = 8;
    const V = @Vector(lanes, f32);
    const inf: V = @splat(std.math.inf(f32));
    var i: usize = 0;
    while (i + lanes <= values.len) : (i += lanes) {
        const v: V = values[i..][0..lanes].*;
        if (!@reduce(.And, @abs(v) < inf)) return false;
    }
    while (i < values.len) : (i += 1) {
        if (!std.math.isFinite(values[i])) return false;
    }
    return true;
}
```

`|x| < inf` is false for NaN (NaN compares false to everything) and false for ±inf — one vector compare catches both, eight lanes at a time.

## 11.11 The export tool: transcode, merge, and models bigger than RAM

Everything in this chapter assembles into one tool: `tools/export_gguf.zig` (`zig build export-gguf`), which "closes the train→export→serve-anywhere loop" (its own header comment). Three modes (`docs/REFERENCE.md` §12.4; shell comments condensed):

```sh
# (a) re-emit / transcode
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf Qwen3-0.6B-f16.gguf --out Qwen3-0.6B-Q4_K_S.gguf --dtype q4_k

# (b) merge Fucina LoRA adapters into dense base weights and re-emit
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf base-f16.gguf --adapters ckpt-dir --alpha 16 --out merged.gguf

# (c) shard-streaming PTQTP quantization: models far bigger than RAM,
#     one tensor at a time (docs/PTQTP.md — Chapter 14)
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf big-BF16.gguf --out big-ptqtp3.gguf --ptqtp=3
```

Each mode is a policy document as much as a feature, and the policies teach.

**Transcode is deliberately conservative** (`tools/export_gguf.zig:14–23`). Only matrix weights transcode — `n_dims >= 2`, name ends `.weight`, name not containing `norm` (llama.cpp's own convention; norms and 1-D tensors keep their stored type). Sources must be f32/f16/bf16: transcoding an already-quantized source would *chain-requantize* — decode Q6_K's error, then add Q4_K's error on top — so the global `--dtype` refuses with an error; re-emit quantized sources verbatim instead. And a tensor whose row length does not divide the target block size keeps its **source** dtype rather than falling back to a smaller quant the way `llama-quantize` does: "no extra quantization loss, small size cost" (`docs/REFERENCE.md` §12.4). Every one of those choices trades convenience for never silently degrading a model.

**One measured exception**: `--experts-dtype` overrides the target for MoE expert tensors (`*_exps.weight`) and *is* allowed to requantize pre-quantized sources — because shipped MoE GGUFs store experts pre-quantized, and "experts are exactly where shrinking bytes pays in decode bandwidth while the redundancy keeps quality risk lowest" (`tools/export_gguf.zig:24–33`). The policy bends exactly where the bandwidth arithmetic of §11.6 says the payoff is largest and the risk smallest — and the exception is documented at the flag that grants it.

**LoRA merge** folds trained adapters (safetensors, from `zig build finetune` — [Chapter 15](15-training-llms-on-cpu.md)) into base weights: `W' = W + (α/r)·B·A` via `lora.Adapter.mergeInto`/`mergeF16`. `--alpha` is *required* because the adapter file stores A and B but not the training-time α. Quantized bases error for the chain-requantize reason above: merge on a float base, transcode in a second pass. The result is a plain GGUF that any ggml-ecosystem runtime can serve — the loop the tool exists to close.

**Streaming: the payoff of computable offsets.** Mode (c) quantizes models "far bigger than RAM" on a small machine. The mechanism is the writer split we saw in §11.4: `declareTensor` records name/type/dims only — no bytes — and since `tensorByteLen` fixes every offset at declaration time, `beginStream` can write the *complete header immediately* and return a `DataStreamer` that accepts each tensor's bytes in declaration order, freeing each buffer before the next is produced (`src/gguf.zig:1116–1156`). Bytes must arrive in order, exact length (a mismatch is `Error.InvalidTensorInfo`), all of them — `finish` errors with `Error.TensorDataMissing` otherwise, and the streamed output is byte-identical to the buffered `finish` path (pinned in `src/gguf_tests.zig`). On the read side, source tensors arrive through the §11.3 mmap, `prefetch`ed before decode and `release`d after, so at any moment the tool holds one source tensor's f32 buffer plus its quantized output — "residency stays bounded no matter the model size" and the run ends with a peak-RSS report (`tools/export_gguf.zig:54–65`). What PTQTP itself computes — K ternary planes approximating each weight matrix — is [Chapter 14](14-the-low-bit-frontier.md)'s story; this chapter's contribution is that the container made the streaming trivial: *because a GGUF's directory is fully determined before any data is written, writing is a plan plus a stream.*

## 11.12 A field guide to the traps

Model I/O is a domain of sharp edges, and this chapter's subsystems document theirs precisely. Collected here — each is a documented contract, with the place it is stated:

**Lifetime and ownership**

- Everything a GGUF `File` hands out — metadata strings, vocab arrays, tensor data — is a slice into `File.bytes` and dies at `deinit`. Even `Array.stringSlices` allocates only the *outer* slice; the strings still borrow (`src/gguf.zig:155–162`).
- `copyAllMetadata`/`addMetaCopy` re-read the source file's bytes, so call them **before** `from.deinit()` or `takeMapping()`; after a mapping transfer, use `copyAllMetadataRaw` over the still-alive region (`docs/REFERENCE.md` §12.2).
- The GGUF `Writer` duplicates metadata and tensor *names* but **borrows tensor data until `finish` returns** (`docs/REFERENCE.md` §12.2). Free a source buffer early and `finish` serializes garbage.
- `fromBorrowedBlocks` means what it says: the blocks must outlive the tensor. `materialize` converts a borrowed tensor into an owned copy when the mapping is going away (`docs/REFERENCE.md` §10.2).
- safetensors `parse` borrows; only `parseOwned`/`load`/`loadMmap` own (`src/safetensors.zig:222–260`).

**Format contracts**

- GGUF dims are ne-order — a logical `[out, in]` matrix is `&.{ in, out }` at `addTensor`. Forgetting silently transposes shapes (`src/gguf.zig:1019–1023`).
- Parsed metadata integers are widened to `i64`; a wire `uint64 >= 2^63` fails the whole parse with `MetadataValueOutOfRange`. Width-sensitive passthrough must use the raw copy path (`docs/REFERENCE.md` §12.1.2).
- `encodeF32`: `dst.len` must equal `tensorByteLen`; block formats reject NaN/inf (`error.NonFiniteValue`) in release builds too; `dst` must satisfy the block struct's alignment; little-endian targets only (`docs/REFERENCE.md` §12.3).
- Shapes are logical, blocks are physical: `fromBlocks` errors when block count and shape disagree (`InvalidDataLength`) or when `k` is not a whole number of blocks (`InvalidShape`) — 256 for every K-quant path (`docs/REFERENCE.md` §10.2).
- Streaming writes must arrive **in declaration order**, exact length, all of them — a wrong length is `Error.InvalidTensorInfo`; writing past the declared set or finishing an incomplete stream is `Error.TensorDataMissing` (`src/gguf.zig:1140–1155`).
- The safetensors writer sorts; never depend on tensor order in a file you wrote (§11.5).

**Concurrency and caching**

- None of the §12 I/O types have internal locking: a parsed `File` is read-safe from many threads, but `deinit`, `takeMapping`, and every writer mutation need external serialization (`docs/REFERENCE.md` §12).
- `RhsLifetime.stable_process` is a promise about *storage stability*, not "this operand is a weight" — pooled or recycled storage must never claim it (`src/exec/quant_matmul.zig:43–45`).
- Split GGUFs: `takeMapping` returns `null` (one region cannot carry N mappings); part 1 carries all metadata (`src/gguf.zig:477–483`, `docs/REFERENCE.md` §12.1.1).

None of these is a "gotcha" in the pejorative sense: each is the visible edge of a real constraint — zero-copy needs lifetimes, wire formats need exact widths, caching needs stable addresses. The library's job, done well here, is to make the edge *documented and loud* rather than smooth and treacherous.

## 11.13 Where this leaves us

You can now read every byte of a model file, write one that llama.cpp will accept, and multiply by weights that never leave their compressed form. What this chapter deliberately did *not* do is assemble those weights into anything — that is [Chapter 12](12-a-transformer-from-scratch.md), where a GGUF's tensor directory becomes a transformer: embeddings looked up with `getRows`, projections packed once at load and driven with `dotPacked` under `.stable_process`, and a tokenizer built from the metadata's vocab arrays. [Chapter 13](13-inference-tricks.md) then leans on the same machinery from another angle — the Q8_0 activation codec maintaining quantized KV caches, borrowed expert blocks streaming a 142 GB mixture-of-experts through a 64 GB machine (README.md's headline demo, `docs/RUNNING-MODELS.md`). And [Chapter 14](14-the-low-bit-frontier.md) pushes the block-format idea to its current edge: ternary weights at 2.06 bits, kernels with no multiplications, and quantization *as* training. All of it stands on this chapter's two primitives: a byte-exact block struct, and a table of contents whose offsets you can compute before you have the bytes.

## What you now know

- A model file is tensors + metadata: GGUF is magic → version → KVs → tensor directory → aligned data; safetensors is a length-prefixed JSON header + one buffer. Everything else is bookkeeping.
- Parsers of untrusted files use checked arithmetic (`std.math.add/mul/cast`), validate hostile values *before* lossy widening, and turn bounds checks into self-diagnosing errors (the truncated-download message).
- mmap makes the file the memory: zero-copy slices, file-backed evictable pages, `prefetch`/`release` hints, and `takeMapping` as an explicit ownership handoff. Everything a `File` hands out dies at `deinit`.
- GGUF dims are ne-order (innermost first) — `logicalMatrixShape` undoes the transposition trap.
- Metadata surfaces as a tagged union with widened scalars (every int → `i64`) — a lossy convenience view, paired with a lossless raw path for byte-verbatim passthrough.
- The writer precomputes llama.cpp-exact padded offsets, emits version 3, keeps metadata byte-verbatim via the raw re-read path, and proves itself by *byte-identical re-emit* (`src/gguf_tests.zig`).
- safetensors in Fucina: F32/F16/BF16 core mapping only, sorted output (input order not preserved) golden-pinned against upstream, and atomic temp-file-then-rename saves.
- CPU decode is weight-bandwidth-bound (`docs/BENCHMARK.md`), so bits-per-weight is the speed knob: Q8_0 = 34 bytes per 32 weights (8.5 bpw) via absmax-scale blocks; K-quants amortize scale storage through 256-element super-blocks with packed sub-scales (Q4_K = 4.5 bpw).
- The block structs are `extern struct`s pinned by comptime size asserts — the wire format is the in-memory type; loading is reinterpreting mapped bytes.
- All 27 formats decode; `gguf.encodeF32` produces ten block formats (+ scalar casts) — the rest are decode-only (`error.EncoderUnavailable`).
- Quantized matmul = dynamic activation quantization (Q8_0/Q8_1/Q8_K per weight family) + integer dots + per-block rescale. Quantized weights are constants: LHS gradients flow through transient dequantization; the weights never receive grad, and `dotPacked` refuses grad outright.
- Pack once at load, `dotPacked` per step; `RhsLifetime.stable_process` is a caller promise about *address stability* that unlocks address-keyed caching.
- Encoders are held to byte-exact ggml parity by embedded goldens; the export tool's conservative transcode policy, LoRA merge, and shard-streaming quantization all sit on the same two primitives — `tensorByteLen` and the block struct.

## Explore the source

- `src/gguf.zig` — the whole GGUF story in under 1,300 lines: `Cursor`, mmap loaders, `takeMapping`, the writer with precomputed offsets, `DataStreamer`, `encodeF32`/`decodeF32`.
- `src/safetensors.zig` — JSON-header parsing with total validation, the sorting writer, `saveFileAtomic`.
- `src/dtype.zig:54–248` — every block struct and the comptime size asserts: the ABI contract, one screen tall.
- `src/backend/quant/q8k.zig` — Q8_0/Q8_K encode/decode: the 11-line reference quantizer and its NEON twin.
- `src/backend/quant/types.zig` — the format-trait table, `QuantizedRowsFor` (a type-returning function with the `?Allocator` borrow convention), packed RHS containers.
- `src/exec/quant_matmul.zig` — `RhsLifetime`, `QuantizedMatmulOptions`, and the dispatch tier between facade and kernels.
- `src/backend/quant/encode_golden_test.zig` — what "byte-exact ggml parity" looks like as a test file.
- `src/ag/backward.zig` — find `ConstRhsDotBackward` to see how an LHS gradient flows through a weight that stays quantized.
- `tools/export_gguf.zig` — the header comment alone is a lesson in documenting policy; then read `--ptqtp`'s streaming loop.
- `docs/REFERENCE.md` §10 and §12 — the full machine-verified API contract for everything this chapter summarized.

## Exercises

1. **Inspect a real model.** Using `fucina.gguf.File.loadMmap`, write a small tool that prints a GGUF's `general.architecture`, layer count, and every tensor's name, *logical* shape, wire type, and byte size. (No model handy? `finish` one with the §11.4 writer test as a template and inspect that.) Verify the total of your byte sizes plus padding matches the file size.
2. **Finish the course parser.** Extend §11.2's `Cursor` with `readString`, `readValue`, and the KV loop, and parse the metadata section of a real GGUF's header (mmap it, or slice the first few KB). Reproduce the "validate alignment before widening" special case, then feed your parser a file of random bytes and confirm every failure is an error return, never a crash.
3. **Round-trip re-emit.** Load any GGUF, copy all metadata with `copyAllMetadata`, re-add every tensor verbatim (`info.name`, `info.ggml_type`, `info.dims[0..info.n_dims]`, `info.data`), `finish`, and `cmp` against the original. Byte-identical? Now you have reproduced the discipline of `src/gguf_tests.zig` — and you will discover firsthand why `addTensor` takes ne-order dims.
4. **A Q4_0 encoder, held to parity.** Port ggml's `quantize_row_q4_0_ref` (18-byte blocks: f16 scale + 32 four-bit codes) as course code, then verify it against `fucina.gguf.encodeF32(.q4_0, ...)` byte-for-byte on a few hundred random rows. When a byte differs, find the rounding rule you got wrong — that hunt is the whole point of the exercise.
5. **(Hard) A streaming transcoder.** Using `declareTensor` + `beginStream` + `decodeF32`/`encodeF32`, write a bounded-memory f16→q8_0 transcoder that never holds more than one tensor's f32 buffer, applying the §11.11 eligibility policy (matrix `.weight` tensors, no `norm`, rows divisible by 32 — else keep source dtype). Check your output loads in Fucina *and* matches `zig build export-gguf -- --dtype q8_0` byte-for-byte. Instrument peak RSS to prove the bound.

---

[Previous: The guitar amp — real-time neural audio](10-the-guitar-amp.md) ·
[Next: A transformer from scratch](12-a-transformer-from-scratch.md)
