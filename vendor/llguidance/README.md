# Vendored llguidance

[llguidance](https://github.com/guidance-ai/llguidance) is the constrained-
decoding engine behind Fucina's grammar/JSON-schema token masking
(`llm.llguidance`, enabled with `-Dllguidance=true` — see
[REFERENCE.md](../../docs/REFERENCE.md)). It is a Rust library; this
directory vendors its two crates so the build needs no clone of the upstream
repository.

| | |
|---|---|
| Upstream | <https://github.com/guidance-ai/llguidance> |
| Version | 1.7.6 |
| Commit | `e98236e877125522028223ad5a86caa752874fb6` (2026-06-25) |
| License | MIT (`LICENSE`, Microsoft Corporation) |

## Layout

- `parser/` — the `llguidance` crate: grammar compiler (Lark variant, JSON
  schema, regex), Earley parser, token-mask computation, and the C FFI
  (`parser/src/ffi.rs`; `parser/llguidance.h` is the checked-in generated
  header, kept byte-verbatim from upstream).
- `toktrie/` — the token-trie crate `parser` depends on.
- `Cargo.toml` — a minimal Fucina-local workspace over the two crates (not an
  upstream file).
- `Cargo.lock` — pins the exact crates.io versions of every transitive
  dependency. Committed so builds are reproducible; cargo fetches these
  SHA-verified packages from crates.io on first build.

Crate sources (`src/`, `llguidance.h`, `build.rs`, `LICENSE`, `README.md`)
are byte-verbatim upstream copies. Only the two `Cargo.toml` manifests
deviate:

- `toktrie` is a path dependency instead of a workspace dependency.
- Dropped: `[dev-dependencies]`, `[[bench]]` (their source dirs are not
  vendored), the `cbindgen` build-dependency and `generate-header` feature
  (the generated header is checked in), and the optional `rayon`,
  `jsonschema_validation` and `wasm` features with their dependencies.
- Default features are `lark` + `referencing` (upstream adds `rayon` — only
  used by the parallel multi-sequence `llg_par_compute_mask`, which Fucina
  does not call and which would embed a second thread pool — and `ahash`, a
  regex-compilation perf nicety; both stay available as opt-ins).
- `crate-type` drops `cdylib` (Fucina links the static library only).
- `rust-version = "1.87"` is inlined (upstream inherits it from its
  workspace; same value), and a `[lints.rust] unexpected_cfgs = "allow"`
  entry silences check-cfg warnings for the dropped features' `#[cfg]`
  gates.

## Building

`zig build <step> -Dllguidance=true` runs, via `build.zig`:

```sh
cargo build --release --package llguidance
```

in this directory and links
`vendor/llguidance/target/release/libllguidance.a`. Requires a Rust
toolchain >= 1.87. `target/` is git-ignored.

## Going fully offline (committing all dependencies)

The tree currently vendors only llguidance's own crates; crates.io
dependencies are fetched (SHA-pinned by `Cargo.lock`) on first build. To make
the build fully hermetic (~25 MB extra), run in this directory:

```sh
cargo vendor vendor
mkdir -p .cargo
cat > .cargo/config.toml <<'EOF'
[source.crates-io]
replace-with = "vendored-sources"
[source.vendored-sources]
directory = "vendor"
EOF
```

commit `vendor/` and `.cargo/`, remove the `/vendor/` line from
`.gitignore`, and add `--offline` to the cargo invocation in `build.zig`.

## Updating to a new upstream version

1. Clone upstream at the new tag/commit.
2. Re-copy `parser/{src,llguidance.h,build.rs,LICENSE,README.md}` and
   `toktrie/{src,LICENSE,README.md}` verbatim.
3. Re-apply the manifest deviations above to the new upstream
   `Cargo.toml`s (diff against this tree's manifests).
4. `rm Cargo.lock && cargo generate-lockfile`, then
   `cargo build --release --package llguidance`.
5. If `parser/llguidance.h` changed, re-check the hand-written extern
   declarations in `src/llm/llguidance.zig` against it (struct layouts and
   signatures; the gated tests include an ABI smoke check).
6. Update the version/commit table above.
