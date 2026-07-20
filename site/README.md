# Fucina — landing site

A single static page for the Fucina deep-learning library, in the same "forge" visual
language as the course videos (ember/iron palette, Archivo/Inter/JetBrains Mono, the drifting
grid + ember sparks background). It presents the library the way the repo does: a **CPU-first**
eager runtime and LLM inference engine whose distinctive mechanisms are **named axes**, the
**recycling refcounted buffer**, **no-tape autograd** with **exec scopes** (write the forward
once), the **matmul-only GPU offload** (Metal/CUDA at the op seam), the **validated model
families** (llama.cpp parity + benchmarks), and **gradient-free neuroevolution** — with the
code-line-highlight treatment from the videos. All code and claims are taken from the repo's
`README.md`, `docs/REFERENCE.md`, `docs/MEMORY-MODEL.md`, and `docs/GPU-OFFLOAD.md`.

## Files (self-contained, no build step)

```
index.html            the whole page (inline CSS + JS)
assets/
  fonts/              Archivo · Inter · JetBrains Mono (woff2 + fonts.css)
  fucina_logo.png     anvil mark (unused — favicon is an inline SVG, no on-page logo)
  code_*.js           tokenized code shots (hero, axes, buffer, replace, autograd)
snip/                 the .zig source of each code shot (for re-tokenizing/editing)
```

All paths are relative, so the site works from any base URL (repo root, `/docs`, a project
subpath, or a user/org site).

## Preview locally

```bash
cd site && python3 -m http.server 8000    # then open http://localhost:8000
```
(Opening `index.html` directly with `file://` also works.)

## Deploy on GitHub Pages — pick one

- **From `/docs` of the repo:** copy these files into `docs/` on the default branch, then
  *Settings → Pages → Build and deployment → Source: Deploy from a branch → main → /docs*.
- **`gh-pages` branch:** put these files at the root of a `gh-pages` branch and set Pages to
  serve that branch's root.
- **Dedicated site repo:** push to `matteo-grella.github.io` (served at the root) or any repo
  (served at `/<repo>/`). Relative paths handle the subpath automatically.

## Editing

- **Text/design:** edit `index.html` (palette vars live in `:root`; they mirror the videos'
  `brand.css`).
- **A code shot:** edit the matching `snip/<name>.zig`, then re-tokenize:
  `python3 ../videos/_shared/build_code.py snip/<name>.zig 1 40 assets/code_<NAME>.js <NAME>`.
  Line highlights are the index arrays passed to `renderCode(...)` near the bottom of `index.html`
  (and the hero's rotating `beats`).
