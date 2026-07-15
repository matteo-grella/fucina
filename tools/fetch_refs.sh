#!/bin/sh
# Fetch the reference implementations that the benchmark and parity records
# compare against, pinned to the exact commits of the published snapshot.
# Everything lands under refs/, which is gitignored: references are cloned
# and benchmarked against, never vendored.
#
#   tools/fetch_refs.sh                 # clone/pin every reference
#   tools/fetch_refs.sh llama.cpp zinc  # just the named ones
#   tools/fetch_refs.sh --build         # also build llama.cpp CPU-only
#   tools/fetch_refs.sh --patch         # apply tools/ref-patches/ dump patches
#
# The llama.cpp CPU build keeps each platform's production defaults, which is
# what the records were measured against: Accelerate BLAS stays ON on macOS,
# plain native CPU elsewhere (no CUDA/Metal).
#
# --patch applies the instrumentation patches under tools/ref-patches/
# (currently: parakeet.cpp tensor-dump hooks). They exist ONLY to regenerate
# parity dumps; benchmarks always run stock pinned references — never patch
# a reference you are about to benchmark.
#
# locate-anything.cpp builds stock (cmake, ggml submodule initialized below);
# its parity dumps come from tools/ref-patches/la_dump.cpp, an out-of-tree
# harness compiled AGAINST the stock build (build line in that file's header)
# rather than a patch applied to it.
#
# nanochat (+ rustbpe, its tokenizer trainer) is the reference for
# examples/nanochat. It stays stock too: the parity dumps come from
# examples/nanochat/tools/nanochat_dump.py and nanochat_export.py, out-of-tree
# harnesses run with the stock pinned checkout on PYTHONPATH (invocation lines
# in their headers; regen recipe in examples/nanochat/goldens/README.md).
#
# ds4 is the architecture reference and validation oracle for the deepseek4
# port. It stays stock — no patches: examples/deepseek4.zig consumes its
# SHIPPED fixtures in place (--vectors refs/ds4/tests/test-vectors/official,
# --golden refs/ds4/tests/test-vectors/local-golden.vec). Do NOT run its
# CPU build path (`make cpu`) on macOS — it can kernel-panic the VM system.
#
# colibri is a design reference only (the out-of-core MoE expert streaming
# was inspired by it): pinned for the record, never built or benchmarked.
#
# ik_llama.cpp is a design reference only (the CUDA quantized-prefill
# split-K occupancy adaptation was audited from its MMQ implementation):
# pinned for the record, never built or benchmarked.
#
# cartridges (HazyResearch) is the semantics reference for the Cartridges
# port (src/llm/cartridge.zig, examples/cartridge.zig, docs/CARTRIDGES.md):
# the distillation loss (train.py), the five seed-prompt meta-texts
# (data resources), the token chunker, and the KV-init recipe were audited
# against it. It stays stock and is never run: numerical parity comes from
# tools/gen_cartridge_goldens.py, an INDEPENDENT PyTorch implementation of
# the mechanism (torch 2.12) whose output is committed as
# src/llm/cartridge_golden_tests.zig.
#
# engram (deepseek-ai, Apache-2.0) is the semantics reference for the Engram
# port (src/llm/engram.zig, docs/ENGRAM.md): engram_demo_v1.py defines the
# tokenizer-compression table, layer-seeded odd multipliers, the
# multiply-XOR n-gram hash with per-head prime table sizes, and the
# gated multi-head-embedding + causal short-conv module. It stays stock
# and is never run: numerical parity comes from tools/gen_engram_goldens.py,
# an INDEPENDENT PyTorch implementation of the demo semantics (torch 2.12)
# whose output is committed as src/llm/engram_golden_tests.zig.
set -eu
cd "$(dirname "$0")/.."
mkdir -p refs

# prism-llama.cpp (PrismML's llama.cpp fork, prism branch) is the parity
# oracle for the Ternary-Bonsai-27B port (Q2_0 g128 ternary weights on the
# qwen35 arch): tools/llama_logits.cpp compiles against its CPU build for
# the logits gate, llama-tokenize for the token-ID gate. bonsai-demo pins
# the whitepapers + run recipes. Both stock, never patched.

# name|url|pinned commit (the snapshot's reference state)
REFS='llama.cpp|https://github.com/ggml-org/llama.cpp|30af6e2b98b00eee01a8f76249fe1399a724702e
prism-llama.cpp|https://github.com/PrismML-Eng/llama.cpp|62061f91088281e65071cc38c5f69ee95c39f14e
bonsai-demo|https://github.com/PrismML-Eng/Bonsai-demo|cfd842af57d7f458d5a4ea28312f1dc62e02395e
locate-anything.cpp|https://github.com/mudler/locate-anything.cpp|92c1682da792c1e8a5dec91acc2be4b02c742ded
parakeet.cpp|https://github.com/mudler/parakeet.cpp|89f5e2977b4d8bccd45e7bcc6f2ef7c4ed49e89a
face-detect.cpp|https://github.com/mudler/face-detect.cpp|e22260d5d5490b37b021b7f795079f386d553afd
omnivoice.cpp|https://github.com/ServeurpersoCom/omnivoice.cpp|0f37401bebe9b20c0160a888e592108fc1d17607
nanochat|https://github.com/karpathy/nanochat|92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
rustbpe|https://github.com/karpathy/rustbpe|ddf848f6961a0655dc8693742fc338e5682c0d3b
zinc|https://github.com/zolotukhin/zinc|986c2390bdf337d1fb46aa611e12ab1b7a74a05e
NeuralAmpModelerCore|https://github.com/sdatkinson/NeuralAmpModelerCore|e49c93e678549230d09efbb0beeb50511e387874
neural-amp-modeler|https://github.com/sdatkinson/neural-amp-modeler|a11ed88a128031c306faba79878eade51a209c48
ds4|https://github.com/antirez/ds4|80ebbc35237f77e51ce7e57970ba9a6a112c4faa
colibri|https://github.com/JustVugg/colibri|a5fc89e88f113fc9d1c9d8752861b158d7c303e7
es-at-scale|https://github.com/VsonicV/es-at-scale|574a9d134da1ffce2a8bb812019899e5c96b588a
es-awd|https://github.com/kschweig/es-awd|f432ff823a7d59f91d4ac2cf99e4923654c6f464
ik_llama.cpp|https://github.com/ikawrakow/ik_llama.cpp|b90939934add9ba4fbb37e8c6470809a70b78f0a
cartridges|https://github.com/HazyResearch/cartridges|ef34ba97a06049c34820506e2c283746284ae5f0
engram|https://github.com/deepseek-ai/Engram|fb7f84a21f91223715394a33a1dc24bbfb7f788e'

build_llama=0
apply_patches=0
selected=""
for arg in "$@"; do
    case "$arg" in
    --build) build_llama=1 ;;
    --patch) apply_patches=1 ;;
    *) selected="$selected $arg" ;;
    esac
done

echo "$REFS" | while IFS='|' read -r name url pin; do
    if [ -n "$selected" ]; then
        case " $selected " in *" $name "*) ;; *) continue ;; esac
    fi
    if [ ! -d "refs/$name/.git" ]; then
        echo "cloning refs/$name ..."
        git clone --quiet "$url" "refs/$name"
    fi
    git -C "refs/$name" fetch --quiet origin
    git -C "refs/$name" checkout --quiet "$pin"
    if [ -f "refs/$name/.gitmodules" ]; then
        git -C "refs/$name" submodule --quiet update --init
    fi
    echo "refs/$name @ $(git -C "refs/$name" rev-parse --short HEAD)  ($url)"
    if [ "$apply_patches" = 1 ]; then
        for pf in "tools/ref-patches/$name"-*.patch; do
            [ -e "$pf" ] || continue
            if git -C "refs/$name" apply --check "../../$pf" 2>/dev/null; then
                git -C "refs/$name" apply "../../$pf"
                echo "  patched: $pf"
            else
                echo "  skipped: $pf (already applied or does not match)"
            fi
        done
    fi
done

if [ "$build_llama" = 1 ]; then
    echo "building llama.cpp CPU-only into refs/llama.cpp/build-cpu ..."
    cmake -S refs/llama.cpp -B refs/llama.cpp/build-cpu \
        -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DLLAMA_CURL=OFF
    cmake --build refs/llama.cpp/build-cpu -j \
        --target llama-bench llama-cli llama-tokenize
    echo "done: refs/llama.cpp/build-cpu/bin/"
fi
