#!/bin/sh
# Regenerate the committed PTX for the -Dgpu=cuda provider's kernels
# (src/backend/cuda/kernels.ptx from kernels.cu). Needs any Linux box with a
# CUDA toolkit (NVRTC); the default build never runs this — the PTX is a
# vendored artifact like the generated unicode tables, and the driver JIT
# compiles it at runtime. NVRTC deliberately matches FUCINA_GPU_KERNELS=src:
# CUDA 12.0 NVCC emitted a measurably slower module from the same source.
#
#   tools/gen_cuda_ptx.sh            # regenerate in place
#
# compute_70 keeps every tensor-core GPU since Volta in play; the CUDA toolkit
# supplies cuda_fp16.h and mma.h to both nvcc and the optional NVRTC dev path.
set -eu
cd "$(dirname "$0")/.."

ZIG=${ZIG:-zig}
ARCH=${ARCH:-compute_70}

"$ZIG" run -lc --dep cuda_api -Mroot=tools/gen_cuda_ptx.zig \
    -Mcuda_api=src/backend/cuda/api.zig -- \
    src/backend/cuda/kernels.cu src/backend/cuda/kernels.ptx "$ARCH"
echo "wrote src/backend/cuda/kernels.ptx ($(wc -c < src/backend/cuda/kernels.ptx) bytes, $ARCH)"
