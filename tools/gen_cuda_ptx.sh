#!/bin/sh
# Regenerate the committed PTX for the -Dgpu=cuda provider's kernels
# (src/backend/cuda/kernels.ptx from kernels.cu). Needs any Linux box with a
# CUDA toolkit (nvcc); the default build never runs this — the PTX is a
# vendored artifact like the generated unicode tables, and the driver JIT
# compiles it at runtime.
#
#   tools/gen_cuda_ptx.sh            # regenerate in place
#
# compute_70 keeps every tensor-core GPU since Volta in play; the CUDA toolkit
# supplies cuda_fp16.h and mma.h to both nvcc and the optional NVRTC dev path.
set -eu
cd "$(dirname "$0")/.."

NVCC=${NVCC:-nvcc}
CCBIN=${CCBIN:-}
ARCH=${ARCH:-compute_70}

ccbin_flag=""
if [ -n "$CCBIN" ]; then
    ccbin_flag="-ccbin $CCBIN"
fi

# shellcheck disable=SC2086
"$NVCC" -ptx -arch="$ARCH" -std=c++17 -O3 $ccbin_flag \
    src/backend/cuda/kernels.cu -o src/backend/cuda/kernels.ptx
echo "wrote src/backend/cuda/kernels.ptx ($(wc -c < src/backend/cuda/kernels.ptx) bytes, $ARCH)"
