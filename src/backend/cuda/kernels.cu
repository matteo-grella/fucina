// Quantized (dequant-in-kernel) CUDA kernels for the -Dgpu=cuda provider.
//
// The mul_mm family is a CUDA translation of the vendored Metal kernel
// src/backend/metal/ggml_mul_mm.metal (itself adapted from llama.cpp's ggml
// Metal backend, MIT, Copyright (c) 2023-2026 The ggml authors): same block
// structs, same dequantization bit-logic, same CPU-built `fucina_qmm_tile`
// grouped dispatch, and the same numerics contract — weights dequantize to
// half, activations convert f32 -> half in shared memory, accumulation is
// f32 — so the Metal parity tolerances apply unchanged. Two compute cores are
// shipped behind the same tile-table ABI: the portable plain-FFMA fallback and
// a WMMA f16-input/f32-accumulate path selected on tensor-core devices. Both
// consume the exact same half-rounded shared operands; only the accumulation
// association differs.

//
// The gemv family (decode m <= 8) is a warp-per-row
// dequant-dot in f32 (no f16 rounding of weights: decode competes with the
// CPU int8 kernels under the same 5e-3 quant tier).
//
// Shipped as vendored PTX (tools/gen_cuda_ptx.sh, compute_70) loaded via
// cuModuleLoadData; NVRTC recompile of this source is the dev-loop fallback.

#include <cuda_fp16.h>
#include <mma.h>

#define QK_K 256
#define QK8_0 32

typedef struct __align__(2) {
    __half d;
    signed char qs[QK8_0];
} block_q8_0;

typedef struct __align__(2) {
    unsigned char ql[QK_K / 2];
    unsigned char qh[QK_K / 4];
    signed char scales[QK_K / 16];
    __half d;
} block_q6_K;

#define K_SCALE_SIZE 12
typedef struct __align__(2) {
    __half d;
    __half dmin;
    unsigned char scales[K_SCALE_SIZE];
    unsigned char qs[QK_K / 2];
} block_q4_K;

// One 32-row output tile of one expert group; must mirror QMMTile in
// backend/cuda.zig and metal.zig (i32 x4).
typedef struct {
    int expert;
    int base_row;
    int m;
    int tile_m;
} fucina_qmm_tile;

// --- Dequantizers: 16 consecutive elements (il = 16-element chunk within the
// block) into float regs. Bit logic verbatim from the vendored Metal kernel.

__device__ __forceinline__ void dequant_q8_0(const block_q8_0 *xb, int il, float *reg) {
    const signed char *qs = xb->qs;
    const float d = __half2float(xb->d);
#pragma unroll
    for (int i = 0; i < 16; i++) reg[i] = qs[i + 16 * il] * d;
}

__device__ __forceinline__ void dequant_q6_K(const block_q6_K *xb, int il, float *reg) {
    const float d_all = __half2float(xb->d);
    const unsigned short *ql = (const unsigned short *)xb->ql;
    const unsigned short *qh = (const unsigned short *)xb->qh;
    const signed char *scales = xb->scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    const float sc = scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;

    const unsigned kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0u : 0x30303030u) : (il > 0 ? 0x0C0C0C0Cu : 0x03030303u);
    const unsigned kmask2 = il > 1 ? 0xF0F0F0F0u : 0x0F0F0F0Fu;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const int shr_h = il > 2 ? 2 : 0;
    const int shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const int shr_l = il > 1 ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned low = (ql[2 * i] | (unsigned)(ql[2 * i + 1] << 16)) & kmask2;
        const unsigned high = (qh[2 * i] | (unsigned)(qh[2 * i + 1] << 16)) & kmask1;
        const unsigned q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[4 * i + 0] = dl0 * (float)(q & 0xFFu) - ml;
        reg[4 * i + 1] = dl1 * (float)(q & 0xFF00u) - ml;
        reg[4 * i + 2] = dl2 * (float)(q & 0xFF0000u) - ml;
        reg[4 * i + 3] = dl3 * (float)(q & 0xFF000000u) - ml;
    }
}

__device__ __forceinline__ void get_scale_min_k4_just2(int j, int k, const unsigned char *q, unsigned char *sc, unsigned char *mn) {
    if (j < 4) {
        *sc = q[j + 0 + k] & 63;
        *mn = q[j + 4 + k] & 63;
    } else {
        *sc = (q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2);
        *mn = (q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2);
    }
}

__device__ __forceinline__ void dequant_q4_K(const block_q4_K *xb, int il, float *reg) {
    const unsigned char *q = xb->qs;
    const int is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    unsigned char sc, mn;
    get_scale_min_k4_just2(is, il / 2, xb->scales, &sc, &mn);
    // Metal computes xb->d / 16.h in half: exact (power-of-two scale).
    const float d = il < 2 ? __half2float(xb->d) : __half2float(xb->d) / 16.f;
    const float minv = __half2float(xb->dmin);
    const float dl = d * sc;
    const float ml = minv * mn;
    const unsigned mask = il < 2 ? 0x0Fu : 0xF0u;
#pragma unroll
    for (int i = 0; i < 16; ++i) reg[i] = dl * (q[i] & mask) - ml;
}

// --- Grouped mul_mm ----------------------------------------------------------
// dst[base_row + r, c] = sum_k src1[base_row + r, k] * dequant(W[expert])[c, k]
// Tile: 64 weight rows (NR0) x 32 panel rows (NR1) per block, K in chunks of
// 32. 256 threads as a 16x16 grid; each thread owns a 2(m) x 4(n) microtile.
// Numerics: sa/sb hold HALF (the Metal contract), accumulate f32.

#define NR0 64
#define NR1 32
#define NK 32

template <typename block_q, int nl, void (*dequant)(const block_q *, int, float *)>
__device__ __forceinline__ void mul_mm_scalar_body(
    const char *__restrict__ src0,
    const float *__restrict__ src1,
    const fucina_qmm_tile *__restrict__ tiles,
    float *__restrict__ dst,
    int ne00, // K
    int ne01, // n_out
    unsigned long long nb01,
    unsigned long long nb02) {
    __shared__ __half sa[NR0][NK + 1];
    __shared__ __half sb[NR1][NK + 1];

    const fucina_qmm_tile tile = tiles[blockIdx.x];
    const int r0 = blockIdx.y * NR0;
    const int r1 = tile.tile_m * NR1;
    const unsigned long long row0 = (unsigned long long)tile.base_row + (unsigned long long)r1;

    const int nr0 = (ne01 - r0 < NR0) ? (ne01 - r0) : NR0; // valid weight rows
    const int nr1 = (tile.m - r1 < NR1) ? (tile.m - r1) : NR1; // valid panel rows

    const int t = threadIdx.y * 16 + threadIdx.x; // 0..255
    const char *wbase = src0 + nb02 * (unsigned long long)tile.expert;

    float acc[2][4];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[r][c] = 0.f;

    for (int k0 = 0; k0 < ne00; k0 += NK) {
        // Weights: 64 rows x 32 k = 128 16-element chunks; threads 0..127.
        if (t < NR0 * 2) {
            const int wr = t / 2;       // weight row within tile
            const int which = t & 1;    // which 16-chunk of this k-step
            const int row = r0 + (wr < nr0 ? wr : nr0 - 1); // clamp like Metal
            const int chunk = (k0 / 16) + which; // global 16-chunk index
            const block_q *xb = (const block_q *)(wbase + nb01 * (unsigned long long)row) + chunk / nl;
            float regs[16];
            dequant(xb, chunk % nl, regs);
            __half *dstsh = &sa[wr][16 * which];
#pragma unroll
            for (int i = 0; i < 16; i++) dstsh[i] = __float2half(regs[i]);
        }
        // Activations: 32 rows x 32 k halfs; 256 threads x 4 elements.
        {
            const int ar = t / 8;            // 0..31 panel row within tile
            const int ac = (t % 8) * 4;      // 0..28 column base
            const int row = (ar < nr1 ? ar : (nr1 - 1)); // clamp
            const float *y = src1 + (row0 + (unsigned long long)row) * (unsigned long long)ne00 + k0 + ac;
#pragma unroll
            for (int i = 0; i < 4; i++) sb[ar][ac + i] = __float2half(y[i]);
        }
        __syncthreads();

        const int mrow = threadIdx.y * 2; // panel-row base of this thread
        const int ncol = threadIdx.x * 4; // weight-row base of this thread
#pragma unroll
        for (int kk = 0; kk < NK; kk++) {
            const float a0 = __half2float(sb[mrow + 0][kk]);
            const float a1 = __half2float(sb[mrow + 1][kk]);
            const float w0 = __half2float(sa[ncol + 0][kk]);
            const float w1 = __half2float(sa[ncol + 1][kk]);
            const float w2 = __half2float(sa[ncol + 2][kk]);
            const float w3 = __half2float(sa[ncol + 3][kk]);
            acc[0][0] += a0 * w0;
            acc[0][1] += a0 * w1;
            acc[0][2] += a0 * w2;
            acc[0][3] += a0 * w3;
            acc[1][0] += a1 * w0;
            acc[1][1] += a1 * w1;
            acc[1][2] += a1 * w2;
            acc[1][3] += a1 * w3;
        }
        __syncthreads();
    }

    // Store the valid nr1 x nr0 region (per-element guards; panel dst has row
    // stride ne01, same as the Metal kernel).
    const int mrow = threadIdx.y * 2;
    const int ncol = threadIdx.x * 4;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int j = mrow + r;
        if (j >= nr1) continue;
        float *D = dst + (row0 + (unsigned long long)j) * (unsigned long long)ne01 + r0;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = ncol + c;
            if (i < nr0) D[i] = acc[r][c];
        }
    }
}

extern "C" __global__ void fucina_mul_mm_q8_0(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_scalar_body<block_q8_0, 2, dequant_q8_0>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}
extern "C" __global__ void fucina_mul_mm_q6_K(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_scalar_body<block_q6_K, 16, dequant_q6_K>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}
extern "C" __global__ void fucina_mul_mm_q4_K(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_scalar_body<block_q4_K, 16, dequant_q4_K>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}

// --- Tensor-core grouped mul_mm ---------------------------------------------
// The scalar kernel above already rounds dequantized weights and f32
// activations to half in shared memory. WMMA therefore changes no operand
// precision: four or eight warps cover a 32x32/32x64 output tile with
// independent 16x16x16 matrix fragments, accumulating into f32. The narrower
// n tile is retained for severely underfilled grids; the 64-column tile avoids
// duplicating activation loads on ordinary LLM shapes. A shared f32 epilogue
// preserves guarded stores on partial m/n edge tiles.

template <int mma_nr0, typename block_q, int nl, void (*dequant)(const block_q *, int, float *)>
__device__ __forceinline__ void mul_mm_mma_body(
    const char *__restrict__ src0,
    const float *__restrict__ src1,
    const fucina_qmm_tile *__restrict__ tiles,
    float *__restrict__ dst,
    int ne00,
    int ne01,
    unsigned long long nb01,
    unsigned long long nb02) {
    __shared__ __align__(32) __half sa[mma_nr0][NK];
    __shared__ __align__(32) __half sb[NR1][NK];
    __shared__ __align__(32) float sc[NR1][mma_nr0];

    const fucina_qmm_tile tile = tiles[blockIdx.x];
    const int r0 = blockIdx.y * mma_nr0;
    const int r1 = tile.tile_m * NR1;
    const unsigned long long row0 = (unsigned long long)tile.base_row + (unsigned long long)r1;
    const int nr0 = (ne01 - r0 < mma_nr0) ? (ne01 - r0) : mma_nr0;
    const int nr1 = (tile.m - r1 < NR1) ? (tile.m - r1) : NR1;

    const int t = threadIdx.y * 16 + threadIdx.x;
    const int warp = t >> 5;
    const int warps_n = mma_nr0 / 16;
    const int warp_m = (warp / warps_n) * 16;
    const int warp_n = (warp % warps_n) * 16;
    const int threads = 4 * mma_nr0;
    const char *wbase = src0 + nb02 * (unsigned long long)tile.expert;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < ne00; k0 += NK) {
        if (t < mma_nr0 * 2) {
            const int wr = t / 2;
            const int which = t & 1;
            const int row = r0 + (wr < nr0 ? wr : nr0 - 1);
            const int chunk = (k0 / 16) + which;
            const block_q *xb = (const block_q *)(wbase + nb01 * (unsigned long long)row) + chunk / nl;
            float regs[16];
            dequant(xb, chunk % nl, regs);
            __half *dstsh = &sa[wr][16 * which];
#pragma unroll
            for (int i = 0; i < 16; i++) dstsh[i] = __float2half(regs[i]);
        }
        {
            const int values_per_thread = (NR1 * NK) / threads;
            const int flat = t * values_per_thread;
            const int ar = flat / NK;
            const int ac = flat - ar * NK;
            const int row = ar < nr1 ? ar : nr1 - 1;
            const float *y = src1 + (row0 + (unsigned long long)row) * (unsigned long long)ne00 + k0 + ac;
#pragma unroll
            for (int i = 0; i < values_per_thread; i++) sb[ar][ac + i] = __float2half(y[i]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < NK; kk += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> bf;
            nvcuda::wmma::load_matrix_sync(af, &sb[warp_m][kk], NK);
            nvcuda::wmma::load_matrix_sync(bf, &sa[warp_n][kk], NK);
            nvcuda::wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }

    // Full LLM tiles can go straight to the row-major output. CUDA allocations
    // are at least 256-bit aligned; ne01/r0/warp_n preserve that alignment when
    // ne01 is a multiple of eight floats. Partial edge tiles retain the shared
    // epilogue below so no WMMA store can cross a logical row boundary.
    if (nr0 == mma_nr0 && nr1 == NR1 && (ne01 & 7) == 0) {
        float *D = dst + (row0 + (unsigned long long)warp_m) * (unsigned long long)ne01 + r0 + warp_n;
        nvcuda::wmma::store_matrix_sync(D, acc, ne01, nvcuda::wmma::mem_row_major);
        return;
    }

    nvcuda::wmma::store_matrix_sync(&sc[warp_m][warp_n], acc, mma_nr0, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int index = t; index < mma_nr0 * NR1; index += threads) {
        const int j = index / mma_nr0;
        const int i = index - j * mma_nr0;
        if (j < nr1 && i < nr0) {
            dst[(row0 + (unsigned long long)j) * (unsigned long long)ne01 + r0 + i] = sc[j][i];
        }
    }
}

extern "C" __global__ void fucina_mul_mm_mma_q8_0(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_mma_body<64, block_q8_0, 2, dequant_q8_0>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}
extern "C" __global__ void fucina_mul_mm_mma_q6_K(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_mma_body<64, block_q6_K, 16, dequant_q6_K>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}
extern "C" __global__ void fucina_mul_mm_mma_q4_K(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_mma_body<64, block_q4_K, 16, dequant_q4_K>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}

extern "C" __global__ void fucina_mul_mm_mma_n32_q8_0(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_mma_body<32, block_q8_0, 2, dequant_q8_0>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}
extern "C" __global__ void fucina_mul_mm_mma_n32_q6_K(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_mma_body<32, block_q6_K, 16, dequant_q6_K>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}
extern "C" __global__ void fucina_mul_mm_mma_n32_q4_K(
    const char *src0, const float *src1, const fucina_qmm_tile *tiles, float *dst,
    int ne00, int ne01, unsigned long long nb01, unsigned long long nb02) {
    mul_mm_mma_body<32, block_q4_K, 16, dequant_q4_K>(src0, src1, tiles, dst, ne00, ne01, nb01, nb02);
}

// --- Decode GEMV (m <= 8) ----------------------------------------------------
// One warp per output row; lanes stride 16-element chunks of the row, f32
// dequant + f32 dot against up to 8 activation rows simultaneously.
// y[j, r] = sum_k dequant(W)[r, k] * x[j, k]   (dst row-major [m, n_out])

#define GEMV_WARPS 4

template <typename block_q, int nl, void (*dequant)(const block_q *, int, float *)>
__device__ __forceinline__ void gemv_body(
    const char *__restrict__ src0,
    const float *__restrict__ x,
    float *__restrict__ y,
    int ne00, // K
    int ne01, // n_out
    int m,    // 1..8 activation rows
    unsigned long long nb01) {
    const int row = blockIdx.x * GEMV_WARPS + (threadIdx.x >> 5);
    if (row >= ne01) return;
    const int lane = threadIdx.x & 31;
    const int nchunks = ne00 / 16;

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = 0.f;

    const block_q *brow = (const block_q *)(src0 + nb01 * (unsigned long long)row);
    for (int c = lane; c < nchunks; c += 32) {
        float w[16];
        dequant(brow + c / nl, c % nl, w);
        const float *xc = x + 16 * c;
        for (int j = 0; j < m; j++) {
            const float *xj = xc + (unsigned long long)j * (unsigned long long)ne00;
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < 16; i++) s += w[i] * xj[i];
            acc[j] += s;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] += __shfl_down_sync(0xffffffffu, acc[j], off);
    if (lane == 0) {
        for (int j = 0; j < m; j++) y[(unsigned long long)j * (unsigned long long)ne01 + row] = acc[j];
    }
}

// --- Prefill attention (f16 KV) ----------------------------------------------
// Fused online-softmax grouped attention, semantics identical to the CPU
// tiled kernel (exec/attention.zig): query row i at absolute position
// p = source_offset + i attends keys [max(0, p+1-window), p] when causal
// (window 0 = full causal, pre-clamped host-side), or all kv_seq keys when
// bidirectional. Layouts: q/o [q_seq, heads, d] f32, k/v [kv_seq, kv_heads,
// d] f16, kvmap [heads] -> kv head. One warp per query row per head; f32
// accumulate; d <= 256.

#define ATTN_WARPS 4
#define ATTN_MAX_D 256

extern "C" __global__ void fucina_attn_f16(
    const float *__restrict__ q,
    const __half *__restrict__ k,
    const __half *__restrict__ v,
    float *__restrict__ o,
    const int *__restrict__ kvmap,
    int q_seq, int kv_seq, int heads, int kv_heads, int d,
    int source_offset, float scale, int window, int causal) {
    __shared__ float sh_q[ATTN_WARPS][ATTN_MAX_D];
    __shared__ float sh_p[ATTN_WARPS][32];

    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qi = blockIdx.x * ATTN_WARPS + w;
    if (qi >= q_seq) return; // warp-uniform exit; no block-wide syncs below
    const int h = blockIdx.y;
    const int kvh = kvmap[h];

    // Stage the (pre-scaled) query row once per warp.
    const float *qrow = q + ((size_t)qi * heads + h) * d;
    for (int t = lane; t < d; t += 32) sh_q[w][t] = qrow[t] * scale;
    __syncwarp();

    const int p_abs = source_offset + qi;
    const int end = causal ? (p_abs + 1 < kv_seq ? p_abs + 1 : kv_seq) : kv_seq;
    const int start = (causal && window > 0 && p_abs + 1 > window) ? (p_abs + 1 - window) : 0;

    const int nd = (d + 31) / 32;
    float acc[ATTN_MAX_D / 32];
#pragma unroll
    for (int t = 0; t < ATTN_MAX_D / 32; t++) acc[t] = 0.f;
    float m_run = -3.4e38f;
    float l_run = 0.f;

    for (int j0 = start; j0 < end; j0 += 32) {
        const int j = j0 + lane;
        float s = -3.4e38f;
        if (j < end) {
            const __half *krow = k + ((size_t)j * kv_heads + kvh) * d;
            float dot = 0.f;
            // d is even for every supported head_dim; half2 loads halve the
            // instruction count of this latency-bound inner product.
            const __half2 *krow2 = (const __half2 *)krow;
#pragma unroll 4
            for (int t = 0; t < d / 2; t++) {
                const float2 kk = __half22float2(krow2[t]);
                dot += sh_q[w][2 * t] * kk.x + sh_q[w][2 * t + 1] * kk.y;
            }
            if (d & 1) dot += sh_q[w][d - 1] * __half2float(krow[d - 1]);
            s = dot;
        }
        // Warp max of this tile, then online-softmax rescale.
        float m_tile = s;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, off));
        const float m_new = fmaxf(m_run, m_tile);
        const float p = (j < end) ? expf(s - m_new) : 0.f;
        const float rescale = (m_run > -3.0e38f) ? expf(m_run - m_new) : 0.f;
        float p_sum = p;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            p_sum += __shfl_xor_sync(0xffffffffu, p_sum, off);
        l_run = l_run * rescale + p_sum;
        m_run = m_new;

        sh_p[w][lane] = p;
        __syncwarp();
        const int jn = (end - j0 < 32) ? (end - j0) : 32;
        for (int t = 0; t < nd; t++) {
            const int dim = lane + t * 32;
            float a = acc[t] * rescale;
            if (dim < d) {
                for (int jj = 0; jj < jn; jj++) {
                    const float pj = sh_p[w][jj];
                    a += pj * __half2float(v[((size_t)(j0 + jj) * kv_heads + kvh) * d + dim]);
                }
            }
            acc[t] = a;
        }
        __syncwarp();
    }

    float *orow = o + ((size_t)qi * heads + h) * d;
    const float inv_l = (l_run > 0.f) ? 1.f / l_run : 0.f;
    for (int t = 0; t < nd; t++) {
        const int dim = lane + t * 32;
        if (dim < d) orow[dim] = acc[t] * inv_l;
    }
}

extern "C" __global__ void fucina_gemv_q8_0(
    const char *src0, const float *x, float *y, int ne00, int ne01, int m, unsigned long long nb01) {
    gemv_body<block_q8_0, 2, dequant_q8_0>(src0, x, y, ne00, ne01, m, nb01);
}
extern "C" __global__ void fucina_gemv_q6_K(
    const char *src0, const float *x, float *y, int ne00, int ne01, int m, unsigned long long nb01) {
    gemv_body<block_q6_K, 16, dequant_q6_K>(src0, x, y, ne00, ne01, m, nb01);
}
extern "C" __global__ void fucina_gemv_q4_K(
    const char *src0, const float *x, float *y, int ne00, int ne01, int m, unsigned long long nb01) {
    gemv_body<block_q4_K, 16, dequant_q4_K>(src0, x, y, ne00, ne01, m, nb01);
}

// --- Evolution-strategies kernels (fucina.es device arm) --------------------
//
// Exact device port of the ES noise contract: src/rng.zig gaussianFillAtFast
// (counter-based splitmix64 pairs + f32 polynomial Box-Muller) and the
// perturb/update/anchored-weight-decay element math of src/es.zig. Every
// float step uses explicit round-to-nearest intrinsics so the compiler can
// never contract mul+add into FMA — the kernels are BITWISE identical to
// the CPU path for any launch geometry (each pair is a pure function of its
// counter index), which is what keeps checkpoints and the parity suites
// device-independent. Grid-stride, one thread per Box-Muller pair.

#define FUCINA_SPLITMIX_GAMMA 0x9E3779B97F4A7C15ULL

static __device__ __forceinline__ unsigned long long fucina_splitmix_at(
    unsigned long long seed, unsigned long long i) {
    unsigned long long z = seed + (i + 1ULL) * FUCINA_SPLITMIX_GAMMA;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Cephes logf polynomial over (0, 1] — op order pinned to rng.zig lnPoly.
static __device__ __forceinline__ float fucina_ln_poly(float x) {
    unsigned int bits = __float_as_uint(x);
    float e = (float)((int)(bits >> 23) - 126);
    float m = __uint_as_float((bits & 0x007fffffu) | 0x3f000000u);
    if (m < 0.70710678118654752440f) {
        e = __fsub_rn(e, 1.0f);
        m = __fadd_rn(m, m);
    }
    float f = __fsub_rn(m, 1.0f);
    float z = __fmul_rn(f, f);
    float p = 7.0376836292e-2f;
    p = __fadd_rn(__fmul_rn(p, f), -1.1514610310e-1f);
    p = __fadd_rn(__fmul_rn(p, f), 1.1676998740e-1f);
    p = __fadd_rn(__fmul_rn(p, f), -1.2420140846e-1f);
    p = __fadd_rn(__fmul_rn(p, f), 1.4249322787e-1f);
    p = __fadd_rn(__fmul_rn(p, f), -1.6668057665e-1f);
    p = __fadd_rn(__fmul_rn(p, f), 2.0000714765e-1f);
    p = __fadd_rn(__fmul_rn(p, f), -2.4999993993e-1f);
    p = __fadd_rn(__fmul_rn(p, f), 3.3333331174e-1f);
    float y = __fmul_rn(__fmul_rn(f, z), p);
    y = __fadd_rn(y, __fmul_rn(e, -2.12194440e-4f));
    y = __fsub_rn(y, __fmul_rn(0.5f, z));
    y = __fadd_rn(f, y);
    return __fadd_rn(y, __fmul_rn(e, 0.693359375f));
}

// sin/cos of a full turn via quadrant reduction — rng.zig sinCosTurn.
static __device__ __forceinline__ void fucina_sincos_turn(float u, float* s_out, float* c_out) {
    float t4 = __fmul_rn(u, 4.0f);
    float jf = floorf(__fadd_rn(t4, 0.5f));
    float y = __fsub_rn(t4, jf);
    float arg = __fmul_rn(y, 1.5707963267948966f);
    float z = __fmul_rn(arg, arg);

    float sp = -1.9515295891e-4f;
    sp = __fadd_rn(__fmul_rn(sp, z), 8.3321608736e-3f);
    sp = __fadd_rn(__fmul_rn(sp, z), -1.6666654611e-1f);
    float sin_arg = __fadd_rn(arg, __fmul_rn(__fmul_rn(arg, z), sp));

    float cp = 2.443315711809948e-5f;
    cp = __fadd_rn(__fmul_rn(cp, z), -1.388731625493765e-3f);
    cp = __fadd_rn(__fmul_rn(cp, z), 4.166664568298827e-2f);
    float cos_arg = __fadd_rn(__fsub_rn(1.0f, __fmul_rn(0.5f, z)), __fmul_rn(__fmul_rn(z, z), cp));

    unsigned int q = ((unsigned int)(int)jf) & 3u;
    float sin_base = (q & 1u) ? cos_arg : sin_arg;
    float cos_base = (q & 1u) ? sin_arg : cos_arg;
    *s_out = (q & 2u) ? -sin_base : sin_base;
    *c_out = (((q + 1u) & 2u) != 0u) ? -cos_base : cos_base;
}

// The Box-Muller pair at even counter index `pair_index` (unscaled).
static __device__ __forceinline__ void fucina_gauss_pair(
    unsigned long long seed, unsigned long long pair_index, float* even, float* odd) {
    unsigned long long a = fucina_splitmix_at(seed, pair_index);
    unsigned long long b = fucina_splitmix_at(seed, pair_index + 1ULL);
    float ua = __double2float_rn(((double)(a >> 11) + 1.0) * 0x1.0p-53);
    float ub = __double2float_rn((double)(b >> 11) * 0x1.0p-53);
    float radius = __fsqrt_rn(__fmul_rn(-2.0f, fucina_ln_poly(ua)));
    float s, c;
    fucina_sincos_turn(ub, &s, &c);
    *even = __fmul_rn(radius, c);
    *odd = __fmul_rn(radius, s);
}

// data[j] += narrow(scaled * eps(j)) with widen-add-narrow (es.zig perturbSlot).
#define FUCINA_ES_PERTURB(NAME, T, WIDEN, NARROW)                                       \
    extern "C" __global__ void NAME(T* data, unsigned long long seed, float scaled,     \
                                    unsigned long long n) {                             \
        unsigned long long pairs = (n + 1ULL) / 2ULL;                                   \
        for (unsigned long long p = (unsigned long long)blockIdx.x * blockDim.x +       \
                                    threadIdx.x;                                        \
             p < pairs; p += (unsigned long long)gridDim.x * blockDim.x) {              \
            float even, odd;                                                            \
            fucina_gauss_pair(seed, 2ULL * p, &even, &odd);                             \
            unsigned long long j = 2ULL * p;                                            \
            T t0 = NARROW(__fmul_rn(scaled, even));                                     \
            data[j] = NARROW(__fadd_rn(WIDEN(data[j]), WIDEN(t0)));                     \
            if (j + 1ULL < n) {                                                         \
                T t1 = NARROW(__fmul_rn(scaled, odd));                                  \
                data[j + 1] = NARROW(__fadd_rn(WIDEN(data[j + 1]), WIDEN(t1)));         \
            }                                                                           \
        }                                                                               \
    }

#define FUCINA_WIDEN_F16(x) __half2float(x)
#define FUCINA_NARROW_F16(x) __float2half_rn(x)
#define FUCINA_WIDEN_F32(x) (x)
#define FUCINA_NARROW_F32(x) (x)

FUCINA_ES_PERTURB(fucina_es_perturb_f16, __half, FUCINA_WIDEN_F16, FUCINA_NARROW_F16)
FUCINA_ES_PERTURB(fucina_es_perturb_f32, float, FUCINA_WIDEN_F32, FUCINA_NARROW_F32)

// data[j] += narrow(scale * sum_s coeffs[s]*eps_s(j)), fp32 accumulation in
// stream order (es.zig updateSlot; antithetic folding happens host-side).
#define FUCINA_ES_UPDATE(NAME, T, WIDEN, NARROW)                                        \
    extern "C" __global__ void NAME(T* data, const unsigned long long* seeds,           \
                                    const float* coeffs, unsigned int n_streams,        \
                                    float scale, unsigned long long n) {                \
        unsigned long long pairs = (n + 1ULL) / 2ULL;                                   \
        for (unsigned long long p = (unsigned long long)blockIdx.x * blockDim.x +       \
                                    threadIdx.x;                                        \
             p < pairs; p += (unsigned long long)gridDim.x * blockDim.x) {              \
            float acc0 = 0.0f;                                                          \
            float acc1 = 0.0f;                                                          \
            for (unsigned int s = 0; s < n_streams; s++) {                              \
                float even, odd;                                                        \
                fucina_gauss_pair(seeds[s], 2ULL * p, &even, &odd);                     \
                acc0 = __fadd_rn(acc0, __fmul_rn(coeffs[s], even));                     \
                acc1 = __fadd_rn(acc1, __fmul_rn(coeffs[s], odd));                      \
            }                                                                           \
            unsigned long long j = 2ULL * p;                                            \
            T d0 = NARROW(__fmul_rn(scale, acc0));                                      \
            data[j] = NARROW(__fadd_rn(WIDEN(data[j]), WIDEN(d0)));                     \
            if (j + 1ULL < n) {                                                         \
                T d1 = NARROW(__fmul_rn(scale, acc1));                                  \
                data[j + 1] = NARROW(__fadd_rn(WIDEN(data[j + 1]), WIDEN(d1)));         \
            }                                                                           \
        }                                                                               \
    }

FUCINA_ES_UPDATE(fucina_es_update_f16, __half, FUCINA_WIDEN_F16, FUCINA_NARROW_F16)
FUCINA_ES_UPDATE(fucina_es_update_f32, float, FUCINA_WIDEN_F32, FUCINA_NARROW_F32)

// Anchored weight decay, the reference's per-op rounding chain
// (es.zig anchorSlot): d = narrow(p - a); l2: narrow(widen(d)*keep);
// l1: soft-threshold at decay_step; then narrow(shrunk + a).
#define FUCINA_ES_ANCHOR(NAME, T, WIDEN, NARROW)                                        \
    extern "C" __global__ void NAME(T* data, const T* anchor, float decay_step,         \
                                    int is_l1, unsigned long long n) {                  \
        float keep = __fsub_rn(1.0f, decay_step);                                       \
        for (unsigned long long j = (unsigned long long)blockIdx.x * blockDim.x +       \
                                    threadIdx.x;                                        \
             j < n; j += (unsigned long long)gridDim.x * blockDim.x) {                  \
            float aw = WIDEN(anchor[j]);                                                \
            T d = NARROW(__fsub_rn(WIDEN(data[j]), aw));                                \
            if (is_l1) {                                                                \
                float dw = WIDEN(d);                                                    \
                T th = NARROW(__fsub_rn(fabsf(dw), decay_step));                        \
                float cl = fmaxf(WIDEN(th), 0.0f);                                      \
                float shrunk = (dw < 0.0f) ? -cl : cl;                                  \
                data[j] = NARROW(__fadd_rn(shrunk, aw));                                \
            } else {                                                                    \
                T kept = NARROW(__fmul_rn(WIDEN(d), keep));                             \
                data[j] = NARROW(__fadd_rn(WIDEN(kept), aw));                           \
            }                                                                           \
        }                                                                               \
    }

FUCINA_ES_ANCHOR(fucina_es_anchor_f16, __half, FUCINA_WIDEN_F16, FUCINA_NARROW_F16)
FUCINA_ES_ANCHOR(fucina_es_anchor_f32, float, FUCINA_WIDEN_F32, FUCINA_NARROW_F32)
