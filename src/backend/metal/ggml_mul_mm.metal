// Quantized (dequant-in-kernel) Metal GEMM — vendored/adapted from llama.cpp's
// ggml Metal backend (MIT, Copyright (c) 2023-2026 The ggml authors):
// `kernel_mul_mm` (legacy simdgroup path) +
// `dequantize_q6_K`/`dequantize_q4_K`/`dequantize_q8_0` from
// ggml/src/ggml-metal/ggml-metal.metal, block structs from
// ggml/src/ggml-common.h (refs/llama.cpp @ 30af6e2b9).
//
// Differences from upstream:
//  - Grouped-by-tile dispatch: a CPU-built `fucina_qmm_tile` table replaces
//    both plain batching and the mul_mm_id map0/ids machinery — Fucina already
//    count-sorts (token, expert) pairs on the CPU, so every threadgroup gets
//    exact work (no early-exit waste, no GPU row mapping pass).
//  - src1 (activations, f32) and dst (f32) are packed row panels in shim-owned
//    staging buffers: src1 row stride = K, dst row stride = n_out. Weights are
//    the raw GGUF blocks (zero-copy page wrap), row-major [n_out, K] per
//    expert with uniform byte strides nb01 (row) / nb02 (expert).
//  - No broadcast (ne12/r2/r3) and no bc_inp arm: K must be a multiple of 32
//    and a whole number of blocks (always true for whole quantized rows).
//
// Numerics match upstream: weights dequantize to half, activations convert
// f32 -> half in threadgroup memory, accumulation is f32 (simdgroup_float8x8).

#include <metal_stdlib>

using namespace metal;

#define FUCINA_FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef half ggml_half;

#define QK_K 256
#define QK8_0 32

typedef struct {
    ggml_half d;        // delta
    int8_t qs[QK8_0];   // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

// 6-bit quantization: 16 sub-blocks of 16 elements, x = d * scale * q.
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K/16 + 3*QK_K/4, "wrong q6_K block size/padding");

// 4-bit quantization: 8 sub-blocks of 32 elements, x = d * scale * q - dmin * min
// (upstream stores d/dmin in a ggml_half2 union; plain fields here, same layout).
#define K_SCALE_SIZE 12
typedef struct {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// Dequantize 16 consecutive elements (`il` = 16-element chunk within the
// block) into a half4x4 register. Verbatim from upstream.
template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// Verbatim from upstream (the q4_K packed scale/min extraction).
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// One 32-row output tile of one expert group. The CPU builds the full list:
// for every expert e with m_e > 0 rows, ceil(m_e/32) entries that all share
// (expert, base_row, m). Rows are panel rows: src1 row `base_row + i` and dst
// row `base_row + i` belong to expert `expert` for i < m.
// Must mirror FucinaQMMTile in shim.m and metal.QMMTile in metal.zig.
typedef struct {
    int32_t expert;     // weight index: src0 += expert * nb02
    int32_t base_row;   // first panel row of this expert's group
    int32_t m;          // rows in this expert's group
    int32_t tile_m;     // which 32-row tile within the group
} fucina_qmm_tile;

// Must mirror FucinaQMMArgs in shim.m (C layout).
typedef struct {
    int32_t  ne00;  // K; multiple of 32 and a whole number of blocks
    int32_t  ne01;  // n_out = weight rows per expert; multiple of 4
    uint64_t nb01;  // weight row stride, bytes
    uint64_t nb02;  // weight expert stride, bytes
} fucina_qmm_args;

// dst[base_row + r, c] = sum_k src1[base_row + r, k] * dequant(W[expert])[c, k]
// Upstream tile geometry: 64 weight rows x 32 panel rows per threadgroup,
// 128 threads = 4 simdgroups, K consumed in chunks of 32.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void fucina_mul_mm_q(
        constant fucina_qmm_args & args,
        device const char  * src0,
        device const float * src1,
        device const fucina_qmm_tile * tiles,
        device float       * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const fucina_qmm_tile tile = tiles[tgpig.x];

    const int r0 = tgpig.y*NR0;
    const int r1 = tile.tile_m*NR1;        // row offset within the expert group
    const ulong row0 = (ulong)tile.base_row + (ulong)r1; // global panel row

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne01 - r0 < NR0) ? (short)(args.ne01 - r0) : (short)NR0;
    const short nr1 = (tile.m    - r1 < NR1) ? (short)(tile.m    - r1) : (short)NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const short offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0
        + args.nb02*(ulong)tile.expert
        + args.nb01*(ulong)(r0 + lr0)) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = src1 + (row0 + (ulong)lr1)*(ulong)args.ne00 + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // dequantize a 16-element chunk of the weight row and store it to
        // threadgroup memory in the simdgroup-tile layout
        {
            half4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FUCINA_FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // activations: f32 -> half, 8 elements per thread
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (half2x4)(*((device const float2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const half * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb = (sb + 2*64*(sgitg/2));

        FUCINA_FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FUCINA_FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FUCINA_FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FUCINA_FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= args.ne01 && nr1 == NR1) {
        // full tile: store the 8x8 accumulators straight to device memory
        device float * C = dst
            + (ulong)(r0 + 32*(sgitg &  1))
            + (row0 + (ulong)(16*(sgitg >> 1))) * (ulong)args.ne01;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*(ulong)args.ne01*(i/4), (ulong)args.ne01, 0, false);
        }
    } else {
        // partial tile: stage in threadgroup memory, then copy the valid
        // nr1 x nr0 region row by row
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = dst + (ulong)r0 + (row0 + (ulong)j)*(ulong)args.ne01;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = ((threadgroup float *) shmem) + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

typedef decltype(fucina_mul_mm_q<block_q8_0, 2, dequantize_q8_0>) fucina_mul_mm_q_t;

template [[host_name("fucina_mul_mm_q8_0_f32")]] kernel fucina_mul_mm_q_t fucina_mul_mm_q<block_q8_0, 2,  dequantize_q8_0>;
template [[host_name("fucina_mul_mm_q6_K_f32")]] kernel fucina_mul_mm_q_t fucina_mul_mm_q<block_q6_K, 16, dequantize_q6_K>;
template [[host_name("fucina_mul_mm_q4_K_f32")]] kernel fucina_mul_mm_q_t fucina_mul_mm_q<block_q4_K, 16, dequantize_q4_K>;
