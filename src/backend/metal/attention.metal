// Grouped causal attention FORWARD — the GPU arm of exec/attention.zig's
// `groupedCausalAttentionDispatch`. Reduction idioms (simd_max/simd_sum
// lane reductions, fma dot accumulation, threadgroup score staging) follow
// llama.cpp's ggml Metal flash-attention kernel (MIT, Copyright (c)
// 2023-2026 The ggml authors; refs/llama.cpp
// ggml/src/ggml-metal/ggml-metal.metal `kernel_flash_attn_ext`), but the
// contract is Fucina's CPU kernel's, not ggml's: causal/window bounds are
// computed analytically per row (no mask buffer), GQA is the uniform
// heads_per_kv mapping, softmax statistics {row max, sum of exp} are
// captured for the training backward, and the math is f32 accumulation
// throughout. The kernel is templated over the K/V element type — f32
// (training graphs) and half (the inference f16 KV cache, widened per
// load) instantiate the same body.
//
// Shape: one threadgroup per (query row, head), FUCINA_ATTN_THREADS
// threads. Scores for the row's live key span live in dynamic threadgroup
// memory (index 0, kv_seq floats — the host gates kv_seq against the 32 KB
// threadgroup budget). Pass 1 computes scaled q.k dots and the running
// max; pass 2 exponentiates and sums; pass 3 contracts P.V with the
// 4-slab / 64-lane split so every thread stays busy at d <= 64 and larger
// heads run in 64-column chunks. Summation order differs from the CPU
// kernels (the documented ~1e-6 relative class shared by every kernel
// tier); a simdgroup-matrix upgrade in ggml's style is the recorded
// follow-up if the scalar-dot form shows up in a trace.
//
// Must mirror FucinaAttnArgs in shim.m (C layout).

#define FUCINA_ATTN_THREADS 256
#define FUCINA_ATTN_SIMDGROUPS (FUCINA_ATTN_THREADS / 32)
#define FUCINA_ATTN_PV_SLABS 4
#define FUCINA_ATTN_PV_LANES (FUCINA_ATTN_THREADS / FUCINA_ATTN_PV_SLABS)

typedef struct {
    int32_t q_seq;
    int32_t kv_seq;
    int32_t heads;
    int32_t kv_heads;
    int32_t d;
    int32_t source_offset;
    int32_t window;
    int32_t causal;
    int32_t heads_per_kv;
    int32_t has_stats;
    float scale;
} fucina_attn_args;

template <typename KVT>
kernel void fucina_attention_fwd(
    constant fucina_attn_args & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const KVT * k [[buffer(2)]],
    device const KVT * v [[buffer(3)]],
    device float * out [[buffer(4)]],
    device float * stats [[buffer(5)]],
    threadgroup float * scores [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]) {

    threadgroup float red[FUCINA_ATTN_SIMDGROUPS];
    threadgroup float pvred[FUCINA_ATTN_THREADS];

    const int q_i = (int)tgpig.x;
    const int head_i = (int)tgpig.y;
    const int d = args.d;
    const int q_stride = args.heads * d;
    const int kv_stride = args.kv_heads * d;
    const int p_abs = args.source_offset + q_i;
    const int hi = args.causal != 0 ? min(p_abs + 1, args.kv_seq) : args.kv_seq;
    const int lo = (args.causal == 0 || args.window == 0) ? 0 : max(0, p_abs + 1 - args.window);
    const int live = hi - lo;
    const int kv_base = (head_i / args.heads_per_kv) * d;
    device const float * qrow = q + q_i * q_stride + head_i * d;
    device float * orow = out + q_i * q_stride + head_i * d;

    if (live <= 0) { // windowed corner; a causal row always sees >= 1 key
        for (int c = (int)tid; c < d; c += FUCINA_ATTN_THREADS) orow[c] = 0.0f;
        if (tid == 0 && args.has_stats != 0) {
            const int sb = (head_i * args.q_seq + q_i) * 2;
            stats[sb] = 0.0f;
            stats[sb + 1] = 0.0f;
        }
        return;
    }

    // Pass 1: scaled scores + running max. One SIMDGROUP per key row, lanes
    // striding consecutive columns, so the K loads coalesce into full-width
    // transactions (a thread-per-row mapping puts adjacent lanes whole rows
    // apart and runs latency-bound on scalar loads — measured ~25x slower).
    float m_local = -INFINITY;
    for (int j = (int)sgitg; j < live; j += FUCINA_ATTN_SIMDGROUPS) {
        device const KVT * krow = k + (lo + j) * kv_stride + kv_base;
        float partial = 0.0f;
        for (int c = (int)tiisg; c < d; c += 32) partial = fma(qrow[c], (float)krow[c], partial);
        const float s = simd_sum(partial) * args.scale; // broadcast to every lane
        if (tiisg == 0) scores[j] = s;
        m_local = max(m_local, s);
    }
    m_local = simd_max(m_local);
    if (tiisg == 0) red[sgitg] = m_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_row = red[0];
    for (short i = 1; i < FUCINA_ATTN_SIMDGROUPS; ++i) m_row = max(m_row, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup); // red reuse below

    // Pass 2: exponentiate in place + row sum.
    float s_local = 0.0f;
    for (int j = (int)tid; j < live; j += FUCINA_ATTN_THREADS) {
        const float e = exp(scores[j] - m_row);
        scores[j] = e;
        s_local += e;
    }
    s_local = simd_sum(s_local);
    if (tiisg == 0) red[sgitg] = s_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_row = 0.0f;
    for (short i = 0; i < FUCINA_ATTN_SIMDGROUPS; ++i) sum_row += red[i];
    if (tid == 0 && args.has_stats != 0) {
        const int sb = (head_i * args.q_seq + q_i) * 2;
        stats[sb] = m_row;
        stats[sb + 1] = sum_row;
    }
    const float inv = 1.0f / sum_row;

    // Pass 3: O = (P.V) / sum. 4 slabs stride the key span, 64 lanes cover
    // one 64-column chunk of d; lanes read consecutive v columns, so the
    // per-key loads coalesce across the threadgroup.
    const short lane = (short)(tid % FUCINA_ATTN_PV_LANES);
    const short slab = (short)(tid / FUCINA_ATTN_PV_LANES);
    for (int c0 = 0; c0 < d; c0 += FUCINA_ATTN_PV_LANES) {
        const int cw = min((int)FUCINA_ATTN_PV_LANES, d - c0);
        float acc = 0.0f;
        if ((int)lane < cw) {
            for (int j = (int)slab; j < live; j += FUCINA_ATTN_PV_SLABS) {
                acc = fma(scores[j], (float)v[(lo + j) * kv_stride + kv_base + c0 + lane], acc);
            }
        }
        pvred[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slab == 0 && (int)lane < cw) {
            float o = pvred[lane];
            for (short s = 1; s < FUCINA_ATTN_PV_SLABS; ++s) o += pvred[s * FUCINA_ATTN_PV_LANES + lane];
            orow[c0 + lane] = o * inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // pvred reuse next chunk
    }
}

template [[host_name("fucina_attention_fwd_f32")]] kernel decltype(fucina_attention_fwd<float>) fucina_attention_fwd<float>;
template [[host_name("fucina_attention_fwd_f16kv")]] kernel decltype(fucina_attention_fwd<half>) fucina_attention_fwd<half>;
