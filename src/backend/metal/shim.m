// Fucina Metal GEMM shim — the entire Objective-C surface of -Dgpu=metal.
//
// One opaque context = MTLDevice + queue + MTLLibrary (the MLX steel GEMM
// source, compiled once at init) + a lazy pipeline cache keyed on
// (transpose variant, M/N/K tile-alignment function constants). The exported
// C ABI is consumed by src/backend/metal.zig as plain `extern fn`s.
//
// Buffer strategy (Apple Silicon unified memory): wrap the host operand's
// containing pages with newBufferWithBytesNoCopy (zero copy; the kernel gets
// buffer + byte offset). GPU stores stay inside [c, c+m*n) — the surrounding
// page bytes are mapped but never written. If a wrap fails the call returns
// nonzero and the caller falls back to the CPU path, so correctness never
// depends on the GPU.
//
// Synchronization: the native tensor path commits immediately and returns an
// owned completion ticket; CPU visibility waits through that ticket. The
// direct slice ABI retains the old commit+wait behavior for parity/benchmarks.
// Batched GEMM = ONE dispatch with grid depth = batch via the params batch
// strides (the kernel adds batch_stride_* * tid.z itself).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <os/lock.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// Must mirror the MSL GEMMParams declaration order exactly (C++ layout).
typedef struct {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t lda;
    int32_t ldb;
    int32_t ldd;
    int32_t tiles_n;
    int32_t tiles_m;
    size_t  batch_stride_a;
    size_t  batch_stride_b;
    size_t  batch_stride_d;
    int32_t swizzle_log;
    int32_t gemm_k_iterations_aligned;
    int32_t batch_ndim;
} FucinaGEMMParams;

// variant: 0 = nn, 1 = tn (A stored [k,m]), 2 = nt (B stored [n,k]).
enum { FUCINA_GEMM_NN = 0, FUCINA_GEMM_TN = 1, FUCINA_GEMM_NT = 2 };
// dtype: 0 = f32 operands/output; 1 = f16 operands + f16 output staging;
// 2 = f16 operands + direct f32 output; 3 = bf16 operands + direct f32
// output (needs __HAVE_BFLOAT__ in the shader — pipeline lookup fails
// gracefully on older toolchains and the caller falls back to CPU). Steel
// accumulates in f32 for all.
enum {
    FUCINA_GEMM_F32 = 0,
    FUCINA_GEMM_F16 = 1,
    FUCINA_GEMM_F16_F32 = 2,
    FUCINA_GEMM_BF16_F32 = 3,
};

// Quantized-weights GEMM formats (ggml_mul_mm.metal). Must mirror
// metal.QFormat in metal.zig.
enum { FUCINA_QFMT_Q8_0 = 0, FUCINA_QFMT_Q6_K = 1, FUCINA_QFMT_Q4_K = 2, FUCINA_QFMT_TQ2_0 = 3 };
#define FUCINA_QMM_FORMATS 4

// Must mirror fucina_qmm_args in ggml_mul_mm.metal (C layout).
typedef struct {
    int32_t  ne00; // K
    int32_t  ne01; // n_out (weight rows per expert)
    uint64_t nb01; // weight row stride, bytes
    uint64_t nb02; // weight expert stride, bytes
} FucinaQMMArgs;

// Must mirror fucina_qmm_tile in ggml_mul_mm.metal and metal.QMMTile.
typedef struct {
    int32_t expert;
    int32_t base_row;
    int32_t m;
    int32_t tile_m;
} FucinaQMMTile;

typedef struct {
    uint64_t gpu_ns;
    uint64_t sched_ns;
} FucinaCommandTiming;

// ARC-owned wrappers returned through the C ABI.  A storage wrapper lives for
// the backing Fucina buffer's lifetime; a ticket lives from eager submission
// until that output is synchronized or discarded.
@interface FucinaMetalWrap : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) uintptr_t base;
@property(nonatomic) uintptr_t end;
@end
@implementation FucinaMetalWrap
@end

@interface FucinaMetalTicket : NSObject
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) id<MTLBuffer> aBuffer;
@property(nonatomic, strong) id<MTLBuffer> bBuffer;
@property(nonatomic, strong) id<MTLBuffer> cBuffer;
@end
@implementation FucinaMetalTicket
@end

#define FUCINA_GEMM_VARIANTS 3
#define FUCINA_GEMM_DTYPES 4
#define FUCINA_GEMM_PIPELINES (FUCINA_GEMM_DTYPES * FUCINA_GEMM_VARIANTS * 8)
#define FUCINA_WRAP_CACHE 512

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    // Lazily built pipelines: [dtype][variant][align_m][align_n][align_k].
    id<MTLComputePipelineState> pipelines[FUCINA_GEMM_PIPELINES];
    // Guards lazy pipeline construction. f32 dispatches are not caller-serialized
    // and f16/qmoe use different caller locks, so cache slots need their own lock.
    os_unfair_lock pipeline_lock;
    // Grow-only output staging for the legacy blocking f16 entry. Valid until
    // its next call; the caller holds f16_lock across dispatch + CPU widen.
    // The eager async entry uses the mixed f16-input/f32-output steel variant
    // and binds tensor output storage directly, so it never touches this.
    id<MTLBuffer> f16_out;
    size_t f16_out_cap;
    // Wrap cache for the f16 RHS operand: a fresh bytesNoCopy wrap costs
    // ~32 us/MB of GPU residency mapping on its FIRST dispatch (measured:
    // 10 ms for a 311 MB RHS, 0.2 ms once cached) — re-wrapping a 1.5 GB
    // lm head every call would dominate the GEMM. Only the f16 B operand is
    // cached: resident-f16 RHS storage lives for the whole process, so the cache
    // can never go stale unless an owner explicitly frees a resident buffer
    // during teardown (pool-churned activations are NOT cached).
    struct { uintptr_t base; uintptr_t end; } wrap_keys[FUCINA_WRAP_CACHE];
    id<MTLBuffer> wrap_bufs[FUCINA_WRAP_CACHE];
    // Guards wrap_keys/wrap_bufs: lookups run from eager async dispatch,
    // f16_lock/qmoe_lock legacy paths, and concurrent resident allocation
    // during model loading. calloc zero-init == OS_UNFAIR_LOCK_INIT.
    os_unfair_lock wrap_lock;
    size_t page_size;
    // Quantized grouped GEMM (ggml_mul_mm.metal): one pipeline per format,
    // grow-only shared staging for the activation/result panels (the CPU
    // gathers/reads through .contents, so per-call page wraps — and their
    // first-dispatch residency cost — never apply to the panels) and for the
    // tile table. All guarded by the caller's qmoe lock (metal.zig).
    id<MTLComputePipelineState> qmm_pipelines[FUCINA_QMM_FORMATS];
    id<MTLBuffer> qmoe_in;
    size_t qmoe_in_cap;
    id<MTLBuffer> qmoe_out;
    size_t qmoe_out_cap;
    id<MTLBuffer> qmm_tiles;
    size_t qmm_tiles_cap;
    char device_name[256];
} FucinaMetalCtx;

static const char *fucina_gemm_fn_names[FUCINA_GEMM_DTYPES][FUCINA_GEMM_VARIANTS] = {
    {
        "gemm_nn_f32_f32_32_32_16_2_2",
        "gemm_tn_f32_f32_32_32_16_2_2",
        "gemm_nt_f32_f32_32_32_16_2_2",
    },
    {
        "gemm_nn_f16_f16_32_32_16_2_2",
        "gemm_tn_f16_f16_32_32_16_2_2",
        "gemm_nt_f16_f16_32_32_16_2_2",
    },
    {
        "gemm_nn_f16_f32_32_32_16_2_2",
        "gemm_tn_f16_f32_32_32_16_2_2",
        "gemm_nt_f16_f32_32_32_16_2_2",
    },
    {
        "gemm_nn_bf16_f32_32_32_16_2_2",
        "gemm_tn_bf16_f32_32_32_16_2_2",
        "gemm_nt_bf16_f32_32_32_16_2_2",
    },
};

static uint64_t fucina_seconds_to_ns(CFTimeInterval seconds) {
    return seconds > 0.0 ? (uint64_t)(seconds * 1000000000.0) : 0;
}

static void fucina_record_timing(FucinaCommandTiming *timing, id<MTLCommandBuffer> cmd) {
    if (timing == NULL) {
        return;
    }
    timing->gpu_ns = fucina_seconds_to_ns(cmd.GPUEndTime - cmd.GPUStartTime);
    timing->sched_ns = fucina_seconds_to_ns(cmd.kernelEndTime - cmd.kernelStartTime);
}

void *fucina_metal_init(const char *msl_source) {
    @autoreleasepool {
        // MTLCreateSystemDefaultDevice consults the WindowServer and returns
        // nil in plain CLI/SSH sessions; MTLCopyAllDevices still enumerates
        // the GPU there (Apple Silicon has exactly one).
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
            if (all.count > 0) {
                device = all[0];
            }
        }
        if (device == nil) {
            return NULL;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            return NULL;
        }
        NSError *error = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:msl_source]
                                 options:opts
                                   error:&error];
        if (library == nil) {
            NSLog(@"fucina-metal: shader compile failed: %@", error);
            return NULL;
        }
        FucinaMetalCtx *ctx = calloc(1, sizeof(FucinaMetalCtx));
        if (ctx == NULL) {
            return NULL;
        }
        ctx->device = device;
        ctx->queue = queue;
        ctx->library = library;
        ctx->page_size = (size_t)getpagesize();
        const char *name = [[device name] UTF8String];
        if (name != NULL) {
            strncpy(ctx->device_name, name, sizeof(ctx->device_name) - 1);
        }
        return ctx;
    }
}

void fucina_metal_deinit(void *ctx_opaque) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL) {
        return;
    }
    @autoreleasepool {
        for (int i = 0; i < FUCINA_GEMM_PIPELINES; i++) {
            ctx->pipelines[i] = nil;
        }
        for (int i = 0; i < FUCINA_WRAP_CACHE; i++) {
            ctx->wrap_bufs[i] = nil;
        }
        for (int i = 0; i < FUCINA_QMM_FORMATS; i++) {
            ctx->qmm_pipelines[i] = nil;
        }
        ctx->qmoe_in = nil;
        ctx->qmoe_out = nil;
        ctx->qmm_tiles = nil;
        ctx->f16_out = nil;
        ctx->library = nil;
        ctx->queue = nil;
        ctx->device = nil;
    }
    free(ctx);
}

const char *fucina_metal_device_name(void *ctx_opaque) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    return ctx->device_name;
}

static id<MTLComputePipelineState> fucina_gemm_pipeline(
    FucinaMetalCtx *ctx, int dtype, int variant, bool align_m, bool align_n, bool align_k) {
    int slot = (((dtype * FUCINA_GEMM_VARIANTS + variant) * 2 + (align_m ? 1 : 0)) * 2 + (align_n ? 1 : 0)) * 2 + (align_k ? 1 : 0);
    id<MTLComputePipelineState> cached = ctx->pipelines[slot];
    if (cached != nil) {
        return cached;
    }
    os_unfair_lock_lock(&ctx->pipeline_lock);
    cached = ctx->pipelines[slot];
    if (cached != nil) {
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return cached;
    }
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    // has_batch stays false: batching uses the params batch strides + grid
    // depth (the kernel's non-batch path multiplies them by tid.z).
    bool bfalse = false;
    [constants setConstantValue:&bfalse type:MTLDataTypeBool atIndex:10];  // has_batch
    [constants setConstantValue:&bfalse type:MTLDataTypeBool atIndex:100]; // use_out_source
    [constants setConstantValue:&bfalse type:MTLDataTypeBool atIndex:110]; // do_axpby
    [constants setConstantValue:&align_m type:MTLDataTypeBool atIndex:200];
    [constants setConstantValue:&align_n type:MTLDataTypeBool atIndex:201];
    [constants setConstantValue:&align_k type:MTLDataTypeBool atIndex:202];
    [constants setConstantValue:&bfalse type:MTLDataTypeBool atIndex:300]; // do_gather
    NSError *error = nil;
    id<MTLFunction> fn =
        [ctx->library newFunctionWithName:[NSString stringWithUTF8String:fucina_gemm_fn_names[dtype][variant]]
                           constantValues:constants
                                    error:&error];
    if (fn == nil) {
        NSLog(@"fucina-metal: function %s: %@", fucina_gemm_fn_names[dtype][variant], error);
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [ctx->device newComputePipelineStateWithFunction:fn error:&error];
    if (pipeline == nil) {
        NSLog(@"fucina-metal: pipeline %s: %@", fucina_gemm_fn_names[dtype][variant], error);
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return nil;
    }
    ctx->pipelines[slot] = pipeline;
    os_unfair_lock_unlock(&ctx->pipeline_lock);
    return pipeline;
}

// Zero-copy wrap of the pages containing [ptr, ptr+len): page-floored base,
// page-rounded length, byte offset returned for setBuffer:offset:.
// Hazard tracking is off: every dispatch in this shim is synchronous
// (commit + waitUntilCompleted under the caller's locks), so Metal's
// per-commit dependency/residency validation — which was measured at
// 8-46 ms per dispatch on the multi-hundred-MB expert RHS wraps — buys
// nothing here.
static id<MTLBuffer> fucina_wrap(FucinaMetalCtx *ctx, const void *ptr, size_t len, size_t *offset_out) {
    uintptr_t mask = (uintptr_t)ctx->page_size - 1;
    uintptr_t base = (uintptr_t)ptr & ~mask;
    uintptr_t end = ((uintptr_t)ptr + len + mask) & ~mask;
    *offset_out = (uintptr_t)ptr - base;
    return [ctx->device newBufferWithBytesNoCopy:(void *)base
                                          length:(NSUInteger)(end - base)
                                         options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                     deallocator:nil];
}

// As fucina_wrap, but cached by (page base, length-covers) — ONLY for memory
// the caller guarantees stays mapped for the process lifetime (resident f16
// weights, device-owned expert buffers). Linear probe; a full table degrades
// to uncached wraps. The cache has its own lock: its mutators run under
// DIFFERENT caller locks (f16_lock, qmoe_lock) and fucina_metal_alloc_resident_bytes
// is called from concurrent model-loading workers with no caller lock at all.
static id<MTLBuffer> fucina_wrap_cached(FucinaMetalCtx *ctx, const void *ptr, size_t len, size_t *offset_out) {
    uintptr_t mask = (uintptr_t)ctx->page_size - 1;
    uintptr_t base = (uintptr_t)ptr & ~mask;
    uintptr_t end = ((uintptr_t)ptr + len + mask) & ~mask;
    *offset_out = (uintptr_t)ptr - base;
    size_t h = (size_t)((base >> 14) % FUCINA_WRAP_CACHE);
    size_t first_empty = FUCINA_WRAP_CACHE;
    os_unfair_lock_lock(&ctx->wrap_lock);
    for (size_t probe = 0; probe < FUCINA_WRAP_CACHE; probe++) {
        size_t slot = (h + probe) % FUCINA_WRAP_CACHE;
        if (ctx->wrap_bufs[slot] == nil) {
            if (first_empty == FUCINA_WRAP_CACHE) {
                first_empty = slot;
            }
            continue;
        }
        if (ctx->wrap_keys[slot].base == base && ctx->wrap_keys[slot].end >= end) {
            id<MTLBuffer> buf = ctx->wrap_bufs[slot];
            os_unfair_lock_unlock(&ctx->wrap_lock);
            return buf;
        }
    }
    if (first_empty != FUCINA_WRAP_CACHE) {
        id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void *)base
                                                           length:(NSUInteger)(end - base)
                                                          options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                                      deallocator:nil];
        if (buf != nil) {
            ctx->wrap_keys[first_empty].base = base;
            ctx->wrap_keys[first_empty].end = end;
            ctx->wrap_bufs[first_empty] = buf;
        }
        os_unfair_lock_unlock(&ctx->wrap_lock);
        return buf;
    }
    os_unfair_lock_unlock(&ctx->wrap_lock);
    return fucina_wrap(ctx, ptr, len, offset_out);
}

// Lookup-only variant of fucina_wrap_cached: returns the cached wrap when
// [ptr, ptr+len) lies in a pre-registered resident allocation (exact
// page-base match), otherwise falls back to a transient wrap WITHOUT
// inserting. Safe for arbitrary transient operands: only buffers whose
// lifetime is owner-managed (alloc/free_resident_bytes evict on free) can
// ever be served from the cache.
static id<MTLBuffer> fucina_wrap_resident_or_transient(FucinaMetalCtx *ctx, const void *ptr, size_t len, size_t *offset_out) {
    uintptr_t mask = (uintptr_t)ctx->page_size - 1;
    uintptr_t base = (uintptr_t)ptr & ~mask;
    uintptr_t end = ((uintptr_t)ptr + len + mask) & ~mask;
    size_t h = (size_t)((base >> 14) % FUCINA_WRAP_CACHE);
    os_unfair_lock_lock(&ctx->wrap_lock);
    for (size_t probe = 0; probe < FUCINA_WRAP_CACHE; probe++) {
        size_t slot = (h + probe) % FUCINA_WRAP_CACHE;
        if (ctx->wrap_bufs[slot] == nil) {
            continue;
        }
        if (ctx->wrap_keys[slot].base == base && ctx->wrap_keys[slot].end >= end) {
            id<MTLBuffer> buf = ctx->wrap_bufs[slot];
            os_unfair_lock_unlock(&ctx->wrap_lock);
            *offset_out = (uintptr_t)ptr - base;
            return buf;
        }
    }
    os_unfair_lock_unlock(&ctx->wrap_lock);
    return fucina_wrap(ctx, ptr, len, offset_out);
}

// Cache one page wrapper on the storage object rather than in an address-only
// process table.  Its owner destroys it before freeing/remapping the backing
// allocation, so pooled buffers may safely reuse the mapping with new values.
void *fucina_metal_wrap_storage(void *ctx_opaque, const void *ptr, int64_t len) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || ptr == NULL || len <= 0) return NULL;
    @autoreleasepool {
        size_t offset = 0;
        id<MTLBuffer> buffer = fucina_wrap_resident_or_transient(ctx, ptr, (size_t)len, &offset);
        if (buffer == nil) return NULL;
        FucinaMetalWrap *wrap = [[FucinaMetalWrap alloc] init];
        wrap.buffer = buffer;
        wrap.base = (uintptr_t)ptr - offset;
        wrap.end = wrap.base + buffer.length;
        return (__bridge_retained void *)wrap;
    }
}

void fucina_metal_free_storage_wrap(void *wrap_opaque) {
    if (wrap_opaque == NULL) return;
    @autoreleasepool {
        CFBridgingRelease(wrap_opaque);
    }
}

static id<MTLBuffer> fucina_buffer_from_storage_wrap(
    FucinaMetalCtx *ctx, void *wrap_opaque, const void *ptr, size_t len,
    size_t *offset_out) {
    if (wrap_opaque != NULL) {
        FucinaMetalWrap *wrap = (__bridge FucinaMetalWrap *)wrap_opaque;
        uintptr_t begin = (uintptr_t)ptr;
        uintptr_t end = begin + len;
        if (begin >= wrap.base && end >= begin && end <= wrap.end) {
            *offset_out = begin - wrap.base;
            return wrap.buffer;
        }
    }
    return fucina_wrap_resident_or_transient(ctx, ptr, len, offset_out);
}

// Eager asynchronous f32 GEMM submission.  The operation is encoded and
// committed before return; the ticket is only a completion/lifetime token.
// Command-queue order carries dependencies between consecutive calls.
void *fucina_metal_gemm_f32_async(
    void *ctx_opaque, int variant,
    const float *a, const float *b, float *c,
    void *a_wrap, void *b_wrap, void *c_wrap,
    int64_t m, int64_t n, int64_t k,
    int64_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || variant < 0 || variant >= FUCINA_GEMM_VARIANTS ||
        m <= 0 || n <= 0 || k <= 0 || batch <= 0 ||
        m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) return NULL;
    @autoreleasepool {
        const int bm = 32, bn = 32, bk = 16;
        id<MTLComputePipelineState> pipeline =
            fucina_gemm_pipeline(ctx, FUCINA_GEMM_F32, variant,
                                 m % bm == 0, n % bn == 0, k % bk == 0);
        if (pipeline == nil) return NULL;

        size_t a_len = ((size_t)(batch - 1) * (size_t)stride_a + (size_t)m * (size_t)k) * sizeof(float);
        size_t b_len = ((size_t)(batch - 1) * (size_t)stride_b + (size_t)k * (size_t)n) * sizeof(float);
        size_t c_len = ((size_t)(batch - 1) * (size_t)stride_c + (size_t)m * (size_t)n) * sizeof(float);
        size_t a_off, b_off, c_off;
        id<MTLBuffer> a_buf = fucina_buffer_from_storage_wrap(ctx, a_wrap, a, a_len, &a_off);
        id<MTLBuffer> b_buf = fucina_buffer_from_storage_wrap(ctx, b_wrap, b, b_len, &b_off);
        id<MTLBuffer> c_buf = fucina_buffer_from_storage_wrap(ctx, c_wrap, c, c_len, &c_off);
        if (a_buf == nil || b_buf == nil || c_buf == nil) return NULL;

        int32_t tiles_n = (int32_t)((n + bn - 1) / bn);
        int32_t tiles_m = (int32_t)((m + bm - 1) / bm);
        FucinaGEMMParams params = {
            .M = (int32_t)m, .N = (int32_t)n, .K = (int32_t)k,
            .lda = (int32_t)(variant == FUCINA_GEMM_TN ? m : k),
            .ldb = (int32_t)(variant == FUCINA_GEMM_NT ? k : n),
            .ldd = (int32_t)n,
            .tiles_n = tiles_n, .tiles_m = tiles_m,
            .batch_stride_a = (size_t)stride_a,
            .batch_stride_b = (size_t)stride_b,
            .batch_stride_d = (size_t)stride_c,
            .swizzle_log = 0,
            .gemm_k_iterations_aligned = (int32_t)(k / bk),
            .batch_ndim = 1,
        };
        int32_t batch_shape = (int32_t)batch;
        size_t batch_strides[2] = { (size_t)stride_a, (size_t)stride_b };

        id<MTLCommandBuffer> cmd = [ctx->queue commandBufferWithUnretainedReferences];
        if (cmd == nil) return NULL;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:a_buf offset:a_off atIndex:0];
        [enc setBuffer:b_buf offset:b_off atIndex:1];
        [enc setBuffer:c_buf offset:c_off atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc setBytes:&batch_shape length:sizeof(batch_shape) atIndex:6];
        [enc setBytes:batch_strides length:sizeof(batch_strides) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tiles_n, (NSUInteger)tiles_m, (NSUInteger)batch)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        [enc endEncoding];

        FucinaMetalTicket *ticket = [[FucinaMetalTicket alloc] init];
        ticket.command = cmd;
        ticket.aBuffer = a_buf;
        ticket.bBuffer = b_buf;
        ticket.cBuffer = c_buf;
        [cmd commit];
        return (__bridge_retained void *)ticket;
    }
}

// Eager asynchronous 16-bit-operand NT GEMM with direct f32 output.  The
// public tensor result is f32, so these mixed steel instantiations avoid the
// old shared f16 staging buffer, its process lock, and the CPU widening
// pass.  `gemm_dtype` picks the operand encoding (FUCINA_GEMM_F16_F32 or
// FUCINA_GEMM_BF16_F32 — both are 16-bit rows, so the buffer math is
// identical).
static void *fucina_metal_gemm_16bit_nt_async(
    void *ctx_opaque, int gemm_dtype,
    const uint16_t *a, const uint16_t *b, float *c,
    void *a_wrap, void *b_wrap, void *c_wrap,
    int64_t m, int64_t n, int64_t k) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || m <= 0 || n <= 0 || k <= 0 ||
        m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) return NULL;
    @autoreleasepool {
        const int bm = 32, bn = 32, bk = 16;
        id<MTLComputePipelineState> pipeline =
            fucina_gemm_pipeline(ctx, gemm_dtype, FUCINA_GEMM_NT,
                                 m % bm == 0, n % bn == 0, k % bk == 0);
        if (pipeline == nil) return NULL;

        size_t a_len = (size_t)m * (size_t)k * sizeof(uint16_t);
        size_t b_len = (size_t)n * (size_t)k * sizeof(uint16_t);
        size_t c_len = (size_t)m * (size_t)n * sizeof(float);
        size_t a_off, b_off, c_off;
        id<MTLBuffer> a_buf = fucina_buffer_from_storage_wrap(ctx, a_wrap, a, a_len, &a_off);
        id<MTLBuffer> b_buf = fucina_buffer_from_storage_wrap(ctx, b_wrap, b, b_len, &b_off);
        id<MTLBuffer> c_buf = fucina_buffer_from_storage_wrap(ctx, c_wrap, c, c_len, &c_off);
        if (a_buf == nil || b_buf == nil || c_buf == nil) return NULL;

        int32_t tiles_n = (int32_t)((n + bn - 1) / bn);
        int32_t tiles_m = (int32_t)((m + bm - 1) / bm);
        FucinaGEMMParams params = {
            .M = (int32_t)m, .N = (int32_t)n, .K = (int32_t)k,
            .lda = (int32_t)k, .ldb = (int32_t)k, .ldd = (int32_t)n,
            .tiles_n = tiles_n, .tiles_m = tiles_m,
            .batch_stride_a = 0, .batch_stride_b = 0, .batch_stride_d = 0,
            .swizzle_log = 0,
            .gemm_k_iterations_aligned = (int32_t)(k / bk),
            .batch_ndim = 1,
        };
        int32_t batch_shape = 1;
        size_t batch_strides[2] = { 0, 0 };

        id<MTLCommandBuffer> cmd = [ctx->queue commandBufferWithUnretainedReferences];
        if (cmd == nil) return NULL;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:a_buf offset:a_off atIndex:0];
        [enc setBuffer:b_buf offset:b_off atIndex:1];
        [enc setBuffer:c_buf offset:c_off atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc setBytes:&batch_shape length:sizeof(batch_shape) atIndex:6];
        [enc setBytes:batch_strides length:sizeof(batch_strides) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tiles_n, (NSUInteger)tiles_m, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        [enc endEncoding];

        FucinaMetalTicket *ticket = [[FucinaMetalTicket alloc] init];
        ticket.command = cmd;
        ticket.aBuffer = a_buf;
        ticket.bBuffer = b_buf;
        ticket.cBuffer = c_buf;
        [cmd commit];
        return (__bridge_retained void *)ticket;
    }
}

void *fucina_metal_gemm_f16_nt_async(
    void *ctx_opaque,
    const uint16_t *a, const uint16_t *b, float *c,
    void *a_wrap, void *b_wrap, void *c_wrap,
    int64_t m, int64_t n, int64_t k) {
    return fucina_metal_gemm_16bit_nt_async(ctx_opaque, FUCINA_GEMM_F16_F32,
                                            a, b, c, a_wrap, b_wrap, c_wrap, m, n, k);
}

void *fucina_metal_gemm_bf16_nt_async(
    void *ctx_opaque,
    const uint16_t *a, const uint16_t *b, float *c,
    void *a_wrap, void *b_wrap, void *c_wrap,
    int64_t m, int64_t n, int64_t k) {
    return fucina_metal_gemm_16bit_nt_async(ctx_opaque, FUCINA_GEMM_BF16_F32,
                                            a, b, c, a_wrap, b_wrap, c_wrap, m, n, k);
}

int fucina_metal_ticket_wait(void *ticket_opaque, FucinaCommandTiming *timing) {
    if (ticket_opaque == NULL) return 1;
    FucinaMetalTicket *ticket = (__bridge FucinaMetalTicket *)ticket_opaque;
    [ticket.command waitUntilCompleted];
    if (ticket.command.status == MTLCommandBufferStatusError) {
        NSLog(@"fucina-metal: async gemm command failed: %@", ticket.command.error);
        return 1;
    }
    fucina_record_timing(timing, ticket.command);
    return 0;
}

void fucina_metal_ticket_free(void *ticket_opaque) {
    if (ticket_opaque == NULL) return;
    @autoreleasepool {
        CFBridgingRelease(ticket_opaque);
    }
}

// C[m,n] = op(A) * op(B) per variant, f32, row-major, beta=0 overwrite.
// `batch` > 1 runs `batch` independent GEMMs in one dispatch with the given
// element strides between consecutive matrices. Returns 0 on success;
// nonzero = not handled (caller must run the CPU path).
int fucina_metal_gemm_f32(
    void *ctx_opaque, int variant,
    const float *a, const float *b, float *c,
    int64_t m, int64_t n, int64_t k,
    int64_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
    FucinaCommandTiming *timing) {
    if (timing != NULL) {
        timing->gpu_ns = 0;
        timing->sched_ns = 0;
    }
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || variant < 0 || variant >= FUCINA_GEMM_VARIANTS) {
        return 1;
    }
    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0) {
        return 1;
    }
    if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
        return 1;
    }
    @autoreleasepool {
        const int bm = 32, bn = 32, bk = 16;
        id<MTLComputePipelineState> pipeline =
            fucina_gemm_pipeline(ctx, FUCINA_GEMM_F32, variant, m % bm == 0, n % bn == 0, k % bk == 0);
        if (pipeline == nil) {
            return 1;
        }

        size_t a_len = ((size_t)(batch - 1) * (size_t)stride_a + (size_t)m * (size_t)k) * sizeof(float);
        size_t b_len = ((size_t)(batch - 1) * (size_t)stride_b + (size_t)k * (size_t)n) * sizeof(float);
        size_t c_len = ((size_t)(batch - 1) * (size_t)stride_c + (size_t)m * (size_t)n) * sizeof(float);
        size_t a_off, b_off, c_off;
        id<MTLBuffer> a_buf = fucina_wrap_resident_or_transient(ctx, a, a_len, &a_off);
        id<MTLBuffer> b_buf = fucina_wrap_resident_or_transient(ctx, b, b_len, &b_off);
        id<MTLBuffer> c_buf = fucina_wrap_resident_or_transient(ctx, c, c_len, &c_off);
        if (a_buf == nil || b_buf == nil || c_buf == nil) {
            return 1;
        }

        int32_t lda = (int32_t)(variant == FUCINA_GEMM_TN ? m : k);
        int32_t ldb = (int32_t)(variant == FUCINA_GEMM_NT ? k : n);
        int32_t tiles_n = (int32_t)((n + bn - 1) / bn);
        int32_t tiles_m = (int32_t)((m + bm - 1) / bm);

        FucinaGEMMParams params = {
            .M = (int32_t)m,
            .N = (int32_t)n,
            .K = (int32_t)k,
            .lda = lda,
            .ldb = ldb,
            .ldd = (int32_t)n,
            .tiles_n = tiles_n,
            .tiles_m = tiles_m,
            .batch_stride_a = (size_t)stride_a,
            .batch_stride_b = (size_t)stride_b,
            .batch_stride_d = (size_t)stride_c,
            .swizzle_log = 0,
            .gemm_k_iterations_aligned = (int32_t)(k / bk),
            .batch_ndim = 1,
        };
        int32_t batch_shape = (int32_t)batch;
        size_t batch_strides[2] = { (size_t)stride_a, (size_t)stride_b };

        // Unretained references: skip per-commit resource retention/residency
        // bookkeeping (measured at ~45 us/MB of referenced buffers — tens of
        // ms per dispatch on the expert weights). All referenced buffers are
        // owned by the context or by process-lifetime weights, and every
        // dispatch is synchronous, so lifetimes are guaranteed by the caller.
        id<MTLCommandBuffer> cmd = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:a_buf offset:a_off atIndex:0];
        [enc setBuffer:b_buf offset:b_off atIndex:1];
        [enc setBuffer:c_buf offset:c_off atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc setBytes:&batch_shape length:sizeof(batch_shape) atIndex:6];
        [enc setBytes:batch_strides length:sizeof(batch_strides) atIndex:7];
        // Tiled kernels need dispatchThreadgroups (bounds checks are inside);
        // threadgroup = (32, WN, WM) = 128 threads = 4 simdgroups.
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tiles_n, (NSUInteger)tiles_m, (NSUInteger)batch)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            NSLog(@"fucina-metal: gemm command failed: %@", cmd.error);
            return 1;
        }
        fucina_record_timing(timing, cmd);
        return 0;
    }
}

// C16[m,n] (f16, staged in a context-owned buffer) = A16[m,k] * B16[n,k]^T —
// the f16-weights NT GEMM (the steel kernel accumulates in f32; only the
// stored result rounds to f16). On success *out_staging points at the m*n
// f16 results; it stays valid until the NEXT f16 call, so the caller must
// hold its f16 lock across call + widen. Returns nonzero when not handled.
int fucina_metal_gemm_f16_nt(
    void *ctx_opaque,
    const uint16_t *a, const uint16_t *b,
    int64_t m, int64_t n, int64_t k,
    int cache_rhs,
    const uint16_t **out_staging,
    FucinaCommandTiming *timing) {
    if (timing != NULL) {
        timing->gpu_ns = 0;
        timing->sched_ns = 0;
    }
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || m <= 0 || n <= 0 || k <= 0) {
        return 1;
    }
    if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
        return 1;
    }
    @autoreleasepool {
        const int bm = 32, bn = 32, bk = 16;
        id<MTLComputePipelineState> pipeline =
            fucina_gemm_pipeline(ctx, FUCINA_GEMM_F16, FUCINA_GEMM_NT, m % bm == 0, n % bn == 0, k % bk == 0);
        if (pipeline == nil) {
            return 1;
        }

        size_t a_off, b_off;
        id<MTLBuffer> a_buf = fucina_wrap(ctx, a, (size_t)m * (size_t)k * 2, &a_off);
        // Cache the RHS wrap only when the caller guarantees process-lifetime
        // storage. Transient RHS operands must pass cache_rhs = 0.
        id<MTLBuffer> b_buf = cache_rhs
            ? fucina_wrap_cached(ctx, b, (size_t)n * (size_t)k * 2, &b_off)
            : fucina_wrap(ctx, b, (size_t)n * (size_t)k * 2, &b_off);
        if (a_buf == nil || b_buf == nil) {
            return 1;
        }
        size_t out_bytes = (size_t)m * (size_t)n * 2;
        if (ctx->f16_out_cap < out_bytes) {
            ctx->f16_out = [ctx->device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
            if (ctx->f16_out == nil) {
                ctx->f16_out_cap = 0;
                return 1;
            }
            ctx->f16_out_cap = out_bytes;
        }

        int32_t tiles_n = (int32_t)((n + bn - 1) / bn);
        int32_t tiles_m = (int32_t)((m + bm - 1) / bm);
        FucinaGEMMParams params = {
            .M = (int32_t)m,
            .N = (int32_t)n,
            .K = (int32_t)k,
            .lda = (int32_t)k,
            .ldb = (int32_t)k,
            .ldd = (int32_t)n,
            .tiles_n = tiles_n,
            .tiles_m = tiles_m,
            .batch_stride_a = 0,
            .batch_stride_b = 0,
            .batch_stride_d = 0,
            .swizzle_log = 0,
            .gemm_k_iterations_aligned = (int32_t)(k / bk),
            .batch_ndim = 1,
        };
        int32_t batch_shape = 1;
        size_t batch_strides[2] = { 0, 0 };

        // Unretained references: skip per-commit resource retention/residency
        // bookkeeping (measured at ~45 us/MB of referenced buffers — tens of
        // ms per dispatch on the expert weights). All referenced buffers are
        // owned by the context or by process-lifetime weights, and every
        // dispatch is synchronous, so lifetimes are guaranteed by the caller.
        id<MTLCommandBuffer> cmd = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:a_buf offset:a_off atIndex:0];
        [enc setBuffer:b_buf offset:b_off atIndex:1];
        [enc setBuffer:ctx->f16_out offset:0 atIndex:3];
        [enc setBytes:&params length:sizeof(params) atIndex:4];
        [enc setBytes:&batch_shape length:sizeof(batch_shape) atIndex:6];
        [enc setBytes:batch_strides length:sizeof(batch_strides) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tiles_n, (NSUInteger)tiles_m, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            NSLog(@"fucina-metal: f16 gemm command failed: %@", cmd.error);
            return 1;
        }
        fucina_record_timing(timing, cmd);
        *out_staging = (const uint16_t *)ctx->f16_out.contents;
        return 0;
    }
}

static id<MTLBuffer> fucina_grow_buffer(FucinaMetalCtx *ctx, id<MTLBuffer> __strong *buf, size_t *cap, size_t bytes) {
    if (*cap < bytes) {
        *buf = [ctx->device newBufferWithLength:bytes
                                        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        *cap = (*buf == nil) ? 0 : bytes;
    }
    return *buf;
}

// Acquire the quantized-GEMM staging panels: `in` is the activations panel the
// CPU writes (and later overwrites with the gated rows), `out` receives the
// GEMM results. Pointers are the buffers' shared contents; they stay valid
// until the NEXT stage call grows the buffers, so the caller holds its qmoe
// lock across stage + dispatches + readback. Returns nonzero when not handled.
int fucina_metal_qmoe_stage(
    void *ctx_opaque, int64_t in_bytes, int64_t out_bytes,
    void **in_ptr, void **out_ptr) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || in_bytes <= 0 || out_bytes <= 0) {
        return 1;
    }
    @autoreleasepool {
        id<MTLBuffer> in_buf = fucina_grow_buffer(ctx, &ctx->qmoe_in, &ctx->qmoe_in_cap, (size_t)in_bytes);
        id<MTLBuffer> out_buf = fucina_grow_buffer(ctx, &ctx->qmoe_out, &ctx->qmoe_out_cap, (size_t)out_bytes);
        if (in_buf == nil || out_buf == nil) {
            return 1;
        }
        *in_ptr = in_buf.contents;
        *out_ptr = out_buf.contents;
        return 0;
    }
}

// Device-owned resident weight storage. Client-memory wraps
// (bytesNoCopy) are PAGEABLE: Metal re-wires them into the GPU address space
// on every commit — measured ~45 us/MB of referenced weights per dispatch,
// i.e. tens of ms on multi-hundred-MB expert tensors, regardless of hazard
// tracking or unretained references. Device-owned buffers stay mapped. The
// buffer registers in the wrap cache keyed by its (page-aligned) contents,
// so the dispatch-time cached-wrap lookup finds it with no further changes;
// the CPU reads the same bytes through the returned contents pointer
// (unified memory). Owners may release these buffers with
// fucina_metal_free_resident_bytes; callers that cannot prove ownership must
// still treat them as process-lifetime and fall back when the bounded wrap cache
// is full.
void *fucina_metal_alloc_resident_bytes(void *ctx_opaque, int64_t len) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || len <= 0) {
        return NULL;
    }
    @autoreleasepool {
        id<MTLBuffer> buf = [ctx->device newBufferWithLength:(NSUInteger)len
                                                     options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        if (buf == nil) {
            return NULL;
        }
        uintptr_t mask = (uintptr_t)ctx->page_size - 1;
        uintptr_t base = (uintptr_t)buf.contents;
        uintptr_t end = (base + (uintptr_t)len + mask) & ~mask;
        size_t h = (size_t)((base >> 14) % FUCINA_WRAP_CACHE);
        // Model loading registers weights from concurrent pool workers — the
        // probe/insert must be atomic or two workers can claim one slot and
        // ARC releases the overwritten buffer while its contents pointer is
        // already handed out.
        os_unfair_lock_lock(&ctx->wrap_lock);
        for (size_t probe = 0; probe < FUCINA_WRAP_CACHE; probe++) {
            size_t slot = (h + probe) % FUCINA_WRAP_CACHE;
            if (ctx->wrap_bufs[slot] == nil) {
                ctx->wrap_keys[slot].base = base;
                ctx->wrap_keys[slot].end = end;
                ctx->wrap_bufs[slot] = buf;
                os_unfair_lock_unlock(&ctx->wrap_lock);
                return buf.contents;
            }
        }
        os_unfair_lock_unlock(&ctx->wrap_lock);
        return NULL; // cache full — caller falls back to plain host memory
    }
}

int fucina_metal_free_resident_bytes(void *ctx_opaque, const void *ptr) {
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || ptr == NULL) {
        return 1;
    }
    uintptr_t base = (uintptr_t)ptr;
    os_unfair_lock_lock(&ctx->wrap_lock);
    for (size_t slot = 0; slot < FUCINA_WRAP_CACHE; slot++) {
        if (ctx->wrap_bufs[slot] != nil && ctx->wrap_keys[slot].base == base) {
            ctx->wrap_bufs[slot] = nil;
            ctx->wrap_keys[slot].base = 0;
            ctx->wrap_keys[slot].end = 0;
            os_unfair_lock_unlock(&ctx->wrap_lock);
            return 0;
        }
    }
    os_unfair_lock_unlock(&ctx->wrap_lock);
    return 1;
}

static id<MTLComputePipelineState> fucina_qmm_pipeline(FucinaMetalCtx *ctx, int format) {
    id<MTLComputePipelineState> cached = ctx->qmm_pipelines[format];
    if (cached != nil) {
        return cached;
    }
    os_unfair_lock_lock(&ctx->pipeline_lock);
    cached = ctx->qmm_pipelines[format];
    if (cached != nil) {
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return cached;
    }
    const char *name = NULL;
    switch (format) {
    case FUCINA_QFMT_Q8_0: name = "fucina_mul_mm_q8_0_f32"; break;
    case FUCINA_QFMT_Q6_K: name = "fucina_mul_mm_q6_K_f32"; break;
    case FUCINA_QFMT_Q4_K: name = "fucina_mul_mm_q4_K_f32"; break;
    case FUCINA_QFMT_TQ2_0: name = "fucina_mul_mm_tq2_0_f32"; break;
    default:
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return nil; // bounds-checked by the caller
    }
    NSError *error = nil;
    id<MTLFunction> fn = [ctx->library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (fn == nil) {
        NSLog(@"fucina-metal: function %s not found", name);
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [ctx->device newComputePipelineStateWithFunction:fn error:&error];
    if (pipeline == nil) {
        NSLog(@"fucina-metal: pipeline %s: %@", name, error);
        os_unfair_lock_unlock(&ctx->pipeline_lock);
        return nil;
    }
    ctx->qmm_pipelines[format] = pipeline;
    os_unfair_lock_unlock(&ctx->pipeline_lock);
    return pipeline;
}

// Eager asynchronous dense/shared-input quantized NT GEMM. Unlike the MoE
// protocol below, ordinary tensor input/output storage is wrapped directly:
// no shared panels, CPU memcpy, process lock, or immediate wait. A compact
// <=4 KiB tile table is copied into command-buffer-owned setBytes storage.
void *fucina_metal_gemm_q_dense_nt_async(
    void *ctx_opaque, int format,
    const void *rhs_bytes, int64_t rhs_len,
    int64_t nb01, int64_t nb02,
    const float *a, float *c,
    void *a_wrap, void *c_wrap,
    int64_t batch_count, int64_t m, int64_t n, int64_t k) {
    enum { max_dense_tiles = 256 };
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || format < 0 || format >= FUCINA_QMM_FORMATS ||
        rhs_bytes == NULL || rhs_len <= 0 || a == NULL || c == NULL ||
        batch_count <= 0 || m <= 0 || n <= 0 || k <= 0 ||
        batch_count > INT32_MAX || m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) return NULL;
    int64_t n_tiles = (m + 31) / 32;
    if (n_tiles <= 0 || n_tiles > max_dense_tiles) return NULL;
    if ((uint64_t)m > SIZE_MAX / (uint64_t)k / sizeof(float) ||
        (uint64_t)batch_count > SIZE_MAX / (uint64_t)m ||
        (uint64_t)(batch_count * m) > SIZE_MAX / (uint64_t)n / sizeof(float)) return NULL;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = fucina_qmm_pipeline(ctx, format);
        if (pipeline == nil) return NULL;

        size_t a_len = (size_t)m * (size_t)k * sizeof(float);
        size_t c_len = (size_t)batch_count * (size_t)m * (size_t)n * sizeof(float);
        size_t a_off, c_off, w_off;
        id<MTLBuffer> a_buf = fucina_buffer_from_storage_wrap(ctx, a_wrap, a, a_len, &a_off);
        id<MTLBuffer> c_buf = fucina_buffer_from_storage_wrap(ctx, c_wrap, c, c_len, &c_off);
        // This ABI is only used for stable GGUF/model weights. Registering the
        // mapping once is therefore safe and removes dispatch-time page wiring.
        id<MTLBuffer> w_buf = fucina_wrap_cached(ctx, rhs_bytes, (size_t)rhs_len, &w_off);
        if (a_buf == nil || c_buf == nil || w_buf == nil) return NULL;

        FucinaQMMArgs args = {
            .ne00 = (int32_t)k,
            .ne01 = (int32_t)n,
            .nb01 = (uint64_t)nb01,
            .nb02 = (uint64_t)nb02,
        };
        FucinaQMMTile tiles[max_dense_tiles];
        id<MTLCommandBuffer> cmd = [ctx->queue commandBufferWithUnretainedReferences];
        if (cmd == nil) return NULL;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:w_buf offset:w_off atIndex:1];
        [enc setBuffer:a_buf offset:a_off atIndex:2];
        [enc setThreadgroupMemoryLength:8192 atIndex:0];
        for (int64_t bi = 0; bi < batch_count; bi++) {
            for (int64_t ti = 0; ti < n_tiles; ti++) {
                tiles[ti] = (FucinaQMMTile){
                    .expert = (int32_t)bi,
                    .base_row = 0,
                    .m = (int32_t)m,
                    .tile_m = (int32_t)ti,
                };
            }
            [enc setBytes:tiles length:(NSUInteger)n_tiles * sizeof(FucinaQMMTile) atIndex:3];
            [enc setBuffer:c_buf
                    offset:c_off + (NSUInteger)bi * (NSUInteger)m * (NSUInteger)n * sizeof(float)
                   atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tiles, (NSUInteger)((n + 63) / 64), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [enc endEncoding];

        FucinaMetalTicket *ticket = [[FucinaMetalTicket alloc] init];
        ticket.command = cmd;
        ticket.aBuffer = a_buf;
        ticket.bBuffer = w_buf;
        ticket.cBuffer = c_buf;
        [cmd commit];
        return (__bridge_retained void *)ticket;
    }
}

// Grouped quantized NT GEMM over the staged panels: for every tile t,
// out[rows of t, 0..n_out) = in[rows of t, 0..k) * dequant(W[t.expert])^T.
// `rhs_bytes` = raw GGUF blocks, row-major [n_out, k] per expert with uniform
// byte strides nb01 (row) / nb02 (expert). `cache_rhs` routes the wrap
// through the page cache — ONLY for stable process-lifetime storage
// (device-owned fucina_metal_alloc_resident_bytes buffers or a model-owned mmap);
// transient buffers (tests) must pass 0 or a later allocation at the same page
// base would be read through a stale mapping. The activations panel must
// already be staged via fucina_metal_qmoe_stage (rows of k f32), results land
// in the out panel (rows of n_out f32). Returns nonzero when not handled
// (caller falls back to the CPU path).
int fucina_metal_gemm_q_grouped_nt(
    void *ctx_opaque, int format,
    const void *rhs_bytes, int64_t rhs_len, int cache_rhs,
    int64_t nb01, int64_t nb02,
    int64_t n_out, int64_t k,
    const FucinaQMMTile *tiles, int64_t n_tiles,
    FucinaCommandTiming *timing) {
    if (timing != NULL) {
        timing->gpu_ns = 0;
        timing->sched_ns = 0;
    }
    FucinaMetalCtx *ctx = (FucinaMetalCtx *)ctx_opaque;
    if (ctx == NULL || format < 0 || format >= FUCINA_QMM_FORMATS) {
        return 1;
    }
    if (n_out <= 0 || k <= 0 || n_tiles <= 0 || rhs_len <= 0) {
        return 1;
    }
    if (n_out > INT32_MAX || k > INT32_MAX) {
        return 1;
    }
    if (ctx->qmoe_in == nil || ctx->qmoe_out == nil) {
        return 1;
    }
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = fucina_qmm_pipeline(ctx, format);
        if (pipeline == nil) {
            return 1;
        }

        size_t w_off;
        id<MTLBuffer> w_buf = cache_rhs
            ? fucina_wrap_cached(ctx, rhs_bytes, (size_t)rhs_len, &w_off)
            : fucina_wrap(ctx, rhs_bytes, (size_t)rhs_len, &w_off);
        if (w_buf == nil) {
            return 1;
        }

        if ((uint64_t)n_tiles > SIZE_MAX / sizeof(FucinaQMMTile)) {
            return 1;
        }
        size_t tiles_bytes = (size_t)n_tiles * sizeof(FucinaQMMTile);
        id<MTLBuffer> tiles_buf = fucina_grow_buffer(ctx, &ctx->qmm_tiles, &ctx->qmm_tiles_cap, tiles_bytes);
        if (tiles_buf == nil) {
            return 1;
        }
        memcpy(tiles_buf.contents, tiles, tiles_bytes);

        FucinaQMMArgs args = {
            .ne00 = (int32_t)k,
            .ne01 = (int32_t)n_out,
            .nb01 = (uint64_t)nb01,
            .nb02 = (uint64_t)nb02,
        };

        // Unretained references: skip per-commit resource retention/residency
        // bookkeeping (measured at ~45 us/MB of referenced buffers — tens of
        // ms per dispatch on the expert weights). All referenced buffers are
        // owned by the context or by process-lifetime weights, and every
        // dispatch is synchronous, so lifetimes are guaranteed by the caller.
        id<MTLCommandBuffer> cmd = [ctx->queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:w_buf offset:w_off atIndex:1];
        [enc setBuffer:ctx->qmoe_in offset:0 atIndex:2];
        [enc setBuffer:tiles_buf offset:0 atIndex:3];
        [enc setBuffer:ctx->qmoe_out offset:0 atIndex:4];
        // sa (64x32 half) + sb (32x32 half) for the K loop; the partial-tile
        // store restripes the whole region as 64x32 f32 = 8192 bytes.
        [enc setThreadgroupMemoryLength:8192 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_tiles, (NSUInteger)((n_out + 63) / 64), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        CFAbsoluteTime wall0 = CFAbsoluteTimeGetCurrent();
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status == MTLCommandBufferStatusError) {
            NSLog(@"fucina-metal: quant gemm command failed: %@", cmd.error);
            return 1;
        }
        fucina_record_timing(timing, cmd);
        if (getenv("FUCINA_GPU_DEBUG") != NULL) {
            NSLog(@"fucina-metal: qmm fmt=%d n=%lld k=%lld tiles=%lld wall=%.2fms gpu=%.2fms sched=%.2fms",
                  format, n_out, k, n_tiles,
                  (CFAbsoluteTimeGetCurrent() - wall0) * 1000.0,
                  (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0,
                  (cmd.kernelEndTime - cmd.kernelStartTime) * 1000.0);
        }
        return 0;
    }
}
