const std = @import("std");

const BackendKind = enum { scalar, native, cpu };
const BlasKind = enum { none, accelerate, openblas, mkl, blis, nvpl, blas };
const GpuKind = enum { none, metal, cuda };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const backend_kind = b.option(
        BackendKind,
        "backend",
        "Backend implementation: native (Zig SIMD + optional BLAS, default), scalar (reference only); cpu is a deprecated alias for scalar",
    ) orelse .native;
    const requested_accelerate = b.option(
        bool,
        "accelerate",
        "Compatibility alias: false is equivalent to -Dblas=none; true selects Accelerate on macOS",
    );
    const default_blas: BlasKind = if (target.result.os.tag == .macos) .accelerate else .none;
    const blas_kind = b.option(
        BlasKind,
        "blas",
        "Native BLAS provider: none, accelerate, openblas, mkl, blis, nvpl, blas",
    ) orelse if (requested_accelerate) |use_accelerate|
        if (use_accelerate) BlasKind.accelerate else BlasKind.none
    else
        default_blas;
    const blas_threads = b.option(
        u32,
        "blas-threads",
        "Threads for explicit native BLAS providers; 0 keeps the provider default",
    ) orelse 0;
    if (blas_kind == .accelerate and target.result.os.tag != .macos) {
        @panic("-Dblas=accelerate is only available on macOS; use -Dblas=openblas, -Dblas=mkl, -Dblas=blis, -Dblas=nvpl, or -Dblas=blas");
    }
    const max_threads = b.option(
        usize,
        "max-threads",
        "Comptime worker-team ceiling and runtime default thread count (1-64, default 8); FUCINA_MAX_THREADS can still lower it at runtime",
    ) orelse 8;
    if (max_threads < 1 or max_threads > 64) {
        std.debug.panic("-Dmax-threads must be between 1 and 64, got {d}", .{max_threads});
    }
    const gpu_kind = b.option(
        GpuKind,
        "gpu",
        "GPU GEMM offload provider: none (default), metal (Apple Silicon; big f32/f16 GEMMs, dense quantized linears (Q4_K/Q6_K/Q8_0 prefill), and the MoE expert FFN run on the GPU; decode and training stay on CPU), cuda (Linux/NVIDIA; f32/f16 GEMMs via dlopen'd cuBLAS, Q4_K/Q5_K/Q6_K/Q8_0 prefill + fused prefill attention via vendored PTX kernels, opt-in decode — no CUDA SDK at build time)",
    ) orelse .none;
    if (gpu_kind == .metal and target.result.os.tag != .macos) {
        @panic("-Dgpu=metal is only available on macOS");
    }
    if (gpu_kind == .cuda and target.result.os.tag != .linux) {
        @panic("-Dgpu=cuda currently targets Linux (the provider dlopens libcuda.so.1 at runtime; cross-compile with -Dtarget=x86_64-linux-gnu)");
    }

    const parakeet_mic = b.option(
        bool,
        "parakeet-mic",
        "Link the vendored miniaudio capture stack into the parakeet example so `--mic` (live microphone streaming) works (default false — keeps the default parakeet build fast).",
    ) orelse false;

    const vector_scan = b.option(
        bool,
        "vector-scan",
        "Vectorize the scan kernels (cumsum/cumprod and cumsum's reverse VJP pass). Default false = the documented serial-per-row scans. When true, non-last-axis scans vectorize across independent columns (bitwise identical to serial) and last-axis scans use an in-register prefix scan — still bitwise deterministic for any thread count, but the accumulation ORDER differs from the serial default (the sum-SIMD-lanes rounding class).",
    ) orelse false;

    const llguidance_enabled = b.option(
        bool,
        "llguidance",
        "Build and link the vendored llguidance constrained-decoding engine (vendor/llguidance, Rust — requires cargo >= 1.87 on PATH) so `llm.llguidance` grammar/JSON-schema token masking works. Default false: the build stays pure Zig and `llm.llguidance.Constraint.init` returns error.LlguidanceNotEnabled.",
    ) orelse false;

    const options = b.addOptions();
    options.addOption(BackendKind, "backend_kind", backend_kind);
    options.addOption(BlasKind, "blas_kind", blas_kind);
    options.addOption(bool, "use_blas", blas_kind != .none);
    options.addOption(u32, "blas_threads", blas_threads);
    options.addOption(usize, "max_threads", max_threads);
    options.addOption(bool, "use_gpu", gpu_kind != .none);
    options.addOption(GpuKind, "gpu_kind", gpu_kind);
    options.addOption(bool, "vector_scan", vector_scan);

    const module = b.addModule("fucina", .{
        .root_source_file = b.path("src/fucina.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addOptions("build_options", options);

    // fucina_llm's own build options (the fucina options above are per-kernel
    // knobs the llm tier never reads). Every module built from src/llm.zig
    // must receive one of these under the name "llm_build_options".
    const llm_options = b.addOptions();
    llm_options.addOption(bool, "llguidance", llguidance_enabled);
    const llm_options_off = b.addOptions(); // compile-only legs: never link the Rust lib
    llm_options_off.addOption(bool, "llguidance", false);

    // -Dllguidance: build the vendored Rust staticlib once per `zig build`
    // invocation (cargo's own incremental cache makes the no-change case
    // sub-second) and link it into the executables/test roots that actually
    // reference the `llm.llguidance` externs. Consumers that merely import
    // fucina_llm don't need the link — extern symbols resolve lazily.
    const llguidance_dep: ?LlguidanceDep = if (llguidance_enabled) blk: {
        const cargo = b.addSystemCommand(&.{ "cargo", "build", "--release", "--package", "llguidance" });
        cargo.setCwd(b.path("vendor/llguidance"));
        cargo.has_side_effects = true; // cargo tracks its own inputs; always invoke it
        break :blk .{
            .build_step = &cargo.step,
            .lib = b.path("vendor/llguidance/target/release/libllguidance.a"),
        };
    } else null;

    const llm_module = b.addModule("fucina_llm", .{
        .root_source_file = b.path("src/llm.zig"),
        .target = target,
        .optimize = optimize,
    });
    llm_module.addImport("fucina", module);
    llm_module.addOptions("llm_build_options", llm_options);

    const exe = b.addExecutable(.{
        .name = "fucina-zig-smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/smoke.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("fucina", module);
    configureBlas(exe, blas_kind);
    configureGpu(b, exe, gpu_kind);
    const exe_install = installArtifactStep(b, exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(exe_install);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the smoke example");
    run_step.dependOn(&run_cmd.step);

    const facedetect_exe = b.addExecutable(.{
        .name = "fucina-zig-facedetect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/facedetect.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    facedetect_exe.root_module.addImport("fucina", module);
    configureBlas(facedetect_exe, blas_kind);
    configureGpu(b, facedetect_exe, gpu_kind);
    const facedetect_install = installArtifactStep(b, facedetect_exe);

    const facedetect_cmd = b.addRunArtifact(facedetect_exe);
    facedetect_cmd.step.dependOn(facedetect_install);
    if (b.args) |args| {
        facedetect_cmd.addArgs(args);
    }

    const facedetect_step = b.step("facedetect", "Face detection/recognition (face-detect.cpp buffalo_l port): detect/embed/verify/analyze/landmarks");
    facedetect_step.dependOn(&facedetect_cmd.step);

    const nanochat_exe = b.addExecutable(.{
        .name = "fucina-zig-nanochat",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/nanochat.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    nanochat_exe.root_module.addImport("fucina", module);
    // nanochat's raw-byte BPE pretokenizer reuses the generated \p{L}/\p{N}/\s
    // tables via the fucina_llm re-export (llm.unicode_categories) — sharing
    // the file keeps it in ONE module, so nanochat code can coexist with
    // fucina_llm consumers in the same compilation (the lmserve example).
    nanochat_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(nanochat_exe, blas_kind);
    configureGpu(b, nanochat_exe, gpu_kind);
    const nanochat_install = installArtifactStep(b, nanochat_exe);

    const nanochat_cmd = b.addRunArtifact(nanochat_exe);
    nanochat_cmd.step.dependOn(nanochat_install);
    if (b.args) |args| {
        nanochat_cmd.addArgs(args);
    }

    const nanochat_step = b.step("nanochat", "nanochat port (karpathy/nanochat): tok-train / base-train / sft / eval-bpb / chat");
    nanochat_step.dependOn(&nanochat_cmd.step);

    const spirals_exe = b.addExecutable(.{
        .name = "fucina-zig-spirals",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/spirals.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    spirals_exe.root_module.addImport("fucina", module);
    configureBlas(spirals_exe, blas_kind);
    configureGpu(b, spirals_exe, gpu_kind);
    const spirals_install = installArtifactStep(b, spirals_exe);

    const spirals_cmd = b.addRunArtifact(spirals_exe);
    spirals_cmd.step.dependOn(spirals_install);
    if (b.args) |args| {
        spirals_cmd.addArgs(args);
    }

    const spirals_step = b.step("spirals", "Train a two-spirals MLP with SGD/AdamW/Muon/APOLLO (+groups/schedule/clip), checkpoint, resume, infer");
    spirals_step.dependOn(&spirals_cmd.step);

    const nam_exe = b.addExecutable(.{
        .name = "fucina-zig-nam",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/nam.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    nam_exe.root_module.addImport("fucina", module);
    configureBlas(nam_exe, blas_kind);
    configureGpu(b, nam_exe, gpu_kind);
    configureNamAudio(nam_exe);
    const nam_install = installArtifactStep(b, nam_exe);

    const nam_cmd = b.addRunArtifact(nam_exe);
    nam_cmd.step.dependOn(nam_install);
    if (b.args) |args| {
        nam_cmd.addArgs(args);
    }

    const nam_step = b.step("nam", "Neural Amp Modeler: .nam profiles, profiling/training, live amp sim");
    nam_step.dependOn(&nam_cmd.step);

    const qwen3_exe = b.addExecutable(.{
        .name = "fucina-zig-qwen3",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/qwen3.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    qwen3_exe.root_module.addImport("fucina", module);
    qwen3_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(qwen3_exe, blas_kind);
    configureGpu(b, qwen3_exe, gpu_kind);
    configureLlguidance(qwen3_exe, llguidance_dep);
    const qwen3_install = installArtifactStep(b, qwen3_exe);

    const qwen3_cmd = b.addRunArtifact(qwen3_exe);
    qwen3_cmd.step.dependOn(qwen3_install);
    if (b.args) |args| {
        qwen3_cmd.addArgs(args);
    }

    const qwen3_step = b.step("qwen3", "Run Qwen3 dense/MoE GGUF inference (text chat; --spec/--spec-ref lossless speculative decode, --tokenize tokenizer-parity oracle)");
    qwen3_step.dependOn(&qwen3_cmd.step);

    const deepseek2_exe = b.addExecutable(.{
        .name = "fucina-zig-deepseek2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/deepseek2.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    deepseek2_exe.root_module.addImport("fucina", module);
    deepseek2_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(deepseek2_exe, blas_kind);
    configureGpu(b, deepseek2_exe, gpu_kind);
    const deepseek2_install = installArtifactStep(b, deepseek2_exe);

    const deepseek2_cmd = b.addRunArtifact(deepseek2_exe);
    deepseek2_cmd.step.dependOn(deepseek2_install);
    if (b.args) |args| {
        deepseek2_cmd.addArgs(args);
    }

    const deepseek2_step = b.step("deepseek2", "Run DeepSeek-V2 family (MLA + MoE) GGUF inference");
    deepseek2_step.dependOn(&deepseek2_cmd.step);

    const inkling_exe = b.addExecutable(.{
        .name = "fucina-zig-inkling",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/inkling.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    inkling_exe.root_module.addImport("fucina", module);
    inkling_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(inkling_exe, blas_kind);
    configureGpu(b, inkling_exe, gpu_kind);
    const inkling_install = installArtifactStep(b, inkling_exe);

    const inkling_cmd = b.addRunArtifact(inkling_exe);
    inkling_cmd.step.dependOn(inkling_install);
    if (b.args) |args| {
        inkling_cmd.addArgs(args);
    }

    const inkling_step = b.step("inkling", "Run Inkling (hybrid SWA + rel-bias + MoE) GGUF inference");
    inkling_step.dependOn(&inkling_cmd.step);

    const glm4moe_exe = b.addExecutable(.{
        .name = "fucina-zig-glm4moe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/glm4moe.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    glm4moe_exe.root_module.addImport("fucina", module);
    glm4moe_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(glm4moe_exe, blas_kind);
    configureGpu(b, glm4moe_exe, gpu_kind);
    const glm4moe_install = installArtifactStep(b, glm4moe_exe);

    const glm4moe_cmd = b.addRunArtifact(glm4moe_exe);
    glm4moe_cmd.step.dependOn(glm4moe_install);
    if (b.args) |args| {
        glm4moe_cmd.addArgs(args);
    }

    const glm4moe_step = b.step("glm4moe", "Run GLM-4.5 family GGUF inference (--mtp native multi-token-prediction speculative decode)");
    glm4moe_step.dependOn(&glm4moe_cmd.step);

    const deepseek4_exe = b.addExecutable(.{
        .name = "fucina-zig-deepseek4",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/deepseek4.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    deepseek4_exe.root_module.addImport("fucina", module);
    deepseek4_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(deepseek4_exe, blas_kind);
    configureGpu(b, deepseek4_exe, gpu_kind);
    const deepseek4_install = installArtifactStep(b, deepseek4_exe);

    const deepseek4_cmd = b.addRunArtifact(deepseek4_exe);
    deepseek4_cmd.step.dependOn(deepseek4_install);
    if (b.args) |args| {
        deepseek4_cmd.addArgs(args);
    }

    const deepseek4_step = b.step("deepseek4", "Run DeepSeek V4 Flash GGUF inference (CSA/HCA + streamed experts)");
    deepseek4_step.dependOn(&deepseek4_cmd.step);

    const omnivoice_exe = b.addExecutable(.{
        .name = "fucina-zig-omnivoice",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/omnivoice.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    omnivoice_exe.root_module.addImport("fucina", module);
    omnivoice_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(omnivoice_exe, blas_kind);
    configureGpu(b, omnivoice_exe, gpu_kind);
    configureOmnivoiceAudio(omnivoice_exe);
    const omnivoice_install = installArtifactStep(b, omnivoice_exe);

    const omnivoice_cmd = b.addRunArtifact(omnivoice_exe);
    omnivoice_cmd.step.dependOn(omnivoice_install);
    if (b.args) |args| {
        omnivoice_cmd.addArgs(args);
    }

    const omnivoice_step = b.step("omnivoice", "OmniVoice MaskGIT TTS from GGUF: voice cloning/design, codec encode/decode");
    omnivoice_step.dependOn(&omnivoice_cmd.step);

    const locate_anything_exe = b.addExecutable(.{
        .name = "fucina-zig-locate-anything",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/locate_anything.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    locate_anything_exe.root_module.addImport("fucina", module);
    locate_anything_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(locate_anything_exe, blas_kind);
    configureGpu(b, locate_anything_exe, gpu_kind);
    const locate_anything_install = installArtifactStep(b, locate_anything_exe);

    const locate_anything_cmd = b.addRunArtifact(locate_anything_exe);
    locate_anything_cmd.step.dependOn(locate_anything_install);
    if (b.args) |args| {
        locate_anything_cmd.addArgs(args);
    }

    const locate_anything_step = b.step("locate-anything", "LocateAnything-3B open-vocabulary detection from GGUF: detect/info, parity oracles, bench");
    locate_anything_step.dependOn(&locate_anything_cmd.step);

    const finetune_exe = b.addExecutable(.{
        .name = "fucina-zig-finetune",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/finetune.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    finetune_exe.root_module.addImport("fucina", module);
    finetune_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(finetune_exe, blas_kind);
    configureGpu(b, finetune_exe, gpu_kind);
    const finetune_install = installArtifactStep(b, finetune_exe);

    const finetune_cmd = b.addRunArtifact(finetune_exe);
    finetune_cmd.step.dependOn(finetune_install);
    if (b.args) |args| {
        finetune_cmd.addArgs(args);
    }

    const finetune_step = b.step("finetune", "LoRA fine-tune Qwen3 GGUF on a tiny built-in SFT dataset");
    finetune_step.dependOn(&finetune_cmd.step);

    const cartridge_exe = b.addExecutable(.{
        .name = "fucina-zig-cartridge",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/cartridge.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cartridge_exe.root_module.addImport("fucina", module);
    cartridge_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(cartridge_exe, blas_kind);
    configureGpu(b, cartridge_exe, gpu_kind);
    const cartridge_install = installArtifactStep(b, cartridge_exe);

    const cartridge_cmd = b.addRunArtifact(cartridge_exe);
    cartridge_cmd.step.dependOn(cartridge_install);
    if (b.args) |args| {
        cartridge_cmd.addArgs(args);
    }

    const cartridge_step = b.step("cartridge", "Train/serve a corpus as a trained KV prefix on a Qwen3 GGUF (arXiv 2506.06266)");
    cartridge_step.dependOn(&cartridge_cmd.step);

    const engram_exe = b.addExecutable(.{
        .name = "fucina-zig-engram",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/engram.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    engram_exe.root_module.addImport("fucina", module);
    engram_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(engram_exe, blas_kind);
    configureGpu(b, engram_exe, gpu_kind);
    const engram_install = installArtifactStep(b, engram_exe);

    const engram_cmd = b.addRunArtifact(engram_exe);
    engram_cmd.step.dependOn(engram_install);
    if (b.args) |args| {
        engram_cmd.addArgs(args);
    }

    const engram_step = b.step("engram", "Graft conditional n-gram memory onto a frozen Qwen3 GGUF and train it (arXiv 2601.07372)");
    engram_step.dependOn(&engram_cmd.step);

    const es_finetune_exe = b.addExecutable(.{
        .name = "fucina-zig-es-finetune",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/es_finetune.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    es_finetune_exe.root_module.addImport("fucina", module);
    es_finetune_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(es_finetune_exe, blas_kind);
    configureGpu(b, es_finetune_exe, gpu_kind);
    const es_finetune_install = installArtifactStep(b, es_finetune_exe);

    const es_finetune_cmd = b.addRunArtifact(es_finetune_exe);
    es_finetune_cmd.step.dependOn(es_finetune_install);
    if (b.args) |args| {
        es_finetune_cmd.addArgs(args);
    }

    const es_finetune_step = b.step("es-finetune", "Evolution-strategies fine-tune Qwen3 GGUF (gradient-free; --mode lora|full, --reward rule|nll|acc)");
    es_finetune_step.dependOn(&es_finetune_cmd.step);

    const es_spirals_exe = b.addExecutable(.{
        .name = "fucina-zig-es-spirals",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/es_spirals.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    es_spirals_exe.root_module.addImport("fucina", module);
    configureBlas(es_spirals_exe, blas_kind);
    configureGpu(b, es_spirals_exe, gpu_kind);
    const es_spirals_install = installArtifactStep(b, es_spirals_exe);

    const es_spirals_cmd = b.addRunArtifact(es_spirals_exe);
    es_spirals_cmd.step.dependOn(es_spirals_install);
    if (b.args) |args| {
        es_spirals_cmd.addArgs(args);
    }

    const es_spirals_step = b.step("es-spirals", "Train the two-spirals MLP FROM SCRATCH with evolution strategies (gradient-free; self-verifying)");
    es_spirals_step.dependOn(&es_spirals_cmd.step);

    const es_ternary_spirals_exe = b.addExecutable(.{
        .name = "fucina-zig-es-ternary-spirals",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/es_ternary_spirals.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    es_ternary_spirals_exe.root_module.addImport("fucina", module);
    configureBlas(es_ternary_spirals_exe, blas_kind);
    configureGpu(b, es_ternary_spirals_exe, gpu_kind);
    const es_ternary_spirals_install = installArtifactStep(b, es_ternary_spirals_exe);

    const es_ternary_spirals_cmd = b.addRunArtifact(es_ternary_spirals_exe);
    es_ternary_spirals_cmd.step.dependOn(es_ternary_spirals_install);
    if (b.args) |args| {
        es_ternary_spirals_cmd.addArgs(args);
    }

    const es_ternary_spirals_step = b.step("es-ternary-spirals", "Train a two-spirals MLP FROM SCRATCH with the ternary-native ES (packed TQ2_0 genome = the inference model; self-verifying)");
    es_ternary_spirals_step.dependOn(&es_ternary_spirals_cmd.step);

    const ptqtp_spirals_exe = b.addExecutable(.{
        .name = "fucina-zig-ptqtp-spirals",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/ptqtp_spirals.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    ptqtp_spirals_exe.root_module.addImport("fucina", module);
    configureBlas(ptqtp_spirals_exe, blas_kind);
    configureGpu(b, ptqtp_spirals_exe, gpu_kind);
    const ptqtp_spirals_install = installArtifactStep(b, ptqtp_spirals_exe);

    const ptqtp_spirals_cmd = b.addRunArtifact(ptqtp_spirals_exe);
    ptqtp_spirals_cmd.step.dependOn(ptqtp_spirals_install);
    if (b.args) |args| {
        ptqtp_spirals_cmd.addArgs(args);
    }

    const ptqtp_spirals_step = b.step("ptqtp-spirals", "Train a float two-spirals MLP, then post-training-quantize it to dual trit-planes (PTQTP over packed TQ2_0; self-verifying)");
    ptqtp_spirals_step.dependOn(&ptqtp_spirals_cmd.step);

    const ptqtp_qwen3_exe = b.addExecutable(.{
        .name = "fucina-zig-ptqtp-qwen3",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/ptqtp_qwen3.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    ptqtp_qwen3_exe.root_module.addImport("fucina", module);
    ptqtp_qwen3_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(ptqtp_qwen3_exe, blas_kind);
    configureGpu(b, ptqtp_qwen3_exe, gpu_kind);
    const ptqtp_qwen3_install = installArtifactStep(b, ptqtp_qwen3_exe);

    const ptqtp_qwen3_cmd = b.addRunArtifact(ptqtp_qwen3_exe);
    ptqtp_qwen3_cmd.step.dependOn(ptqtp_qwen3_install);
    if (b.args) |args| {
        ptqtp_qwen3_cmd.addArgs(args);
    }

    const ptqtp_qwen3_step = b.step("ptqtp-qwen3", "PTQTP-decorate a Qwen3 GGUF's linears in place (any source dtype) and compare teacher-forced NLL before/after + greedy completion");
    ptqtp_qwen3_step.dependOn(&ptqtp_qwen3_cmd.step);

    const gemma4_exe = b.addExecutable(.{
        .name = "fucina-zig-gemma4",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/gemma4.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gemma4_exe.root_module.addImport("fucina", module);
    gemma4_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(gemma4_exe, blas_kind);
    configureGpu(b, gemma4_exe, gpu_kind);
    configureLlguidance(gemma4_exe, llguidance_dep);
    const gemma4_install = installArtifactStep(b, gemma4_exe);

    const gemma4_cmd = b.addRunArtifact(gemma4_exe);
    gemma4_cmd.step.dependOn(gemma4_install);
    if (b.args) |args| {
        gemma4_cmd.addArgs(args);
    }

    const gemma4_step = b.step("gemma4", "Run Gemma 4 GGUF inference from token IDs (logit-parity harness)");
    gemma4_step.dependOn(&gemma4_cmd.step);

    const lmserve_exe = b.addExecutable(.{
        .name = "fucina-zig-lmserve",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/lmserve.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lmserve_exe.root_module.addImport("fucina", module);
    lmserve_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(lmserve_exe, blas_kind);
    configureGpu(b, lmserve_exe, gpu_kind);
    configureLlguidance(lmserve_exe, llguidance_dep);
    // Uses std.c.shutdown/recv (signal-driven accept unblock, MSG_PEEK
    // hang-up probe): libc links implicitly on macOS but must be declared
    // for the Linux leg.
    lmserve_exe.root_module.link_libc = true;
    const lmserve_install = installArtifactStep(b, lmserve_exe);

    const lmserve_cmd = b.addRunArtifact(lmserve_exe);
    lmserve_cmd.step.dependOn(lmserve_install);
    if (b.args) |args| {
        lmserve_cmd.addArgs(args);
    }

    const lmserve_step = b.step("lmserve", "OpenAI-compatible language-model HTTP server (chat completions + responses; SSE streaming; JSON-schema constrained output with -Dllguidance=true) over qwen3/gemma4/diffusion-gemma GGUFs + nanochat checkpoints");
    lmserve_step.dependOn(&lmserve_cmd.step);

    const parakeet_exe = b.addExecutable(.{
        .name = "fucina-zig-parakeet",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/parakeet.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    parakeet_exe.root_module.addImport("fucina", module);
    parakeet_exe.root_module.addImport("fucina_llm", llm_module);
    const parakeet_opts = b.addOptions();
    parakeet_opts.addOption(bool, "parakeet_mic", parakeet_mic);
    parakeet_exe.root_module.addOptions("build_options", parakeet_opts);
    configureBlas(parakeet_exe, blas_kind);
    configureGpu(b, parakeet_exe, gpu_kind);
    if (parakeet_mic) configureParakeetAudio(parakeet_exe); // --mic: vendored miniaudio capture
    const parakeet_install = installArtifactStep(b, parakeet_exe);

    const parakeet_cmd = b.addRunArtifact(parakeet_exe);
    parakeet_cmd.step.dependOn(parakeet_install);
    if (b.args) |args| {
        parakeet_cmd.addArgs(args);
    }

    const parakeet_step = b.step("parakeet", "Parakeet ASR (NeMo FastConformer): transcribe a WAV (mel -> encoder -> CTC/TDT decoder -> text); --stream/--manifest/--mic, --compare parity harness");
    parakeet_step.dependOn(&parakeet_cmd.step);

    const bench_gate_cmd = b.addSystemCommand(&.{ "python3", "tools/bench_gate.py" });
    bench_gate_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bench_gate_cmd.addArgs(args);
    }
    const bench_gate_step = b.step("bench-gate", "Run paired Fucina-vs-llama benchmark gate");
    bench_gate_step.dependOn(&bench_gate_cmd.step);

    const diffusion_gemma_exe = b.addExecutable(.{
        .name = "fucina-zig-diffusion-gemma",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/diffusion_gemma.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    diffusion_gemma_exe.root_module.addImport("fucina", module);
    diffusion_gemma_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(diffusion_gemma_exe, blas_kind);
    configureGpu(b, diffusion_gemma_exe, gpu_kind);
    diffusion_gemma_exe.root_module.link_libc = true;
    const diffusion_gemma_install = installArtifactStep(b, diffusion_gemma_exe);

    const diffusion_gemma_cmd = b.addRunArtifact(diffusion_gemma_exe);
    diffusion_gemma_cmd.step.dependOn(diffusion_gemma_install);
    if (b.args) |args| {
        diffusion_gemma_cmd.addArgs(args);
    }

    const diffusion_gemma_step = b.step("diffusion-gemma", "Run DiffusionGemma GGUF block-diffusion inference (parity harness + EB chat)");
    diffusion_gemma_step.dependOn(&diffusion_gemma_cmd.step);

    const qwen35_exe = b.addExecutable(.{
        .name = "fucina-zig-qwen35",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/qwen35.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    qwen35_exe.root_module.addImport("fucina", module);
    qwen35_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(qwen35_exe, blas_kind);
    configureGpu(b, qwen35_exe, gpu_kind);
    const qwen35_install = installArtifactStep(b, qwen35_exe);

    const qwen35_cmd = b.addRunArtifact(qwen35_exe);
    qwen35_cmd.step.dependOn(qwen35_install);
    if (b.args) |args| {
        qwen35_cmd.addArgs(args);
    }

    const qwen35_step = b.step("qwen35", "Run Qwen3.5 (qwen35 hybrid Gated-DeltaNet) GGUF — loader/parity harness");
    qwen35_step.dependOn(&qwen35_cmd.step);

    const export_gguf_exe = b.addExecutable(.{
        .name = "fucina-zig-export-gguf",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/export_gguf.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    export_gguf_exe.root_module.addImport("fucina", module);
    export_gguf_exe.root_module.addImport("fucina_llm", llm_module);
    configureBlas(export_gguf_exe, blas_kind);
    configureGpu(b, export_gguf_exe, gpu_kind);
    const export_gguf_install = installArtifactStep(b, export_gguf_exe);

    const export_gguf_cmd = b.addRunArtifact(export_gguf_exe);
    export_gguf_cmd.step.dependOn(export_gguf_install);
    if (b.args) |args| {
        export_gguf_cmd.addArgs(args);
    }

    const export_gguf_step = b.step("export-gguf", "Export a GGUF: re-emit/transcode a model, merge Fucina LoRA adapters (checkpoint dir or safetensors) into dense weights, or PTQTP-quantize tensor-at-a-time (--ptqtp[=K]; models bigger than RAM)");
    export_gguf_step.dependOn(&export_gguf_cmd.step);

    const arch_check_exe = b.addExecutable(.{
        .name = "fucina-zig-arch-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_import_graph.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const arch_check_cmd = b.addRunArtifact(arch_check_exe);
    const arch_check_step = b.step("arch-check", "Verify the production src/*.zig import graph has zero SCCs");
    arch_check_step.dependOn(&arch_check_cmd.step);

    const doc_check_exe = b.addExecutable(.{
        .name = "fucina-zig-doc-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_doc_links.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const doc_check_cmd = b.addRunArtifact(doc_check_exe);
    const doc_check_step = b.step("doc-check", "Verify AGENTS.md's doc index references only root .md files that exist");
    doc_check_step.dependOn(&doc_check_cmd.step);

    // REFERENCE.md snippet gate: extract every runnable ```zig snippet (a
    // block with a column-0 `test` decl; `<!-- snippet: helper/skip -->`
    // markers documented in tools/gen_snippet_tests.zig) into a generated
    // test root and run it against the real fucina/fucina_llm modules, so a
    // doc example that stops compiling or asserting fails the build — the
    // doc-check counterpart for snippet rot.
    const snippet_gen_exe = b.addExecutable(.{
        .name = "fucina-zig-gen-snippet-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_snippet_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const snippet_gen_run = b.addRunArtifact(snippet_gen_exe);
    snippet_gen_run.addFileArg(b.path("docs/REFERENCE.md"));
    const snippet_dir = snippet_gen_run.addOutputDirectoryArg("snippets");
    const snippet_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = snippet_dir.path(b, "root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    snippet_tests.root_module.addImport("fucina", module);
    snippet_tests.root_module.addImport("fucina_llm", llm_module);
    configureBlas(snippet_tests, blas_kind);
    configureGpu(b, snippet_tests, gpu_kind);
    configureLlguidance(snippet_tests, llguidance_dep);

    const run_snippet_tests = b.addRunArtifact(snippet_tests);
    const snippet_check_step = b.step("snippet-check", "Extract and run every runnable docs/REFERENCE.md snippet against the real modules");
    snippet_check_step.dependOn(&run_snippet_tests.step);

    // Cross-ISA parity vehicle for the int8 dot primitives + Q4_K/Q8_0 dot
    // kernels (src/x86dot_check.zig — run-book and per-arm coverage table in
    // its header). Always ReleaseSafe: the run-book config. The run leg
    // follows -Dtarget, so the same step drives the emulated x86 legs when
    // cross-invoked (e.g. -Dtarget=x86_64-macos -Dcpu=baseline -frosetta);
    // natively on the aarch64 dev machine it executes the sdot arms.
    const x86dot_check_exe = b.addExecutable(.{
        .name = "fucina-zig-x86dot-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/x86dot_check.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
        }),
    });

    const x86dot_check_cmd = b.addRunArtifact(x86dot_check_exe);
    const x86dot_check_step = b.step("x86dot-check", "Run the cross-ISA int8/Q4_K/Q8_0/TQ2_0 dot parity checker (follows -Dtarget) + compile-only AVX2/VNNI/smmla bit-rot legs");
    x86dot_check_step.dependOn(&x86dot_check_cmd.step);

    // Compile-only legs: one CPU model per feature gate that no local
    // substrate can execute (backend/quant/common.zig has_x86_avx2/avxvnni/
    // avx512vnni + has_aarch64_i8mm). These catch bit-rot of those arms at
    // build time; EXECUTION coverage for them stays the dated attestations in
    // the checker's header (src/x86dot_check.zig).
    const x86dot_check_compile_legs = [_]std.Target.Query{
        // AVX2 sign-trick arm (vpsignb + vpmaddubsw ladder)
        .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu, .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 } },
        // AVX-VNNI (VEX vpdpbusd)
        .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu, .cpu_model = .{ .explicit = &std.Target.x86.cpu.alderlake } },
        // AVX512-VNNI (EVEX vpdpbusd)
        .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu, .cpu_model = .{ .explicit = &std.Target.x86.cpu.znver4 } },
        // aarch64 smmla asm (FEAT_I8MM; M1 lacks it — needs Graviton3+/Grace-class cores)
        .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.neoverse_v1 } },
    };
    for (x86dot_check_compile_legs) |leg_query| {
        const leg_exe = b.addExecutable(.{
            .name = b.fmt("fucina-zig-x86dot-check-{s}", .{leg_query.cpu_model.explicit.name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/x86dot_check.zig"),
                .target = b.resolveTargetQuery(leg_query),
                .optimize = .ReleaseSafe,
            }),
        });
        // Force binary emission (an unconsumed artifact gets -fno-emit-bin):
        // the gated inline-asm arms are only instruction-selected at emit.
        _ = leg_exe.getEmittedBin();
        x86dot_check_step.dependOn(&leg_exe.step);
    }

    // Compile-only CUDA-provider/tooling leg (`zig build cuda-check`):
    // semantically analyzes the -Dgpu=cuda provider, its tests, and the PTX
    // generator for x86_64-linux-gnu without running them, so neither can
    // bit-rot on GPU-less/macOS dev machines — the same discipline as the
    // x86dot-check legs.
    // The dead-switch-arm selection in src/backend/gpu.zig means no other
    // build configuration ever analyzes cuda.zig.
    const cuda_check_step = b.step("cuda-check", "Compile-only -Dgpu=cuda legs (x86_64-linux-gnu fucina + llm roots and NVRTC PTX generator, not run): catches CUDA-provider bit-rot on GPU-less machines");
    {
        const cuda_target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
        const cuda_options = b.addOptions();
        cuda_options.addOption(BackendKind, "backend_kind", backend_kind);
        cuda_options.addOption(BlasKind, "blas_kind", .none);
        cuda_options.addOption(bool, "use_blas", false);
        cuda_options.addOption(u32, "blas_threads", 0);
        cuda_options.addOption(usize, "max_threads", max_threads);
        cuda_options.addOption(bool, "vector_scan", vector_scan);
        cuda_options.addOption(bool, "use_gpu", true);
        cuda_options.addOption(GpuKind, "gpu_kind", .cuda);

        // Leg 1: the fucina root — backend/exec/provider code + provider tests.
        const cuda_fucina_module = b.createModule(.{
            .root_source_file = b.path("src/fucina.zig"),
            .target = cuda_target,
            .optimize = optimize,
        });
        cuda_fucina_module.addOptions("build_options", cuda_options);
        cuda_fucina_module.link_libc = true;
        const cuda_check_fucina = b.addTest(.{ .root_module = cuda_fucina_module });
        _ = cuda_check_fucina.getEmittedBin();
        cuda_check_step.dependOn(&cuda_check_fucina.step);

        // Leg 2: the llm root — the tier that consumes the provider surface
        // exec never touches (residency, qmoeStage, gemmQGroupedNt, ...);
        // without it, drift in those decls stays invisible until a GPU-box
        // build (verified by @compileError probes during review).
        const cuda_llm_module = b.createModule(.{
            .root_source_file = b.path("src/llm.zig"),
            .target = cuda_target,
            .optimize = optimize,
        });
        cuda_llm_module.addImport("fucina", cuda_fucina_module);
        cuda_llm_module.addOptions("llm_build_options", llm_options_off);
        cuda_llm_module.link_libc = true;
        const cuda_check_llm = b.addTest(.{ .root_module = cuda_llm_module });
        _ = cuda_check_llm.getEmittedBin();
        cuda_check_step.dependOn(&cuda_check_llm.step);

        // Leg 3: the toolkit-optional PTX generator. It reuses the provider's
        // dlopen-only NVRTC binding, so compiling this leg needs no CUDA SDK.
        const cuda_api_module = b.createModule(.{
            .root_source_file = b.path("src/backend/cuda/api.zig"),
            .target = cuda_target,
            .optimize = optimize,
        });
        const cuda_ptx_gen_module = b.createModule(.{
            .root_source_file = b.path("tools/gen_cuda_ptx.zig"),
            .target = cuda_target,
            .optimize = optimize,
        });
        cuda_ptx_gen_module.addImport("cuda_api", cuda_api_module);
        cuda_ptx_gen_module.link_libc = true;
        const cuda_ptx_gen = b.addExecutable(.{
            .name = "fucina-zig-gen-cuda-ptx",
            .root_module = cuda_ptx_gen_module,
        });
        _ = cuda_ptx_gen.getEmittedBin();
        cuda_check_step.dependOn(&cuda_ptx_gen.step);
    }

    const bench_exe = b.addExecutable(.{
        .name = "fucina-zig-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/mlp.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const bench_raw_module = b.addModule("bench_raw", .{
        .root_source_file = b.path("src/bench_raw.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_raw_module.addOptions("build_options", options);
    bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(bench_exe, blas_kind);
    configureGpu(b, bench_exe, gpu_kind);

    const bench_cmd = b.addRunArtifact(bench_exe);
    if (b.args) |args| {
        bench_cmd.addArgs(args);
    }

    const bench_step = b.step("bench", "Run MLP-shaped inference and backward benchmarks");
    bench_step.dependOn(&bench_cmd.step);

    const optim_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-optim-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/optim.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    optim_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(optim_bench_exe, blas_kind);
    configureGpu(b, optim_bench_exe, gpu_kind);

    const optim_bench_cmd = b.addRunArtifact(optim_bench_exe);
    if (b.args) |args| {
        optim_bench_cmd.addArgs(args);
    }

    const optim_bench_step = b.step("bench-optim", "Optimizer step kernels (SGD/AdamW/Muon/APOLLO) at LLM shapes");
    optim_bench_step.dependOn(&optim_bench_cmd.step);

    const ce_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-ce-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/ce.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    ce_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(ce_bench_exe, blas_kind);
    configureGpu(b, ce_bench_exe, gpu_kind);

    const ce_bench_cmd = b.addRunArtifact(ce_bench_exe);
    if (b.args) |args| {
        ce_bench_cmd.addArgs(args);
    }

    const ce_bench_step = b.step("bench-ce", "Softmax / cross-entropy row kernels at LLM shapes");
    ce_bench_step.dependOn(&ce_bench_cmd.step);

    const conv_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-conv-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/conv.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    conv_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(conv_bench_exe, blas_kind);
    configureGpu(b, conv_bench_exe, gpu_kind);

    const conv_bench_cmd = b.addRunArtifact(conv_bench_exe);
    if (b.args) |args| {
        conv_bench_cmd.addArgs(args);
    }

    const conv_bench_step = b.step("bench-conv", "conv2d forward/backward-input/backward-weight at CNN shapes");
    conv_bench_step.dependOn(&conv_bench_cmd.step);

    const scatter_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-scatter-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/scatter.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    scatter_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(scatter_bench_exe, blas_kind);
    configureGpu(b, scatter_bench_exe, gpu_kind);

    const scatter_bench_cmd = b.addRunArtifact(scatter_bench_exe);
    if (b.args) |args| {
        scatter_bench_cmd.addArgs(args);
    }

    const scatter_bench_step = b.step("bench-scatter", "Scatter-add (embedding-gradient) kernel at vocab x dim shapes");
    scatter_bench_step.dependOn(&scatter_bench_cmd.step);

    const backward_diamond_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-backward-diamond-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/backward_diamond.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    backward_diamond_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(backward_diamond_bench_exe, blas_kind);
    configureGpu(b, backward_diamond_bench_exe, gpu_kind);

    const backward_diamond_bench_cmd = b.addRunArtifact(backward_diamond_bench_exe);
    if (b.args) |args| {
        backward_diamond_bench_cmd.addArgs(args);
    }

    const backward_diamond_bench_step = b.step("bench-backward-diamond", "Measure serial vs manual-parallel independent GEMM VJPs");
    backward_diamond_bench_step.dependOn(&backward_diamond_bench_cmd.step);

    const attention_backward_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-attention-backward-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/attention_backward.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    attention_backward_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(attention_backward_bench_exe, blas_kind);
    configureGpu(b, attention_backward_bench_exe, gpu_kind);

    const attention_backward_bench_cmd = b.addRunArtifact(attention_backward_bench_exe);
    if (b.args) |args| {
        attention_backward_bench_cmd.addArgs(args);
    }

    const attention_backward_bench_step = b.step("bench-attention-backward", "Measure grouped causal attention backward");
    attention_backward_bench_step.dependOn(&attention_backward_bench_cmd.step);

    // Backend comparison benchmark. No ggml C kernels are linked by the Zig
    // project; pure Zig vector kernels are internal to the native backend.
    const backend_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-backend-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/backend.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const raw_backend_module = b.addModule("raw_backend", .{
        .root_source_file = b.path("src/backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    raw_backend_module.addOptions("build_options", options);
    backend_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    const bench_options = b.addOptions();
    bench_options.addOption(BlasKind, "native_blas_kind", blas_kind);
    bench_options.addOption(bool, "native_uses_blas", blas_kind != .none);
    bench_options.addOption(u32, "native_blas_threads", blas_threads);
    backend_bench_exe.root_module.addOptions("bench_options", bench_options);
    configureBlas(backend_bench_exe, blas_kind);
    configureGpu(b, backend_bench_exe, gpu_kind);

    const backend_bench_cmd = b.addRunArtifact(backend_bench_exe);
    if (b.args) |args| {
        backend_bench_cmd.addArgs(args);
    }

    const backend_bench_step = b.step("bench-backend", "Compare scalar / native backends on representative ops");
    backend_bench_step.dependOn(&backend_bench_cmd.step);

    const f16gemm_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-f16gemm-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/f16gemm.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    f16gemm_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    configureBlas(f16gemm_bench_exe, blas_kind);
    configureGpu(b, f16gemm_bench_exe, gpu_kind);
    const f16gemm_bench_cmd = b.addRunArtifact(f16gemm_bench_exe);
    if (b.args) |args| {
        f16gemm_bench_cmd.addArgs(args);
    }
    const f16gemm_bench_step = b.step("bench-f16gemm", "f16 TransB GEMM parallel-efficiency microbench (Qwen3 shapes)");
    f16gemm_bench_step.dependOn(&f16gemm_bench_cmd.step);

    const gemm_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-gemm-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gemm.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gemm_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    configureBlas(gemm_bench_exe, blas_kind);
    configureGpu(b, gemm_bench_exe, gpu_kind);
    const gemm_bench_cmd = b.addRunArtifact(gemm_bench_exe);
    if (b.args) |args| {
        gemm_bench_cmd.addArgs(args);
    }
    const gemm_bench_step = b.step("bench-gemm", "Large-shape f32 GEMM: row kernels vs cache-blocked packed kernel (+BLAS reference)");
    gemm_bench_step.dependOn(&gemm_bench_cmd.step);

    const gpu_dispatch_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-gpu-dispatch-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gpu_dispatch.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gpu_dispatch_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    configureBlas(gpu_dispatch_bench_exe, blas_kind);
    configureGpu(b, gpu_dispatch_bench_exe, gpu_kind);
    const gpu_dispatch_bench_cmd = b.addRunArtifact(gpu_dispatch_bench_exe);
    if (b.args) |args| gpu_dispatch_bench_cmd.addArgs(args);
    const gpu_dispatch_bench_step = b.step("bench-gpu-dispatch", "CPU BLAS vs synchronous/asynchronous eager GPU GEMM/GEMV dispatch");
    gpu_dispatch_bench_step.dependOn(&gpu_dispatch_bench_cmd.step);

    const gpu_formats_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-gpu-formats-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/gpu_formats.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gpu_formats_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    configureBlas(gpu_formats_bench_exe, blas_kind);
    configureGpu(b, gpu_formats_bench_exe, gpu_kind);
    const gpu_formats_bench_cmd = b.addRunArtifact(gpu_formats_bench_exe);
    if (b.args) |args| gpu_formats_bench_cmd.addArgs(args);
    const gpu_formats_bench_step = b.step("bench-gpu-formats", "Packed CPU vs eager GPU f16/Q4_K/Q5_K/Q6_K/Q8_0 LLM linears");
    gpu_formats_bench_step.dependOn(&gpu_formats_bench_cmd.step);

    const q5kmoe_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-q5kmoe-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/q5kmoe.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    q5kmoe_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    // Uses std.heap.c_allocator directly: libc links implicitly on macOS but
    // must be declared for the Linux bench-check leg.
    q5kmoe_bench_exe.root_module.link_libc = true;
    configureBlas(q5kmoe_bench_exe, blas_kind);
    configureGpu(b, q5kmoe_bench_exe, gpu_kind);
    const q5kmoe_bench_cmd = b.addRunArtifact(q5kmoe_bench_exe);
    if (b.args) |args| {
        q5kmoe_bench_cmd.addArgs(args);
    }
    const q5kmoe_bench_step = b.step("bench-q5kmoe", "Q5_K MoE-expert matmul: per-row vs 4-row lane-packed col-outer");
    q5kmoe_bench_step.dependOn(&q5kmoe_bench_cmd.step);

    const ternary_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-ternary-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/ternary.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    ternary_bench_exe.root_module.addImport("raw_backend", raw_backend_module);
    // Uses std.heap.c_allocator directly (see q5kmoe above).
    ternary_bench_exe.root_module.link_libc = true;
    configureBlas(ternary_bench_exe, blas_kind);
    configureGpu(b, ternary_bench_exe, gpu_kind);
    const ternary_bench_cmd = b.addRunArtifact(ternary_bench_exe);
    if (b.args) |args| {
        ternary_bench_cmd.addArgs(args);
    }
    const ternary_bench_step = b.step("bench-ternary", "TQ2_0 ternary matmul: hot sdot/vpdpbusd tiles vs cold table path, f32-act path, Q4_K, dense f32");
    ternary_bench_step.dependOn(&ternary_bench_cmd.step);

    const facade_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-facade-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/facade.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    facade_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(facade_bench_exe, blas_kind);
    configureGpu(b, facade_bench_exe, gpu_kind);

    const facade_bench_cmd = b.addRunArtifact(facade_bench_exe);
    if (b.args) |args| {
        facade_bench_cmd.addArgs(args);
    }

    const facade_bench_step = b.step("bench-facade", "Compare raw tensor ops with the public no-grad Tensor facade");
    facade_bench_step.dependOn(&facade_bench_cmd.step);

    const einsum_bench_exe = b.addExecutable(.{
        .name = "fucina-zig-einsum-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/einsum.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    einsum_bench_exe.root_module.addImport("bench_raw", bench_raw_module);
    configureBlas(einsum_bench_exe, blas_kind);
    configureGpu(b, einsum_bench_exe, gpu_kind);

    const einsum_bench_cmd = b.addRunArtifact(einsum_bench_exe);
    if (b.args) |args| {
        einsum_bench_cmd.addArgs(args);
    }

    const einsum_bench_step = b.step("bench-einsum", "einsum vs hand-written dot/permute contraction pipelines (parity + advantage cases)");
    einsum_bench_step.dependOn(&einsum_bench_cmd.step);

    // Compile every bench executable without running it. Bench mains are
    // reachable only through their run steps, so nothing else in the build
    // graph exercises them; this step is the cheap gate that keeps the suite
    // compiling.
    const bench_check_step = b.step("bench-check", "Compile all bench executables without running them");
    for ([_]*std.Build.Step.Compile{
        bench_exe,
        optim_bench_exe,
        ce_bench_exe,
        conv_bench_exe,
        scatter_bench_exe,
        backward_diamond_bench_exe,
        attention_backward_bench_exe,
        backend_bench_exe,
        f16gemm_bench_exe,
        gemm_bench_exe,
        q5kmoe_bench_exe,
        ternary_bench_exe,
        facade_bench_exe,
        einsum_bench_exe,
    }) |bench_check_exe| {
        bench_check_step.dependOn(&bench_check_exe.step);
    }

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/fucina.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    tests.root_module.addOptions("build_options", options);
    configureBlas(tests, blas_kind);
    configureGpu(b, tests, gpu_kind);

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Kernel/spec leg: the fucina root alone — every exec/backend/optim/ag
    // kernel test lives under it, which is the scalar reference backend's
    // whole specification surface. Routine `-Dbackend=scalar` runs use this;
    // the full nine-root `test` matrix stays the pre-merge gate.
    const test_fucina_step = b.step("test-fucina", "Run the fucina-root unit tests only (routine -Dbackend=scalar leg)");
    test_fucina_step.dependOn(&run_tests.step);

    const llm_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/llm.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    llm_tests.root_module.addImport("fucina", module);
    llm_tests.root_module.addOptions("llm_build_options", llm_options);
    configureBlas(llm_tests, blas_kind);
    configureGpu(b, llm_tests, gpu_kind);
    configureLlguidance(llm_tests, llguidance_dep);

    const run_llm_tests = b.addRunArtifact(llm_tests);
    test_step.dependOn(&run_llm_tests.step);

    const lmserve_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/lmserve.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lmserve_tests.root_module.addImport("fucina", module);
    lmserve_tests.root_module.addImport("fucina_llm", llm_module);
    configureBlas(lmserve_tests, blas_kind);
    configureGpu(b, lmserve_tests, gpu_kind);
    configureLlguidance(lmserve_tests, llguidance_dep);
    lmserve_tests.root_module.link_libc = true;

    const run_lmserve_tests = b.addRunArtifact(lmserve_tests);
    test_step.dependOn(&run_lmserve_tests.step);

    const nam_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/nam.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    nam_tests.root_module.addImport("fucina", module);
    configureBlas(nam_tests, blas_kind);
    configureGpu(b, nam_tests, gpu_kind);
    configureNamAudio(nam_tests);

    const run_nam_tests = b.addRunArtifact(nam_tests);
    test_step.dependOn(&run_nam_tests.step);

    const parakeet_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/parakeet.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    parakeet_tests.root_module.addImport("fucina", module);
    parakeet_tests.root_module.addImport("fucina_llm", llm_module);
    parakeet_tests.root_module.addOptions("build_options", parakeet_opts);
    configureBlas(parakeet_tests, blas_kind);
    configureGpu(b, parakeet_tests, gpu_kind);
    if (parakeet_mic) configureParakeetAudio(parakeet_tests);

    const run_parakeet_tests = b.addRunArtifact(parakeet_tests);
    test_step.dependOn(&run_parakeet_tests.step);

    const omnivoice_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/omnivoice.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    omnivoice_tests.root_module.addImport("fucina", module);
    omnivoice_tests.root_module.addImport("fucina_llm", llm_module);
    configureBlas(omnivoice_tests, blas_kind);
    configureGpu(b, omnivoice_tests, gpu_kind);
    configureOmnivoiceAudio(omnivoice_tests);

    const run_omnivoice_tests = b.addRunArtifact(omnivoice_tests);
    test_step.dependOn(&run_omnivoice_tests.step);

    const locate_anything_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/locate_anything.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    locate_anything_tests.root_module.addImport("fucina", module);
    locate_anything_tests.root_module.addImport("fucina_llm", llm_module);
    configureBlas(locate_anything_tests, blas_kind);
    configureGpu(b, locate_anything_tests, gpu_kind);

    const run_locate_anything_tests = b.addRunArtifact(locate_anything_tests);
    test_step.dependOn(&run_locate_anything_tests.step);

    const facedetect_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/facedetect.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    facedetect_tests.root_module.addImport("fucina", module);
    configureBlas(facedetect_tests, blas_kind);
    configureGpu(b, facedetect_tests, gpu_kind);

    const run_facedetect_tests = b.addRunArtifact(facedetect_tests);
    test_step.dependOn(&run_facedetect_tests.step);

    const nanochat_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/nanochat.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    nanochat_tests.root_module.addImport("fucina", module);
    nanochat_tests.root_module.addImport("fucina_llm", llm_module);
    configureBlas(nanochat_tests, blas_kind);
    configureGpu(b, nanochat_tests, gpu_kind);

    const run_nanochat_tests = b.addRunArtifact(nanochat_tests);
    test_step.dependOn(&run_nanochat_tests.step);
}

const LlguidanceDep = struct {
    build_step: *std.Build.Step,
    lib: std.Build.LazyPath,
};

/// Link the vendored llguidance staticlib (built by the cargo step) into a
/// compile step that references the `llm.llguidance` externs. No-op when
/// -Dllguidance is off. The Rust staticlib needs libc, and — because its FFI
/// converts panics to error strings via catch_unwind (panic=unwind) — an
/// unwinder: macOS's libSystem ships one, but glibc's libc does not export
/// `_Unwind_*`, so non-macOS targets link Zig's bundled LLVM libunwind
/// through link_libcpp (hermetic; no system libgcc_s dependency).
fn configureLlguidance(step: *std.Build.Step.Compile, dep: ?LlguidanceDep) void {
    const d = dep orelse return;
    step.root_module.addObjectFile(d.lib);
    step.root_module.link_libc = true;
    if (step.root_module.resolved_target.?.result.os.tag != .macos) {
        step.root_module.link_libcpp = true;
    }
    step.step.dependOn(d.build_step);
}

/// Install `exe` under the default install step (plain `zig build` still
/// installs every artifact) and return the artifact's own InstallArtifact
/// step. Per-example run commands depend on this instead of the global
/// install step so `zig build <example>` builds only that executable.
fn installArtifactStep(b: *std.Build, exe: *std.Build.Step.Compile) *std.Build.Step {
    const install = b.addInstallArtifact(exe, .{});
    b.getInstallStep().dependOn(&install.step);
    return &install.step;
}

/// The NAM example's audio/MIDI device layer: two C TUs (vendored
/// miniaudio + shim ABI, and a CoreMIDI shim that compiles to stubs off
/// macOS). On macOS the CoreAudio/CoreMIDI frameworks are linked directly
/// (MA_NO_RUNTIME_LINKING in the audio shim); elsewhere miniaudio dlopens
/// its backend at runtime through libc.
fn configureNamAudio(step: *std.Build.Step.Compile) void {
    const module = step.root_module;
    module.link_libc = true;
    module.addCSourceFile(.{
        .file = step.step.owner.path("examples/nam/audio_shim.c"),
        .flags = &.{ "-fno-sanitize=undefined", "-O2" },
    });
    module.addCSourceFile(.{
        .file = step.step.owner.path("examples/nam/midi_shim.c"),
        .flags = &.{ "-fno-sanitize=undefined", "-O2" },
    });
    const target = module.resolved_target.?.result;
    if (target.os.tag == .macos) {
        module.linkFramework("CoreFoundation", .{});
        module.linkFramework("CoreAudio", .{});
        module.linkFramework("AudioToolbox", .{});
        module.linkFramework("CoreMIDI", .{});
    }
}

/// The OmniVoice example's speaker-playback layer (`--play`): NAM's vendored
/// miniaudio TU (`examples/nam/audio_shim.c`, the single
/// MINIAUDIO_IMPLEMENTATION build) plus the playback-only shim
/// (`examples/omnivoice/play_shim.c`) that links against it. No MIDI. On
/// macOS the CoreAudio frameworks are linked directly (MA_NO_RUNTIME_LINKING
/// in the audio shim); elsewhere miniaudio dlopens its backend through libc.
fn configureOmnivoiceAudio(step: *std.Build.Step.Compile) void {
    const module = step.root_module;
    module.link_libc = true;
    module.addCSourceFile(.{
        .file = step.step.owner.path("examples/nam/audio_shim.c"),
        .flags = &.{ "-fno-sanitize=undefined", "-O2" },
    });
    module.addCSourceFile(.{
        .file = step.step.owner.path("examples/omnivoice/play_shim.c"),
        .flags = &.{ "-fno-sanitize=undefined", "-O2" },
    });
    const target = module.resolved_target.?.result;
    if (target.os.tag == .macos) {
        module.linkFramework("CoreFoundation", .{});
        module.linkFramework("CoreAudio", .{});
        module.linkFramework("AudioToolbox", .{});
    }
}

/// Link ONLY the miniaudio capture shim (no MIDI) into the parakeet example for
/// `--mic` (`-Dparakeet-mic`). Reuses NAM's vendored `examples/nam/audio_shim.c`
/// + `third_party/miniaudio.h`; macOS needs the CoreAudio frameworks (elsewhere
/// miniaudio dlopens its backend).
fn configureParakeetAudio(step: *std.Build.Step.Compile) void {
    const module = step.root_module;
    module.link_libc = true;
    module.addCSourceFile(.{
        .file = step.step.owner.path("examples/nam/audio_shim.c"),
        .flags = &.{ "-fno-sanitize=undefined", "-O2" },
    });
    const target = module.resolved_target.?.result;
    if (target.os.tag == .macos) {
        module.linkFramework("CoreFoundation", .{});
        module.linkFramework("CoreAudio", .{});
        module.linkFramework("AudioToolbox", .{});
    }
}

fn configureBlas(
    step: *std.Build.Step.Compile,
    blas_kind: BlasKind,
) void {
    const module = step.root_module;
    switch (blas_kind) {
        .none => {},
        .accelerate => {
            module.link_libc = true;
            module.linkFramework("Accelerate", .{});
        },
        .openblas => {
            module.link_libc = true;
            addLibrarySearchPath(step, "/opt/homebrew/opt/openblas");
            addLibrarySearchPath(step, "/usr/local/opt/openblas");
            module.linkSystemLibrary("openblas", .{});
        },
        .mkl => {
            module.link_libc = true;
            addLibrarySearchPath(step, "/opt/intel/oneapi/mkl/latest");
            module.linkSystemLibrary("mkl_rt", .{});
        },
        .blis => {
            module.link_libc = true;
            addLibrarySearchPath(step, "/opt/homebrew/opt/blis");
            addLibrarySearchPath(step, "/usr/local/opt/blis");
            module.linkSystemLibrary("blis", .{});
        },
        .nvpl => {
            module.link_libc = true;
            addLibrarySearchPath(step, "/opt/nvidia/hpc_sdk");
            module.linkSystemLibrary("nvpl_blas", .{});
        },
        .blas => {
            module.link_libc = true;
            addLibrarySearchPath(step, "/opt/homebrew/opt/openblas");
            addLibrarySearchPath(step, "/usr/local/opt/openblas");
            addLibrarySearchPath(step, "/opt/homebrew/opt/blis");
            addLibrarySearchPath(step, "/usr/local/opt/blis");
            module.linkSystemLibrary("blas", .{});
        },
    }
}

fn configureGpu(
    b: *std.Build,
    step: *std.Build.Step.Compile,
    gpu_kind: GpuKind,
) void {
    const module = step.root_module;
    switch (gpu_kind) {
        .none => {},
        .metal => {
            module.link_libc = true;
            module.linkFramework("Metal", .{});
            module.linkFramework("Foundation", .{});
            module.addCSourceFile(.{
                .file = b.path("src/backend/metal/shim.m"),
                .flags = &.{"-fobjc-arc"},
            });
        },
        // No SDK, no shim, no link-time CUDA dependency: the provider
        // resolves libcuda/libcublas at runtime via std.DynLib, which only
        // needs libc.
        .cuda => {
            module.link_libc = true;
        },
    }
}

fn addLibrarySearchPath(step: *std.Build.Step.Compile, prefix: []const u8) void {
    // Only add directories that exist: zig 0.16's build runner treats any
    // stderr from a compile step (e.g. "unable to open library directory"
    // warnings for the missing Homebrew prefixes on Linux) as a step
    // failure, so a speculative search path breaks `-Dblas=openblas` exe
    // builds on Linux outright.
    const lib_dir = bPath(prefix, "lib");
    std.Io.Dir.accessAbsolute(step.step.owner.graph.io, lib_dir, .{}) catch return;
    const lib_path = std.Build.LazyPath{ .cwd_relative = lib_dir };
    step.root_module.addLibraryPath(lib_path);
    step.root_module.addRPath(lib_path);
}

fn bPath(prefix: []const u8, suffix: []const u8) []const u8 {
    return std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}", .{ prefix, suffix }) catch @panic("failed to allocate build path");
}
