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

    // Cross-example reuse: a module cannot @import above its root source
    // file's directory, so example code shared across folders is exposed as
    // named modules, created ONCE from the SAME fucina/fucina_llm modules
    // above (type identity across every consumer).
    const facedetect_image_module = b.createModule(.{
        .root_source_file = b.path("examples/facedetect/image.zig"),
        .target = target,
        .optimize = optimize,
    });
    const nam_audio_module = b.createModule(.{
        .root_source_file = b.path("examples/nam/audio.zig"),
        .target = target,
        .optimize = optimize,
    });
    const nanochat_module = b.createModule(.{
        .root_source_file = b.path("examples/nanochat/nanochat.zig"),
        .target = target,
        .optimize = optimize,
    });
    nanochat_module.addImport("fucina", module);
    nanochat_module.addImport("fucina_llm", llm_module);

    const tool_ctx: ToolCtx = .{ .target = target, .optimize = optimize, .module = module, .llm_module = llm_module, .blas_kind = blas_kind, .gpu_kind = gpu_kind };

    const smoke = addExample(b, tool_ctx, .{ .step = "smoke", .desc = "Run the smoke example", .exe = "fucina-smoke", .root = "examples/smoke/main.zig", .llm = false });
    const run_step = b.step("run", "Run the smoke example (alias of smoke)");
    run_step.dependOn(&smoke.run.step);

    _ = addExample(b, tool_ctx, .{ .step = "facedetect", .desc = "Face detection/recognition (face-detect.cpp buffalo_l port): detect/embed/verify/analyze/landmarks", .exe = "fucina-facedetect", .root = "examples/facedetect/main.zig", .llm = false });
    // nanochat's raw-byte BPE pretokenizer reuses the generated \p{L}/\p{N}/\s
    // tables via the fucina_llm re-export (llm.unicode_categories) — sharing
    // the file keeps it in ONE module, so nanochat code can coexist with
    // fucina_llm consumers in the same compilation (the lmserve example).
    _ = addExample(b, tool_ctx, .{ .step = "nanochat", .desc = "nanochat port (karpathy/nanochat): tok-train / base-train / sft / eval-bpb / chat", .exe = "fucina-nanochat", .root = "examples/nanochat/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "spirals", .desc = "Train a two-spirals MLP with SGD/AdamW/Muon/APOLLO (+groups/schedule/clip), checkpoint, resume, infer", .exe = "fucina-spirals", .root = "examples/spirals/main.zig", .llm = false });
    const nam = addExample(b, tool_ctx, .{ .step = "nam", .desc = "Neural Amp Modeler: .nam profiles, profiling/training, live amp sim", .exe = "fucina-nam", .root = "examples/nam/main.zig", .llm = false });
    configureNamAudio(nam.exe);
    const qwen3 = addExample(b, tool_ctx, .{ .step = "qwen3", .desc = "Run Qwen3 dense/MoE GGUF inference (text chat; --spec/--spec-ref lossless speculative decode, --tokenize tokenizer-parity oracle)", .exe = "fucina-qwen3", .root = "examples/qwen3/main.zig", .llm = true });
    configureLlguidance(qwen3.exe, llguidance_dep);
    _ = addExample(b, tool_ctx, .{ .step = "deepseek2", .desc = "Run DeepSeek-V2 family (MLA + MoE) GGUF inference", .exe = "fucina-deepseek2", .root = "examples/deepseek2/main.zig", .llm = true });
    const inkling = addExample(b, tool_ctx, .{ .step = "inkling", .desc = "Run Inkling (hybrid SWA + rel-bias + MoE) GGUF inference", .exe = "fucina-inkling", .root = "examples/inkling/main.zig", .llm = true });
    inkling.exe.root_module.addImport("facedetect_image", facedetect_image_module);
    _ = addExample(b, tool_ctx, .{ .step = "glm4moe", .desc = "Run GLM-4.5 family GGUF inference (--mtp native multi-token-prediction speculative decode)", .exe = "fucina-glm4moe", .root = "examples/glm4moe/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "deepseek4", .desc = "Run DeepSeek V4 Flash GGUF inference (CSA/HCA + streamed experts)", .exe = "fucina-deepseek4", .root = "examples/deepseek4/main.zig", .llm = true });
    const omnivoice = addExample(b, tool_ctx, .{ .step = "omnivoice", .desc = "OmniVoice MaskGIT TTS from GGUF: voice cloning/design, codec encode/decode", .exe = "fucina-omnivoice", .root = "examples/omnivoice/main.zig", .llm = true });
    configureOmnivoiceAudio(omnivoice.exe);
    _ = addExample(b, tool_ctx, .{ .step = "locate-anything", .desc = "LocateAnything-3B open-vocabulary detection from GGUF: detect/info, parity oracles, bench", .exe = "fucina-locate-anything", .root = "examples/locate_anything/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "finetune", .desc = "LoRA fine-tune Qwen3 GGUF on a tiny built-in SFT dataset", .exe = "fucina-finetune", .root = "examples/finetune/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "cartridge", .desc = "Train/serve a corpus as a trained KV prefix on a Qwen3 GGUF (arXiv 2506.06266)", .exe = "fucina-cartridge", .root = "examples/cartridge/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "cartridge-fleet", .desc = "Per-document cartridge fleets: mixed-visibility training, RAM/disk budget manager, cosine cartridge-RAG (arXiv 2606.04557)", .exe = "fucina-cartridge-fleet", .root = "examples/cartridge_fleet/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "engram", .desc = "Graft conditional n-gram memory onto a frozen Qwen3 GGUF and train it (arXiv 2601.07372)", .exe = "fucina-engram", .root = "examples/engram/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "es-finetune", .desc = "Evolution-strategies fine-tune Qwen3 GGUF (gradient-free; --mode lora|full, --reward rule|nll|acc)", .exe = "fucina-es-finetune", .root = "examples/es_finetune/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "es-spirals", .desc = "Train the two-spirals MLP FROM SCRATCH with evolution strategies (gradient-free; self-verifying)", .exe = "fucina-es-spirals", .root = "examples/es_spirals/main.zig", .llm = false });
    _ = addExample(b, tool_ctx, .{ .step = "es-ternary-spirals", .desc = "Train a two-spirals MLP FROM SCRATCH with the ternary-native ES (packed TQ2_0 genome = the inference model; self-verifying)", .exe = "fucina-es-ternary-spirals", .root = "examples/es_ternary_spirals/main.zig", .llm = false });
    _ = addExample(b, tool_ctx, .{ .step = "ptqtp-spirals", .desc = "Train a float two-spirals MLP, then post-training-quantize it to dual trit-planes (PTQTP over packed TQ2_0; self-verifying)", .exe = "fucina-ptqtp-spirals", .root = "examples/ptqtp_spirals/main.zig", .llm = false });
    _ = addExample(b, tool_ctx, .{ .step = "ptqtp-qwen3", .desc = "PTQTP-decorate a Qwen3 GGUF's linears in place (any source dtype) and compare teacher-forced NLL before/after + greedy completion", .exe = "fucina-ptqtp-qwen3", .root = "examples/ptqtp_qwen3/main.zig", .llm = true });
    const gemma4 = addExample(b, tool_ctx, .{ .step = "gemma4", .desc = "Run Gemma 4 GGUF inference from token IDs (logit-parity harness)", .exe = "fucina-gemma4", .root = "examples/gemma4/main.zig", .llm = true });
    configureLlguidance(gemma4.exe, llguidance_dep);
    const lmserve = addExample(b, tool_ctx, .{ .step = "lmserve", .desc = "OpenAI-compatible language-model HTTP server (chat completions + responses; SSE streaming; JSON-schema constrained output with -Dllguidance=true) over qwen3/gemma4/diffusion-gemma GGUFs + nanochat checkpoints", .exe = "fucina-lmserve", .root = "examples/lmserve/main.zig", .llm = true });
    lmserve.exe.root_module.addImport("nanochat", nanochat_module);
    configureLlguidance(lmserve.exe, llguidance_dep);
    // Uses std.c.shutdown/recv (signal-driven accept unblock, MSG_PEEK
    // hang-up probe): libc links implicitly on macOS but must be declared
    // for the Linux leg.
    lmserve.exe.root_module.link_libc = true;
    const parakeet = addExample(b, tool_ctx, .{ .step = "parakeet", .desc = "Parakeet ASR (NeMo FastConformer): transcribe a WAV (mel -> encoder -> CTC/TDT decoder -> text); --stream/--manifest/--mic, --compare parity harness", .exe = "fucina-parakeet", .root = "examples/parakeet/main.zig", .llm = true });
    parakeet.exe.root_module.addImport("nam_audio", nam_audio_module);
    const parakeet_opts = b.addOptions();
    parakeet_opts.addOption(bool, "parakeet_mic", parakeet_mic);
    parakeet.exe.root_module.addOptions("build_options", parakeet_opts);
    if (parakeet_mic) configureParakeetAudio(parakeet.exe); // --mic: vendored miniaudio capture

    const bench_gate_cmd = b.addSystemCommand(&.{ "python3", "tools/bench_gate.py" });
    bench_gate_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bench_gate_cmd.addArgs(args);
    }
    const bench_gate_step = b.step("bench-gate", "Run paired Fucina-vs-llama benchmark gate");
    bench_gate_step.dependOn(&bench_gate_cmd.step);

    const diffusion_gemma = addExample(b, tool_ctx, .{ .step = "diffusion-gemma", .desc = "Run DiffusionGemma GGUF block-diffusion inference (parity harness + EB chat)", .exe = "fucina-diffusion-gemma", .root = "examples/diffusion_gemma/main.zig", .llm = true });
    diffusion_gemma.exe.root_module.link_libc = true;
    _ = addExample(b, tool_ctx, .{ .step = "qwen35", .desc = "Run Qwen3.5 (qwen35 hybrid Gated-DeltaNet) GGUF — loader/parity harness", .exe = "fucina-qwen35", .root = "examples/qwen35/main.zig", .llm = true });
    _ = addExample(b, tool_ctx, .{ .step = "export-gguf", .desc = "Export a GGUF: re-emit/transcode a model, merge Fucina LoRA adapters (checkpoint dir or safetensors) into dense weights, or PTQTP-quantize tensor-at-a-time (--ptqtp[=K]; models bigger than RAM)", .exe = "fucina-export-gguf", .root = "tools/export_gguf.zig", .llm = true });

    const arch_check_exe = b.addExecutable(.{
        .name = "fucina-arch-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_import_graph.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const arch_check_cmd = b.addRunArtifact(arch_check_exe);
    const arch_check_step = b.step("arch-check", "Verify the production src/*.zig import graph has zero SCCs and every test file is forwarded");
    arch_check_step.dependOn(&arch_check_cmd.step);

    const doc_check_exe = b.addExecutable(.{
        .name = "fucina-doc-check",
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
        .name = "fucina-gen-snippet-tests",
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
        .name = "fucina-x86dot-check",
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
            .name = b.fmt("fucina-x86dot-check-{s}", .{leg_query.cpu_model.explicit.name}),
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
            .name = "fucina-gen-cuda-ptx",
            .root_module = cuda_ptx_gen_module,
        });
        _ = cuda_ptx_gen.getEmittedBin();
        cuda_check_step.dependOn(&cuda_ptx_gen.step);
    }

    // Compile every bench executable without running it. Bench mains are
    // reachable only through their run steps, so nothing else in the build
    // graph exercises them; this step is the cheap gate that keeps the suite
    // compiling. addBench registers every bench into it.
    const bench_check_step = b.step("bench-check", "Compile all bench executables without running them");

    const bench_raw_module = b.addModule("bench_raw", .{
        .root_source_file = b.path("src/bench_raw.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_raw_module.addOptions("build_options", options);
    const raw_backend_module = b.addModule("raw_backend", .{
        .root_source_file = b.path("src/backend.zig"),
        .target = target,
        .optimize = optimize,
    });
    raw_backend_module.addOptions("build_options", options);

    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench", .desc = "Run MLP-shaped inference and backward benchmarks", .exe = "fucina-bench", .root = "bench/mlp.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-optim", .desc = "Optimizer step kernels (SGD/AdamW/Muon/APOLLO) at LLM shapes", .exe = "fucina-optim-bench", .root = "bench/optim.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-ce", .desc = "Softmax / cross-entropy row kernels at LLM shapes", .exe = "fucina-ce-bench", .root = "bench/ce.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-conv", .desc = "conv2d forward/backward-input/backward-weight at CNN shapes", .exe = "fucina-conv-bench", .root = "bench/conv.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-scatter", .desc = "Scatter-add (embedding-gradient) kernel at vocab x dim shapes", .exe = "fucina-scatter-bench", .root = "bench/scatter.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-backward-diamond", .desc = "Measure serial vs manual-parallel independent GEMM VJPs", .exe = "fucina-backward-diamond-bench", .root = "bench/backward_diamond.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-attention-backward", .desc = "Measure grouped causal attention backward", .exe = "fucina-attention-backward-bench", .root = "bench/attention_backward.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    const backend_bench = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-backend", .desc = "Compare scalar / native backends on representative ops", .exe = "fucina-backend-bench", .root = "bench/backend.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module } });
    const bench_options = b.addOptions();
    bench_options.addOption(BlasKind, "native_blas_kind", blas_kind);
    bench_options.addOption(bool, "native_uses_blas", blas_kind != .none);
    bench_options.addOption(u32, "native_blas_threads", blas_threads);
    backend_bench.root_module.addOptions("bench_options", bench_options);
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-f16gemm", .desc = "f16 TransB GEMM parallel-efficiency microbench (Qwen3 shapes)", .exe = "fucina-f16gemm-bench", .root = "bench/f16gemm.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-gemm", .desc = "Large-shape f32 GEMM: row kernels vs cache-blocked packed kernel (+BLAS reference)", .exe = "fucina-gemm-bench", .root = "bench/gemm.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-train-step", .desc = "Full GPT autograd training step; pairs with tools/torch_train_step.py", .exe = "fucina-train-step-bench", .root = "bench/train_step.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-packed-gemm", .desc = "Pack-once dense GEMM at skinny-m inference shapes", .exe = "fucina-packed-gemm-bench", .root = "bench/packed_gemm.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-gpu-dispatch", .desc = "CPU BLAS vs synchronous/asynchronous eager GPU GEMM/GEMV dispatch", .exe = "fucina-gpu-dispatch-bench", .root = "bench/gpu_dispatch.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-gpu-formats", .desc = "Packed CPU vs eager GPU f16/Q4_K/Q5_K/Q6_K/Q8_0 LLM linears", .exe = "fucina-gpu-formats-bench", .root = "bench/gpu_formats.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module } });
    // q5kmoe/q8gemv/ternary use std.c timing and membw is a raw libc probe:
    // libc links implicitly on macOS but must be declared for the Linux
    // bench-check leg.
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-q5kmoe", .desc = "Q5_K MoE-expert matmul: per-row vs 4-row lane-packed col-outer", .exe = "fucina-q5kmoe-bench", .root = "bench/q5kmoe.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module }, .libc = true });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-q8gemv", .desc = "q8_0 skinny-m decode GEMV: per-row vs x4 interleaved vs lane-packed lhs", .exe = "fucina-q8gemv-bench", .root = "bench/q8gemv.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module }, .libc = true });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-ternary", .desc = "TQ2_0 ternary matmul: hot sdot/vpdpbusd tiles vs x4 interleaved pack (A/B pair) vs cold table path, f32-act path, Q4_K, dense f32", .exe = "fucina-ternary-bench", .root = "bench/ternary.zig", .import = .{ .name = "raw_backend", .module = raw_backend_module }, .libc = true });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-membw", .desc = "Measured DRAM read-bandwidth ceiling: single-thread + all-core roofline probe", .exe = "fucina-membw-bench", .root = "bench/membw.zig", .libc = true, .backend = false });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-facade", .desc = "Compare raw tensor ops with the public no-grad Tensor facade", .exe = "fucina-facade-bench", .root = "bench/facade.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-einsum", .desc = "einsum vs hand-written dot/permute contraction pipelines (parity + advantage cases)", .exe = "fucina-einsum-bench", .root = "bench/einsum.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });

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
            .root_source_file = b.path("examples/lmserve/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lmserve_tests.root_module.addImport("fucina", module);
    lmserve_tests.root_module.addImport("fucina_llm", llm_module);
    lmserve_tests.root_module.addImport("nanochat", nanochat_module);
    configureBlas(lmserve_tests, blas_kind);
    configureGpu(b, lmserve_tests, gpu_kind);
    configureLlguidance(lmserve_tests, llguidance_dep);
    lmserve_tests.root_module.link_libc = true;

    const run_lmserve_tests = b.addRunArtifact(lmserve_tests);
    test_step.dependOn(&run_lmserve_tests.step);

    const nam_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/nam/main.zig"),
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
            .root_source_file = b.path("examples/parakeet/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    parakeet_tests.root_module.addImport("fucina", module);
    parakeet_tests.root_module.addImport("fucina_llm", llm_module);
    parakeet_tests.root_module.addImport("nam_audio", nam_audio_module);
    parakeet_tests.root_module.addOptions("build_options", parakeet_opts);
    configureBlas(parakeet_tests, blas_kind);
    configureGpu(b, parakeet_tests, gpu_kind);
    if (parakeet_mic) configureParakeetAudio(parakeet_tests);

    const run_parakeet_tests = b.addRunArtifact(parakeet_tests);
    test_step.dependOn(&run_parakeet_tests.step);

    const omnivoice_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/omnivoice/main.zig"),
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
            .root_source_file = b.path("examples/locate_anything/main.zig"),
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
            .root_source_file = b.path("examples/facedetect/main.zig"),
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
            .root_source_file = b.path("examples/nanochat/main.zig"),
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

const ToolCtx = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    module: *std.Build.Module,
    llm_module: *std.Build.Module,
    blas_kind: BlasKind,
    gpu_kind: GpuKind,
};

const ExampleArtifacts = struct {
    exe: *std.Build.Step.Compile,
    run: *std.Build.Step.Run,
};

/// Standard example/tool wiring: exe + fucina (+ fucina_llm) imports,
/// BLAS/GPU config, install, and a run step forwarding `-- args`. Special
/// per-target wiring (extra imports, libc, llguidance, option modules)
/// attaches to the returned artifacts at the call site — the build graph is
/// declarative, so late additions are equivalent to inline ones.
fn addExample(
    b: *std.Build,
    ctx: ToolCtx,
    spec: struct {
        step: []const u8,
        desc: []const u8,
        exe: []const u8,
        root: []const u8,
        llm: bool = false,
    },
) ExampleArtifacts {
    const exe = b.addExecutable(.{
        .name = spec.exe,
        .root_module = b.createModule(.{
            .root_source_file = b.path(spec.root),
            .target = ctx.target,
            .optimize = ctx.optimize,
        }),
    });
    exe.root_module.addImport("fucina", ctx.module);
    if (spec.llm) exe.root_module.addImport("fucina_llm", ctx.llm_module);
    configureBlas(exe, ctx.blas_kind);
    configureGpu(b, exe, ctx.gpu_kind);
    const install = installArtifactStep(b, exe);
    const cmd = b.addRunArtifact(exe);
    cmd.step.dependOn(install);
    if (b.args) |args| cmd.addArgs(args);
    const step = b.step(spec.step, spec.desc);
    step.dependOn(&cmd.step);
    return .{ .exe = exe, .run = cmd };
}

/// Standard bench wiring: exe (+ its raw module import), optional libc,
/// BLAS/GPU config, registration into bench-check, and a run step
/// forwarding `-- args`. Benches are never installed.
fn addBench(
    b: *std.Build,
    ctx: ToolCtx,
    bench_check_step: *std.Build.Step,
    spec: struct {
        step: []const u8,
        desc: []const u8,
        exe: []const u8,
        root: []const u8,
        import: ?struct { name: []const u8, module: *std.Build.Module } = null,
        libc: bool = false,
        backend: bool = true,
    },
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = spec.exe,
        .root_module = b.createModule(.{
            .root_source_file = b.path(spec.root),
            .target = ctx.target,
            .optimize = ctx.optimize,
        }),
    });
    if (spec.import) |imp| exe.root_module.addImport(imp.name, imp.module);
    if (spec.libc) exe.root_module.link_libc = true;
    if (spec.backend) {
        configureBlas(exe, ctx.blas_kind);
        configureGpu(b, exe, ctx.gpu_kind);
    }
    bench_check_step.dependOn(&exe.step);
    const cmd = b.addRunArtifact(exe);
    if (b.args) |args| cmd.addArgs(args);
    const step = b.step(spec.step, spec.desc);
    step.dependOn(&cmd.step);
    return exe;
}
