const std = @import("std");

const BackendKind = enum { scalar, native };
const BlasKind = enum { none, accelerate, openblas, mkl, blis, nvpl, blas };
const GpuKind = enum { none, metal, cuda };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const backend_kind = b.option(
        BackendKind,
        "backend",
        "Backend implementation: native (Zig SIMD + optional BLAS, default), scalar (reference only)",
    ) orelse .native;
    // macOS always has Accelerate; native Linux probes the system for a
    // provider (cross builds cannot inspect the target machine and default
    // to none, matching the cuda-check legs).
    const default_blas: BlasKind = if (target.result.os.tag == .macos)
        .accelerate
    else if (target.result.os.tag == .linux and target.query.isNative())
        detectNativeLinuxBlas(b, target.result.cpu.arch)
    else
        .none;
    const blas_kind = b.option(
        BlasKind,
        "blas",
        "Native BLAS provider: none, accelerate, openblas, mkl, blis, nvpl, blas",
    ) orelse default_blas;
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
        "GPU GEMM offload provider: none (default), metal (Apple Silicon; big f32/f16 GEMMs, dense quantized linears (Q4_K/Q6_K/Q8_0 prefill), and the MoE expert FFN run on the GPU; decode and training stay on CPU), cuda (Linux/NVIDIA; f32/f16 GEMMs via dlopen'd cuBLAS, Q4_K/Q5_K/Q6_K/Q8_0 prefill + streaming attention forward via vendored PTX kernels, opt-in decode — no CUDA SDK at build time)",
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

    const option_values = OptionValues{
        .backend_kind = backend_kind,
        .blas_kind = blas_kind,
        .blas_threads = blas_threads,
        .max_threads = max_threads,
        .gpu_kind = gpu_kind,
        .vector_scan = vector_scan,
    };
    const options = option_values.addTo(b);

    const module = b.addModule("fucina", .{
        .root_source_file = b.path("src/fucina.zig"),
        .target = target,
        .optimize = optimize,
    });
    module.addOptions("build_options", options);

    // Package-dependency builds (`b.dependency("fucina", ...)`) receive the
    // exported modules without the in-tree executables, so the BLAS/GPU
    // link inputs must travel WITH the module — link inputs propagate
    // through module imports. Dependency-context only: in-tree, the
    // per-executable configure* calls below stay authoritative, and
    // attaching here too would compile the Metal shim into one binary
    // twice.
    if (b.pkg_hash.len != 0) {
        configureBlasModule(b, module, blas_kind);
        configureGpuModule(b, module, gpu_kind);
    }

    // fucina_models's own build options (the fucina options above are per-kernel
    // knobs the models tier never reads). Every module built from src/models.zig
    // must receive one of these under the name "models_build_options".
    const models_options = b.addOptions();
    models_options.addOption(bool, "llguidance", llguidance_enabled);
    const models_options_off = b.addOptions(); // compile-only legs: never link the Rust lib
    models_options_off.addOption(bool, "llguidance", false);

    // -Dllguidance: build the vendored Rust staticlib once per `zig build`
    // invocation (cargo's own incremental cache makes the no-change case
    // sub-second) and link it into the executables/test roots that actually
    // reference the `llm.llguidance` externs. Consumers that merely import
    // fucina_models don't need the link — extern symbols resolve lazily.
    const llguidance_dep: ?LlguidanceDep = if (llguidance_enabled) blk: {
        const cargo = b.addSystemCommand(&.{ "cargo", "build", "--release", "--package", "llguidance" });
        cargo.setCwd(b.path("vendor/llguidance"));
        cargo.has_side_effects = true; // cargo tracks its own inputs; always invoke it
        break :blk .{
            .build_step = &cargo.step,
            .lib = b.path("vendor/llguidance/target/release/libllguidance.a"),
        };
    } else null;

    const models_module = b.addModule("fucina_models", .{
        .root_source_file = b.path("src/models.zig"),
        .target = target,
        .optimize = optimize,
    });
    models_module.addImport("fucina", module);
    models_module.addOptions("models_build_options", models_options);

    // Cross-example reuse: a module cannot @import above its root source
    // file's directory, so example code shared across folders is exposed as
    // named modules, created ONCE from the SAME fucina/fucina_models modules
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
    nanochat_module.addImport("fucina_models", models_module);

    const tool_ctx: ToolCtx = .{ .target = target, .optimize = optimize, .module = module, .models_module = models_module, .blas_kind = blas_kind, .gpu_kind = gpu_kind };

    // SubQ research tools ride addExample for install/run, and their compile
    // steps register into bench-check below so the research surface cannot
    // silently rot out of the compile gate.
    const bench_subq = addExample(b, tool_ctx, .{ .step = "bench-subq", .desc = "Dense vs SubQ decode benchmark on a Qwen3 GGUF (research attention evaluator)", .exe = "fucina-bench-subq", .root = "tools/bench_subq_decode.zig", .models = true });
    const bench_subq_kernels = addExample(b, tool_ctx, .{ .step = "bench-subq-kernels", .desc = "Microbenchmark for the f16 row-block attention primitives", .exe = "fucina-bench-subq-kernels", .root = "tools/bench_subq_kernels.zig", .models = false });
    const bench_subq_scaling = addExample(b, tool_ctx, .{ .step = "bench-subq-scaling", .desc = "Selection-scaling probe: flat vs hierarchical frontier on synthetic clustered KV", .exe = "fucina-bench-subq-scaling", .root = "tools/bench_subq_scaling.zig", .models = true });
    const eval_subq = addExample(b, tool_ctx, .{ .step = "eval-subq-freerun", .desc = "Gate C stage 2: free-running SubQ vs dense generation, loop metrics, dense-judged NLL", .exe = "fucina-eval-subq-freerun", .root = "tools/eval_subq_freerun.zig", .models = true });
    const smoke = addExample(b, tool_ctx, .{ .step = "smoke", .desc = "Run the smoke example", .exe = "fucina-smoke", .root = "examples/smoke/main.zig", .models = false });
    const run_step = b.step("run", "Run the smoke example (alias of smoke)");
    run_step.dependOn(&smoke.run.step);

    _ = addExample(b, tool_ctx, .{ .step = "facedetect", .desc = "Face detection/recognition (face-detect.cpp buffalo_l port): detect/embed/verify/analyze/landmarks", .exe = "fucina-facedetect", .root = "examples/facedetect/main.zig", .models = false });
    // nanochat's raw-byte BPE pretokenizer reuses the generated \p{L}/\p{N}/\s
    // tables via the fucina_models re-export (models.text.unicode_categories) — sharing
    // the file keeps it in ONE module, so nanochat code can coexist with
    // fucina_models consumers in the same compilation (the lmserve example).
    _ = addExample(b, tool_ctx, .{ .step = "nanochat", .desc = "nanochat port (karpathy/nanochat): tok-train / base-train / sft / eval-bpb / chat", .exe = "fucina-nanochat", .root = "examples/nanochat/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "spirals", .desc = "Train a two-spirals MLP with SGD/AdamW/Muon/APOLLO (+groups/schedule/clip), checkpoint, resume, infer", .exe = "fucina-spirals", .root = "examples/spirals/main.zig", .models = false });
    const nam = addExample(b, tool_ctx, .{ .step = "nam", .desc = "Neural Amp Modeler: .nam profiles, profiling/training, live amp sim", .exe = "fucina-nam", .root = "examples/nam/main.zig", .models = false });
    configureNamAudio(nam.exe);
    const qwen3 = addExample(b, tool_ctx, .{ .step = "qwen3", .desc = "Run Qwen3 dense/MoE GGUF inference (text chat; --spec/--spec-ref lossless speculative decode, --tokenize tokenizer-parity oracle, --shine in-context hypernetwork adapters)", .exe = "fucina-qwen3", .root = "examples/qwen3/main.zig", .models = true });
    configureLlguidance(qwen3.exe, llguidance_dep);
    _ = addExample(b, tool_ctx, .{ .step = "deepseek2", .desc = "Run DeepSeek-V2 family (MLA + MoE) GGUF inference", .exe = "fucina-deepseek2", .root = "examples/deepseek2/main.zig", .models = true });
    const inkling = addExample(b, tool_ctx, .{ .step = "inkling", .desc = "Run Inkling (hybrid SWA + rel-bias + MoE) GGUF inference", .exe = "fucina-inkling", .root = "examples/inkling/main.zig", .models = true });
    inkling.exe.root_module.addImport("facedetect_image", facedetect_image_module);
    _ = addExample(b, tool_ctx, .{ .step = "glm4moe", .desc = "Run GLM-4.5 family GGUF inference (--mtp native multi-token-prediction speculative decode)", .exe = "fucina-glm4moe", .root = "examples/glm4moe/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "deepseek4", .desc = "Run DeepSeek V4 Flash GGUF inference (CSA/HCA + streamed experts)", .exe = "fucina-deepseek4", .root = "examples/deepseek4/main.zig", .models = true });
    const voiceagent = addExample(b, tool_ctx, .{ .step = "voiceagent", .desc = "Native cascade voice agent TUI: mic -> parakeet EOU STT -> qwen3 chat -> qwen3-tts -> speakers", .exe = "fucina-voiceagent", .root = "examples/voiceagent/main.zig", .models = true });
    voiceagent.exe.root_module.addImport("nam_audio", nam_audio_module);
    configureAudioShim(voiceagent.exe);
    configureLlguidance(voiceagent.exe, llguidance_dep);
    // The agent hosts the models.text.serving chat server in-process, on a thread;
    // the band's http layer needs libc on Linux (std.c.recv hang-up probe).
    voiceagent.exe.root_module.link_libc = true;
    _ = addExample(b, tool_ctx, .{ .step = "pockettts", .desc = "Pocket TTS v2 from GGUF (kyutai port): continuous-latent flow-matching TTS, streaming Mimi decode", .exe = "fucina-pockettts", .root = "examples/pockettts/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "qwen3tts", .desc = "Qwen3-TTS from GGUF (qwentts.cpp port): CustomVoice text-to-speech, streamed codec decode", .exe = "fucina-qwen3tts", .root = "examples/qwen3tts/main.zig", .models = true });
    const omnivoice = addExample(b, tool_ctx, .{ .step = "omnivoice", .desc = "OmniVoice MaskGIT TTS from GGUF: voice cloning/design, codec encode/decode", .exe = "fucina-omnivoice", .root = "examples/omnivoice/main.zig", .models = true });
    configureOmnivoiceAudio(omnivoice.exe);
    _ = addExample(b, tool_ctx, .{ .step = "locate-anything", .desc = "LocateAnything-3B open-vocabulary detection from GGUF: detect/info, parity oracles, bench", .exe = "fucina-locate-anything", .root = "examples/locate_anything/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "finetune", .desc = "LoRA fine-tune Qwen3 GGUF on a tiny built-in SFT dataset", .exe = "fucina-finetune", .root = "examples/finetune/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "shine-train", .desc = "Train the SHINE hypernetwork (LoRA or cartridge readout) over a frozen qwen3 base", .exe = "fucina-shine-train", .root = "examples/shine_train/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "cartridge", .desc = "Train/serve a corpus as a trained KV prefix on a Qwen3 GGUF (arXiv 2506.06266)", .exe = "fucina-cartridge", .root = "examples/cartridge/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "cartridge-fleet", .desc = "Per-document cartridge fleets: mixed-visibility training, RAM/disk budget manager, cosine cartridge-RAG (arXiv 2606.04557)", .exe = "fucina-cartridge-fleet", .root = "examples/cartridge_fleet/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "engram", .desc = "Graft conditional n-gram memory onto a frozen Qwen3 GGUF and train it (arXiv 2601.07372)", .exe = "fucina-engram", .root = "examples/engram/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "es-finetune", .desc = "Evolution-strategies fine-tune Qwen3 GGUF (gradient-free; --mode lora|full, --reward rule|nll|acc)", .exe = "fucina-es-finetune", .root = "examples/es_finetune/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "es-spirals", .desc = "Train the two-spirals MLP FROM SCRATCH with evolution strategies (gradient-free; self-verifying)", .exe = "fucina-es-spirals", .root = "examples/es_spirals/main.zig", .models = false });
    _ = addExample(b, tool_ctx, .{ .step = "es-ternary-spirals", .desc = "Train a two-spirals MLP FROM SCRATCH with the ternary-native ES (packed TQ2_0 genome = the inference model; self-verifying)", .exe = "fucina-es-ternary-spirals", .root = "examples/es_ternary_spirals/main.zig", .models = false });
    _ = addExample(b, tool_ctx, .{ .step = "ptqtp-spirals", .desc = "Train a float two-spirals MLP, then post-training-quantize it to dual trit-planes (PTQTP over packed TQ2_0; self-verifying)", .exe = "fucina-ptqtp-spirals", .root = "examples/ptqtp_spirals/main.zig", .models = false });
    _ = addExample(b, tool_ctx, .{ .step = "ptqtp-qwen3", .desc = "PTQTP-decorate a Qwen3 GGUF's linears in place (any source dtype) and compare teacher-forced NLL before/after + greedy completion", .exe = "fucina-ptqtp-qwen3", .root = "examples/ptqtp_qwen3/main.zig", .models = true });
    const gemma4 = addExample(b, tool_ctx, .{ .step = "gemma4", .desc = "Run Gemma 4 GGUF inference from token IDs (logit-parity harness)", .exe = "fucina-gemma4", .root = "examples/gemma4/main.zig", .models = true });
    configureLlguidance(gemma4.exe, llguidance_dep);
    const lmserve = addExample(b, tool_ctx, .{ .step = "lmserve", .desc = "OpenAI-compatible language-model HTTP server (chat completions + responses; SSE streaming; JSON-schema constrained output with -Dllguidance=true) over qwen3/gemma4/diffusion-gemma GGUFs + nanochat checkpoints", .exe = "fucina-lmserve", .root = "examples/lmserve/main.zig", .models = true });
    lmserve.exe.root_module.addImport("nanochat", nanochat_module);
    configureLlguidance(lmserve.exe, llguidance_dep);
    // Uses std.c.shutdown/recv (signal-driven accept unblock, MSG_PEEK
    // hang-up probe): libc links implicitly on macOS but must be declared
    // for the Linux leg.
    lmserve.exe.root_module.link_libc = true;
    const parakeet = addExample(b, tool_ctx, .{ .step = "parakeet", .desc = "Parakeet ASR (NeMo FastConformer): transcribe a WAV (mel -> encoder -> CTC/TDT decoder -> text); --stream/--manifest/--mic, --compare parity harness", .exe = "fucina-parakeet", .root = "examples/parakeet/main.zig", .models = true });
    parakeet.exe.root_module.addImport("nam_audio", nam_audio_module);
    const parakeet_opts = b.addOptions();
    parakeet_opts.addOption(bool, "parakeet_mic", parakeet_mic);
    parakeet.exe.root_module.addOptions("build_options", parakeet_opts);
    if (parakeet_mic) configureAudioShim(parakeet.exe); // --mic: vendored miniaudio capture

    const bench_gate_cmd = b.addSystemCommand(&.{ "python3", "tools/bench_gate.py" });
    bench_gate_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bench_gate_cmd.addArgs(args);
    }
    const bench_gate_step = b.step("bench-gate", "Run paired Fucina-vs-llama benchmark gate");
    bench_gate_step.dependOn(&bench_gate_cmd.step);

    const diffusion_gemma = addExample(b, tool_ctx, .{ .step = "diffusion-gemma", .desc = "Run DiffusionGemma GGUF block-diffusion inference (parity harness + EB chat)", .exe = "fucina-diffusion-gemma", .root = "examples/diffusion_gemma/main.zig", .models = true });
    diffusion_gemma.exe.root_module.link_libc = true;
    _ = addExample(b, tool_ctx, .{ .step = "qwen35", .desc = "Run Qwen3.5 (qwen35 hybrid Gated-DeltaNet) GGUF — loader/parity harness", .exe = "fucina-qwen35", .root = "examples/qwen35/main.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "export-gguf", .desc = "Export a GGUF: re-emit/transcode a model, merge Fucina LoRA adapters (checkpoint dir or safetensors) into dense weights, or PTQTP-quantize tensor-at-a-time (--ptqtp[=K]; models bigger than RAM)", .exe = "fucina-export-gguf", .root = "tools/export_gguf.zig", .models = true });
    _ = addExample(b, tool_ctx, .{ .step = "convert-ds4-fp4", .desc = "Convert DeepSeek-V4 fp4 safetensors experts into tied-PTQTP plane stacks over a trunk GGUF (trunk bytes verbatim, experts solved from the released fp4)", .exe = "fucina-convert-ds4-fp4", .root = "tools/convert_ds4_fp4.zig", .models = true });

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

    const replay_experts_exe = b.addExecutable(.{
        .name = "fucina-replay-experts",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/replay_experts.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    const replay_experts_cmd = b.addRunArtifact(replay_experts_exe);
    if (b.args) |args| replay_experts_cmd.addArgs(args);
    const replay_experts_step = b.step("replay-experts", "Replay a --moe-trace routing trace through LRU/Belady/pinned cache policies across capacities");
    replay_experts_step.dependOn(&replay_experts_cmd.step);

    const doc_check_exe = b.addExecutable(.{
        .name = "fucina-doc-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_doc_links.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const doc_check_cmd = b.addRunArtifact(doc_check_exe);
    const doc_check_step = b.step("doc-check", "Verify doc-index references exist and README's fetch pin matches build.zig.zon's version");
    doc_check_step.dependOn(&doc_check_cmd.step);

    // REFERENCE.md snippet gate: extract every runnable ```zig snippet (a
    // block with a column-0 `test` decl; `<!-- snippet: helper/skip -->`
    // markers documented in tools/gen_snippet_tests.zig) into a generated
    // test root and run it against the real fucina/fucina_models modules, so a
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
    snippet_tests.root_module.addImport("fucina_models", models_module);
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
    const cuda_check_step = b.step("cuda-check", "Compile-only -Dgpu=cuda legs (x86_64-linux-gnu fucina + models roots and NVRTC PTX generator, not run): catches CUDA-provider bit-rot on GPU-less machines");
    {
        const cuda_target = b.resolveTargetQuery(.{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .gnu });
        // The cuda leg reuses the SAME option-value set with only the
        // provider fields overridden, so a new build option added to
        // OptionValues can never silently leave this leg checking a stale
        // configuration.
        var cuda_values = option_values;
        cuda_values.blas_kind = .none;
        cuda_values.blas_threads = 0;
        cuda_values.gpu_kind = .cuda;
        const cuda_options = cuda_values.addTo(b);

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

        // Leg 2: the models root — the tier that consumes the provider surface
        // exec never touches (residency, qmoeStage, gemmQGroupedNt, ...);
        // without it, drift in those decls stays invisible until a GPU-box
        // build (verified by @compileError probes during review).
        const cuda_models_module = b.createModule(.{
            .root_source_file = b.path("src/models.zig"),
            .target = cuda_target,
            .optimize = optimize,
        });
        cuda_models_module.addImport("fucina", cuda_fucina_module);
        cuda_models_module.addOptions("models_build_options", models_options_off);
        cuda_models_module.link_libc = true;
        const cuda_check_models = b.addTest(.{ .root_module = cuda_models_module });
        _ = cuda_check_models.getEmittedBin();
        cuda_check_step.dependOn(&cuda_check_models.step);

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
    bench_check_step.dependOn(&bench_subq.exe.step);
    bench_check_step.dependOn(&bench_subq_kernels.exe.step);
    bench_check_step.dependOn(&bench_subq_scaling.exe.step);
    bench_check_step.dependOn(&eval_subq.exe.step);

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
    _ = addBench(b, tool_ctx, bench_check_step, .{ .step = "bench-masked-reduce", .desc = "Masked reductions: fused vs maskedFill+reduce vs unmasked", .exe = "fucina-masked-reduce-bench", .root = "bench/masked_reduce.zig", .import = .{ .name = "bench_raw", .module = bench_raw_module } });
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
    // the full ten-root `test` matrix stays the pre-merge gate.
    const test_fucina_step = b.step("test-fucina", "Run the fucina-root unit tests only (routine -Dbackend=scalar leg)");
    test_fucina_step.dependOn(&run_tests.step);

    const models_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/models.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    models_tests.root_module.addImport("fucina", module);
    models_tests.root_module.addOptions("models_build_options", models_options);
    configureBlas(models_tests, blas_kind);
    configureGpu(b, models_tests, gpu_kind);
    configureLlguidance(models_tests, llguidance_dep);

    const run_models_tests = b.addRunArtifact(models_tests);
    test_step.dependOn(&run_models_tests.step);

    const test_models_step = b.step("test-models", "Run the models-root unit tests only");
    test_models_step.dependOn(&run_models_tests.step);

    // The models.text.serving band's tests ride test-models (Zig collects tests from
    // the root module only). That forces no libc on the models root: the band's
    // tests never reach http's std.c hang-up probe, which lazy analysis
    // leaves out of the test binary.
    const lmserve_tests = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-lmserve", .desc = "Run the lmserve-root unit tests only", .root = "examples/lmserve/main.zig", .models = true });
    lmserve_tests.root_module.addImport("nanochat", nanochat_module);
    configureLlguidance(lmserve_tests, llguidance_dep);
    lmserve_tests.root_module.link_libc = true;

    const nam_tests = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-nam", .desc = "Run the nam-root unit tests only", .root = "examples/nam/main.zig" });
    configureNamAudio(nam_tests);

    const parakeet_tests = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-parakeet", .desc = "Run the parakeet-root unit tests only", .root = "examples/parakeet/main.zig", .models = true });
    parakeet_tests.root_module.addImport("nam_audio", nam_audio_module);
    parakeet_tests.root_module.addOptions("build_options", parakeet_opts);
    if (parakeet_mic) configureAudioShim(parakeet_tests);

    const omnivoice_tests = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-omnivoice", .desc = "Run the omnivoice-root unit tests only", .root = "examples/omnivoice/main.zig", .models = true });
    configureOmnivoiceAudio(omnivoice_tests);

    _ = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-locate-anything", .desc = "Run the locate_anything-root unit tests only", .root = "examples/locate_anything/main.zig", .models = true });

    _ = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-facedetect", .desc = "Run the facedetect-root unit tests only", .root = "examples/facedetect/main.zig" });

    // AEC/duplex leg: the GTCRN parity test gates every stage against the
    // exporter fixtures, so kernel work on aec.zig iterates on the solo step
    // instead of the full matrix.
    const voiceagent_tests = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-voiceagent", .desc = "Run the voiceagent-root unit tests only (GTCRN-AEC parity + duplex gates)", .root = "examples/voiceagent/main.zig", .models = true });
    voiceagent_tests.root_module.addImport("nam_audio", nam_audio_module);
    configureAudioShim(voiceagent_tests);

    _ = addTestRoot(b, tool_ctx, test_step, .{ .step = "test-nanochat", .desc = "Run the nanochat-root unit tests only", .root = "examples/nanochat/main.zig", .models = true });
}

/// Every value behind the `build_options` module, in one struct: the main
/// build and the cuda-check leg both construct their `addOptions` through
/// `addTo`, so the option list exists exactly once.
const OptionValues = struct {
    backend_kind: BackendKind,
    blas_kind: BlasKind,
    blas_threads: u32,
    max_threads: usize,
    gpu_kind: GpuKind,
    vector_scan: bool,

    fn addTo(v: OptionValues, b: *std.Build) *std.Build.Step.Options {
        const options = b.addOptions();
        options.addOption(BackendKind, "backend_kind", v.backend_kind);
        options.addOption(BlasKind, "blas_kind", v.blas_kind);
        options.addOption(bool, "use_blas", v.blas_kind != .none);
        options.addOption(u32, "blas_threads", v.blas_threads);
        options.addOption(usize, "max_threads", v.max_threads);
        options.addOption(bool, "use_gpu", v.gpu_kind != .none);
        options.addOption(GpuKind, "gpu_kind", v.gpu_kind);
        options.addOption(bool, "vector_scan", v.vector_scan);
        return options;
    }
};

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

/// Link ONLY NAM's vendored miniaudio TU (`examples/nam/audio_shim.c` +
/// `third_party/miniaudio.h`: enumeration, capture, duplex — no MIDI, no
/// OmniVoice play shim). Used by parakeet `--mic` (`-Dparakeet-mic`) and the
/// voiceagent duplex stream; macOS links the CoreAudio frameworks directly
/// (MA_NO_RUNTIME_LINKING), elsewhere miniaudio dlopens its backend.
fn configureAudioShim(step: *std.Build.Step.Compile) void {
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
    configureBlasModule(step.step.owner, step.root_module, blas_kind);
}

fn configureBlasModule(
    b: *std.Build,
    module: *std.Build.Module,
    blas_kind: BlasKind,
) void {
    switch (blas_kind) {
        .none => {},
        .accelerate => {
            module.link_libc = true;
            module.linkFramework("Accelerate", .{});
        },
        .openblas => {
            module.link_libc = true;
            addLibrarySearchPath(b, module, "/opt/homebrew/opt/openblas");
            addLibrarySearchPath(b, module, "/usr/local/opt/openblas");
            module.linkSystemLibrary("openblas", .{});
        },
        .mkl => {
            module.link_libc = true;
            addLibrarySearchPath(b, module, "/opt/intel/oneapi/mkl/latest");
            module.linkSystemLibrary("mkl_rt", .{});
        },
        .blis => {
            module.link_libc = true;
            addLibrarySearchPath(b, module, "/opt/homebrew/opt/blis");
            addLibrarySearchPath(b, module, "/usr/local/opt/blis");
            module.linkSystemLibrary("blis", .{});
        },
        .nvpl => {
            module.link_libc = true;
            addLibrarySearchPath(b, module, "/opt/nvidia/hpc_sdk");
            module.linkSystemLibrary("nvpl_blas", .{});
        },
        .blas => {
            module.link_libc = true;
            addLibrarySearchPath(b, module, "/opt/homebrew/opt/openblas");
            addLibrarySearchPath(b, module, "/usr/local/opt/openblas");
            addLibrarySearchPath(b, module, "/opt/homebrew/opt/blis");
            addLibrarySearchPath(b, module, "/usr/local/opt/blis");
            module.linkSystemLibrary("blas", .{});
        },
    }
}

/// Probe the build host for a linkable BLAS provider (native Linux only:
/// cross builds cannot inspect the target machine). The probe is the
/// dynamic-linker cache — the honest proxy for "linkSystemLibrary will
/// find it without extra search paths". Priority is by expected GEMM
/// throughput: NVPL first on aarch64, MKL first on x86-64, then OpenBLAS,
/// then BLIS. The generic netlib `blas` is never auto-selected: reference
/// BLAS is routinely slower than the in-house SIMD kernels, so linking it
/// stays an explicit choice. One stderr line reports the outcome.
fn detectNativeLinuxBlas(b: *std.Build, arch: std.Target.Cpu.Arch) BlasKind {
    var code: u8 = undefined;
    const listing = b.runAllowFail(&.{ "ldconfig", "-p" }, &code, .ignore) catch
        (b.runAllowFail(&.{ "/sbin/ldconfig", "-p" }, &code, .ignore) catch "");
    const kind: BlasKind = if (arch == .aarch64 and std.mem.indexOf(u8, listing, "libnvpl_blas") != null)
        .nvpl
    else if (arch == .x86_64 and std.mem.indexOf(u8, listing, "libmkl_rt.so") != null)
        .mkl
    else if (std.mem.indexOf(u8, listing, "libopenblas.so") != null)
        .openblas
    else if (std.mem.indexOf(u8, listing, "libblis.so") != null)
        .blis
    else
        .none;
    if (kind == .none) {
        std.debug.print("blas: none (no MKL/OpenBLAS/BLIS in the linker cache; -Dblas=<provider> links one explicitly)\n", .{});
    } else {
        std.debug.print("blas: {s} (auto-detected on this Linux host; -Dblas overrides, -Dblas=none disables)\n", .{@tagName(kind)});
    }
    return kind;
}

fn configureGpu(
    b: *std.Build,
    step: *std.Build.Step.Compile,
    gpu_kind: GpuKind,
) void {
    configureGpuModule(b, step.root_module, gpu_kind);
}

fn configureGpuModule(
    b: *std.Build,
    module: *std.Build.Module,
    gpu_kind: GpuKind,
) void {
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

fn addLibrarySearchPath(b: *std.Build, module: *std.Build.Module, prefix: []const u8) void {
    // Only add directories that exist: zig 0.16's build runner treats any
    // stderr from a compile step (e.g. "unable to open library directory"
    // warnings for the missing Homebrew prefixes on Linux) as a step
    // failure, so a speculative search path breaks `-Dblas=openblas` exe
    // builds on Linux outright.
    const lib_dir = b.pathJoin(&.{ prefix, "lib" });
    std.Io.Dir.accessAbsolute(b.graph.io, lib_dir, .{}) catch return;
    const lib_path = std.Build.LazyPath{ .cwd_relative = lib_dir };
    module.addLibraryPath(lib_path);
    module.addRPath(lib_path);
}

const ToolCtx = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    module: *std.Build.Module,
    models_module: *std.Build.Module,
    blas_kind: BlasKind,
    gpu_kind: GpuKind,
};

/// Standard test-root wiring: addTest + fucina (+ fucina_models) imports,
/// BLAS/GPU config, registration into `zig build test`, and a solo
/// `test-<name>` step so any root iterates without the full matrix.
/// Special per-root wiring (extra imports, option modules, libc,
/// llguidance, audio shims) attaches to the returned artifact at the call
/// site, exactly like `addExample`.
fn addTestRoot(
    b: *std.Build,
    ctx: ToolCtx,
    test_step: *std.Build.Step,
    spec: struct {
        step: []const u8,
        desc: []const u8,
        root: []const u8,
        models: bool = false,
    },
) *std.Build.Step.Compile {
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path(spec.root),
            .target = ctx.target,
            .optimize = ctx.optimize,
        }),
    });
    tests.root_module.addImport("fucina", ctx.module);
    if (spec.models) tests.root_module.addImport("fucina_models", ctx.models_module);
    configureBlas(tests, ctx.blas_kind);
    configureGpu(b, tests, ctx.gpu_kind);
    const run = b.addRunArtifact(tests);
    test_step.dependOn(&run.step);
    b.step(spec.step, spec.desc).dependOn(&run.step);
    return tests;
}

const ExampleArtifacts = struct {
    exe: *std.Build.Step.Compile,
    run: *std.Build.Step.Run,
};

/// Standard example/tool wiring: exe + fucina (+ fucina_models) imports,
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
        models: bool = false,
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
    if (spec.models) exe.root_module.addImport("fucina_models", ctx.models_module);
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
