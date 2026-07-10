// la_dump — parity-oracle dump harness for the locate-anything port.
//
// Out-of-tree consumer of the pinned reference (refs/locate-anything.cpp @
// 92c1682, built per its README): runs the reference pipeline on one image +
// prompt and writes every stage checkpoint the Zig port gates against into a
// single GGUF (read back by examples/locate_anything.zig `--compare`).
//
// Build (from the repo root, with the reference built at refs/locate-anything.cpp/build):
//   c++ -std=c++17 -O2 tools/ref-patches/la_dump.cpp \
//       -Irefs/locate-anything.cpp/src -Irefs/locate-anything.cpp/third_party/ggml/include \
//       refs/locate-anything.cpp/build/liblocate_anything.a \
//       refs/locate-anything.cpp/build/third_party/ggml/src/libggml.a \
//       refs/locate-anything.cpp/build/third_party/ggml/src/libggml-cpu.a \
//       refs/locate-anything.cpp/build/third_party/ggml/src/ggml-blas/libggml-blas.a \
//       refs/locate-anything.cpp/build/third_party/ggml/src/libggml-base.a \
//       -framework Accelerate -framework Foundation -o /tmp/la_dump
//
// Run:
//   /tmp/la_dump <model.gguf> <image> <prompt> <out.gguf> [max_new] [mtp_rounds]
//
// Dump layout (ggml ne convention: ne0 = fastest/innermost axis):
//   prompt_ids            i32 [n_prompt]
//   pixel_values          f32 [588, n_tok]          (per-patch (c,i,j), token-major)
//   vit_patch_pos         f32 [1152, n_tok]         patch_embed + interpolated pos-emb
//   vit_block0            f32 [1152, n_tok]         output of encoder block 0
//   vit_layer_26          f32 [1152, n_tok]         output of encoder block 26
//   vit_final             f32 [1152, n_tok]         after final_layernorm
//   merged                f32 [4608, n_merged]      2x2 patch merge
//   projected             f32 [2048, n_merged]      projector output
//   embeds_spliced        f32 [2048, n_prompt]      token embeds + vision splice
//   lm_layer_00           f32 [2048, n_prompt]      Qwen2 decoder layer 0 output
//   lm_layer_35           f32 [2048, n_prompt]      Qwen2 decoder layer 35 output
//   logits_step0          f32 [vocab]               last-position prefill logits
//   stream_slow           i32 [...]                 greedy AR tokens (resident KV)
//   stream_hybrid         i32 [...]                 hybrid PBD tokens (early_stop=false)
//   stream_fast           i32 [...]                 fast MTP-only tokens (early_stop=false)
//   mtp_logits6_rNNN      f32 [vocab, 6]            hybrid MTP round NNN block logits
//   tok_case_NN           i32 [...]                 tokenizer parity cases (see kv tok_case_NN.text)
// KV: la.gh, la.gw, la.target_w, la.target_h, la.n_prompt, la.mtp_rounds, la.max_new
#include "model_loader.hpp"
#include "backend.hpp"
#include "tokenizer.hpp"
#include "prompt.hpp"
#include "image_io.hpp"
#include "vit_encoder.hpp"
#include "projector.hpp"
#include "lm.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static ggml_context* g_ctx = nullptr;
static gguf_context* g_out = nullptr;

static void put_f32(const char* name, const std::vector<float>& v, int64_t ne0, int64_t ne1) {
    ggml_tensor* t = ne1 > 0 ? ggml_new_tensor_2d(g_ctx, GGML_TYPE_F32, ne0, ne1)
                             : ggml_new_tensor_1d(g_ctx, GGML_TYPE_F32, ne0);
    if ((size_t)ggml_nelements(t) != v.size()) {
        std::fprintf(stderr, "FATAL: %s size mismatch: %zu vs %lld\n", name, v.size(), (long long)ggml_nelements(t));
        std::exit(1);
    }
    std::memcpy(t->data, v.data(), v.size() * sizeof(float));
    ggml_set_name(t, name);
    gguf_add_tensor(g_out, t);
}

static void put_i32(const char* name, const std::vector<int32_t>& v) {
    // gguf rejects zero-size tensors; encode emptiness as a [1] sentinel of -1
    // plus a kv flag the reader checks.
    if (v.empty()) {
        ggml_tensor* t = ggml_new_tensor_1d(g_ctx, GGML_TYPE_I32, 1);
        ((int32_t*)t->data)[0] = -1;
        ggml_set_name(t, name);
        gguf_add_tensor(g_out, t);
        gguf_set_val_bool(g_out, (std::string(name) + ".empty").c_str(), true);
        return;
    }
    ggml_tensor* t = ggml_new_tensor_1d(g_ctx, GGML_TYPE_I32, (int64_t)v.size());
    std::memcpy(t->data, v.data(), v.size() * sizeof(int32_t));
    ggml_set_name(t, name);
    gguf_add_tensor(g_out, t);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <image> <prompt> <out.gguf> [max_new] [mtp_rounds]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1], image_path = argv[2], prompt = argv[3], out_path = argv[4];
    const int max_new = argc > 5 ? std::atoi(argv[5]) : 256;
    const int mtp_rounds_keep = argc > 6 ? std::atoi(argv[6]) : 12;

    ggml_init_params ip{};
    ip.mem_size = (size_t)1536 * 1024 * 1024;
    ip.no_alloc = false;
    g_ctx = ggml_init(ip);
    g_out = gguf_init_empty();

    la::ModelLoader ml;
    if (!ml.load(model_path)) { std::fprintf(stderr, "model load failed\n"); return 1; }
    la::Backend be;
    be.set_n_threads(8);
    if (!ml.offload_weights(be)) { std::fprintf(stderr, "offload failed\n"); return 1; }
    la::Tokenizer tok;
    if (!tok.load(ml)) { std::fprintf(stderr, "tokenizer load failed\n"); return 1; }

    // ---- tokenizer parity cases (rung 1) ----
    const char* tok_cases[] = {
        "cat</c>remote.",
        "Locate all the instances that matches the following description: person</c>traffic light.",
        "hello world",
        "  leading and   trailing  runs   \n\nnewlines\r\n mixed",
        "I'm can't they'll we'd it's you're I've O'Neill",
        "emoji \xF0\x9F\x98\x80\xF0\x9F\xA7\x91\xE2\x80\x8D\xF0\x9F\xA4\x9D\xE2\x80\x8D\xF0\x9F\xA7\x91 zwj",
        "unicode: \xC3\xA9\xC3\xA0\xC3\xBC \xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95 \xD0\xBA\xD0\xB8\xD1\x80\xD0\xB8\xD0\xBB\xD0\xBB\xD0\xB8\xD1\x86\xD0\xB0",
        "int main(){return 0;}//x+=1e-9f;\n\tfor(;;){}",
        "numbers 1234567890 3.14159 0x1p-3",
        "<|im_start|>user<IMG_CONTEXT></img><|im_end|><box><ref>fake specials</ref>",
        "bare <| pipes |> and <notatoken> here",
        "<0><500><1000>",
    };
    for (size_t i = 0; i < sizeof(tok_cases) / sizeof(tok_cases[0]); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "tok_case_%02zu", i);
        put_i32(name, tok.encode(tok_cases[i]));
        gguf_set_val_str(g_out, (std::string(name) + ".text").c_str(), tok_cases[i]);
    }

    // ---- preprocess ----
    la::Image img;
    if (!la::load_image_rgb(image_path, img)) { std::fprintf(stderr, "image load failed\n"); return 1; }
    la::Preprocessed P;
    if (!la::preprocess(img, P)) { std::fprintf(stderr, "preprocess failed\n"); return 1; }
    const int n_tok = P.gh * P.gw;
    gguf_set_val_u32(g_out, "la.gh", (uint32_t)P.gh);
    gguf_set_val_u32(g_out, "la.gw", (uint32_t)P.gw);
    gguf_set_val_u32(g_out, "la.target_w", (uint32_t)P.target_w);
    gguf_set_val_u32(g_out, "la.target_h", (uint32_t)P.target_h);
    gguf_set_val_u32(g_out, "la.max_new", (uint32_t)max_new);
    put_f32("pixel_values", P.pixel_values, 588, n_tok);

    // ---- prompt ----
    std::vector<int32_t> prompt_ids = la::build_prompt(tok, P.gh, P.gw, prompt);
    put_i32("prompt_ids", prompt_ids);
    gguf_set_val_u32(g_out, "la.n_prompt", (uint32_t)prompt_ids.size());

    // ---- vision tower ----
    std::vector<float> patch_pos, block0, vfinal;
    std::vector<std::vector<float>> caps;
    la::VitEncoder vit(ml, be);
    if (!vit.patch_and_pos(P.pixel_values, P.gh, P.gw, patch_pos)) return 1;
    put_f32("vit_patch_pos", patch_pos, 1152, n_tok);
    if (!vit.block0(P.pixel_values, P.gh, P.gw, block0)) return 1;
    put_f32("vit_block0", block0, 1152, n_tok);
    if (!vit.forward(P.pixel_values, P.gh, P.gw, vfinal, {26}, caps)) return 1;
    put_f32("vit_layer_26", caps[0], 1152, n_tok);
    put_f32("vit_final", vfinal, 1152, n_tok);

    std::vector<float> merged = la::merge_patches(vfinal, P.gh, P.gw, 1152);
    const int n_merged = (P.gh / 2) * (P.gw / 2);
    put_f32("merged", merged, 4608, n_merged);

    la::Projector proj(ml, be);
    std::vector<float> projected;
    if (!proj.project(merged, projected)) return 1;
    put_f32("projected", projected, 2048, n_merged);

    // ---- LM prefill ----
    la::LMForward lm(ml, be);
    std::vector<float> spliced;
    if (!lm.embed_and_splice(prompt_ids, projected, spliced)) return 1;
    put_f32("embeds_spliced", spliced, 2048, (int64_t)prompt_ids.size());
    std::vector<float> logits;
    std::vector<std::vector<float>> lmcaps;
    if (!lm.forward(prompt_ids, projected, logits, {0, 35}, lmcaps)) return 1;
    put_f32("lm_layer_00", lmcaps[0], 2048, (int64_t)prompt_ids.size());
    put_f32("lm_layer_35", lmcaps[1], 2048, (int64_t)prompt_ids.size());
    put_f32("logits_step0", logits, (int64_t)logits.size(), 0);

    // ---- decode streams (early_stop=false: the full reference streams) ----
    std::vector<int32_t> ids_slow, ids_hybrid, ids_fast;
    if (!lm.decode_greedy_resident(prompt_ids, projected, max_new, ids_slow)) return 1;
    put_i32("stream_slow", ids_slow);
    std::vector<std::vector<float>> mtp_logits;
    if (!lm.decode_hybrid(prompt_ids, projected, max_new, ids_hybrid, &mtp_logits, false, false)) return 1;
    put_i32("stream_hybrid", ids_hybrid);
    const int rounds = (int)mtp_logits.size();
    gguf_set_val_u32(g_out, "la.mtp_rounds", (uint32_t)rounds);
    for (int r = 0; r < rounds && r < mtp_rounds_keep; ++r) {
        char name[32];
        std::snprintf(name, sizeof(name), "mtp_logits6_r%03d", r);
        put_f32(name, mtp_logits[r], (int64_t)(mtp_logits[r].size() / 6), 6);
    }
    if (!lm.decode_hybrid(prompt_ids, projected, max_new, ids_fast, nullptr, true, false)) return 1;
    put_i32("stream_fast", ids_fast);

    if (!gguf_write_to_file(g_out, out_path.c_str(), false)) { std::fprintf(stderr, "gguf write failed\n"); return 1; }
    std::fprintf(stderr, "wrote %s (gh=%d gw=%d prompt=%zu slow=%zu hybrid=%zu fast=%zu rounds=%d)\n",
                 out_path.c_str(), P.gh, P.gw, prompt_ids.size(), ids_slow.size(), ids_hybrid.size(), ids_fast.size(), rounds);
    gguf_free(g_out);
    ggml_free(g_ctx);
    return 0;
}
