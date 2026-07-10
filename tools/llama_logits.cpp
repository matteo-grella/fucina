#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    // usage: <model.gguf> <comma-ids> <out.bin>
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <comma-ids> <out.bin>\n", argv[0]);
        return 2;
    }

    std::vector<llama_token> ids;
    std::string s = argv[2];
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        std::string t = (j == std::string::npos) ? s.substr(i) : s.substr(i, j - i);
        if (!t.empty()) ids.push_back(std::atoi(t.c_str()));
        if (j == std::string::npos) break;
        i = j + 1;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    auto *m = llama_model_load_from_file(argv[1], mp);
    if (m == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 1;
    }

    const auto *v = llama_model_get_vocab(m);
    int nv = llama_vocab_n_tokens(v);
    auto cp = llama_context_default_params();
    cp.n_ctx = ids.size() + 8;
    cp.n_batch = ids.size() + 8;
    auto *ctx = llama_init_from_model(m, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(m);
        return 1;
    }

    auto b = llama_batch_get_one(ids.data(), (int)ids.size());
    if (llama_decode(ctx, b) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        llama_free(ctx);
        llama_model_free(m);
        return 1;
    }

    float *lg = llama_get_logits_ith(ctx, (int)ids.size() - 1);
    FILE *f = std::fopen(argv[3], "wb");
    if (f == nullptr) {
        std::perror(argv[3]);
        llama_free(ctx);
        llama_model_free(m);
        return 1;
    }
    std::fwrite(lg, 4, nv, f);
    std::fclose(f);

    int am = 0;
    for (int k = 1; k < nv; k++) {
        if (lg[k] > lg[am]) am = k;
    }
    std::fprintf(stderr, "argmax=%d (%.5f)\n", am, lg[am]);

    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
