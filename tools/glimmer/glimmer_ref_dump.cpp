// tools/glimmer/glimmer_ref_dump.cpp - the reference side of xolotl-glimmer's parity check: llama.cpp (with Muse Glimmer
// support, #26841) reads a text file, writes its token ids (comma separated) and the logits of every `stride`-th position
// plus the last (float32 [n_written][n_vocab]) so tools/glimmer/parity.py can compare them with
// `xolotl-glimmer --dump ... --stride S`.  Write the text with its own <|begin_of_text|>: nothing is added.
//
//   g++ -O2 -std=c++17 tools/glimmer/glimmer_ref_dump.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//       -L$LLAMA/build/bin -lllama -Wl,-rpath,$LLAMA/build/bin -o build-xolotl/glimmer_ref_dump
//   glimmer_ref_dump <model.gguf> <text file> <max tokens> <ids out> <logits out> [stride=1] [n_gpu_layers=99]
//   env GLIMMER_REF_UBATCH (default 512)
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: glimmer_ref_dump <model.gguf> <text> <max tokens> <ids out> <logits out> [stride] [ngl]\n");
        return 2;
    }
    const int max_tok = std::atoi(argv[3]);
    const int stride = argc > 6 ? std::max(1, std::atoi(argv[6])) : 1;
    const int ngl = argc > 7 ? std::atoi(argv[7]) : 99;
    std::ifstream tf(argv[2]);
    std::stringstream ss;
    ss << tf.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> ids(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), ids.data(), (int) ids.size(), false, true);
    if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    ids.resize(std::min(n, max_tok));
    n = (int) ids.size();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + 16;
    const char* ub = std::getenv("GLIMMER_REF_UBATCH");
    cp.n_ubatch = ub && std::atoi(ub) > 0 ? std::atoi(ub) : 512;
    cp.n_batch = cp.n_ubatch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create context\n"); return 1; }

    FILE* fo = std::fopen(argv[5], "wb");
    const int nb = (int) cp.n_batch;
    int written = 0;
    for (int p = 0; p < n; p += nb) {
        const int T = std::min(nb, n - p);
        llama_batch b = llama_batch_init(T, 0, 1);
        for (int t = 0; t < T; ++t) {
            b.token[t] = ids[p + t]; b.pos[t] = p + t; b.n_seq_id[t] = 1; b.seq_id[t][0] = 0; b.logits[t] = 1;
        }
        b.n_tokens = T;
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed at %d\n", p); return 1; }
        for (int t = 0; t < T; ++t)
            if ((p + t) % stride == 0 || p + t + 1 == n) { std::fwrite(llama_get_logits_ith(ctx, t), sizeof(float), n_vocab, fo); ++written; }
        llama_batch_free(b);
    }
    std::fclose(fo);
    FILE* fi = std::fopen(argv[4], "w");
    for (int i = 0; i < n; ++i) std::fprintf(fi, i ? ",%d" : "%d", ids[i]);
    std::fclose(fi);
    std::fprintf(stderr, "glimmer_ref_dump: %d tokens, %d positions x %d vocab written (stride %d)\n", n, written, n_vocab, stride);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
