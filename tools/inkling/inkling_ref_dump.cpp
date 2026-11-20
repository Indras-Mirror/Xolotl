// tools/inkling/inkling_ref_dump.cpp - the reference side of xolotl-inkling's parity check (stages S2/S3).
// llama.cpp with Inkling 2 + the loop (#32217 + PR #32694) reads a text file and writes:
//   <out>/ids.txt               token ids, comma separated (the text is tokenized with specials parsed, nothing added)
//   <out>/logits.f32            float32 [n_tok][n_vocab], every position
//   <out>/<name>-<il>.f32       float32 [n_tok][ne0] for each captured graph tensor (default: l_out of every cache slot;
//                               slot order = layers 0-5, core pass 1 = slots 6-17, 18-23 = tail, pass 2 = 24-35, pass 3 = 36-47)
// One ubatch holds the whole prompt, so every captured tensor covers every token.
//
//   g++ -O2 -std=c++17 tools/inkling/inkling_ref_dump.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//       -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o build-xolotl/inkling_ref_dump
//   inkling_ref_dump <model.gguf> <text file> <out dir> [max tokens=512] [ngl=99]
//   env INKLING_DUMP_NAMES="l_out,ffn_out,attn_out"  (prefixes captured; default "l_out,result_norm")
//   env INKLING_CPU_EXPS="blk\\.(2[0-3])\\.ffn_.*_exps"  (regex of tensors kept in host RAM: the 22.9 GB parent GGUF does
//       not fit a 24 GB card whole; default = the four BF16 (ex-MXFP8) expert layers 20-23, "" = everything on the GPU)
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct Capture {
    std::vector<std::string> prefixes;
    std::map<std::string, std::vector<float>> data;  // name -> [n_tok][ne0]
    std::map<std::string, int64_t> ne0;

    bool wanted(const char* name) const {
        for (const auto& p : prefixes) {
            const size_t n = p.size();
            if (std::strncmp(name, p.c_str(), n) == 0 && (name[n] == '-' || name[n] == '\0')) return true;
        }
        return false;
    }
};

bool on_tensor(ggml_tensor* t, bool ask, void* ud) {
    auto* c = static_cast<Capture*>(ud);
    if (!c->wanted(t->name)) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "inkling_ref_dump: %s is %s, skipped\n", t->name, ggml_type_name(t->type));
        return true;
    }
    // copy the whole tensor (made contiguous by reading row by row through its strides)
    std::vector<char> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    const int64_t ne0 = t->ne[0], rows = ggml_nrows(t);
    std::vector<float> out((size_t) (ne0 * rows));
    for (int64_t i3 = 0, r = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1, ++r) {
                const char* src = raw.data() + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                for (int64_t i0 = 0; i0 < ne0; ++i0)
                    std::memcpy(&out[(size_t) (r * ne0 + i0)], src + i0 * t->nb[0], sizeof(float));
            }
    c->data[t->name] = std::move(out);
    c->ne0[t->name] = ne0;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: inkling_ref_dump <model.gguf> <text> <out dir> [max tokens] [ngl]\n");
        return 2;
    }
    const std::string out = argv[3];
    const int max_tok = argc > 4 ? std::atoi(argv[4]) : 512;
    const int ngl = argc > 5 ? std::atoi(argv[5]) : 99;
    mkdir(out.c_str(), 0755);

    Capture cap;
    const char* names = std::getenv("INKLING_DUMP_NAMES");
    std::stringstream ns(names && *names ? names : "l_out,result_norm");
    for (std::string s; std::getline(ns, s, ',');)
        if (!s.empty()) cap.prefixes.push_back(s);

    std::ifstream tf(argv[2]);
    std::stringstream ss;
    ss << tf.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    const char* cpu_exps = std::getenv("INKLING_CPU_EXPS");
    const std::string cpu_pat = cpu_exps ? cpu_exps : "blk\\.(2[0-3])\\.ffn_.*_exps";
    std::vector<llama_model_tensor_buft_override> ov;
    if (!cpu_pat.empty()) {
        ov.push_back({cpu_pat.c_str(), ggml_backend_cpu_buffer_type()});
        ov.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = ov.data();
    }
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
    cp.n_ubatch = cp.n_batch = (uint32_t) n;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &cap;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create context\n"); return 1; }

    llama_batch b = llama_batch_init(n, 0, 1);
    for (int t = 0; t < n; ++t) {
        b.token[t] = ids[t]; b.pos[t] = t; b.n_seq_id[t] = 1; b.seq_id[t][0] = 0; b.logits[t] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }

    FILE* fo = std::fopen((out + "/logits.f32").c_str(), "wb");
    for (int t = 0; t < n; ++t) std::fwrite(llama_get_logits_ith(ctx, t), sizeof(float), n_vocab, fo);
    std::fclose(fo);
    FILE* fi = std::fopen((out + "/ids.txt").c_str(), "w");
    for (int i = 0; i < n; ++i) std::fprintf(fi, i ? ",%d" : "%d", ids[i]);
    std::fclose(fi);
    for (const auto& [name, v] : cap.data) {
        FILE* f = std::fopen((out + "/" + name + ".f32").c_str(), "wb");
        std::fwrite(v.data(), sizeof(float), v.size(), f);
        std::fclose(f);
    }
    // greedy continuation for a readable sanity line (not part of the parity data)
    std::string cont;
    llama_token tok = 0;
    {
        const float* lg = llama_get_logits_ith(ctx, n - 1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        tok = best;
    }
    for (int s = 0; s < 48 && !llama_vocab_is_eog(vocab, tok); ++s) {
        char buf[256];
        const int k = llama_token_to_piece(vocab, tok, buf, sizeof buf, 0, true);
        if (k > 0) cont.append(buf, (size_t) k);
        llama_batch one = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, one) != 0) break;
        const float* lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        tok = best;
    }
    std::fprintf(stderr, "inkling_ref_dump: %d tokens, %zu tensors captured, logits [%d x %d]\n", n, cap.data.size(), n, n_vocab);
    std::printf("greedy: %s\n", cont.c_str());
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
