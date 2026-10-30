// tools/qwen35/q36_mtmd_dump.cpp - llama.cpp's side of the Qwen3.6 vision parity check: the same GGUF and mmproj, the
// prompt with its pictures through llama.cpp's mtmd, then a teacher-forced (or greedy) continuation.  Writes:
//   --out-ids F    "prompt: <ids>" with every picture as its run of <|image_pad|> ids (the cells strata-q36 reads),
//                  "gen: <ids>" the continuation;
//   --out-sve F    the pictures' embeddings as strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then f32 rows):
//                  strata-q36 reads exactly these rows (GENI), so the check is the language model's alone;
//   --dump-full-logits F   every step's logits (f32 [step][vocab]): the prompt's last position, then each forced token.
// Built against a llama.cpp build tree (static libs; not part of Strata's build), e.g.:
//   g++ -O2 -std=c++17 q36_mtmd_dump.cpp -I<llama>/include -I<llama>/ggml/include -I<llama>/tools/mtmd \
//       <llama>/build/tools/mtmd/libmtmd.a <llama>/build/src/libllama.a <llama>/build/ggml/src/ggml-cuda/libggml-cuda.a \
//       <llama>/build/ggml/src/libggml.a <llama>/build/ggml/src/libggml-cpu.a <llama>/build/ggml/src/libggml-base.a \
//       <llama>/build/vendor/hash/libvendor-hash.a (inside -Wl,--start-group / --end-group) -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcublasLt -lcuda -lgomp -lpthread -ldl -o q36_mtmd_dump
// Usage: q36_mtmd_dump --model GGUF --mmproj GGUF --prompt-file F [--image P ...] [--force IDS | --n N] [--ngl 99]
//        [--ctx N] [--max-tokens N] ... ; the prompt text has one <__media__> per picture (mtmd adds the
//        <|vision_start|> / <|vision_end|> markers around it) and its special tokens are parsed.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model, mmproj, prompt_file, force, out_ids, out_sve, dump;
    std::vector<std::string> images;
    int ngl = 99, n_gen = 16, n_ctx = 32768, max_tokens = 0, min_tokens = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--image") images.push_back(next());
        else if (a == "--force") force = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--ngl") ngl = std::atoi(next().c_str());
        else if (a == "--ctx") n_ctx = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") min_tokens = std::atoi(next().c_str());
        else if (a == "--out-ids") out_ids = next();
        else if (a == "--out-sve") out_sve = next();
        else if (a == "--dump-full-logits") dump = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::ifstream pf(prompt_file);
    std::stringstream ss;
    ss << pf.rdbuf();
    const std::string prompt = ss.str();
    std::vector<int> forced;
    for (std::stringstream fs(force); fs.good();) {
        std::string t;
        std::getline(fs, t, ',');
        if (!t.empty()) forced.push_back(std::atoi(t.c_str()));
    }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { std::fprintf(stderr, "cannot load %s\n", model.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    llama_context* lctx = llama_init_from_model(m, cp);
    const llama_vocab* vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd = llama_model_n_embd_inp(m);
    int pad = -1;
    for (int t = 0; t < n_vocab; ++t) {
        char buf[64];
        const int k = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
        if (k == 13 && std::memcmp(buf, "<|image_pad|>", 13) == 0) { pad = t; break; }
    }
    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = ngl > 0;
    mcp.print_timings = false;
    mcp.media_marker = "<__media__>";
    if (max_tokens > 0) mcp.image_max_tokens = max_tokens;
    if (min_tokens > 0) mcp.image_min_tokens = min_tokens;
    mtmd_context* mc = mtmd_init_from_file(mmproj.c_str(), m, mcp);
    if (!mc) { std::fprintf(stderr, "cannot load %s\n", mmproj.c_str()); return 1; }
    std::vector<mtmd::bitmap> bms;
    std::vector<const mtmd_bitmap*> bmp;
    for (const std::string& p : images) {
        mtmd_bitmap* b = mtmd_helper_bitmap_init_from_file(mc, p.c_str(), false, mtmd_helper_init_opt_default()).bitmap;
        if (!b) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); return 1; }
        bms.emplace_back(b);
        bmp.push_back(b);
    }
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (mtmd_tokenize(mc, chunks.ptr.get(), &txt, bmp.data(), bmp.size()) != 0) {
        std::fprintf(stderr, "mtmd_tokenize failed\n");
        return 1;
    }
    std::vector<int> ids;
    std::FILE* sve = out_sve.empty() ? nullptr : std::fopen(out_sve.c_str(), "wb");
    llama_pos n_past = 0;
    const size_t nch = mtmd_input_chunks_size(chunks.ptr.get());
    for (size_t c = 0; c < nch; ++c) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks.ptr.get(), c);
        const bool last = c + 1 == nch;
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t nt = 0;
            const llama_token* t = mtmd_input_chunk_get_tokens_text(ch, &nt);
            ids.insert(ids.end(), t, t + nt);
            if (mtmd_helper_eval_chunk_single(mc, lctx, ch, n_past, 0, 2048, last, &n_past) != 0) {
                std::fprintf(stderr, "text chunk eval failed\n");
                return 1;
            }
            continue;
        }
        if (mtmd_encode_chunk(mc, ch) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
        float* embd = mtmd_get_output_embd(mc);
        const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
        const int n = (int) mtmd_image_tokens_get_n_tokens(it);
        const mtmd_decoder_pos lp = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
        const int nx = (int) lp.x + 1, ny = (int) lp.y + 1;
        std::fprintf(stderr, "image chunk: %d tokens, grid %d x %d, n_pos %d, at position %d\n", n, nx, ny,
                     (int) mtmd_image_tokens_get_n_pos(it), (int) n_past);
        if (sve) {
            const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
            std::fwrite(hdr, 4, 5, sve);
            std::fwrite(embd, 4, (size_t) n * n_embd, sve);
        }
        for (int j = 0; j < n; ++j) ids.push_back(pad);
        if (mtmd_helper_decode_image_chunk(mc, lctx, ch, embd, n_past, 0, 2048, &n_past, nullptr, nullptr) != 0) {
            std::fprintf(stderr, "image decode failed\n");
            return 1;
        }
    }
    if (sve) std::fclose(sve);
    std::FILE* ff = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    std::vector<int> gen;
    llama_batch b = llama_batch_init(1, 0, 1);
    for (int s = 0; s < n_gen; ++s) {
        const float* lg = llama_get_logits_ith(lctx, -1);
        if (ff) std::fwrite(lg, 4, (size_t) n_vocab, ff);
        int best = 0;
        for (int v = 1; v < n_vocab; ++v)
            if (lg[v] > lg[best]) best = v;
        gen.push_back(best);
        if (s + 1 == n_gen) break;
        const int nx = s < (int) forced.size() ? forced[(size_t) s] : best;
        b.n_tokens = 1;
        b.token[0] = nx;
        b.pos[0] = n_past;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        if (llama_decode(lctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
        ++n_past;
    }
    if (ff) std::fclose(ff);
    if (!out_ids.empty()) {
        std::FILE* f = std::fopen(out_ids.c_str(), "w");
        std::fprintf(f, "prompt:");
        for (int t : ids) std::fprintf(f, " %d", t);
        std::fprintf(f, "\ngen:");
        for (int t : gen) std::fprintf(f, " %d", t);
        std::fprintf(f, "\n");
        std::fclose(f);
    }
    std::printf("prompt %zu cells, final position %d, gen:", ids.size(), (int) n_past);
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    llama_batch_free(b);
    mtmd_free(mc);
    llama_free(lctx);
    llama_model_free(m);
    return 0;
}
