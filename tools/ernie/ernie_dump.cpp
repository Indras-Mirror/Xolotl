// ernie_dump - dump named ggml tensors of one llama_decode() pass to raw float32 files.
//
// Built against the llama.cpp ERNIE 5.5 PR (#31925) libllama/libggml.  CPU only.
// The graph's cb() names every interesting intermediate (l_last-N, q-N, kv-N,
// ffn_moe_topk_ernie-N, ...) and llama_context::graph_get_cb() turns those into
// ggml tensor names ("<name>-<layer>"), so the eval callback can select them.
//
// usage: ernie_dump -m model.gguf --tokens 1,2,3 --dump DIR [--names a,b,c]
//                   [--no-flash-attn] [--threads N]
//
// Output: DIR/<tensor name>.f32 (little-endian float32, ggml order: ne0 fastest)
//         DIR/manifest.json  {"tensors": {name: {"file", "ne", "ggml_type"}}}
//
// Every tensor is converted to float32.  I32 tensors (router ids) are written
// as float32 too; the manifest keeps the source type so a reader can round.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    std::vector<std::string> names;  // prefixes: exact name, or "<prefix>-<il>"
    std::vector<std::pair<std::string, std::string>> written;  // name -> file
    std::map<std::string, int> count;  // name -> how many times evaluated (views, sites)
    FILE * man = nullptr;
    int n_tensors = 0;
    size_t n_bytes = 0;
};

static bool match_name(const dump_state & st, const std::string & name) {
    if (st.names.empty()) {
        return true;
    }
    for (const auto & p : st.names) {
        if (name == p) {
            return true;
        }
        // "<prefix>-<layer>": a digit-only suffix.  View tensors inherit the
        // base name plus a parenthesised suffix ("q-0 (view)") and are skipped.
        if (name.size() > p.size() + 1 && name.compare(0, p.size(), p) == 0 && name[p.size()] == '-') {
            bool digits = true;
            for (size_t i = p.size() + 1; i < name.size(); ++i) {
                if (!isdigit((unsigned char) name[i])) {
                    digits = false;
                    break;
                }
            }
            if (digits) {
                return true;
            }
        }
    }
    return false;
}

static std::string sanitize(const std::string & s) {
    std::string r = s;
    for (auto & c : r) {
        if (!(isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return r;
}

static const char * type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_I32:  return "I32";
        case GGML_TYPE_I64:  return "I64";
        default:             return ggml_type_name(t);
    }
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;

    if (ask) {
        return true;  // we need every node's data
    }
    if (t->name[0] == 0 || !match_name(*st, t->name)) {
        return true;
    }

    const int64_t n = ggml_nelements(t);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);

    std::vector<float> f(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(f.data(), raw.data(), n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f.data(), n);
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), f.data(), n);
    } else if (t->type == GGML_TYPE_I32) {
        const int32_t * src = (const int32_t *) raw.data();
        for (int64_t i = 0; i < n; ++i) {
            f[i] = (float) src[i];
        }
    } else if (t->type == GGML_TYPE_I64) {
        const int64_t * src = (const int64_t *) raw.data();
        for (int64_t i = 0; i < n; ++i) {
            f[i] = (float) src[i];
        }
    } else {
        fprintf(stderr, "ernie_dump: %s: unsupported type %s\n", t->name, ggml_type_name(t->type));
        return true;
    }

    const int seq = ++st->count[t->name];
    const std::string file = sanitize(t->name) + (seq > 1 ? "__" + std::to_string(seq) : "") + ".f32";
    const std::string path = st->dir + "/" + file;
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "ernie_dump: cannot write %s\n", path.c_str());
        return true;
    }
    out.write((const char *) f.data(), (std::streamsize) (n * sizeof(float)));
    out.close();

    if (st->man) {
        fprintf(st->man, "%s    {\"name\": \"%s\", \"file\": \"%s\", \"seq\": %d, \"ne\": [%lld, %lld, %lld, %lld], "
                         "\"ggml_type\": \"%s\", \"n_elements\": %lld}",
                st->n_tensors ? ",\n" : "", t->name, file.c_str(), seq,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                type_name(t->type), (long long) n);
        st->written.emplace_back(t->name, file);
        ++st->n_tensors;
        st->n_bytes += n * sizeof(float);
    }
    return true;
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf --tokens 1,2,3 --dump DIR [--names a,b,c]\n"
            "          [--no-flash-attn] [--threads N] [--ctx N]\n"
            "  --names  comma separated tensor names or name prefixes (exact name, or\n"
            "           '<prefix>-<layer>'); empty = every named tensor\n",
            argv0);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tokens_arg;
    std::string names_arg;
    std::string dump_dir;
    bool flash_attn = true;
    int  n_threads = 8;
    int  n_ctx = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            model_path = next();
        } else if (a == "--tokens") {
            tokens_arg = next();
        } else if (a == "--dump") {
            dump_dir = next();
        } else if (a == "--names") {
            names_arg = next();
        } else if (a == "--threads") {
            n_threads = atoi(next());
        } else if (a == "--ctx") {
            n_ctx = atoi(next());
        } else if (a == "--no-flash-attn") {
            flash_attn = false;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "ernie_dump: unknown argument %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || tokens_arg.empty() || dump_dir.empty()) {
        usage(argv[0]);
        return 1;
    }

    std::vector<llama_token> tokens;
    {
        size_t pos = 0;
        while (pos < tokens_arg.size()) {
            size_t comma = tokens_arg.find(',', pos);
            if (comma == std::string::npos) {
                comma = tokens_arg.size();
            }
            tokens.push_back((llama_token) atoi(tokens_arg.substr(pos, comma - pos).c_str()));
            pos = comma + 1;
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "ernie_dump: no tokens\n");
        return 1;
    }

    dump_state st;
    st.dir = dump_dir;
    {
        size_t pos = 0;
        while (pos < names_arg.size()) {
            size_t comma = names_arg.find(',', pos);
            if (comma == std::string::npos) {
                comma = names_arg.size();
            }
            std::string s = names_arg.substr(pos, comma - pos);
            if (!s.empty()) {
                st.names.push_back(s);
            }
            pos = comma + 1;
        }
    }

    const std::string man_path = dump_dir + "/manifest.json";
    st.man = fopen(man_path.c_str(), "w");
    if (!st.man) {
        fprintf(stderr, "ernie_dump: cannot write %s\n", man_path.c_str());
        return 1;
    }
    fprintf(st.man, "{\"tensors\": [\n");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;   // no CPU_REPACK: it copies the MXFP4 experts into anonymous RAM (a 154 GB MXFP4 GGUF pushed a 128 GB box into swap)

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "ernie_dump: failed to load %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) (n_ctx > 0 ? n_ctx : ((int) tokens.size() + 32));
    cparams.n_batch = (uint32_t) tokens.size();
    cparams.n_ubatch = (uint32_t) tokens.size();
    cparams.n_seq_max = 1;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = &st;
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ernie_dump: failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == (int32_t) tokens.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = (int32_t) tokens.size();

    fprintf(stderr, "ernie_dump: %zu tokens, ctx %u, flash_attn %s, threads %d\n",
            tokens.size(), cparams.n_ctx, flash_attn ? "on" : "off", n_threads);

    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "ernie_dump: llama_decode failed: %d\n", rc);
        return 1;
    }

    fprintf(st.man, "\n], \"tokens\": [");
    for (size_t i = 0; i < tokens.size(); ++i) {
        fprintf(st.man, "%s%d", i ? ", " : "", tokens[i]);
    }
    fprintf(st.man, "], \"flash_attn\": %s, \"n_threads\": %d", flash_attn ? "true" : "false", n_threads);
    fprintf(st.man, "}\n");
    fclose(st.man);

    fprintf(stderr, "ernie_dump: wrote %d tensors (%.1f MiB) to %s\n",
            st.n_tensors, (double) st.n_bytes / (1024.0 * 1024.0), dump_dir.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return st.n_tensors == 0 ? 2 : 0;
}
