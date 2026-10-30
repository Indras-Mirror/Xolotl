#include "../../m31_run.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-cpu.h"

#include <climits>
#include <iterator>
#include <memory>
#include <set>

namespace {
using namespace strata::m31;

struct Capture {
    run::Dump * dump = nullptr;
    int64_t pos0 = 0;
    std::string error;
};
bool capture_name(const std::string & name) {
    const auto dash = name.rfind('-');
    if (dash != std::string::npos && (dash + 1 == name.size() ||
        name.find_first_not_of("0123456789", dash + 1) != std::string::npos)) return false;
    const std::string base = dash == std::string::npos ? name : name.substr(0, dash);
    static const std::set<std::string> names = {
        "attn_norm", "Qcur_normed", "Kcur_normed", "Qcur", "Kcur", "Vcur",
        "msa_bs", "msa_topk", "msa_block_mask", "kqv_out", "ffn_inp", "ffn_norm",
        "ffn_shexp", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_down_scaled",
        "ffn_moe_topk", "ffn_moe_out", "ffn_out", "l_out", "result_norm", "result_output"
    };
    return names.count(base) != 0;
}
bool callback(ggml_tensor * t, bool ask, void * data) {
    auto & c = *static_cast<Capture *>(data);
    const std::string name = t->name;
    // build_qkv names the projection and reshape Qcur/Kcur before the model
    // reuses that name for post-norm RoPE. Capture the final semantic tensor.
    const bool final_qk = !(name.starts_with("Qcur-") || name.starts_with("Kcur-")) || t->op == GGML_OP_ROPE;
    const bool final_v = !name.starts_with("Vcur-") || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW;
    const bool wanted = c.error.empty() && c.dump && final_qk && final_v &&
                        capture_name(name) && c.dump->wants_tap(name);
    if (ask) return wanted;
    if (!wanted) return true;
    try { c.dump->tensor(t->name, c.pos0, t); return true; }
    // Returning false aborts only a scheduler subgraph; later CPU splits then
    // consume unfinished inputs. Finish the decode safely, report error afterward.
    catch (const std::exception & e) { c.error = e.what(); return true; }
}

struct ModelDeleter { void operator()(llama_model * m) const { llama_model_free(m); } };
struct ContextDeleter { void operator()(llama_context * c) const { llama_free(c); } };
using Model = std::unique_ptr<llama_model, ModelDeleter>;
using Context = std::unique_ptr<llama_context, ContextDeleter>;
struct Batch {
    llama_batch b;
    explicit Batch(int n) : b(llama_batch_init(n, 0, 1)) {}
    ~Batch() { llama_batch_free(b); }
};
struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

struct ChunkReplay {
    std::vector<std::pair<int, int>> chunks;
    std::string source;
    int maximum = 0;
    explicit operator bool() const { return !source.empty(); }
};

ChunkReplay read_chunk_replay(const run::Args & args, const std::vector<run::fs::path> & cases) {
    ChunkReplay replay;
    const char * path = std::getenv("M31_ORACLE_CHUNK_PLAN");
    if (!path) return replay;
    if (!*path || cases.size() != 1)
        throw std::runtime_error("chunk replay requires one case and a nonempty plan path");
    replay.source = path;
    if (run::fs::file_size(path) > 64 * 1024 * 1024)
        throw std::runtime_error("oracle chunk plan exceeds 64 MiB");
    const auto ids = run::read_ids(cases.front());
    if (ids.size() > size_t(args.context)) throw std::runtime_error("replay case exceeds context");
    const int64_t prefix = std::max<int64_t>(0, int64_t(ids.size()) - args.decode_tail);
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open oracle chunk plan");
    int64_t expected = 0, position = 0, count = 0;
    while (stream >> position) {
        if (!(stream >> count) || position != expected || count <= 0 || count > args.chunk ||
            position < 0 || position > int64_t(ids.size()) || count > int64_t(ids.size()) - position)
            throw std::runtime_error("invalid/noncontiguous oracle chunk plan");
        if (args.decode_tail && ((position < prefix && position + count > prefix) ||
                                  (position >= prefix && count != 1)))
            throw std::runtime_error("oracle chunk plan contradicts decode-tail policy");
        replay.chunks.emplace_back(int(position), int(count));
        replay.maximum = std::max(replay.maximum, int(count));
        expected += count;
    }
    if (!stream.eof() || replay.chunks.empty() || expected != int64_t(ids.size()))
        throw std::runtime_error("oracle chunk plan is malformed or does not cover all input tokens");
    return replay;
}

int tokenize(int argc, char ** argv) {
    if (argc != 5) throw std::runtime_error("usage: m31_oracle tokenize MODEL.gguf TEXT-DIR NEW-ID-DIR");
    // Vocab-only operation must not probe a visible GPU.
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    ggml_backend_load_all();
    Backend backend;
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    Model model(llama_model_load_from_file(argv[2], mp));
    if (!model) throw std::runtime_error("cannot load vocabulary");
    const auto * vocab = llama_model_get_vocab(model.get());
    if (!run::fs::create_directory(argv[4])) throw std::runtime_error("ID output directory already exists");
    std::vector<run::fs::path> texts;
    for (const auto & e : run::fs::directory_iterator(argv[3]))
        if (e.is_regular_file() && e.path().extension() == ".txt") texts.push_back(e.path());
    std::sort(texts.begin(), texts.end());
    if (texts.empty()) throw std::runtime_error("no .txt prompts");
    for (const auto & path : texts) {
        std::ifstream f(path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (text.empty() || text.size() > INT_MAX) throw std::runtime_error("invalid prompt size");
        int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, true, true);
        if (n >= 0) throw std::runtime_error("expected a negative required token count");
        std::vector<int32_t> ids(size_t(-n), 0);
        n = llama_tokenize(vocab, text.data(), int(text.size()), ids.data(), int(ids.size()), true, true);
        if (n <= 0) throw std::runtime_error("tokenization failed");
        ids.resize(size_t(n));
        std::ifstream length(path.parent_path() / (path.stem().string() + ".length"));
        int wanted = 0;
        int offset = 0;
        std::ifstream offset_file(path.parent_path() / (path.stem().string() + ".offset"));
        if (offset_file >> offset) {
            if (offset < 0 || offset >= n) throw std::runtime_error("invalid token offset");
            ids.erase(ids.begin(), ids.begin() + offset);
            n -= offset;
        }
        if (length >> wanted) {
            if (wanted <= 0 || wanted > n) throw std::runtime_error("prompt too short for requested .length");
            ids.resize(size_t(wanted));
        }
        std::ofstream out(run::fs::path(argv[4]) / (path.stem().string() + ".i32"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(ids.data()), std::streamsize(ids.size() * sizeof(int32_t)));
        if (!out) throw std::runtime_error("cannot write token IDs");
        std::fprintf(stderr, "tokenized %s: %zu IDs\n", path.stem().string().c_str(), ids.size());
    }
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "tokenize") return tokenize(argc, argv);
        auto args = run::parse(argc, argv);
        if (args.arena_gib) throw std::runtime_error("--arena-gib is native-only; oracle retains original GGUF layout");
        // Shared native defaults must not select or reject a reference path.
        // Explicit native-only resource/optimization flags still fail loudly.
        static const std::set<std::string> native_only = {
            "--gpu-experts-gib", "--gpu-experts-min-tokens", "--reuse-allocations",
            "--persistent-cpu-pool", "--cpu-residual", "--cpu-decode-cache",
            "--dense-decode-cache", "--profile"
        };
        for (int i = 1; i < argc; i += 2)
            if (native_only.count(argv[i])) throw std::runtime_error("native-only option: " + std::string(argv[i]));
        args.reuse_allocations = false;
        args.persistent_cpu_pool = false;
        args.cpu_residual = false;
        args.cpu_decode_cache = false;
        args.dense_decode_cache = false;
        const auto cases = run::cases(args);
        // The native adaptive planner can use a different size at every step.
        // Replay its actual TSV (position, count), rather than quietly comparing
        // a nominal fixed-chunk oracle with different GEMM/attention geometry.
        const auto replay = read_chunk_replay(args, cases);
        if (args.backend == "cpu") setenv("CUDA_VISIBLE_DEVICES", "", 1);
        if (!run::fs::create_directory(args.out)) throw std::runtime_error("output directory exists; use a new path");
        if (replay) {
            std::ofstream plan(run::fs::path(args.out) / "chunk-replay.tsv");
            for (const auto & item : replay.chunks) plan << item.first << '\t' << item.second << '\n';
            if (!plan) throw std::runtime_error("cannot save oracle replay plan");
            std::fprintf(stderr, "M31 oracle replay: entries=%zu max_chunk=%d nominal_chunk=%d source=%s\n",
                         replay.chunks.size(), replay.maximum, args.chunk, replay.source.c_str());
        }
        ggml_backend_load_all();
        Backend backend;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = args.backend == "cuda" ? 999 : 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model_tensor_buft_override overrides[] = {
            {".*ffn_.*_exps\\..*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
        };
        if (args.backend == "cuda") mp.tensor_buft_overrides = overrides;
        Model model(llama_model_load_from_file(args.model.c_str(), mp));
        if (!model) throw std::runtime_error("cannot load PR model");
        Capture capture;
        auto cp = llama_context_default_params();
        cp.n_ctx = args.context;
        cp.n_batch = replay ? replay.maximum : args.chunk;
        cp.n_ubatch = replay ? replay.maximum : args.chunk;
        // The bounded head requests at most this many logits per ubatch.
        // Otherwise context initialization reserves vocab*n_batch outputs before
        // seeing the batch's logits mask (8 GiB at10240 tokens for this model).
        // Layer59 still executes for every token via the layer-input tap below.
        if (args.head_tail) cp.n_outputs_max = uint32_t(args.head_tail);
        cp.n_seq_max = 1;
        cp.n_threads = args.threads;
        cp.n_threads_batch = args.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        const bool capacity_equivalence = std::getenv("M31_ORACLE_CAPACITY_EQUIVALENCE") &&
            std::string(std::getenv("M31_ORACLE_CAPACITY_EQUIVALENCE")) == "1";
        if (capacity_equivalence) {
            // Explicitly retain this model's trained linear RoPE parameters
            // while changing only physical KV allocation for reserve sizing.
            cp.yarn_ext_factor = 0;
            cp.yarn_orig_ctx = uint32_t(llama_model_n_ctx_train(model.get()));
            cp.rope_freq_scale = llama_model_rope_freq_scale_train(model.get());
        }
        cp.offload_kqv = args.backend == "cuda";
        // M1's native routed path is CPU ggml, including activation quantization.
        // Keep that backend identical in the oracle; otherwise CPU-placed expert
        // tensors may still be copied to CUDA for mul_mat_id by op_offload.
        // M2 benchmarks use the reference server's normal op_offload=true.
        cp.op_offload = args.oracle_op_offload;
        cp.kv_unified = false;
        cp.no_perf = true;
        if (args.taps != "none") { cp.cb_eval = callback; cp.cb_eval_user_data = &capture; }
        Context ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) throw std::runtime_error("cannot initialize PR context");
        if (args.head_tail) {
            // Selecting logits normally crops before the final FFN. Preserve
            // native's full-token last-layer routing/histogram and crop only
            // before the bounded head, using the PR's supported layer-tap API.
            llama_set_embeddings_layer_inp(ctx.get(), uint32_t(llama_model_n_layer(model.get())), true);
        }
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        Batch batch(args.chunk);
        for (const auto & path : cases) {
            const auto ids = run::read_ids(path);
            if (ids.size() > size_t(args.context)) throw std::runtime_error("case exceeds context");
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            run::Dump dump(run::fs::path(args.out) / path.stem(), args, ids, vocab);
            std::ofstream capacity_trace;
            if (capacity_equivalence) {
                capacity_trace.open(run::fs::path(args.out) / path.stem() / "capacity-equivalence.jsonl");
                if (!capacity_trace) throw std::runtime_error("cannot create capacity equivalence trace");
            }
            capture.dump = &dump;
            size_t replay_index = 0;
            for (int pos = 0; pos < int(ids.size()); ) {
                const int n = replay ? replay.chunks.at(replay_index++).second :
                                       run::chunk_size(args, int(ids.size()), pos);
                dump.record_chunk(pos, n);
                batch.b.n_tokens = n;
                for (int r = 0; r < n; ++r) {
                    batch.b.token[r] = ids[size_t(pos + r)];
                    batch.b.pos[r] = pos + r;
                    batch.b.n_seq_id[r] = 1;
                    batch.b.seq_id[r][0] = 0;
                    // Default all outputs preserve the existing M1 contract.
                    // The explicit bounded mode matches native's contiguous head
                    // width without allocating vocab*8192 output/scratch.
                    batch.b.logits[r] = !args.head_tail || r >= n - std::min(n, args.head_tail);
                }
                capture.pos0 = pos;
                if (llama_decode(ctx.get(), batch.b) != 0) throw std::runtime_error("PR llama_decode failed");
                if (!capture.error.empty()) throw std::runtime_error(capture.error);
                if (capacity_equivalence) {
                    const auto memory = llama_get_memory(ctx.get());
                    const auto first = llama_memory_seq_pos_min(memory, 0);
                    const auto last = llama_memory_seq_pos_max(memory, 0);
                    if (first != 0 || last != pos + n - 1)
                        throw std::runtime_error("capacity-equivalence requires contiguous KV without wrap/shift");
                    // PR llama_kv_cache::get_n_kv uses PAD(used_max_p1,256),
                    // clamped by physical capacity. The verified contiguous
                    // KV range makes the padding below identical to native.
                    const int keys = ((pos + n + 255) / 256) * 256;
                    capacity_trace << "{\"position\":" << pos << ",\"tokens\":" << n
                        << ",\"seq_min\":" << first << ",\"seq_max\":" << last
                        << ",\"active_keys_padded_from_verified_positions\":" << keys
                        << ",\"physical_context_requested\":" << args.context
                        << ",\"physical_context_actual\":" << llama_n_ctx(ctx.get())
                        << ",\"rope_ext_factor\":0,\"rope_orig_context\":" << cp.yarn_orig_ctx
                        << ",\"rope_frequency_scale\":" << cp.rope_freq_scale << "}\n";
                    if (!capacity_trace) throw std::runtime_error("cannot write capacity equivalence trace");
                }
                for (int r = 0; r < n; ++r) if (dump.wants(pos + r)) {
                    const float * logits = llama_get_logits_ith(ctx.get(), r);
                    if (!logits) throw std::runtime_error("missing oracle logits");
                    dump.row(pos + r, logits);
                }
                pos += n;
            }
            dump.finish("llama.cpp-PR31192", int(ids.size()));
            capture.dump = nullptr;
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "m31_oracle: %s\n", e.what());
        return 1;
    }
}
