// Test-only cache-history oracle, linked exclusively to the audited llama.cpp PR.
#include "../../m31_run.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-cpu.h"
#include "json.hpp"
#include <climits>
#include <memory>
#include <iostream>

namespace {
using namespace strata::m31;
using json = nlohmann::ordered_json;
struct Args {
    std::string model, ids, replacement, out;
    int reuse = -1, context = 4096, chunk = 32768, threads = 32, device = 0, decode = 0;
};
int number(const std::string & text, int low, int high, const std::string & option) {
    size_t end = 0;
    const auto n = std::stoll(text, &end);
    if (end != text.size() || n < low || n > high) throw std::runtime_error("invalid " + option);
    return int(n);
}
Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--model" || key == "-m") a.model = value;
        else if (key == "--ids-file") a.ids = value;
        else if (key == "--replacement-ids") a.replacement = value;
        else if (key == "--out") a.out = value;
        else if (key == "--reuse") a.reuse = number(value, 1, 1048576, key);
        else if (key == "--ctx") a.context = number(value, 1, 1048576, key);
        else if (key == "--chunk") a.chunk = number(value, 1, 32768, key);
        else if (key == "--threads") a.threads = number(value, 1, 4096, key);
        else if (key == "--device") a.device = number(value, 0, 1023, key);
        else if (key == "--decode-before-rewind") a.decode = number(value, 0, 1048576, key);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (a.model.empty() || a.ids.empty() || a.out.empty() || a.reuse < 1)
        throw std::runtime_error("required: --model MODEL --ids-file OLD.i32 --reuse R --out NEW-DIR");
    return a;
}
void ensure(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
void fusion_guard() {
    const auto * value = std::getenv("GGML_CUDA_DISABLE_FUSION");
    if (value && std::string(value) != "1") throw std::runtime_error("GPU oracle requires GGML_CUDA_DISABLE_FUSION=1");
#ifdef _WIN32
    ensure(_putenv_s("GGML_CUDA_DISABLE_FUSION", "1") == 0, "cannot set fusion guard");
#else
    ensure(setenv("GGML_CUDA_DISABLE_FUSION", "1", 1) == 0, "cannot set fusion guard");
#endif
}
void write_binary(const run::fs::path & path, const void * data, size_t bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(static_cast<const char *>(data), std::streamsize(bytes));
    ensure(bool(output), "cannot write " + path.string());
}
struct Backend {
    Backend() { ggml_backend_load_all(); llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};
struct ModelDelete { void operator()(llama_model * p) const { llama_model_free(p); } };
struct ContextDelete { void operator()(llama_context * p) const { llama_free(p); } };
struct Batch {
    llama_batch value;
    explicit Batch(int size) : value(llama_batch_init(size, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
};
json environment() {
    json value = json::object();
    for (const char * key : {"CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_MMQ_PREC",
                            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}) {
        const auto * text = std::getenv(key);
        value[key] = text ? json(text) : json(nullptr);
    }
    return value;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::puts("m31_cache_oracle --model MODEL --ids-file OLD.i32 --reuse R --out NEW-DIR\n"
                  "  [--replacement-ids NEW.i32 --ctx N --chunk N --threads N --device N --decode-before-rewind N]\n"
                  "GPU-offload/F16 oracle: whole old prompt, optional raw greedy decode, seq_rm(R,-1), new suffix.");
        return 0;
    }
    try {
        const auto a = parse(argc, argv);
        const auto old_ids = run::read_ids(a.ids);
        const auto new_ids = a.replacement.empty() ? old_ids : run::read_ids(a.replacement);
        ensure(a.reuse <= int64_t(old_ids.size()) && a.reuse < int64_t(new_ids.size()), "reuse must retain old rows and leave a new suffix");
        ensure(int64_t(old_ids.size()) + a.decode <= a.context && int64_t(new_ids.size()) <= a.context, "history exceeds configured context");
        ensure(std::equal(old_ids.begin(), old_ids.begin() + a.reuse, new_ids.begin()), "replacement changes tokens before reused prefix");
        const int actual_batch = std::max(std::min(a.chunk, int(old_ids.size())),
                                          std::min(a.chunk, int(new_ids.size()) - a.reuse));
        ensure(actual_batch > 0, "no actual batches");
        ensure(run::fs::create_directory(a.out), "output directory exists; use a new path");
        const run::fs::path out = a.out;
        write_binary(out / "old_tokens.i32", old_ids.data(), old_ids.size() * 4);
        write_binary(out / "new_tokens.i32", new_ids.data(), new_ids.size() * 4);
        json manifest{{"format", "m31-cache-oracle-v1"}, {"reference", "llama.cpp PR #31192"},
                      {"reference_sha", M31_CACHE_ORACLE_REFERENCE_SHA}, {"complete", false},
                      {"model", run::fs::canonical(a.model).string()}, {"model_first_shard_bytes", run::fs::file_size(a.model)},
                      {"old_prompt", old_ids.size()}, {"new_prompt", new_ids.size()}, {"reuse", a.reuse},
                      {"nominal_chunk", a.chunk}, {"actual_maximum_batch", actual_batch},
                      {"decode_before_rewind", a.decode}, {"threads", a.threads}, {"device", a.device},
                      {"kv", "f16"}, {"op_offload", true}, {"expert_weights", "CPU original GGUF"},
                      {"dense_weights", "CUDA"}, {"full_last_layer_before_crop", true}, {"n_outputs_max", 1},
                      {"head_file", "logits.f32"}, {"phases", json::array()}};
        auto save_manifest = [&] {
            std::ofstream file(out / "manifest.json"); file << manifest.dump(2) << '\n';
            ensure(bool(file), "cannot write manifest");
        };
        fusion_guard();
        manifest["environment"] = environment();
        save_manifest();
        Backend backend;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999; mp.main_gpu = a.device; mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model_tensor_buft_override overrides[] = {{".*ffn_.*_exps\\..*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        mp.tensor_buft_overrides = overrides;
        std::unique_ptr<llama_model, ModelDelete> model(llama_model_load_from_file(a.model.c_str(), mp));
        ensure(bool(model), "cannot load reference model");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        auto cp = llama_context_default_params();
        cp.n_ctx = a.context; cp.n_batch = actual_batch; cp.n_ubatch = actual_batch;
        cp.n_outputs_max = 1; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = a.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = true; cp.op_offload = true; cp.kv_unified = false; cp.no_perf = true;
        cp.yarn_ext_factor = 0;
        cp.yarn_orig_ctx = uint32_t(llama_model_n_ctx_train(model.get()));
        cp.rope_freq_scale = llama_model_rope_freq_scale_train(model.get());
        std::unique_ptr<llama_context, ContextDelete> ctx(llama_init_from_model(model.get(), cp));
        ensure(bool(ctx), "cannot initialize reference context");
        llama_set_embeddings_layer_inp(ctx.get(), uint32_t(llama_model_n_layer(model.get())), true);
        manifest["vocab"] = vocab;
        manifest["physical_context_requested"] = a.context;
        manifest["physical_context_actual"] = llama_n_ctx(ctx.get());
        manifest["rope"] = {{"ext_factor", 0}, {"orig_context", cp.yarn_orig_ctx}, {"frequency_scale", cp.rope_freq_scale}};
        manifest["system_info"] = llama_print_system_info();
        const auto memory = llama_get_memory(ctx.get());
        llama_memory_clear(memory, true);
        Batch batch(actual_batch);
        std::vector<float> head;
        std::ofstream trace(out / "phases.jsonl");
        ensure(bool(trace), "cannot create phase trace");
        auto phase_record = [&](json record) {
            record["seq_min"] = llama_memory_seq_pos_min(memory, 0);
            record["seq_max"] = llama_memory_seq_pos_max(memory, 0);
            manifest["phases"].push_back(record); trace << record.dump() << '\n'; trace.flush(); save_manifest();
        };
        auto decode = [&](const int32_t * tokens, int position, int count, const std::string & phase) {
            ensure(count > 0 && count <= actual_batch, "invalid actual batch");
            batch.value.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                ensure(tokens[i] >= 0 && tokens[i] < vocab, "token outside vocabulary");
                batch.value.token[i] = tokens[i]; batch.value.pos[i] = position + i;
                batch.value.n_seq_id[i] = 1; batch.value.seq_id[i][0] = 0;
                batch.value.logits[i] = i == count - 1;
            }
            ensure(llama_decode(ctx.get(), batch.value) == 0, "reference decode failed: " + phase);
            ensure(llama_memory_seq_pos_min(memory, 0) == 0 && llama_memory_seq_pos_max(memory, 0) == position + count - 1,
                   "noncontiguous or shifted reference KV: " + phase);
            const auto * data = llama_get_logits_ith(ctx.get(), count - 1);
            ensure(data != nullptr, "reference did not return selected head");
            head.assign(data, data + vocab);
            ensure(std::all_of(head.begin(), head.end(), [](float x) { return std::isfinite(x); }), "nonfinite reference head");
            phase_record({{"phase", phase}, {"position", position}, {"tokens", count},
                          {"active_keys_padded_from_verified_positions", ((position + count + 255) / 256) * 256},
                          {"top1", int(std::max_element(head.begin(), head.end()) - head.begin())}});
        };
        auto feed = [&](const std::vector<int32_t> & ids, int begin, const std::string & phase) {
            for (int position = begin; position < int(ids.size());) {
                const int count = std::min(a.chunk, int(ids.size()) - position);
                decode(ids.data() + position, position, count, phase); position += count;
            }
        };
        feed(old_ids, 0, "old_prompt");
        write_binary(out / "whole.f32", head.data(), head.size() * 4);
        std::vector<int32_t> generated;
        for (int i = 0; i < a.decode; ++i) {
            const int32_t token = int32_t(std::max_element(head.begin(), head.end()) - head.begin());
            generated.push_back(token);
            decode(&token, int(old_ids.size()) + i, 1, "decode_before_rewind");
        }
        manifest["raw_argmax_decoded_ids"] = generated;
        if (!generated.empty()) write_binary(out / "after_decode.f32", head.data(), head.size() * 4);
        ensure(llama_memory_seq_rm(memory, 0, a.reuse, -1), "reference rejected partial KV removal");
        ensure(llama_memory_seq_pos_min(memory, 0) == 0 && llama_memory_seq_pos_max(memory, 0) == a.reuse - 1,
               "reference KV range differs after removal");
        phase_record({{"phase", "remove_suffix"}, {"from", a.reuse}, {"to", -1}, {"retained", a.reuse}});
        feed(new_ids, a.reuse, "new_suffix");
        write_binary(out / "logits.f32", head.data(), head.size() * 4);
        manifest["complete"] = true;
        save_manifest();
        std::cout << "cache history oracle: old=" << old_ids.size() << " reuse=" << a.reuse << " new=" << new_ids.size()
                  << " decode_before_rewind=" << a.decode << " actual_batch=" << actual_batch << " vocab=" << vocab << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "m31_cache_oracle: " << error.what() << '\n';
        return 1;
    }
}
