// Adapted from llama.cpp PR #31192 (MIT); see repository LICENSE.
// Independent PR #31192 LLM oracle for one complete mixed embedding prefill.
// mtmd executes in the encoder process; this process links only the PR ggml ABI.
#include "../../m31_run.hpp"
#include "../../m31_mixed.hpp"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-cpu.h"

#include <cstring>
#include <memory>

// Audited PR internal introspection, read-only. Avoid exposing its model layout.
const std::vector<std::pair<std::string, ggml_tensor *>> &
llama_internal_get_tensor_map(const llama_model * model);

int main(int argc, char ** argv) {
    using namespace strata::m31;
    try {
        const auto args = run::parse(argc, argv);
        if (args.mixed_manifest.empty() || args.decode_tail || args.taps != "none" ||
            args.arena_gib || args.gpu_experts_gib)
            throw std::runtime_error("vision oracle requires --mixed-manifest, no decode-tail/taps/native optimizations");
        if (args.backend == "cpu") setenv("CUDA_VISIBLE_DEVICES", "", 1);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = args.backend == "cuda" ? 999 : 0;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        llama_model_tensor_buft_override overrides[] = {
            {".*ffn_.*_exps\\..*", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        if (args.backend == "cuda") mp.tensor_buft_overrides = overrides;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(args.model.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("PR model loading failed");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        const auto mixed = load_mixed_manifest(args.mixed_manifest, vocab, args.context);
        const int n = int(mixed.tokens.size());
        if (n > args.chunk) throw std::runtime_error("oracle single prefill requires chunk >= mixed rows");
        const int dim = llama_model_n_embd_inp(model.get());
        if (dim != 6144) throw std::runtime_error("unexpected embedding width");
        const ggml_tensor * token_embd = nullptr;
        for (const auto & item : llama_internal_get_tensor_map(model.get()))
            if (item.first == "token_embd.weight") token_embd = item.second;
        if (!token_embd || token_embd->ne[0] != dim || token_embd->ne[1] != vocab)
            throw std::runtime_error("missing/unexpected token embedding tensor");
        const auto * traits = ggml_get_type_traits(token_embd->type);
        std::vector<char> quantized(ggml_row_size(token_embd->type, dim));
        std::vector<float> input(size_t(n) * dim);
        for (int row = 0; row < n; ++row) {
            float * dst = input.data() + size_t(row) * dim;
            if (mixed.tokens[row] == -1) {
                const size_t visual = size_t(mixed.visual_rows[row]);
                std::memcpy(dst, mixed.embeddings.data() + visual * dim, dim * sizeof(float));
            } else {
                ggml_backend_tensor_get(token_embd, quantized.data(),
                    size_t(mixed.tokens[row]) * token_embd->nb[1], quantized.size());
                if (token_embd->type == GGML_TYPE_F32) std::memcpy(dst, quantized.data(), dim * sizeof(float));
                else if (traits->to_float) traits->to_float(quantized.data(), dst, dim);
                else throw std::runtime_error("token embedding type lacks dequantizer");
            }
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = args.context;
        cp.n_batch = cp.n_ubatch = args.chunk;
        if (args.head_tail) cp.n_outputs_max = args.head_tail;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = args.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = args.backend == "cuda";
        cp.op_offload = args.oracle_op_offload;
        cp.kv_unified = false;
        cp.no_perf = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("PR context creation failed");
        if (args.head_tail)
            llama_set_embeddings_layer_inp(ctx.get(), uint32_t(llama_model_n_layer(model.get())), true);
        if (!run::fs::create_directory(args.out)) throw std::runtime_error("output already exists");
        run::Dump dump(run::fs::path(args.out) / run::fs::path(args.mixed_manifest).stem(), args, mixed.tokens, vocab);
        auto batch = llama_batch_init(n, dim, 1);
        batch.n_tokens = n;
        std::memcpy(batch.embd, input.data(), input.size() * sizeof(float));
        for (int row = 0; row < n; ++row) {
            batch.pos[row] = mixed.positions[row];
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = 0;
            batch.logits[row] = !args.head_tail || row >= n - std::min(n, args.head_tail);
        }
        dump.record_chunk(0, n);
        if (llama_decode(ctx.get(), batch)) throw std::runtime_error("PR mixed llama_decode failed");
        for (int row = 0; row < n; ++row) if (dump.wants(row)) {
            const float * logits = llama_get_logits_ith(ctx.get(), row);
            if (!logits) throw std::runtime_error("missing logits row");
            dump.row(row, logits);
        }
        llama_batch_free(batch);
        dump.finish("llama.cpp-PR31192", n);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "m31_vision_oracle: %s\n", error.what());
        return 1;
    }
}
