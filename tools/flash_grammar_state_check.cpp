// CPU-only regression for Flash's shared native GBNF state helper.
// Link exactly like q36-grammar-check. No inference/model weights are loaded.
#include "q36_grammar.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void need(bool v, const char* m) { if (!v) throw std::runtime_error(m); }
int main(int argc, char** argv) {
    try {
        need(argc == 3, "usage: flash-grammar-state-check MODEL SCENE_GBNF");
        llama_backend_init();
        auto params = llama_model_default_params();
        params.vocab_only = true; params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], params), llama_model_free);
        need(bool(model), "vocabulary load failed");
        auto* vocab = llama_model_get_vocab(model.get());
        auto tokens = [&](const char* s) {
            std::vector<llama_token> v(256);
            int n = llama_tokenize(vocab, s, std::strlen(s), v.data(), v.size(), false, true);
            need(n > 0, "tokenize failed"); v.resize(n); return v;
        };
        auto permits = [&](const Q36Grammar& g, int id) {
            float value = 0; g.mask(&value, &id, 1); return std::isfinite(value);
        };
        auto advance = [&](Q36Grammar& g, const char* s) {
            for (int id : tokens(s)) { need(permits(g,id), "valid token rejected"); g.accept(id); }
        };
        int eos = tokens("<|im_end|>").front(), trigger = tokens("</think>").front();
        Q36Grammar::Caches cache;
        Q36Grammar fresh; fresh.init(vocab, argv[2], -1, cache);
        Q36Grammar preview(fresh); advance(preview, "speak=false");
        need(permits(preview,eos), "completed preview rejects EOS");
        need(!permits(fresh,eos), "preview changed authoritative request");
        // Assignment is the actual slot-admission mechanism. Assignment must clone,
        // including self-assignment, and replace the previous request's state.
        Q36Grammar slot; slot = preview; slot = slot;
        need(permits(slot,eos), "copy assignment lost completed state");
        slot = fresh; need(!permits(slot,eos), "slot retained previous request state");
        advance(slot,"speak=true;what=");
        need(!permits(fresh,eos), "slot copy changed its source");
        advance(fresh,"speak=false");
        need(!permits(slot,eos), "source completion changed slot grammar");
        // Cancelling a rejected speculative preview means simply discarding it.
        Q36Grammar current; current.init(vocab, argv[2], -1, cache);
        { Q36Grammar rejected(current); advance(rejected,"speak=false"); }
        need(!permits(current,eos), "discarded proposal leaked accepted tokens");
        Q36Grammar lazy; lazy.init(vocab, argv[2], trigger, cache);
        need(permits(lazy,eos), "aftertrigger masked pre-trigger reasoning");
        Q36Grammar lazy_preview(lazy); lazy_preview.accept(trigger);
        need(!permits(lazy_preview,eos), "preview failed to activate after trigger");
        need(permits(lazy,eos), "preview activated original lazy grammar");
        lazy.accept(trigger); advance(lazy,"speak=false");
        need(permits(lazy,eos), "aftertrigger completed grammar rejects EOS");
        // The tiny literal branch must reject an unrelated token at its initial state,
        // and repeated cache hits must always recreate that initial state.
        for (int round = 0; round < 4; ++round) {
            Q36Grammar tiny; tiny.init(vocab, argv[2], -1, cache);
            need(!permits(tiny,eos), "fresh cached state incorrectly accepts EOS");
            advance(tiny,"speak=false");
            need(permits(tiny,eos), "literal branch failed after repeated reset");
        }
        Q36Grammar disabled; slot = disabled;
        need(!slot.active(), "grammar-free admission retained old grammar");
        need(permits(slot,eos), "grammar-free request still masked");
        std::cout << "PASS clone assignment reset speculative-discard aftertrigger EOS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
