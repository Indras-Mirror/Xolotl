// Explicit-model cache regression. All model interaction uses the public Engine API.
#include "m31_engine.hpp"
#include "m31_run.hpp"
#ifdef M31_ENGINE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstring>
#include <iostream>
#include <map>
#include <set>

using namespace strata::m31;
namespace fs = std::filesystem;
namespace {
void require(bool condition, const std::string & error) { if (!condition) throw std::runtime_error(error); }
bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
struct Backend {
    ggml_backend_t value = nullptr;
    ~Backend() { if (value) ggml_backend_free(value); }
};
}
int main(int argc, char ** argv) {
    try {
        const std::set<std::string> allowed = {"--model", "--ids-file", "--out", "--length", "--reuse",
            "--backend", "--device", "--ctx", "--threads", "--chunk", "--cpu-fused", "--arena-gib",
            "--gpu-experts-gib", "--register-arena", "--register-limit-gib", "--checkpoint-mib", "--decode-steps"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "cache_model_check --model GGUF --ids-file IDS.i32 --length N --reuse R --out NEW-DIR\n"
                             "  [--backend cpu|cuda --device 0 --ctx 4096 --threads N --chunk 32768]\n"
                             "  [--arena-gib N --gpu-experts-gib N --register-arena 0|1 --register-limit-gib N]\n"
                             "  [--checkpoint-mib 512 --decode-steps 7 --cpu-fused 0|1]\n";
                return 0;
            }
            require(allowed.count(key) && i + 1 < argc, "unknown or missing argument: " + key);
            require(args.emplace(key, argv[++i]).second, "duplicate argument: " + key);
        }
        auto text = [&](const std::string & key, const std::string & fallback = "") {
            const auto found = args.find(key); return found == args.end() ? fallback : found->second;
        };
        auto integer = [&](const std::string & key, int64_t fallback, int64_t minimum, int64_t maximum) {
            if (!args.count(key)) return fallback;
            size_t end = 0; const auto value = std::stoll(args.at(key), &end);
            require(end == args.at(key).size() && value >= minimum && value <= maximum, "invalid integer: " + key);
            return int64_t(value);
        };
        auto real = [&](const std::string & key, double fallback) {
            if (!args.count(key)) return fallback;
            size_t end = 0; const double value = std::stod(args.at(key), &end);
            require(end == args.at(key).size() && std::isfinite(value) && value >= 0, "invalid budget: " + key);
            return value;
        };
        const std::string model = text("--model"), ids_file = text("--ids-file"), backend_name = text("--backend", "cpu");
        require(!model.empty() && !ids_file.empty() && !text("--out").empty(), "model, ids-file and new out directory are required");
        const int length = int(integer("--length", 0, 1, INT32_MAX));
        const int reused = int(integer("--reuse", 0, 1, INT32_MAX));
        const int decode_steps = int(integer("--decode-steps", 7, 1, 1024));
        require(length > reused && reused > 0 && reused % 256 == 0, "length must exceed a positive 256-token reuse boundary");
        const auto checkpoint_bytes = size_t(integer("--checkpoint-mib", 512, 1, 65536)) << 20;
        Config cfg;
        cfg.context = integer("--ctx", cfg.context, 1, INT32_MAX);
        cfg.threads = int(integer("--threads", cfg.threads, 1, GGML_MAX_N_THREADS));
        cfg.max_chunk = int(integer("--chunk", cfg.max_chunk, 1, 32768));
        cfg.cpu_fused = integer("--cpu-fused", 1, 0, 1) != 0;
        cfg.arena_gib = real("--arena-gib", 0); cfg.gpu_experts_gib = real("--gpu-experts-gib", 0);
        cfg.register_arena = integer("--register-arena", 0, 0, 1) != 0;
        const double pin_gib = real("--register-limit-gib", 0);
        require(pin_gib <= 65536, "registration budget exceeds the test cap");
        cfg.register_limit_bytes = uint64_t(pin_gib * double(uint64_t(1) << 30));
        require(int64_t(length) + decode_steps <= cfg.context, "context must hold the prompt and decode steps");
        require(backend_name == "cpu" || backend_name == "cuda", "backend must be cpu or cuda");
        prepare_engine_environment(); validate_config(cfg, backend_name == "cuda");
        const fs::path out = text("--out"); require(!fs::exists(out), "output directory must be new");
        fs::create_directories(out);
        std::ifstream input(ids_file, std::ios::binary | std::ios::ate);
        require(bool(input), "cannot open token IDs");
        require(input.tellg() >= std::streamoff(length) * 4, "token IDs file is too short");
        input.seekg(0); std::vector<int32_t> ids(size_t(length), 0);
        input.read(reinterpret_cast<char *>(ids.data()), std::streamsize(length) * 4);
        require(bool(input), "cannot read token IDs");
        Backend backend;
        if (backend_name == "cuda") {
#ifdef M31_ENGINE_CUDA
            backend.value = ggml_backend_cuda_init(int(integer("--device", 0, 0, 1024)));
            require(backend.value != nullptr, "cannot initialize the CUDA backend");
#else
            throw std::runtime_error("CUDA backend is unavailable in this build");
#endif
        }
        cfg.backend = backend.value;
        Engine engine; std::string error;
        require(engine.init(model, cfg, error), error);
        int last_chunk = 0;
        std::vector<std::pair<int, int>> current_chunks;
        std::map<std::string, std::vector<std::pair<int, int>>> recipes;
        auto pass = [&](int first, int count) {
            while (count) {
                last_chunk = engine.prefill_chunk(std::min(count, cfg.max_chunk), error);
                require(last_chunk > 0, error);
                require(engine.forward(ids.data() + first, last_chunk, error), error);
                current_chunks.emplace_back(first, last_chunk);
                first += last_chunk; count -= last_chunk;
            }
        };
        std::vector<std::pair<std::string, int>> heads;
        auto head = [&](const std::string & name) {
            std::vector<float> values;
            require(engine.logits(last_chunk - 1, 1, values, error), error);
            std::ofstream file(out / (name + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
            require(bool(file), "cannot write head: " + name);
            const int top = int(std::max_element(values.begin(), values.end()) - values.begin());
            heads.emplace_back(name, top);
            recipes[name] = current_chunks; current_chunks.clear();
            std::cout << name << " top1=" << top << " position=" << engine.position() << '\n';
            return values;
        };
        auto decode = [&](std::vector<float> values) {
            require(engine.release_prefill_workspace(error), error);
            for (int step = 0; step < decode_steps; ++step) {
                const int32_t token = int32_t(std::max_element(values.begin(), values.end()) - values.begin());
                const int position = int(engine.position());
                require(engine.forward(&token, 1, error), error); last_chunk = 1;
                current_chunks.emplace_back(position, 1);
                require(engine.logits(0, 1, values, error), error);
            }
        };
        auto rewind = [&] { require(engine.rewind(reused, error), error); };
        auto tail_recipe = [&](const std::string & name) {
            std::vector<std::pair<int, int>> result;
            for (auto chunk : recipes.at(name)) if (chunk.first >= reused && chunk.first < length) result.push_back(chunk);
            return result;
        };
        auto check_tail = [&](const std::string & reference_name, const std::vector<float> & reference,
                              const std::string & name, const std::vector<float> & values) {
            require(tail_recipe(reference_name) == tail_recipe(name),
                    "VRAM planner changed the replay recipe; select a smaller --chunk for this check");
            require(same(reference, values), "same-history head bytes changed: " + name);
        };
        engine.reset(); pass(0, length); const auto whole = head("whole");
        rewind(); pass(reused, length - reused); const auto whole_rewind = head("whole_rewind");
        engine.reset(); pass(0, length); const auto whole_again = head("whole_again");
        require(recipes.at("whole") == recipes.at("whole_again"),
                "VRAM planner changed the cold recipe; select a smaller --chunk for this check");
        require(same(whole, whole_again), "repeated cold recipe changed head bytes");
        decode(whole_again); rewind(); pass(reused, length - reused);
        check_tail("whole_rewind", whole_rewind, "whole_decode_rewind", head("whole_decode_rewind"));
        engine.reset(); pass(0, reused);
        Engine::PrefixCheckpointPtr checkpoint;
        require(engine.checkpoint_prefix(checkpoint_bytes, checkpoint, error), error);
        pass(reused, length - reused); const auto split = head("split");
        rewind(); pass(reused, length - reused);
        check_tail("split", split, "split_rewind", head("split_rewind"));
        decode(split); rewind(); pass(reused, length - reused);
        check_tail("split", split, "split_decode_rewind", head("split_decode_rewind"));
        require(engine.restore_prefix(checkpoint, error), error); pass(reused, length - reused);
        check_tail("split", split, "restore", head("restore"));
        require(engine.release_prefill_workspace(error), error); rewind(); pass(reused, length - reused);
        check_tail("split", split, "release_rewind", head("release_rewind"));
        std::ofstream summary(out / "summary.json");
        summary << "{\"same_history_bitwise\":true,\"length\":" << length << ",\"reused\":" << reused
                << ",\"vocab\":" << engine.geometry().vocab << ",\"whole_vs_split_bitwise\":"
                << (same(whole, split) ? "true" : "false") << ",\"whole_vs_replay_bitwise\":"
                << (same(whole, whole_rewind) ? "true" : "false") << ",\"heads\":{";
        for (size_t i = 0; i < heads.size(); ++i) {
            const auto & name = heads[i].first;
            summary << (i ? "," : "") << run::quote(name) << ":{\"top1\":" << heads[i].second << ",\"forward_chunks\":[";
            const auto & chunks = recipes.at(name);
            for (size_t j = 0; j < chunks.size(); ++j)
                summary << (j ? "," : "") << "{\"position\":" << chunks[j].first << ",\"tokens\":" << chunks[j].second << '}';
            summary << "]}";
        }
        summary << "}}\n";
        require(bool(summary), "cannot write summary");
        std::cout << "All same-history cache invariants passed bitwise. Whole and segmented recipes are reported separately.\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
