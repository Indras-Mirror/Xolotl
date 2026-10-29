#include "m31_engine.hpp"
#include "cpu/fused_decode.hpp"
#include "cpu/threadpool_config.hpp"
#include "cpu/task_barrier.hpp"
#include <atomic>
#include <thread>
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace strata::m31;
namespace {
void check(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
Config config() {
    Config cfg;
    cfg.threads = 2;
    cfg.context = 768;
    cfg.max_chunk = 300;
    cfg.cpu_fused = false; // Synthetic F32 fixture, rather than production quantized experts.
    return cfg;
}
void initialize(Engine & engine, const std::string & model, Config cfg = config()) {
    std::string error;
    const bool ready = engine.init(model, cfg, error);
    check(ready, "init: " + error);
}
void forward(Engine & engine, const std::vector<int32_t> & ids) {
    std::string error;
    const bool completed = engine.forward(ids.data(), int(ids.size()), error);
    check(completed, "forward: " + error);
}
std::vector<float> head(Engine & engine, int row, int count = 1) {
    std::string error; std::vector<float> result;
    const bool completed = engine.logits(row, count, result, error);
    check(completed, "logits: " + error);
    return result;
}
std::vector<int32_t> tokens(size_t n, unsigned offset = 0) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = int32_t((i * 17 + offset) % 64);
    return ids;
}
void check_fused_participants() {
    struct Backend { ggml_backend_t value = ggml_backend_cpu_init(); ~Backend() { if (value) ggml_backend_free(value); } } backend;
    check(backend.value != nullptr, "fused fixture backend");
    struct Storage {
        ggml_context * context = ggml_init({65536, nullptr, true});
        ggml_backend_buffer_t buffer = nullptr;
        ~Storage() { if (buffer) ggml_backend_buffer_free(buffer); if (context) ggml_free(context); }
    } storage;
    check(storage.context != nullptr, "fused fixture metadata");
    std::array<ggml_tensor *, 3> weights;
    for (auto & tensor : weights) tensor = ggml_new_tensor_3d(storage.context, GGML_TYPE_IQ4_XS, 256, 256, 2);
    storage.buffer = ggml_backend_alloc_ctx_tensors(storage.context, backend.value);
    check(storage.buffer != nullptr, "fused fixture weights");
    for (size_t role = 0; role < weights.size(); ++role) {
        std::vector<float> values(size_t(ggml_nelements(weights[role]))), importance(256, 1.f);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i * 17 + role * 7) % 101) - 50) * .002f;
        std::vector<uint8_t> quant(ggml_nbytes(weights[role]));
        check(ggml_quantize_chunk(GGML_TYPE_IQ4_XS, values.data(), quant.data(), 0,
            int64_t(values.size() / 256), 256, importance.data()) == quant.size(), "fused fixture quantization");
        ggml_backend_tensor_set(weights[role], quant.data(), 0, quant.size());
    }
    std::vector<float> input(256), reference;
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 13) - 6) * .1f;
    // The last arm asks for four workers from a two-worker pool: ggml clamps
    // the actual callback nth to two. The fused barrier must finish and agree.
    for (const auto setting : {std::pair<int, int>{1, 1}, {4, 4}, {2, 4}}) {
        auto parameters = ggml_threadpool_params_default(setting.first);
        auto * pool = ggml_threadpool_new(&parameters); check(pool != nullptr, "fused fixture pool");
        {
            CpuFusedDecode fused(backend.value, pool, unsigned(setting.second));
            std::vector<float> output;
            for (int repeat = 0; repeat < 3; ++repeat) {
                fused.forward(weights[0], weights[1], weights[2], nullptr, nullptr, nullptr,
                    input, {1, 0}, {.375f, .625f}, output);
                if (reference.empty()) reference = output;
                else check(output == reference, "fused math changed with actual participant count");
            }
        }
        ggml_backend_cpu_set_threadpool(backend.value, nullptr);
        ggml_threadpool_free(pool);
    }
}

}
int main(int argc, char ** argv) {
    try {
        check(argc == 3, "usage: m31_core_check MODEL OUTPUT_LOGITS");
        const std::string model = argv[1]; std::string error;
        Engine engine; initialize(engine, model);
        auto prompt = tokens(29);
        forward(engine, prompt);
        const auto batch = head(engine, 0, 29);
        for (float value : batch) check(std::isfinite(value), "non-finite head");
        std::ofstream output(argv[2], std::ios::binary);
        output.write(reinterpret_cast<const char *>(batch.data()), std::streamsize(batch.size() * sizeof(float)));
        check(bool(output), "cannot write fixture logits"); output.close();

#ifdef M31_ENGINE_VISION
        // Inject exact token-embedding rows as visual F32. This crosses real
        // mixed attention/MSA/experts/head code and must preserve every logit.
        auto artifact = strata::GgufModel::open(model);
        size_t embedding_shard = 0;
        const auto * embedding = artifact.find("token_embd.weight", &embedding_shard);
        check(embedding && embedding->type == GGML_TYPE_F32, "mixed fixture requires F32 token embeddings");
        const auto * embedding_data = artifact.shard(embedding_shard).tensor_data(*embedding);
        const size_t width = size_t(engine.geometry().embd);
        auto mixed_ids = prompt;
        std::vector<int32_t> visual_map(prompt.size(), -1);
        std::vector<float> visual(4 * width);
        for (size_t i = 0; i < 4; ++i) {
            std::memcpy(visual.data() + i * width, embedding_data + size_t(prompt[i + 2]) * width * sizeof(float), width * sizeof(float));
            mixed_ids[i + 2] = -1; visual_map[i + 2] = int32_t(i);
        }
        engine.reset();
        check(engine.forward_mixed(mixed_ids.data(), visual_map.data(), visual.data(), 4, int(prompt.size()), error), error);
        check(head(engine, 0, 29) == batch, "mixed embedding injection changed batch logits");
        const auto mixed_position = engine.position();
        auto invalid_map = visual_map; invalid_map[2] = 4;
        check(!engine.forward_mixed(mixed_ids.data(), invalid_map.data(), visual.data(), 4, int(prompt.size()), error),
              "out-of-range mixed map accepted");
        auto nonfinite_visual = visual; nonfinite_visual[0] = std::numeric_limits<float>::quiet_NaN();
        check(!engine.forward_mixed(mixed_ids.data(), visual_map.data(), nonfinite_visual.data(), 4, int(prompt.size()), error),
              "non-finite mixed embedding accepted");
        check(engine.position() == mixed_position && head(engine, 0, 29) == batch, "mixed validation poisoned state");
        Engine::PrefixCheckpointPtr mixed_checkpoint;
        check(engine.checkpoint_prefix(1 << 20, mixed_checkpoint, error), error);
        const auto mixed_suffix = tokens(3, 19);
        forward(engine, mixed_suffix); const auto mixed_extended = head(engine, 0, 3);
        check(engine.restore_prefix(mixed_checkpoint, error), error);
        forward(engine, mixed_suffix);
        check(head(engine, 0, 3) == mixed_extended, "mixed checkpoint restore changed logits");
        const auto after_release_suffix = tokens(2, 9);
        forward(engine, after_release_suffix); const auto after_release_expected = head(engine, 0, 2);
        check(engine.restore_prefix(mixed_checkpoint, error), error);
        forward(engine, mixed_suffix); // A live head exists when planning discards scratch.
        const auto before_release = engine.position();
        check(engine.release_prefill_workspace(error), error);
        check(engine.position() == before_release, "scratch release changed mixed KV position");
        std::vector<float> released_head;
        check(!engine.logits(0, 1, released_head, error), "scratch release exposed a stale head");
        forward(engine, after_release_suffix);
        check(head(engine, 0, 2) == after_release_expected, "scratch release changed restored mixed KV rows");
#endif

        // Independently exercise decode topology; NumPy checks the batch below.
        engine.reset(); std::vector<float> decode;
        for (auto id : prompt) {
            check(engine.forward(&id, 1, error), error);
            const auto row = head(engine, 0); decode.insert(decode.end(), row.begin(), row.end());
        }
        check(decode.size() == batch.size(), "decode output shape");
        for (size_t i = 0; i < batch.size(); ++i)
            check(std::abs(batch[i] - decode[i]) < .005f + .005f * std::abs(batch[i]), "batch/decode math mismatch");

        // RAM repacking must preserve every expert byte and scalar scale.
        Engine packed; Config packed_cfg = config(); packed_cfg.arena_gib = .01;
        initialize(packed, model, packed_cfg); forward(packed, prompt);
        check(head(packed, 0, 29) == batch, "mapped/packed experts changed logits");

        engine.reset(); forward(engine, tokens(7));
        Engine::PrefixCheckpointPtr checkpoint;
        check(!engine.checkpoint_prefix(1, checkpoint, error) && !checkpoint, "checkpoint budget bypass");
        check(engine.checkpoint_prefix(1 << 20, checkpoint, error), error);
        check(checkpoint->position() == 7 && checkpoint->bytes() == engine.prefix_checkpoint_bytes(), "checkpoint shape");
        const auto extension = tokens(3, 5);
        forward(engine, extension); const auto extended = head(engine, 0, 3);
        check(engine.restore_prefix(checkpoint, error), error);
        check(engine.position() == 7, "restore position");
        std::vector<float> stale;
        check(!engine.logits(0, 1, stale, error), "restore exposed a stale head");
        forward(engine, extension); check(head(engine, 0, 3) == extended, "checkpoint lost main/index KV");
        const auto packed_pos = packed.position();
        check(!packed.restore_prefix(checkpoint, error) && packed.position() == packed_pos, "foreign checkpoint changed state");

        // Rewind keeps main and MSA caches before a safe common-prefix boundary.
        engine.reset(); const auto prefix = tokens(256);
        auto original = prefix; const auto suffix = tokens(7, 11);
        original.insert(original.end(), suffix.begin(), suffix.end());
        forward(engine, original); // 263 rows, so the old pass had a larger padded key window.
        check(!engine.rewind(255, error) && engine.position() == 263, "unsafe boundary accepted");
        check(!engine.rewind(264, error) && engine.position() == 263, "future rewind accepted");
        check(engine.rewind(256, error), error);
        check(!engine.logits(0, 1, stale, error), "rewind exposed a stale head");
        const auto branch = tokens(4, 23); forward(engine, branch); const auto reused = head(engine, 0, 4);
        Engine fresh; initialize(fresh, model); forward(fresh, prefix); forward(fresh, branch);
        check(head(fresh, 0, 4) == reused, "prefix reuse changed head bytes");

        // Preflight rejection preserves the previous valid state and head.
        const auto valid_position = engine.position(); const auto valid_head = head(engine, 0, 4);
        const int32_t invalid = 64;
        check(!engine.forward(&invalid, 1, error) && engine.position() == valid_position, "invalid token changed state");
        check(head(engine, 0, 4) == valid_head, "preflight rejection poisoned valid head");
        check(!engine.logits(std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), stale, error),
              "oversized head range accepted");
        check(head(engine, 0, 4) == valid_head, "head range rejection poisoned valid state");

        // A partially executed cancellation needs reset, then the next request works.
        int polls = 0; Config cancel_cfg = config(); cancel_cfg.cancelled = [&] { return ++polls == 3; };
        Engine cancelled; initialize(cancelled, model, cancel_cfg);
        check(!cancelled.forward(prompt.data(), int(prompt.size()), error) && error.find("cancelled") != std::string::npos,
              "mid-forward cancellation ignored");
        check(!cancelled.forward(prompt.data(), int(prompt.size()), error), "partial forward remained reusable");
        cancelled.reset(); forward(cancelled, prompt);
        check(head(cancelled, 0, 29) == batch, "reset after cancellation changed request");

        check_fused_participants();

        // Barrier rounds use the actual participants, independent of a
        // configured upper bound. This exercises the old pool-mismatch failure.
        for (unsigned participants : {1u, 2u, 4u}) {
            CpuTaskBarrier barrier;
            std::vector<int> values(participants, 0);
            std::atomic<bool> mismatch{false}; std::vector<std::thread> workers;
            for (unsigned ith = 0; ith < participants; ++ith) workers.emplace_back([&, ith] {
                for (int round = 1; round <= 200; ++round) {
                    values[ith] = round;
                    barrier.wait(participants);
                    for (int value : values) if (value != round) mismatch.store(true);
                    barrier.wait(participants);
                }
            });
            for (auto & worker : workers) worker.join();
            check(!mismatch.load(), "actual-participant barrier lost a phase");
        }
        auto * backend = ggml_backend_cpu_init(); check(backend != nullptr, "CPU backend");
        bool rejected = false;
        try { CpuFusedDecode missing_pool(backend, nullptr, 2); } catch (const std::invalid_argument &) { rejected = true; }
        ggml_backend_free(backend);
        check(rejected, "fused CPU accepted a missing persistent pool");
        auto affinity = ggml_threadpool_params_default(2); rejected = false;
        try { apply_cpu_pool_config(affinity, "0", 50); } catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "undersized CPU affinity accepted");
        std::cout << "core math/state/arena/cancellation/pool checks passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
