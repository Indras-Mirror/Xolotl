// GPU integration: layer-major batch prefill across a layer split (stage-1 span_cap after handoff).
//   DS41_TEST_GGUF=/path/model.gguf build-buster/test_ds41_split_batch_prefill
// Skips (77) without GGUF or fewer than two CUDA devices.
#include "ds41_pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef STRATA_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

int main(int argc, char ** argv) {
    const char * gguf = nullptr;
    if (argc >= 2 && argv[1][0] != '\0') gguf = argv[1];
    else if (const char * env = std::getenv("DS41_TEST_GGUF")) {
        if (env[0] != '\0') gguf = env;
    }
    if (!gguf) {
        std::fprintf(stderr, "test_ds41_split_batch_prefill: skipped (set DS41_TEST_GGUF or pass a .gguf path)\n");
        return 77;
    }
#ifdef STRATA_ENABLE_CUDA
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 2) {
        std::fprintf(stderr, "test_ds41_split_batch_prefill: skipped (need 2 CUDA devices)\n");
        return 77;
    }
#else
    std::fprintf(stderr, "test_ds41_split_batch_prefill: skipped (no CUDA build)\n");
    return 77;
#endif
    using namespace strata::v41;
    Ds41FileBlobs blobs(gguf);
    std::string err;
    if (!blobs.open(err)) {
        std::fprintf(stderr, "FAIL: open blobs: %s\n", err.c_str());
        return 1;
    }
    Ds41DenseConfig dc;
    dc.prompt_batch = 8192;
    dc.ctx_max = 262144;
    dc.max_pass = 512;
    dc.elastic_cols = true;
    Ds41MoeConfig mc;
    mc.cpu_only = true;
    Ds41Pipeline pipe;
    const std::vector<int> devs = { 0, 1 };
    if (!ds41_build_pipeline(gguf, "20", devs, -1, 1.5, {}, dc, mc, blobs, pipe, err)) {
        std::fprintf(stderr, "FAIL: pipeline: %s\n", err.c_str());
        return 1;
    }
    const int B = 512;
    const int split = pipe.split_layer;
    std::vector<int> ids((size_t) B, 100);
    for (int st = 0; st < pipe.n_stages(); ++st) {
        if (!pipe.dense(st).set_columns(B)) {
            std::fprintf(stderr, "FAIL: set_columns stage %d: %s\n", st, pipe.dense(st).last_error().c_str());
            return 1;
        }
    }
    if (!pipe.dense(0).begin_batch(ids.data(), B, 0)) {
        std::fprintf(stderr, "FAIL: begin_batch stage0: %s\n", pipe.dense(0).last_error().c_str());
        return 1;
    }
    if (!pipe.sync_handoff(0, B, err)) {
        std::fprintf(stderr, "FAIL: handoff: %s\n", err.c_str());
        return 1;
    }
    if (!pipe.dense(1).begin_batch_continued(ids.data(), B, 0)) {
        std::fprintf(stderr, "FAIL: begin_batch_continued: %s\n", pipe.dense(1).last_error().c_str());
        return 1;
    }
    if (!pipe.dense(1).attn_router_span(split, 0, std::min(512, B))) {
        std::fprintf(stderr, "FAIL: attn_router_span layer %d (bad span if span_cap missing): %s\n", split,
                     pipe.dense(1).last_error().c_str());
        return 1;
    }
    std::fprintf(stderr, "test_ds41_split_batch_prefill: ok (B=%d split=%d)\n", B, split);
    return 0;
}
