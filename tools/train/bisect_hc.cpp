// bisect_hc <mix|combine> [D] [hc] [lr] [T]
// Minimal single-op bisect for the hyper-connection hang: builds ONE of qfn_hc_mix / qfn_hc_combine, its backward,
// and computes on 1 thread. Prints markers so a hang is localised rather than guessed at.
#include "ggml.h"
#include "ggml-cpu.h"
#include "qfn_layer_train.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace xolotl::train;

int main(int argc, char ** argv) {
    const char * which = argc > 1 ? argv[1] : "mix";
    const int D = argc > 2 ? atoi(argv[2]) : 8;
    const int hc = argc > 3 ? atoi(argv[3]) : 3;
    const int lr = argc > 4 ? atoi(argv[4]) : 6;
    const int T = argc > 5 ? atoi(argv[5]) : 8;

    ggml_init_params ip = { 64u << 20, nullptr, false };
    ggml_context * c = ggml_init(ip);
    std::printf("[bisect] %s D %d hc %d lr %d T %d\n", which, D, hc, lr, T);
    std::fflush(stdout);

    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, hc, T);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = 0.1f * (float) (i % 7) - 0.3f;
    ggml_tensor * rr = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, hc, T);
    for (int64_t i = 0; i < ggml_nelements(rr); ++i) ((float *) rr->data)[i] = 0.1f * (float) (i % 5) - 0.2f;

    QfnHc w;
    w.norm = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, hc);
    w.down = ggml_new_tensor_2d(c, GGML_TYPE_F32, D * hc, lr);
    w.up = ggml_new_tensor_2d(c, GGML_TYPE_F32, lr, D * hc);
    w.inject = ggml_new_tensor_2d(c, GGML_TYPE_F32, D * hc, hc);
    for (int64_t i = 0; i < ggml_nelements(w.norm); ++i) ((float *) w.norm->data)[i] = 1.0f + 0.05f * (float) (i % 5);
    for (int64_t i = 0; i < ggml_nelements(w.down); ++i) ((float *) w.down->data)[i] = 0.02f * (float) ((i * 7) % 11) - 0.1f;
    for (int64_t i = 0; i < ggml_nelements(w.up); ++i) ((float *) w.up->data)[i] = 0.02f * (float) ((i * 5) % 13) - 0.1f;
    for (int64_t i = 0; i < ggml_nelements(w.inject); ++i) ((float *) w.inject->data)[i] = 0.02f * (float) ((i * 3) % 9) - 0.08f;

    ggml_tensor * out = nullptr;
    if (!std::strcmp(which, "mix")) {
        ggml_tensor * inj = nullptr;
        out = qfn_hc_mix(c, x, w, hc, 1e-6f, &inj);
        std::printf("[bisect] mix built\n"); std::fflush(stdout);
    } else {
        ggml_tensor * block = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, 1, T);
        for (int64_t i = 0; i < ggml_nelements(block); ++i) ((float *) block->data)[i] = 0.05f * (float) (i % 4);
        ggml_tensor * inj = ggml_new_tensor_2d(c, GGML_TYPE_F32, hc, T);
        for (int64_t i = 0; i < ggml_nelements(inj); ++i) ((float *) inj->data)[i] = 0.1f * (float) (i % 3);
        ggml_tensor * r3 = ggml_reshape_3d(c, x, D, hc, T);
        ggml_tensor * b3 = ggml_reshape_3d(c, block, D, 1, T);
        out = qfn_hc_combine(c, r3, ggml_reshape_2d(c, b3, D, T), inj, hc);
        std::printf("[bisect] combine built\n"); std::fflush(stdout);
    }

    ggml_set_param(x);
    ggml_tensor * o3 = out;   // mix already returns [D, 1, T]; combine returns [D, hc, T] - both match rr
    std::printf("[bisect] out ne %lld %lld %lld\n", (long long) o3->ne[0], (long long) o3->ne[1], (long long) o3->ne[2]);
    ggml_tensor * r = ggml_new_tensor_3d(c, GGML_TYPE_F32, o3->ne[0], o3->ne[1], o3->ne[2]);
    for (int64_t i = 0; i < ggml_nelements(r); ++i) ((float *) r->data)[i] = 0.1f * (float) (i % 5) - 0.2f;
    ggml_tensor * loss = ggml_sum(c, ggml_mul(c, o3, r));
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, true);
    ggml_build_forward_expand(gf, loss);
    std::printf("[bisect] forward graph\n"); std::fflush(stdout);
    ggml_build_backward_expand(c, gf, nullptr);
    ggml_graph_reset(gf);
    std::printf("[bisect] backward graph\n"); std::fflush(stdout);
    ggml_graph_compute_with_ctx(c, gf, 1);
    std::printf("[bisect] computed, loss %.6f\n", ((float *) loss->data)[0]);
    ggml_free(c);
    return 0;
}
