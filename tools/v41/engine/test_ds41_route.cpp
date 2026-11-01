// test_ds41_route [--cuda] (built with -DSTRATA_DSV41_ROUTE_OP=ON on a patched llama.cpp) - the fused
// GGML_OP_DSV41_ROUTE_SCORES / _WEIGHTS vs the stock-op composite ds41_dense.cpp build_attn's flat-bias router builds
// (softplus, sqrt, the vision bias select, two adds; get_rows, sum_rows, ggml_scale_bias, div, ggml_scale) on the same
// backend, bit for bit: the scores, the selection scores, the top-k ids taken from each and the weights.  Logits from
// the model's range plus softplus' > 20 branch, large negatives, zeros and -0; biases and route biases of the engine's
// sizes; image masks of all-text, all-image and mixed passes; with and without the norm.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> synth_logits(int64_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 15);
    std::vector<float> x((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        switch (kind(rng)) {
            case 0:  x[i] = 0.0f; break;
            case 1:  x[i] = -0.0f; break;
            case 2:  x[i] = 20.0f + nd(rng); break;          // both sides of softplus' x > 20 branch
            case 3:  x[i] = -30.0f + nd(rng); break;         // expf underflows toward 0
            case 4:  x[i] = 20.0f; break;
            case 5:  x[i] = nd(rng) * 8.0f; break;
            default: x[i] = nd(rng) * 2.0f; break;           // the router's usual range
        }
    }
    return x;
}

static std::vector<float> synth_bias(int64_t n, float sd, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> x((size_t) n);
    for (int64_t i = 0; i < n; ++i) x[i] = (i % 7 == 0) ? 0.0f : nd(rng);
    return x;
}

int main(int argc, char ** argv) {
    bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) { be = ggml_backend_cpu_init(); cuda = false; }

    const int64_t NE = 256, NU = 6;   // DeepSeek-V4.1 Flash: 256 routed experts, 6 per token
    int fails = 0;
    for (int64_t nt : { (int64_t) 1, (int64_t) 4, (int64_t) 6, (int64_t) 257, (int64_t) 600, (int64_t) 4096 })
    for (int vis = 0; vis < 2; ++vis)
    for (int norm = 0; norm < 2; ++norm) {
        const uint64_t seed = 1017 + (uint64_t) nt * 4 + (uint64_t) vis * 2 + (uint64_t) norm;
        const std::vector<float> lg = synth_logits(NE * nt, seed);
        const std::vector<float> bt = synth_bias(NE, 0.05f, seed + 1), bv = synth_bias(NE, 0.05f, seed + 2);
        std::vector<float> rbv = synth_bias(NE, 0.5f, seed + 3);
        std::vector<float> img((size_t) nt);
        for (int64_t t = 0; t < nt; ++t) img[(size_t) t] = nt == 1 ? (float) vis : (float) ((t * 7 + seed) % 3 == 0);
        const float wscale = norm ? 1.5f : 2.5f;

        ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * rl  = ggml_new_tensor_2d(c, GGML_TYPE_F32, NE, nt);
        ggml_tensor * tbt = ggml_new_tensor_1d(c, GGML_TYPE_F32, NE);
        ggml_tensor * tbv = ggml_new_tensor_1d(c, GGML_TYPE_F32, NE);
        ggml_tensor * trb = ggml_new_tensor_1d(c, GGML_TYPE_F32, NE);
        ggml_tensor * tim = ggml_new_tensor_1d(c, GGML_TYPE_F32, nt);
        for (ggml_tensor * t : { rl, tbt, tbv, trb, tim }) ggml_set_input(t);

        // the composite, exactly as ds41_dense.cpp build_attn builds it
        ggml_tensor * sc = ggml_sqrt(c, ggml_softplus(c, rl));
        ggml_tensor * bias_sel = tbt;
        if (vis) {
            ggml_tensor * b2 = ggml_reshape_2d(c, tbt, NE, 1);
            ggml_tensor * v2 = ggml_reshape_2d(c, tbv, NE, 1);
            ggml_tensor * m = ggml_reshape_2d(c, tim, 1, nt);
            bias_sel = ggml_add(c, ggml_mul(c, ggml_repeat(c, ggml_sub(c, v2, b2), sc), m), b2);
        }
        ggml_tensor * s0 = ggml_add(c, sc, bias_sel);
        ggml_tensor * sel = ggml_add(c, s0, ggml_reshape_2d(c, trb, NE, 1));
        ggml_tensor * ids = ggml_argsort_top_k(c, sel, (int) NU);
        ggml_tensor * wts = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, sc, 1, NE, nt), ids), NU, nt);
        if (norm) wts = ggml_div(c, wts, ggml_scale_bias(c, ggml_sum_rows(c, wts), 1.0f, 1e-20f));
        wts = ggml_scale(c, wts, wscale);

        // the fused ops + the views the engine takes; the weights from the composite's ids AND from the fused ids
        ggml_tensor * rs = ggml_dsv41_route_scores(c, rl, tbt, vis ? tbv : nullptr, vis ? tim : nullptr, trb);
        ggml_tensor * f_sc  = ggml_view_2d(c, rs, NE, nt, rs->nb[1], 0);
        ggml_tensor * f_sel = ggml_view_2d(c, rs, NE, nt, rs->nb[1], rs->nb[2]);
        ggml_tensor * f_ids = ggml_argsort_top_k(c, f_sel, (int) NU);
        ggml_tensor * f_wts = ggml_dsv41_route_weights(c, f_sc, f_ids, norm != 0, wscale);
        ggml_tensor * x_wts = ggml_dsv41_route_weights(c, f_sc, ids, norm != 0, wscale);

        ggml_tensor * sc_c = ggml_cont(c, sc), * sel_c = ggml_cont(c, sel);
        ggml_tensor * f_sc_c = ggml_cont(c, f_sc), * f_sel_c = ggml_cont(c, f_sel);
        ggml_tensor * ids_c = ggml_cont(c, ids), * f_ids_c = ggml_cont(c, f_ids);
        ggml_tensor * outs[] = { sc_c, f_sc_c, sel_c, f_sel_c, ids_c, f_ids_c, wts, f_wts, x_wts };
        for (ggml_tensor * t : outs) ggml_set_output(t);
        ggml_cgraph * gf = ggml_new_graph(c);
        for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al, gf);
        ggml_backend_tensor_set(rl, lg.data(), 0, lg.size() * 4);
        ggml_backend_tensor_set(tbt, bt.data(), 0, bt.size() * 4);
        ggml_backend_tensor_set(trb, rbv.data(), 0, rbv.size() * 4);
        if (vis) {   // (a text pass's graph never reads them: unallocated)
            ggml_backend_tensor_set(tbv, bv.data(), 0, bv.size() * 4);
            ggml_backend_tensor_set(tim, img.data(), 0, img.size() * 4);
        }
        ggml_backend_graph_compute(be, gf);

        auto get = [&](ggml_tensor * t) {
            std::vector<uint32_t> v((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
            return v;
        };
        const char * name[] = { "scores", "selection", "ids", "weights", "weights (composite ids)" };
        const int pair[][2] = { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 6, 8 } };
        int64_t bad = 0;
        for (int q = 0; q < 5; ++q) {
            const std::vector<uint32_t> a = get(outs[pair[q][0]]), b = get(outs[pair[q][1]]);
            int64_t nb = 0;
            for (size_t i = 0; i < a.size(); ++i)
                if (a[i] != b[i] && ++nb <= 3) {
                    float fa, fb;
                    std::memcpy(&fa, &a[i], 4);
                    std::memcpy(&fb, &b[i], 4);
                    std::printf("  %s [%zu]: composite %.9g (0x%08x) fused %.9g (0x%08x)\n", name[q], i, fa, a[i], fb, b[i]);
                }
            bad += nb;
        }
        std::printf("%s ne=%lld k=%lld nt=%lld vision=%d norm=%d: %lld differ bitwise\n", cuda ? "cuda" : "cpu",
                    (long long) NE, (long long) NU, (long long) nt, vis, norm, (long long) bad);
        fails += bad > 0;
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "fused == composite, bit for bit");
    return fails ? 1 : 0;
}
