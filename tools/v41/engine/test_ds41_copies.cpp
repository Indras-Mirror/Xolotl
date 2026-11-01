// test_ds41_copies [--cuda] (built with -DSTRATA_DSV41_COPIES_OP=ON on a patched llama.cpp) - GGML_OP_DSV41_COPIES vs
// one ggml_cpy per pair, byte for byte, on the shapes build_attn writes: a [D, HC, n] state, a [HC, n] mix, a reshaped
// [HC, HC, n] comb, [D, n] rows, the I32 top-k ids (ggml_argsort_top_k's strided view) and F32 weights - into column
// views of larger buffers at a column offset, 1-600 columns, one op of 5 pairs and one of 3.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) { be = ggml_backend_cpu_init(); cuda = false; }

    const int64_t D = 512, HC = 4, NE = 256, NU = 6, CAP = 1024;
    int fails = 0;
    for (int64_t n : { (int64_t) 1, (int64_t) 5, (int64_t) 257, (int64_t) 600 }) {
        const int64_t col0 = n == 600 ? 7 : 3;
        ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        // sources (computed in the graph so they live in the graph's buffer, as in the engine)
        ggml_tensor * x3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, n);
        ggml_tensor * x2 = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, n);
        ggml_tensor * xc = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC * HC, n);
        ggml_tensor * xd = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n);
        ggml_tensor * xs = ggml_new_tensor_2d(c, GGML_TYPE_F32, NE, n);
        for (ggml_tensor * t : { x3, x2, xc, xd, xs }) ggml_set_input(t);
        ggml_tensor * s3 = ggml_scale(c, x3, 1.0f), * s2 = ggml_scale(c, x2, 1.0f), * sd = ggml_scale(c, xd, 1.0f);
        ggml_tensor * scb = ggml_reshape_3d(c, ggml_scale(c, xc, 1.0f), HC, HC, n);
        ggml_tensor * ids = ggml_argsort_top_k(c, xs, (int) NU);   // I32 [NU, n], rows strided by NE
        ggml_tensor * wts = ggml_cont(c, ggml_view_2d(c, xs, NU, n, xs->nb[1], 0));
        ggml_tensor * srcs[8] = { s3, s2, scb, sd, ids, wts, s2, sd };
        // two destination sets: the copies op's and ggml_cpy's, column views at col0 of CAP-column buffers
        ggml_tensor * dst[2][8];
        for (int w = 0; w < 2; ++w) {
            ggml_tensor * b3 = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, CAP);
            ggml_tensor * b2 = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, CAP);
            ggml_tensor * bc = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, CAP);
            ggml_tensor * bd = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, CAP);
            ggml_tensor * bi = ggml_new_tensor_2d(c, GGML_TYPE_I32, NU, CAP);
            ggml_tensor * bw = ggml_new_tensor_2d(c, GGML_TYPE_F32, NU, CAP);
            ggml_tensor * b2b = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, CAP);
            ggml_tensor * bdb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, CAP);
            ggml_tensor * bufs[8] = { b3, b2, bc, bd, bi, bw, b2b, bdb };
            for (int k = 0; k < 8; ++k) {
                ggml_tensor * B = bufs[k];
                ggml_set_input(B);
                ggml_set_output(B);   // (kept: the allocator must not reuse a buffer after its last reader)
                dst[w][k] = k == 0 || k == 2
                    ? ggml_view_3d(c, B, B->ne[0], B->ne[1], n, B->nb[1], B->nb[2], (size_t) col0 * B->nb[2])
                    : ggml_view_2d(c, B, B->ne[0], n, B->nb[1], (size_t) col0 * B->nb[1]);
            }
        }
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, ggml_dsv41_copies(c, srcs, dst[0], 5));
        ggml_build_forward_expand(gf, ggml_dsv41_copies(c, srcs + 5, dst[0] + 5, 3));
        for (int k = 0; k < 8; ++k) ggml_build_forward_expand(gf, ggml_cpy(c, srcs[k], dst[1][k]));
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al, gf);
        std::mt19937 rng((unsigned) n);
        std::uniform_real_distribution<float> ud(-4.0f, 4.0f);
        for (ggml_tensor * t : { x3, x2, xc, xd, xs }) {
            std::vector<float> v((size_t) ggml_nelements(t));
            for (float & f : v) f = ud(rng);
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4);
        }
        for (int w = 0; w < 2; ++w)   // both buffer sets start as the same garbage, so a missed element shows
            for (int k = 0; k < 8; ++k) {
                ggml_tensor * B = dst[w][k]->view_src;
                std::vector<uint32_t> v((size_t) ggml_nelements(B));
                for (size_t i = 0; i < v.size(); ++i) v[i] = 0xdead0000u + (uint32_t) (i * 2654435761u >> 16);
                ggml_backend_tensor_set(B, v.data(), 0, v.size() * 4);
            }
        ggml_backend_graph_compute(be, gf);
        int64_t bad = 0;
        for (int k = 0; k < 8; ++k) {
            ggml_tensor * A = dst[0][k]->view_src, * B = dst[1][k]->view_src;
            std::vector<uint32_t> a((size_t) ggml_nelements(A)), b((size_t) ggml_nelements(B));
            ggml_backend_tensor_get(A, a.data(), 0, a.size() * 4);
            ggml_backend_tensor_get(B, b.data(), 0, b.size() * 4);
            for (size_t i = 0; i < a.size(); ++i)
                if (a[i] != b[i] && ++bad <= 4) std::printf("  pair %d [%zu]: copies 0x%08x cpy 0x%08x\n", k, i, a[i], b[i]);
        }
        std::printf("%s n=%lld col0=%lld: 8 pairs, %lld words differ\n", cuda ? "cuda" : "cpu", (long long) n,
                    (long long) col0, (long long) bad);
        fails += bad > 0;
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "copies == ggml_cpy, byte for byte");
    return fails ? 1 : 0;
}
