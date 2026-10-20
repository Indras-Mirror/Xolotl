// tools/v41/engine/bench_attn_f16_peak.cpp - CPU-only: the attention key subgraph (F32 path vs STRATA_ATTN_F16_GATHER) built with
// the real shapes (n 512, DH 512, SWA 128, K 512, pad 128, 64 heads) and reserved with ggml-alloc: prints each graph buffer.
// 2026-10-18: F32 2135.8 MiB vs F16 1113.8 MiB.  Build: g++ -I<ggml>/include this.cpp libggml{,-cpu,-base}.a -fopenmp
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
static ggml_tensor * rows_as(ggml_context * gc, ggml_tensor * a, ggml_tensor * idx, bool f16) {
    ggml_tensor * t = ggml_get_rows(gc, a, idx);
    if (f16 && t->type == GGML_TYPE_F32) { t->type = GGML_TYPE_F16; t->nb[0] = 2; for (int i = 1; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i-1] * t->ne[i-1]; }
    return t;
}
int main() {
    const int64_t DH = 512, SWA = 128, K = 512, n = 512, CAP = 20000, RAW = 640;
    for (int f16 = 0; f16 < 2; ++f16) {
        ggml_init_params p = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
        ggml_context * gc = ggml_init(p);
        ggml_tensor * wsrc = ggml_new_tensor_2d(gc, GGML_TYPE_F32, DH, RAW + n); ggml_set_input(wsrc);
        ggml_tensor * comp = ggml_new_tensor_2d(gc, GGML_TYPE_F16, DH, CAP);      ggml_set_input(comp);
        ggml_tensor * widx = ggml_new_tensor_1d(gc, GGML_TYPE_I32, SWA * n);      ggml_set_input(widx);
        ggml_tensor * sel  = ggml_new_tensor_1d(gc, GGML_TYPE_I32, K * n);        ggml_set_input(sel);
        ggml_tensor * zpad = ggml_new_tensor_2d(gc, GGML_TYPE_F16, DH, 1);        ggml_set_input(zpad);
        ggml_tensor * zidx = ggml_new_tensor_1d(gc, GGML_TYPE_I32, 255 * 2048);   ggml_set_input(zidx);
        ggml_tensor * Kw = ggml_reshape_3d(gc, rows_as(gc, wsrc, widx, f16), DH, SWA, n);
        ggml_tensor * Kc = ggml_reshape_3d(gc, rows_as(gc, comp, sel, f16), DH, K, n);
        ggml_tensor * Kall = ggml_concat(gc, Kw, Kc, 1);
        const int64_t pad = 256 - (SWA + K) % 256;
        Kall = f16 ? ggml_concat(gc, Kall, ggml_reshape_3d(gc, rows_as(gc, zpad, ggml_view_1d(gc, zidx, pad * n, 0), true), DH, pad, n), 1)
                   : ggml_pad(gc, Kall, 0, (int) pad, 0, 0);
        ggml_tensor * k4 = Kall->type == GGML_TYPE_F16 ? ggml_reshape_4d(gc, Kall, DH, SWA + K + pad, 1, n)
                                                      : ggml_cast(gc, ggml_reshape_4d(gc, Kall, DH, SWA + K + pad, 1, n), GGML_TYPE_F16);
        const int64_t M = SWA + K + pad, NH = 64;
        ggml_tensor * q = ggml_new_tensor_4d(gc, GGML_TYPE_F32, DH, 1, NH, n); ggml_set_input(q);
        ggml_tensor * m = ggml_new_tensor_4d(gc, GGML_TYPE_F16, M, 1, 1, n);   ggml_set_input(m);
        ggml_tensor * out = ggml_flash_attn_ext(gc, q, k4, k4, m, 0.04f, 0.0f, 0.0f);
        ggml_set_output(out);
        ggml_cgraph * g = ggml_new_graph(gc);
        ggml_build_forward_expand(g, out);
        ggml_gallocr_t a = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!ggml_gallocr_reserve(a, g)) { std::printf("reserve failed\n"); return 1; }
        std::printf("%s: k4 %s [%lld,%lld,%lld,%lld]  graph buffer %.1f MiB\n",
                    f16 ? "F16 gather" : "F32 path  ", ggml_type_name(k4->type), (long long) k4->ne[0], (long long) k4->ne[1],
                    (long long) k4->ne[2], (long long) k4->ne[3], ggml_gallocr_get_buffer_size(a, 0) / 1048576.0);
        ggml_gallocr_free(a); ggml_free(gc);
    }
}
