// tools/hy4/engine/hy4_moe_ref.cpp - see hy4_moe_ref.hpp.
#include "hy4_moe_ref.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

namespace strata::hy4 {

struct Hy4RefMoe::Impl {
    const Hy4Geometry * G = nullptr;
    ggml_backend_t be = nullptr;
    ggml_context * wctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    struct Lw { ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr; };
    std::vector<Lw> L;
    ggml_gallocr_t galloc = nullptr;
    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        for (auto b : bufs) ggml_backend_buffer_free(b);
        if (wctx) ggml_free(wctx);
        if (be) ggml_backend_free(be);
    }
};

Hy4RefMoe::Hy4RefMoe() : p_(new Impl) {}
Hy4RefMoe::~Hy4RefMoe() = default;

bool Hy4RefMoe::init(const GgufModel & M, const Hy4Geometry & G, int n_threads, std::string & err) {
    Impl & im = *p_;
    im.G = &G;
    im.be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(im.be, n_threads);
    ggml_init_params ip = { ggml_tensor_overhead() * 512, nullptr, true };
    im.wctx = ggml_init(ip);
    im.L.resize((size_t) G.n_layer);
    std::vector<ggml_backend_buffer_t> shard_buf(M.size(), nullptr);
    auto bind = [&](const std::string & name) -> ggml_tensor * {
        size_t s = 0;
        const TensorInfo * ti = M.find(name, &s);
        if (!ti) { err = "missing " + name; return nullptr; }
        const GgufFile & sh = M.shard(s);
        if (!shard_buf[s]) {
            const auto & ts = sh.tensors();
            uint8_t * base = const_cast<uint8_t *>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
            shard_buf[s] = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            im.bufs.push_back(shard_buf[s]);
        }
        int64_t ne[4] = { 1, 1, 1, 1 };
        for (size_t d = 0; d < ti->shape.size(); ++d) ne[d] = (int64_t) ti->shape[d];
        ggml_tensor * t = ggml_new_tensor(im.wctx, (ggml_type) ti->type, (int) ti->shape.size(), ne);
        if (ggml_backend_tensor_alloc(shard_buf[s], t, const_cast<uint8_t *>(sh.tensor_data(*ti))) != GGML_STATUS_SUCCESS) {
            err = "cannot bind " + name;
            return nullptr;
        }
        return t;
    };
    for (int64_t l = 0; l < G.n_layer; ++l) {
        if (!G.moe(l)) continue;
        const std::string p = "blk." + std::to_string(l) + ".";
        auto & y = im.L[(size_t) l];
        y.gate = bind(p + "ffn_gate_exps.weight");
        y.up = bind(p + "ffn_up_exps.weight");
        y.down = bind(p + "ffn_down_exps.weight");
        if (!err.empty()) return false;
    }
    im.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.be));
    return true;
}

bool Hy4RefMoe::run(int il, const float * x, const int * ids, const float * wts, int n, float * out) {
    Impl & im = *p_;
    const Hy4Geometry & G = *im.G;
    const auto & y = im.L[(size_t) il];
    const int64_t K = G.n_latent, U = G.n_used;

    // fault the used experts in with one readahead each (the mmap is on NVMe)
    {
        std::set<int> used(ids, ids + n * U);
        const long pg = sysconf(_SC_PAGESIZE);
        for (ggml_tensor * t : { y.gate, y.up, y.down })
            for (int e : used) {
                uintptr_t a = (uintptr_t) t->data + (uintptr_t) e * t->nb[2];
                uintptr_t b = a + t->nb[2];
                a &= ~(uintptr_t) (pg - 1);
                madvise((void *) a, b - a, MADV_WILLNEED);
            }
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * tx = ggml_new_tensor_3d(c, GGML_TYPE_F32, K, 1, n);
    ggml_tensor * tid = ggml_new_tensor_2d(c, GGML_TYPE_I32, U, n);
    ggml_tensor * tw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, U, n);
    for (ggml_tensor * t : { tx, tid, tw }) ggml_set_input(t);
    ggml_tensor * up = ggml_mul_mat_id(c, y.up, tx, tid);       // [n_ff, U, n]
    ggml_tensor * gt = ggml_mul_mat_id(c, y.gate, tx, tid);
    const double lim = (size_t) il < G.swiglu_clamp.size() ? G.swiglu_clamp[(size_t) il] : 0.0;
    ggml_tensor * act = lim > 1e-6 ? ggml_swiglu_clamp(c, gt, up, (float) lim) : ggml_swiglu_split(c, gt, up);
    ggml_tensor * dn = ggml_mul_mat_id(c, y.down, act, tid);    // [K, U, n]
    dn = ggml_mul(c, dn, tw);
    ggml_tensor * sum = nullptr;
    for (int64_t u = 0; u < U; ++u) {
        ggml_tensor * v = ggml_view_2d(c, dn, K, n, dn->nb[2], (size_t) u * dn->nb[1]);
        sum = sum ? ggml_add(c, sum, v) : v;
    }
    sum = ggml_cont(c, sum);
    ggml_set_output(sum);
    ggml_build_forward_expand(gf, sum);
    bool ok = ggml_gallocr_alloc_graph(im.galloc, gf);
    if (ok) {
        ggml_backend_tensor_set(tx, x, 0, (size_t) (K * n) * 4);
        ggml_backend_tensor_set(tid, ids, 0, (size_t) (U * n) * 4);
        ggml_backend_tensor_set(tw, wts, 0, (size_t) (U * n) * 4);
        ok = ggml_backend_graph_compute(im.be, gf) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(sum, out, 0, (size_t) (K * n) * 4);
    }
    ggml_free(c);
    if (!ok) std::fprintf(stderr, "hy4_moe_ref: layer %d failed\n", il);
    return ok;
}

}  // namespace strata::hy4
