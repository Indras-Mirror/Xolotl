#include "ds41_pipeline.hpp"

#include "ds41_geometry.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

#ifdef STRATA_ENABLE_CUDA
#include <cuda_runtime.h>
#include "ggml-cuda.h"
#endif

namespace strata::v41 {

std::vector<std::pair<int32_t, int32_t>> filter_ranked_layers(const std::vector<std::pair<int32_t, int32_t>> & ranked,
                                                            int64_t layer_lo, int64_t layer_hi) {
    std::vector<std::pair<int32_t, int32_t>> out;
    for (const auto & p : ranked)
        if (p.first >= layer_lo && p.first < layer_hi) out.push_back(p);
    return out;
}

static ggml_backend_t make_cuda_backend(int device, std::string & err) {
#ifdef STRATA_ENABLE_CUDA
    if (cudaSetDevice(device) != cudaSuccess) {
        err = "cudaSetDevice(" + std::to_string(device) + ") failed";
        return nullptr;
    }
    ggml_backend_t be = ggml_backend_cuda_init(device);
    if (!be) err = "ggml_backend_cuda_init(" + std::to_string(device) + ") failed";
    return be;
#else
    err = "CUDA not compiled";
    return nullptr;
#endif
}

#ifdef STRATA_ENABLE_CUDA
static const char * handoff_mode_label(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) return "same-device";
    int ab = 0, ba = 0;
    cudaDeviceCanAccessPeer(&ab, src_dev, dst_dev);
    cudaDeviceCanAccessPeer(&ba, dst_dev, src_dev);
    return (ab && ba) ? "peer" : "pinned-host";
}

void ds41_log_pipeline_handoffs(const Ds41Pipeline & pipe) {
    if (pipe.n_stages() < 2) return;
    for (int s = 0; s + 1 < pipe.n_stages(); ++s) {
        const int a = pipe.stages[(size_t) s].cuda_device;
        const int b = pipe.stages[(size_t) (s + 1)].cuda_device;
        std::fprintf(stderr, "ds41: handoff stage %d -> %d (GPU %d -> %d): %s\n", s, s + 1, a, b, handoff_mode_label(a, b));
    }
    if (pipe.head_offload()) {
        const int tail = pipe.stages.back().cuda_device;
        const int hd = pipe.head->cuda_device;
        std::fprintf(stderr, "ds41: handoff tail -> head (GPU %d -> %d): %s\n", tail, hd, handoff_mode_label(tail, hd));
    }
}
#else
void ds41_log_pipeline_handoffs(const Ds41Pipeline &) {}
#endif

bool ds41_selftest_handoffs(Ds41Pipeline & pipe, std::string & err) {
    if (pipe.n_stages() < 1) return true;
    constexpr int kCols = 1;
    std::vector<float> expect, got;
    std::vector<std::vector<float>> stage_pat((size_t) pipe.n_stages());
    for (int s = 0; s < pipe.n_stages(); ++s) {
        Ds41Dense & d = pipe.dense(s);
        const size_t nf = d.handoff_bytes_per_column() / sizeof(float);
        stage_pat[(size_t) s].resize(nf);
        for (size_t i = 0; i < nf; ++i) stage_pat[(size_t) s][i] = 0.125f * (float) (s + 1) + 1e-5f * (float) i;
#ifdef STRATA_ENABLE_CUDA
        cudaSetDevice(pipe.stages[(size_t) s].cuda_device);
#endif
        if (!d.import_handoff(0, kCols, stage_pat[(size_t) s].data())) {
            err = "selftest seed stage " + std::to_string(s) + ": " + d.last_error();
            return false;
        }
        if (!d.set_pass_layout(kCols, 0)) {
            err = "selftest layout stage " + std::to_string(s) + ": " + d.last_error();
            return false;
        }
    }
    for (int s = 0; s + 1 < pipe.n_stages(); ++s) {
        if (!pipe.sync_handoff_between(s, s + 1, 0, kCols, err)) {
            err = "selftest stage " + std::to_string(s) + "->" + std::to_string(s + 1) + ": " + err;
            return false;
        }
        expect = stage_pat[(size_t) s];
        got.resize(expect.size());
        Ds41Dense & dst = pipe.dense(s + 1);
#ifdef STRATA_ENABLE_CUDA
        cudaSetDevice(pipe.stages[(size_t) (s + 1)].cuda_device);
#endif
        if (!dst.export_handoff(0, kCols, got.data())) {
            err = "selftest export stage " + std::to_string(s + 1) + ": " + dst.last_error();
            return false;
        }
        for (size_t i = 0; i < expect.size(); ++i)
            if (got[i] != expect[i]) {
                err = "selftest mismatch stage " + std::to_string(s) + "->" + std::to_string(s + 1) + " at float " + std::to_string(i);
                return false;
            }
    }
    if (pipe.head_offload()) {
        const int last = pipe.n_stages() - 1;
#ifdef STRATA_ENABLE_CUDA
        cudaSetDevice(pipe.stages[(size_t) last].cuda_device);
#endif
        if (!pipe.dense(last).import_handoff(0, kCols, stage_pat[(size_t) last].data())) {
            err = "selftest re-seed tail: " + pipe.dense(last).last_error();
            return false;
        }
        if (!pipe.dense(last).set_pass_layout(kCols, 0)) {
            err = "selftest tail layout: " + pipe.dense(last).last_error();
            return false;
        }
        if (!pipe.sync_head_state(0, kCols, err)) {
            err = "selftest head handoff: " + err;
            return false;
        }
        expect = stage_pat[(size_t) last];
        got.resize(expect.size());
#ifdef STRATA_ENABLE_CUDA
        cudaSetDevice(pipe.head->cuda_device);
#endif
        if (!pipe.head->dense->export_handoff(0, kCols, got.data())) {
            err = std::string("selftest head export: ") + pipe.head->dense->last_error();
            return false;
        }
        for (size_t i = 0; i < expect.size(); ++i)
            if (got[i] != expect[i]) {
                err = "selftest mismatch tail -> head at float " + std::to_string(i);
                return false;
            }
    }
    std::fprintf(stderr, "ds41: handoff self-test OK (%d stage(s)%s)\n", pipe.n_stages(),
                 pipe.head_offload() ? ", head offload" : "");
    return true;
}

static double free_vram_gib(ggml_backend_t be, double margin_gib) {
    if (!be || ggml_backend_is_cpu(be)) return 0.0;
    size_t fr = 0, tot = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
    return std::max(0.0, ((double) fr - margin_gib * 1073741824.0) / 1073741824.0);
}

static bool parse_layer_splits(const std::string & layer_split, int64_t n_layers, std::vector<int> & splits, std::string & err) {
    splits.clear();
    if (layer_split.empty()) return false;
    if (layer_split == "auto") {
        splits.push_back((int) (n_layers / 2));
        return true;
    }
    size_t a = 0;
    while (a < layer_split.size()) {
        size_t b = layer_split.find(',', a);
        if (b == std::string::npos) b = layer_split.size();
        const std::string t = layer_split.substr(a, b - a);
        if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) {
            err = "layer-split: expected comma-separated layer indices";
            return false;
        }
        splits.push_back(std::atoi(t.c_str()));
        a = b + 1;
    }
    if (splits.empty()) { err = "layer-split: empty"; return false; }
    for (size_t i = 0; i < splits.size(); ++i) {
        if (splits[i] <= 0 || splits[i] >= (int) n_layers) {
            err = "layer-split: boundary out of range";
            return false;
        }
        if (i > 0 && splits[i] <= splits[i - 1]) {
            err = "layer-split: boundaries must be strictly increasing";
            return false;
        }
    }
    return true;
}

Ds41Pipeline::~Ds41Pipeline() {
#ifdef STRATA_ENABLE_CUDA
    if (head && head->cuda_stream) cudaStreamDestroy((cudaStream_t) head->cuda_stream);
#endif
}

size_t Ds41Pipeline::handoff_bytes(int n) const {
    if (stages.empty()) return 0;
    return stages[0].dense->handoff_bytes_per_column() * (size_t) n;
}

bool Ds41Pipeline::sync_handoff_between(int from_stage, int to_stage, int col0, int n, std::string & err) {
    if (to_stage != from_stage + 1) { err = "handoff: stages must be adjacent"; return false; }
    if (from_stage < 0 || to_stage >= n_stages()) { err = "handoff: bad stage index"; return false; }
    Ds41Dense & d0 = dense(from_stage);
    Ds41Dense & d1 = dense(to_stage);
    const int dev0 = stages[(size_t) from_stage].cuda_device;
    const int dev1 = stages[(size_t) to_stage].cuda_device;
#ifdef STRATA_ENABLE_CUDA
    cudaSetDevice(dev1);
#endif
    if (!d1.import_handoff_from(d0, dev0, dev1, col0, n, nullptr, handoff_host)) {
        err = "import_handoff_from: " + d1.last_error();
        return false;
    }
    for (int64_t gl : stages[(size_t) to_stage].import_layers) {
        if (!d1.import_layer_cache_from(d0, gl, col0, n)) {
            err = "layer cache handoff L" + std::to_string((long long) gl);
            return false;
        }
    }
    if (stages[(size_t) to_stage].import_cand && !d1.import_cand_from(d0, col0, n)) {
        err = "cand handoff failed";
        return false;
    }
    if (d0.pass_size() > 0 && !d1.set_pass_layout(d0.pass_size(), d0.pass_pos0())) {
        err = "handoff pass layout: " + d1.last_error();
        return false;
    }
    return true;
}

bool Ds41Pipeline::sync_head_state(int col0, int n, std::string & err) {
    if (!head_offload()) return true;
    Ds41Dense & tail = dense(n_stages() - 1);
    Ds41Dense & hd = *head->dense;
    const int tail_dev = stages.back().cuda_device;
    const int pass_n = tail.pass_size();
    const int pass_pos0 = tail.pass_pos0();
    if (pass_n < 1 || col0 + n > pass_n) {
        err = "head handoff: tail pass not active (pass_size=" + std::to_string(pass_n) + ")";
        return false;
    }
#ifdef STRATA_ENABLE_CUDA
    cudaSetDevice(head->cuda_device);
#endif
    if (!hd.import_handoff_from(tail, tail_dev, head->cuda_device, col0, n, head->cuda_stream, handoff_host)) {
        err = "head handoff: " + hd.last_error();
        return false;
    }
    if (!hd.set_pass_layout(pass_n, pass_pos0)) {
        err = "head pass layout: " + hd.last_error();
        return false;
    }
    return true;
}

bool Ds41Pipeline::head_logits(int row0, int nrows, float * out, std::string & err) {
    if (!head_offload()) {
        Ds41Dense & d = enabled() ? dense(n_stages() - 1) : dense(0);
        if (!d.logits(row0, nrows, out)) { err = d.last_error(); return false; }
        return true;
    }
    if (!sync_head_state(row0, nrows, err)) return false;
#ifdef STRATA_ENABLE_CUDA
    cudaSetDevice(head->cuda_device);
    if (head->cuda_stream) cudaStreamSynchronize((cudaStream_t) head->cuda_stream);
#endif
    if (!head->dense->logits(row0, nrows, out)) { err = head->dense->last_error(); return false; }
    return true;
}

bool ds41_build_pipeline(const std::string & model, const std::string & layer_split, const std::vector<int> & split_devices,
                         int head_cuda_device, double vram_margin_gib, const std::vector<double> & stage_pcie,
                         Ds41DenseConfig dc_template, Ds41MoeConfig mc_template, Ds41FileBlobs & shared_blobs, Ds41Pipeline & out,
                         std::string & err) {
    out.stages.clear();
    out.head.reset();
    out.split_layers.clear();
    const Ds41MoeGeom & TG = shared_blobs.geom();
    std::vector<int> splits;
    if (!parse_layer_splits(layer_split, TG.n_layers, splits, err)) {
        if (layer_split.empty()) {
            /* single GPU below */
        } else return false;
    }
    const bool want_split = !splits.empty() && split_devices.size() >= 2;
    if (!want_split) {
        Ds41Stage st;
        st.layer_lo = 0;
        st.layer_hi = TG.n_layers;
        st.cuda_device = split_devices.empty() ? 0 : split_devices[0];
        st.backend = dc_template.backend;
        if (!st.backend) {
            st.backend = make_cuda_backend(st.cuda_device, err);
            if (!st.backend) return false;
            st.own_backend = true;
        }
        Ds41DenseConfig dc = dc_template;
        dc.backend = st.backend;
        dc.layer_lo = 0;
        dc.layer_hi = TG.n_layers;
        dc.cuda_device = st.cuda_device;
        st.dense = std::make_unique<Ds41Dense>();
        if (!st.dense->init(model, dc, err)) return false;
        Ds41MoeConfig mc = mc_template;
        mc.cuda_device = st.cuda_device;
        if (!stage_pcie.empty()) mc.pcie_frac = stage_pcie[0];
        if (!mc.cpu_only && mc.slots <= 0) mc.slot_gib = free_vram_gib(st.backend, vram_margin_gib);
        st.tier = std::make_unique<Ds41MoeTier>();
        if (!st.tier->init(TG, &shared_blobs, mc, err)) return false;
        out.stages.push_back(std::move(st));
        out.split_layer = (int) TG.n_layers;
        return true;
    }
    if (split_devices.size() != splits.size() + 1) {
        err = "need " + std::to_string(splits.size() + 1) + " stage devices for " + std::to_string(splits.size()) + " split(s)";
        return false;
    }
    Ds41Geometry Ggeo;
    {
        GgufModel gm = GgufModel::open(model);
        const std::string gerr = ds41_read_geometry(gm, Ggeo);
        if (!gerr.empty()) { err = "geometry: " + gerr; return false; }
    }
    const std::vector<int> valid_bounds = Ggeo.valid_split_boundaries();
    std::fprintf(stderr, "ds41: indexer-safe layer-split boundaries:");
    for (int b : valid_bounds) std::fprintf(stderr, " %d", b);
    std::fprintf(stderr, "\n");
    for (size_t i = 0; i < splits.size(); ++i) {
        if (Ggeo.split_boundary_ok(splits[i])) continue;
        int best = splits[i];
        int best_d = (int) TG.n_layers;
        for (int b : valid_bounds) {
            const int d = std::abs(b - splits[i]);
            if (d < best_d) { best_d = d; best = b; }
        }
        if (best_d >= (int) TG.n_layers) {
            err = "layer-split " + std::to_string(splits[i]) + " crosses indexer groups and no safe boundary exists";
            return false;
        }
        std::fprintf(stderr, "ds41: layer-split %d is not indexer-safe; using %d instead\n", splits[i], best);
        splits[i] = best;
        for (size_t j = i + 1; j < splits.size(); ++j)
            if (splits[j] <= splits[j - 1]) {
                err = "layer-split boundaries must be strictly increasing after adjustment";
                return false;
            }
    }
    out.split_layers = splits;
    out.split_layer = splits[0];
    std::fprintf(stderr, "ds41: %zu-way layer split at", splits.size());
    for (int sp : splits) std::fprintf(stderr, " %d", sp);
    std::fprintf(stderr, " across GPUs");
    for (int d : split_devices) std::fprintf(stderr, " %d", d);
    std::fprintf(stderr, "\n");
    std::string gerr;
    auto stage_has_engram = [&](int64_t lo, int64_t hi) -> bool {
        if (!gerr.empty()) return true;
        for (int64_t el : Ggeo.engram_layers)
            if (el >= lo && el < hi) return true;
        return false;
    };
    const int n_stages = (int) split_devices.size();
    for (int s = 0; s < n_stages; ++s) {
        Ds41Stage st;
        st.cuda_device = split_devices[(size_t) s];
        st.layer_lo = s == 0 ? 0 : splits[(size_t) (s - 1)];
        st.layer_hi = s < (int) splits.size() ? splits[(size_t) s] : TG.n_layers;
        if (s > 0) {
            st.import_layers = Ggeo.stage_import_layers(st.layer_lo, st.layer_hi);
            if (Ggeo.cand_layer >= 0 && Ggeo.cand_layer < st.layer_lo) {
                for (int64_t l = st.layer_lo; l < st.layer_hi; ++l)
                    if (l > Ggeo.cand_layer && Ggeo.is_index_source(l)) { st.import_cand = true; break; }
            }
        }
        if (s > 0 && out.stages[(size_t) (s - 1)].cuda_device == st.cuda_device && out.stages[(size_t) (s - 1)].backend) {
            st.backend = out.stages[(size_t) (s - 1)].backend;
            st.own_backend = false;
        } else {
            st.backend = make_cuda_backend(st.cuda_device, err);
            if (!st.backend) return false;
            st.own_backend = true;
        }
        Ds41DenseConfig dc = dc_template;
        dc.backend = st.backend;
        dc.layer_lo = st.layer_lo;
        dc.layer_hi = st.layer_hi;
        dc.cuda_device = st.cuda_device;
        dc.shadow_layers = st.import_layers;
        dc.omit_output_head = head_cuda_device >= 0 && s == n_stages - 1;
        if (!dc.engram_dirs.empty() && !stage_has_engram(st.layer_lo, st.layer_hi)) dc.engram_dirs.clear();
        st.dense = std::make_unique<Ds41Dense>();
        if (!st.dense->init(model, dc, err)) return false;
        Ds41MoeConfig mc = mc_template;
        mc.cuda_device = st.cuda_device;
        if (stage_pcie.size() == (size_t) n_stages) mc.pcie_frac = stage_pcie[(size_t) s];
        if (!mc.cpu_only && mc.slots <= 0) {
            const double share = (double) (st.layer_hi - st.layer_lo) / (double) TG.n_layers;
            mc.slot_gib = free_vram_gib(st.backend, vram_margin_gib) * std::max(0.35, share);
        }
        st.tier = std::make_unique<Ds41MoeTier>();
        if (!st.tier->init(TG, &shared_blobs, mc, err)) return false;
        out.stages.push_back(std::move(st));
    }
    out.handoff_host.resize(out.handoff_bytes(dc_template.prompt_batch > 0 ? dc_template.prompt_batch : 512) / sizeof(float));
    if (head_cuda_device >= 0) {
        out.head = std::make_unique<Ds41HeadStage>();
        Ds41HeadStage & hs = *out.head;
        hs.cuda_device = head_cuda_device;
        hs.backend = nullptr;
        for (const auto & st : out.stages)
            if (st.cuda_device == head_cuda_device) { hs.backend = st.backend; break; }
        if (!hs.backend) {
            hs.backend = make_cuda_backend(hs.cuda_device, err);
            if (!hs.backend) return false;
            hs.own_backend = true;
        }
#ifdef STRATA_ENABLE_CUDA
        cudaStream_t stream = nullptr;
        if (cudaSetDevice(hs.cuda_device) != cudaSuccess || cudaStreamCreate(&stream) != cudaSuccess) {
            err = "head device cudaStreamCreate failed";
            return false;
        }
        hs.cuda_stream = stream;
#endif
        Ds41DenseConfig hdc = dc_template;
        hdc.backend = hs.backend;
        hdc.head_only = true;
        hdc.omit_output_head = false;
        hdc.cuda_device = hs.cuda_device;
        hdc.engram_dirs.clear();
        hs.dense = std::make_unique<Ds41Dense>();
        if (!hs.dense->init(model, hdc, err)) return false;
        std::fprintf(stderr, "ds41: output head + DSpark on CUDA device %d (trunk stages above)\n", hs.cuda_device);
    }
    ds41_log_pipeline_handoffs(out);
    return true;
}

}  // namespace strata::v41
