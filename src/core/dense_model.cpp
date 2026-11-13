// src/core/dense_model.cpp - see include/strata/core/dense_model.hpp.
#include "strata/core/dense_model.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dense_arch.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#ifdef STRATA_DENSE_MMQ
#include "strata/prefill/moe_mmq.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <exception>
#include <map>
#include <regex>
#include <set>
#include <utility>

namespace strata::core {
namespace {

using namespace strata::kernels;

constexpr uint32_t kF32 = 0, kF16 = 1, kBF16 = 30;
constexpr int NC = DenseModel::kMaxCols;

// The GGUF types a quantized weight may have: exactly what native_mmvq takes.
const char* kSupported = "Q4_0, Q5_0, Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS";

struct QMat {
    void* dev = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    QMat ffn_gate, ffn_up, ffn_down;
    // Hyper-connections (the gated residual): the two mixers' weights, in the engine form gr_read takes.  The
    // matrices are BF16 (uint16_t) and the norms F32, exactly as the MoE path pins them in layout.cpp.
    uint16_t *hc_attn_down = nullptr, *hc_attn_up = nullptr, *hc_attn_inject = nullptr;
    uint16_t *hc_ffn_down = nullptr, *hc_ffn_up = nullptr, *hc_ffn_inject = nullptr;
    float *hc_attn_norm = nullptr, *hc_ffn_norm = nullptr;
    // GDN mixer
    QMat qkv, z, ssm_out;
    float *conv_w = nullptr, *dt = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
    float *alpha_w = nullptr, *beta_w = nullptr;    // ssm_alpha / ssm_beta when the file stores them as floats ...
    QMat alpha_q, beta_q;                           // ... or here when it stores them quantized (UD-Q4_K_M: Q8_0)
    bool ab_quant = false;
    float *state = nullptr, *conv_state = nullptr;
    std::vector<float*> st_slots, cv_slots;         // snapshots after columns 0..draft_max-1; rollback() swaps one in (MTP only)
    // attention mixer
    QMat q, k, v, o;
    float *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
};

/// The multi-token-prediction block: a full-attention layer plus the projection of [embedding | hidden].
struct Mtp {
    QMat eh_proj;                                   // [2 * n_embd -> n_embd]
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    QMat q, k, v, o, ffn_gate, ffn_up, ffn_down;
    float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;          // its own KV cache
    QMat head;                                      // the output head: its own, or the trunk's
};

std::string upper_type(uint32_t t) { return strata::ggml_type_name(t); }

// A tensor of one GGUF file, found by name across the shards.
struct Found {
    const strata::GgufFile* file = nullptr;
    const strata::TensorInfo* tensor = nullptr;
};

// Host-side dequantization of `n` values laid out as whole blocks (a row, or a whole small tensor).
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out) {
    switch (type) {
    case kF32: strata::dequantize_f32(src, out, n); return true;
    case kF16: strata::dequantize_f16(src, out, n); return true;
    case kBF16: strata::dequantize_bf16(src, out, n); return true;
    default: break;
    }
    int elems = 0, bytes = 0;
    if (!strata::block_geometry(type, elems, bytes) || n % elems) return false;
    for (int b = 0; b < n / elems; ++b) {
        const uint8_t* blk = src + (size_t) b * bytes;
        float* o = out + (size_t) b * elems;
        switch (type) {
        case 2: strata::dequantize_q4_0(blk, o); break;
        case 6: strata::dequantize_q5_0(blk, o); break;
        case 8: strata::dequantize_q8_0(blk, o); break;
        case 11: strata::dequantize_q3_K(blk, o); break;
        case 12: strata::dequantize_q4_K(blk, o); break;
        case 13: strata::dequantize_q5_K(blk, o); break;
        case 14: strata::dequantize_q6_K(blk, o); break;
        case 20: strata::dequantize_iq4_nl(blk, o); break;
        case 23: strata::dequantize_iq4_xs(blk, o); break;
        case 42: strata::dequantize_q2_0(blk, o); break;
        default: return false;
        }
    }
    return true;
}

// llama.cpp names a split file <base>-00001-of-0000N.gguf
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re(R"((.*-)(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(m[1].str() + buf + "-of-" + m[3].str() + ".gguf");
    }
    return out;
}

// integer metadata that may be a scalar or (per layer) an array: the first element
bool meta_int(const strata::GgufFile& g, const std::string& key, int64_t& out) {
    const strata::MetaValue* v = g.get(key);
    if (!v) return false;
    if (v->type == strata::MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].u;
        return true;
    }
    out = (int64_t) v->u;
    return true;
}

// Debug aid (STRATA_DENSE_DEBUG): statistics of a device vector, printed to stderr.
void trace_vec(const char* what, int64_t pos, int layer, const char* kind, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<float> h((size_t) n);
    if (cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "dbg pos=%lld %s: copy failed\n", (long long) pos, what);
        return;
    }
    double mx = 0.0, sum2 = 0.0;
    int nan = 0, inf = 0;
    for (float v : h) {
        if (std::isnan(v)) { ++nan; continue; }
        if (std::isinf(v)) { ++inf; continue; }
        mx = std::max(mx, (double) std::fabs(v));
        sum2 += (double) v * v;
    }
    std::fprintf(stderr, "dbg pos=%lld layer=%2d %-4s %-7s max|v|=%-12.5g rms=%-12.5g nan=%d inf=%d\n", (long long) pos, layer,
                 kind, what, mx, std::sqrt(sum2 / std::max(1, n)), nan, inf);
}

void trace_logits(int64_t pos, const float* dev, int n, void* stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<float> h((size_t) n);
    if (cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    std::vector<int> idx;
    int nan = 0;
    for (int i = 0; i < n; ++i) { if (std::isnan(h[(size_t) i])) ++nan; else idx.push_back(i); }
    const size_t k = std::min<size_t>(5, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + (std::ptrdiff_t) k, idx.end(),
                      [&](int a, int b) { return h[(size_t) a] > h[(size_t) b]; });
    std::fprintf(stderr, "dbg pos=%lld logits nan=%d top:", (long long) pos, nan);
    for (size_t i = 0; i < k; ++i) std::fprintf(stderr, " %d(%.3f)", idx[i], h[(size_t) idx[i]]);
    std::fprintf(stderr, "\n");
}

}  // namespace

// STRATA_DENSE_PROF=1: where the time of every run() pass goes (GPU events around the pieces, host enqueue time),
// printed to stderr every 8 passes.  The pass is synchronized at its end, so this slows decoding slightly: it is a
// measuring mode, not a way to run.
struct Prof {
    enum { GEMV_Q4K, GEMV_Q6K, GEMV_Q80, GEMV_OTHER, QUANT, GDN_LOOP, ATTN_LOOP, NB };
    bool enabled = false, on = false;
    std::vector<cudaEvent_t> ev;
    size_t used = 0;
    struct Seg { int bucket; size_t a, b; };
    std::vector<Seg> segs;
    size_t t0 = 0;
    std::chrono::steady_clock::time_point host0;
    double ms[NB] = {}, total_ms = 0, host_ms = 0;
    long long passes = 0, cols = 0;
    size_t get() {
        if (used == ev.size()) { cudaEvent_t e = nullptr; cudaEventCreate(&e); ev.push_back(e); }
        return used++;
    }
    void begin(void* s) {
        on = true; used = 0; segs.clear();
        t0 = get();
        cudaEventRecord(ev[t0], (cudaStream_t) s);
        host0 = std::chrono::steady_clock::now();
    }
    void finish(void* s, int n) {
        host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - host0).count();
        const size_t t1 = get();
        cudaEventRecord(ev[t1], (cudaStream_t) s);
        cudaEventSynchronize(ev[t1]);
        on = false;
        float tot = 0;
        cudaEventElapsedTime(&tot, ev[t0], ev[t1]);
        total_ms += tot;
        for (const Seg& g : segs) { float m = 0; cudaEventElapsedTime(&m, ev[g.a], ev[g.b]); ms[g.bucket] += m; }
        ++passes; cols += n;
        if (passes % 8 == 0) {
            double known = 0;
            for (int i = 0; i < NB; ++i) known += ms[i];
            const double P = (double) passes;
            std::fprintf(stderr,
                         "xolotl-dense prof: %.1f cols/pass, GPU %.1f ms/pass = gemv Q4_K %.1f + Q6_K %.1f + Q8_0 %.1f + other %.1f"
                         " | quantize %.1f | gdn token loop %.1f | attn token loop %.1f | rest %.1f ; host enqueue %.1f ms/pass\n",
                         (double) cols / P, total_ms / P, ms[GEMV_Q4K] / P, ms[GEMV_Q6K] / P, ms[GEMV_Q80] / P, ms[GEMV_OTHER] / P,
                         ms[QUANT] / P, ms[GDN_LOOP] / P, ms[ATTN_LOOP] / P, (total_ms - known) / P, host_ms / P);
            for (double& m : ms) m = 0;
            total_ms = host_ms = 0; passes = cols = 0;
        }
    }
};

struct PScope {
    Prof* p; int bucket; size_t a = 0; void* s;
    PScope(Prof* pp, int b, void* st) : p(pp), bucket(b), s(st) {
        if (p->on) { a = p->get(); cudaEventRecord(p->ev[a], (cudaStream_t) s); }
    }
    ~PScope() {
        if (!p->on) return;
        const size_t e = p->get();
        cudaEventRecord(p->ev[e], (cudaStream_t) s);
        p->segs.push_back({bucket, a, e});
    }
};

struct DenseModel::Impl {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Found> index;
    std::vector<Layer> layers;
    Mtp mtp;
    // xolotl: an external drafter's inputs (DenseOptions::taps): tap_slot[l] = this layer's place in a tap row, -1 none
    std::vector<int> tap_slot;
    int n_taps = 0;
    float* taps_run = nullptr;                      // [NC][n_taps * n_embd]: the last run()'s columns
    float* taps_ring = nullptr;                     // [kTapRing][n_taps * n_embd]: prefill rows at position % kTapRing
    std::vector<void*> allocations;                 // everything cudaMalloc'ed here, freed in the destructor
    std::vector<void*> host_allocs;                 // weights that did not fit in VRAM: mapped pinned host memory
    uint64_t budget = UINT64_MAX;                   // bytes of weights that may go to VRAM
    uint64_t dev_used = 0, host_bytes = 0;
    int host_tensors = 0;
    struct Ckpt { uint8_t* host = nullptr; int64_t pos = -1; };
    std::vector<Ckpt> ckpt;
    uint64_t ckpt_bytes = 0;
    long long dbg_from = -1, dbg_to = -1;           // STRATA_DENSE_DEBUG=A:B traces the steps with A <= position < B
    Prof prof;                                      // STRATA_DENSE_PROF=1
    int kfmt = DENSE_KV_F16, vfmt = DENSE_KV_F16;   // how the KV cache stores K and V (--kv)
    cudaStream_t stream = nullptr;
    cudaEvent_t embed_done = nullptr;

    // a token embedding table: rows are dequantized on the host and staged through pinned memory
    struct EmbdSrc {
        Found f;
        uint32_t type = 0;
        size_t row_bytes = 0;
    };
    EmbdSrc embd, embd_mtp;
    float* embd_stage = nullptr;                    // pinned, NC * n_embd floats
    const float* const* img_rows = nullptr;         // set_embd_rows(): per-cell image rows (nullptr = the table)
    int64_t img_n = 0;

    float* output_norm = nullptr;
    QMat head;
    // ---- hyper-connections (hc > 0): the residual STACK and its mixers.  Absent (null) on a plain dense file.
    uint16_t *out_hc_down = nullptr, *out_hc_up = nullptr;
    float* out_hc_norm = nullptr;
    float* R = nullptr;                 ///< [NC][hc * n_embd]  the residual stack, hc streams per column
    float* hc_xn = nullptr;             ///< [NC][hc * n_embd]  gr_read's activated streams (its workspace, mine here)
    float* hc_gated = nullptr;          ///< [NC][hc * n_embd]  the gated stream product, before the mean
    float* hc_lo = nullptr;             ///< [NC][hc_lr]        the bottleneck
    float* hc_inject = nullptr;         ///< [NC][hc]           one injection per stream, per column
    strata::kernels::GrShapes gr_shape;
    strata::kernels::GrWorkspace gr_ws;  ///< the single-token gr_read path's scratch (carved from gr_ws_mem)
    void* gr_ws_mem = nullptr;
    void* q8_1 = nullptr;                           // shared activation scratch of the native GEMVs
    int64_t max_context = 0;

    // ---- the batched prompt path (prefill_setup() / prefill()); see the block definitions below.
#ifdef STRATA_DENSE_MMQ
    struct Pf {
        int64_t T = 0;
        strata::prefill::mmq::Context ctx;      // MMQ keeps a small scratch pool for its stream-k fixup
        float *x = nullptr, *xn = nullptr, *mix = nullptr;
        float *qkv = nullptr, *h = nullptr, *alpha = nullptr, *beta = nullptr, *gate = nullptr;
        float *z = nullptr, *y = nullptr;
        float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr;
        float *attn = nullptr, *attn32 = nullptr;
        float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
        float* stage = nullptr;                 // pinned embedding stage, T rows
        int32_t* pos_dev = nullptr;             // the position of every row, for RoPE
        void* xq = nullptr;                     // the q8_1 activations MMQ multiplies with
        int32_t* ids = nullptr;                 // iota: MMQ writes each row back to its own index
        int32_t* bounds = nullptr;              // {0, T}: one group, the whole chunk
        // Pinned twins of bounds, cycled so a chunk never overwrites a copy the stream has not read yet.
        static constexpr int kBoundsSlots = 8;
        int32_t* bounds_host = nullptr;
        int bounds_slot = 0;
        std::vector<void*> allocs;              // this arena's cudaMalloc'ed buffers (freed by ~Pf)

        ~Pf() {
            for (void* p : allocs) cudaFree(p);
            if (stage) cudaFreeHost(stage);
            if (bounds_host) cudaFreeHost(bounds_host);
        }
    };
    std::unique_ptr<Pf> pf;
#endif

    // activations (device, f32), NC columns each
    float *x = nullptr, *xn = nullptr, *mix = nullptr;
    float *qkv = nullptr, *h = nullptr, *alpha = nullptr, *beta = nullptr, *gate = nullptr;
    float *z = nullptr, *y = nullptr;
    float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr, *attn32 = nullptr;
    float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
    float* logits = nullptr;
    float* attn_scratch = nullptr;
    int32_t* pos_dev = nullptr;                     // one position per row, for NC columns' worth of rows
    // The rope configuration the analytic kernel is launched with (rope_scaling.hpp).  Seeded from the
    // process config and re-based on the file's rope.freq_base at load(): the dense path has no CLI rope
    // knobs, so this stays `none` (the identity) unless something else has set a scaling.
    RopeScaling rope;
    // the hidden state and the MTP side
    float *hid = nullptr;                           // post-norm hidden of the last run(), NC columns
    float *h_last = nullptr;                        // post-norm hidden at the last fed position
    float *m_e = nullptr, *hin = nullptr, *cat = nullptr, *mtp_logits = nullptr;
    float* mtp_h = nullptr;                         // the MTP block's own output hidden (chains the next draft step)
    int* d_tok = nullptr;
    float* d_prob = nullptr;
    QMat draft_head;                                // the MTP drafts' shortlist head (DenseOptions::draft_vocab), or empty
    std::vector<int32_t> draft_ids;                 // draft_head row -> token id
    int draft_base = 0, draft_cap = 0;              // shortlist rows, and the free rows after them for the prompt's tokens
    std::vector<uint8_t> draft_in;                  // per token id: 1 = in the shortlist
    const uint8_t* draft_src = nullptr;             // the head's rows in the file
    uint64_t draft_row = 0;                         // bytes per head row
    std::vector<int32_t> draft_extra;               // the prompt tokens currently appended
    std::vector<uint8_t> draft_stage;

    // ---- helpers
    template <class T> bool alloc(T** out, uint64_t bytes, std::string& err, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " +
                  cudaGetErrorString(s);
            return false;
        }
        allocations.push_back(p);
        *out = (T*) p;
        return true;
    }

    const Found* find(const std::string& name) const {
        auto it = index.find(name);
        return it == index.end() ? nullptr : &it->second;
    }
    const Found* find_any(std::initializer_list<const char*> names, const std::string& prefix) const {
        for (const char* n : names)
            if (const Found* f = find(prefix + n)) return f;
        return nullptr;
    }

    // A float vector or matrix (F32, F16, BF16 or any type the host can dequantize) uploaded as F32.
    bool upload_floats(const Found* f, const std::string& name, uint64_t expect, float** out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        uint64_t n = 1;
        for (uint64_t d : f->tensor->shape) n *= d;
        if (expect && n != expect) {
            err = "dense model: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(expect);
            return false;
        }
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        std::vector<float> host(n);
        if (n > (uint64_t) INT_MAX || !dequant_row(f->tensor->type, src, (int) n, host.data())) {
            err = "dense model: " + name + " is " + upper_type(f->tensor->type) +
                  ", which cannot be read as floats (expected F32, F16, BF16 or a quantized type)";
            return false;
        }
        if (!alloc(out, n * 4, err, name.c_str())) return false;
        const cudaError_t s = cudaMemcpy(*out, host.data(), n * 4, cudaMemcpyHostToDevice);
        if (s != cudaSuccess) { err = "dense model: upload " + name + ": " + cudaGetErrorString(s); return false; }
        return true;
    }

    // A matrix uploaded in the engine's BF16 form, which is what `gr_read`'s projections take (the MoE path pins the
    // same form: layout.cpp calls these Bf16InF32).  A BF16 source round-trips exactly; an F32 source is rounded
    // once here, on the host, so the kernel sees exactly what the reference does.
    bool upload_bf16(const Found* f, const std::string& name, uint64_t expect, uint16_t** out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        uint64_t n = 1;
        for (uint64_t d : f->tensor->shape) n *= d;
        if (expect && n != expect) {
            err = "dense model: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(expect);
            return false;
        }
        if (n > (uint64_t) INT_MAX) { err = "dense model: " + name + " is too large to convert on the host"; return false; }
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        std::vector<float> host(n);
        if (!dequant_row(f->tensor->type, src, (int) n, host.data())) {
            err = "dense model: " + name + " is " + upper_type(f->tensor->type) + ", which cannot be read as floats";
            return false;
        }
        std::vector<uint16_t> bf(n);
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t x;
            std::memcpy(&x, &host[i], 4);
            if ((x & 0x7fffffffu) > 0x7f800000u) { bf[i] = (uint16_t) ((x >> 16) | 0x40); continue; }   // NaN stays NaN
            x += 0x7fffu + ((x >> 16) & 1u);                                                             // round to nearest even
            bf[i] = (uint16_t) (x >> 16);
        }
        if (!alloc(out, n * 2, err, name.c_str())) return false;
        const cudaError_t s = cudaMemcpy(*out, bf.data(), n * 2, cudaMemcpyHostToDevice);
        if (s != cudaSuccess) { err = "dense model: upload " + name + ": " + cudaGetErrorString(s); return false; }
        return true;
    }

    // A quantized matrix, kept in its GGUF blocks: in VRAM while the budget lasts, in mapped host memory after.
    bool upload_mat(const Found* f, const std::string& name, int n_in, int n_out, QMat& out, uint64_t& total,
                    std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has shape [" + std::to_string(t.shape.empty() ? 0 : t.shape[0]) + ", " +
                  std::to_string(t.shape.size() > 1 ? t.shape[1] : 0) + "], expected [" + std::to_string(n_in) + ", " +
                  std::to_string(n_out) + "]";
            return false;
        }
        if (!native_mmvq_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take. "
                  "Supported: " + kSupported;
            return false;
        }
        int elems = 0, block_bytes = 0;
        if (!strata::block_geometry(t.type, elems, block_bytes) || (uint64_t) n_in % (uint64_t) elems) {
            err = "dense model: " + name + ": invalid block geometry"; return false;
        }
        const uint64_t bytes = native_mmvq_weight_bytes((int) t.type, n_in, n_out);
        const uint64_t in_file = (uint64_t) n_in * n_out / (uint64_t) elems * (uint64_t) block_bytes;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (bytes != in_file || t.offset > payload || bytes > payload - t.offset) {
            err = "dense model: " + name + ": payload size mismatch or truncated file"; return false;
        }
        void* dev = nullptr;
        cudaError_t s = cudaSuccess;
        if (dev_used + bytes <= budget) {
            s = cudaMalloc(&dev, bytes);
            if (s == cudaSuccess) s = cudaMemcpy(dev, f->file->tensor_data(t), bytes, cudaMemcpyHostToDevice);
            if (s != cudaSuccess) {                 // the card is fuller than cudaMemGetInfo said: spill this one
                if (dev) cudaFree(dev);
                dev = nullptr;
                cudaGetLastError();
            } else {
                allocations.push_back(dev);
                dev_used += bytes;
            }
        }
        if (!dev) {
            // Host-resident: pinned + mapped, so the GEMV kernel reads the blocks over PCIe on every token.
            void* host = nullptr;
            s = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
            if (s == cudaSuccess) s = cudaHostGetDevicePointer(&dev, host, 0);
            if (s != cudaSuccess) {
                if (host) cudaFreeHost(host);
                err = "dense model: " + name + " fits neither in VRAM nor in pinned host memory (" +
                      std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
                return false;
            }
            std::memcpy(host, f->file->tensor_data(t), bytes);
            host_allocs.push_back(host);
            host_bytes += bytes;
            ++host_tensors;
        }
        out = QMat{dev, (int) t.type, n_in, n_out};
        total += bytes;
        return true;
    }

    bool setup_embd(EmbdSrc& e, const Found* f, int n_embd, std::string& err) {
        e.f = *f;
        e.type = f->tensor->type;
        int elems = 0, bytes = 0;
        if (!strata::block_geometry(e.type, elems, bytes) || n_embd % elems) {
            err = "dense model: " + std::string("embedding table has an unsupported type ") + upper_type(e.type);
            return false;
        }
        e.row_bytes = (size_t) n_embd / (size_t) elems * (size_t) bytes;
        std::vector<float> probe((size_t) n_embd);      // a dry run on row 0 tells now whether the host knows the type
        if (!dequant_row(e.type, f->file->tensor_data(*f->tensor), n_embd, probe.data())) {
            err = "dense model: cannot dequantize the embedding table of type " + upper_type(e.type);
            return false;
        }
        return true;
    }

    // n embedding rows -> device, one copy; the pinned stage is reused only after the previous copy finished
    // cell0 >= 0: the rows are cells cell0.., and a cell with an image row (set_embd_rows) takes that row instead
    bool upload_embeddings(const EmbdSrc& src, const int32_t* toks, int n, int E, int n_vocab, float* dst,
                           std::string& err, float* stage = nullptr, int64_t cell0 = -1) {
        float* st = stage ? stage : embd_stage;
        if (cudaEventSynchronize(embed_done) != cudaSuccess) { err = "dense model: embedding event"; return false; }
        for (int j = 0; j < n; ++j) {
            if (toks[j] < 0 || toks[j] >= n_vocab) { err = "dense model: token id outside the vocabulary"; return false; }
            if (cell0 >= 0 && img_rows != nullptr && cell0 + j < img_n && img_rows[cell0 + j] != nullptr) {
                std::memcpy(st + (size_t) j * E, img_rows[cell0 + j], (size_t) E * sizeof(float));
                continue;
            }
            const uint8_t* row = src.f.file->tensor_data(*src.f.tensor) + (size_t) toks[j] * src.row_bytes;
            if (!dequant_row(src.type, row, E, st + (size_t) j * E)) { err = "dense model: embedding dequantization"; return false; }
        }
        cudaMemcpyAsync(dst, st, (size_t) n * E * 4, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(embed_done, stream);
        return true;
    }

    // y = W x for n columns; the activation must already be quantized (quantize()) with the same n.
    void gemv(const QMat& w, float* y, int n, void* s) {
        PScope ps(&prof, w.type == 12 ? Prof::GEMV_Q4K : w.type == 14 ? Prof::GEMV_Q6K : w.type == 8 ? Prof::GEMV_Q80 : Prof::GEMV_OTHER, s);
        native_mmvq(w.type, w.dev, q8_1, y, w.n_in, w.n_out, n, s);
    }
    void quantize(const float* x, int n_in, int n, void* s) {
        PScope ps(&prof, Prof::QUANT, s);
        native_quantize_q8_1(x, q8_1, n_in, n, s);
    }

    // ---- the blocks.  Input is I.xn (the normalized residual) unless stated; the output goes to I.mix.
    void gdn_block(const DenseConfig& c, Layer& L, int n, bool snapshot, void* s);
    void attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv, const QMat& wo,
                    const float* qn, const float* kn, uint16_t* kc, uint16_t* vc, int n, int pos0, void* s);
    void ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, const float* post_norm, int n, void* s);
    void ffn_matmul(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, int n, void* s);
    // One hyper-connection read: the residual stack in, the mixer's input written into `xn` and the per-stream
    // injection into `hc_inject`.  Absent on a plain dense file.
    void hc_read(const Layer& L, bool attn_mixer, float eps, int n, void* s);
    bool mtp_block(const DenseConfig& c, int n, int pos0, bool want_logits, void* s);
    bool mtp_kv_only = true;                      // an ingest writes the block's K/V rows only (STRATA_DENSE_MTP_KV_ONLY=0: the whole block)

#ifdef STRATA_DENSE_MMQ
    // ---- the same blocks over a chunk of T tokens, reading and writing Pf instead of the NC-column buffers.
    void pf_quantize(const float* src, int wtype, int n_in, int64_t T, void* s);
    void pf_gemv(const QMat& w, float* y, int64_t T, void* s);
    void pf_gdn_block(const DenseConfig& c, Layer& L, int64_t T, void* s);
    void pf_attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv, const QMat& wo,
                       const float* qn, const float* kn, uint16_t* kc, uint16_t* vc, int64_t T, int pos0, void* s);
    void pf_ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, const float* post_norm,
                      int64_t T, void* s);
#endif
};

// The GDN mixer over n columns.  The recurrence is order-dependent, but "order-dependent" does not mean "one launch
// per column": the chunk kernels walk the columns inside a single launch with the state in registers, so n columns
// cost the same handful of launches as one.  That matters twice over here.  The delta-rule state is 3 MiB per layer,
// and the per-column kernels read and write it once per column; over 48 GDN layers and a 3-column speculative pass
// that is 446 MB of traffic that a chunk does not have.  And the rollback snapshots, which used to be two 3 MiB
// copies per column, are now stores the recurrence makes from registers it already holds.
void DenseModel::Impl::gdn_block(const DenseConfig& c, Layer& L, int n, bool snapshot, void* s) {
    const int C = c.conv_channels(), V = c.value_dim(), E = c.n_embd;
    const int S = c.ssm_state, KH = c.ssm_k_heads, VH = c.ssm_v_heads, qk = S * KH;
    quantize(xn, E, n, s);
    gemv(L.qkv, qkv, n, s);
    gemv(L.z, z, n, s);
    if (L.ab_quant) {                               // the scratch still holds xn: nothing quantized in between
        gemv(L.alpha_q, alpha, n, s);
        gemv(L.beta_q, beta, n, s);
    } else {
        dense_gemv_f32_rows(L.alpha_w, xn, alpha, E, VH, n, s);
        dense_gemv_f32_rows(L.beta_w, xn, beta, E, VH, n, s);
    }
    {
    PScope ps_gdn(&prof, Prof::GDN_LOOP, s);
    // slot[j] receives the state right after column j, which is what rollback(j) restores.  run() guarantees
    // n - 1 <= draft_max, so the slots allocated at load() are exactly enough.
    DenseGdnSnap snaps, snapc;
    if (snapshot) {
        for (int j = 0; j < n - 1; ++j) { snaps.slot[j] = L.st_slots[(size_t) j]; snapc.slot[j] = L.cv_slots[(size_t) j]; }
        snaps.n = n - 1;
        snapc.n = n - 1;
    }
    dense_gdn_conv_chunk(L.conv_state, qkv, L.conv_w, h, C, n, snapc, s);
    dense_gdn_l2_norm(h, n, C, 0, KH, S, c.rms_eps, s);
    dense_gdn_l2_norm(h, n, C, qk, KH, S, c.rms_eps, s);
    dense_gdn_gates(alpha, L.dt, L.ssm_a, gate, beta, VH, n, s);
    dense_gdn_rec_chunk(L.state, h, gate, beta, z, L.ssm_norm, c.rms_eps, y, KH, VH, n, snaps, s);
    }
    quantize(y, V, n, s);
    gemv(L.ssm_out, mix, n, s);
}

void DenseModel::Impl::attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv, const QMat& wo,
                                  const float* qn, const float* kn, uint16_t* kc, uint16_t* vc, int n, int pos0, void* s) {
    const int E = c.n_embd, H = c.n_head, HK = c.n_head_kv, D = c.head_dim;
    quantize(xn, E, n, s);
    gemv(wq, q_full, n, s);
    gemv(wk, kcur, n, s);
    gemv(wv, vcur, n, s);
    const float scale = 1.0f / std::sqrt((float) D);
    {
    PScope ps_attn(&prof, Prof::ATTN_LOOP, s);
    // Everything up to the attention itself is per-row work with no dependency between rows, so it runs over all n
    // columns at once: the strided copy that splits q from its output gate, the per-head norms, RoPE and the KV
    // append are one launch each instead of one per column.  The attention stays the split-K decode kernel, which
    // parallelises over the cache - the one part that wants the columns kept apart.
    dense_split_q(q_full, qcur, n * H, D, s);
    rms_norm_weighted(qcur, qn, n * H, D, c.rms_eps, s);
    rms_norm_weighted(kcur, kn, n * HK, D, c.rms_eps, s);
    // The position of a row is its column, so the table is written per tensor: rows / n_head for q, rows / n_kv for
    // k.  Two writes, exactly as the prompt path does, into a buffer sized for the wider of the two.
    dense_positions_i32(pos_dev, n * H, H, pos0, s);
    native_rope_apply(qcur, qcur, n * H, D, c.n_rot, rope, pos_dev, s);
    dense_positions_i32(pos_dev, n * HK, HK, pos0, s);
    native_rope_apply(kcur, kcur, n * HK, D, c.n_rot, rope, pos_dev, s);
    dense_kv_append_rows_fmt(kc, vc, kfmt, vfmt, kcur, vcur, n, pos0, HK, D, (int) max_context, s);
    // xolotl: every column in one launch (the cache is read once per window, not once per column); a column gets the
    // same bits alone as inside a window, so single steps and verified columns agree
    dense_attn_decode_multi_fmt(qcur, kc, vc, kfmt, vfmt, attn, attn_scratch, H, HK, D, pos0 + 1, n, (int) max_context, scale, s);
    dense_gate_apply(attn, q_full, attn32, n * H, H, D, s);
    }
    quantize(attn32, H * D, n, s);
    gemv(wo, mix, n, s);
}

// The FFN itself over n columns: `xn` holds the mixer's input, `mix` receives the output.  No residual add - the
// caller owns the residual (add_inplace on a plain model, gr_write on a hyper-connection one).
void DenseModel::Impl::ffn_matmul(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, int n, void* s) {
    quantize(xn, c.n_embd, n, s);
    gemv(g, ffn_g, n, s);
    gemv(u, ffn_u, n, s);
    dense_swiglu(ffn_g, ffn_u, ffn_h, (int64_t) n * c.n_ff, s);
    quantize(ffn_h, c.n_ff, n, s);
    gemv(d, mix, n, s);
}

// x += ffn(rms_norm(x, post_norm)), for n columns
void DenseModel::Impl::ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d, const float* post_norm,
                                 int n, void* s) {
    const int E = c.n_embd;
    dense_rms_norm(x, post_norm, xn, n, E, c.rms_eps, s);
    ffn_matmul(c, g, u, d, n, s);
    add_inplace(x, mix, (int64_t) n * E, s);
}

// One hyper-connection read: the residual stack (R) -> the mixer's input (xn) and the per-stream injection.
// The same kernels the MoE path runs (kernels/gr.hpp, `build_hc_mix`), so a dense and a MoE file of this family
// mix their streams identically.
void DenseModel::Impl::hc_read(const Layer& L, bool attn_mixer, float eps, int n, void* s) {
    strata::kernels::gr_read_multi(R, attn_mixer ? L.hc_attn_norm : L.hc_ffn_norm,
                                   attn_mixer ? L.hc_attn_down : L.hc_ffn_down,
                                   attn_mixer ? L.hc_attn_up : L.hc_ffn_up,
                                   attn_mixer ? L.hc_attn_inject : L.hc_ffn_inject,
                                   eps, gr_shape, gr_ws, hc_xn, hc_lo, hc_gated, xn, hc_inject, n, s);
}

#ifdef STRATA_DENSE_MMQ
// ============================================================================ the batched prompt blocks
// The same arithmetic as the blocks above over a chunk of T tokens.  The projections go through MMQ: the weights
// stay in their GGUF blocks, the activations are rounded to q8_1 and the products run on int8 tensor cores, so a
// weight is read once per CHUNK instead of once per eight tokens.

/// q8_1 activations for the next pf_gemv().  The layout MMQ wants depends on the WEIGHT type, so every type change
/// needs its own pass over the activations.
void DenseModel::Impl::pf_quantize(const float* src, int wtype, int n_in, int64_t T, void* s) {
    PScope ps(&prof, Prof::QUANT, s);
    strata::prefill::mmq::quantize(src, nullptr, pf->xq, wtype, n_in, n_in, T, s);
}

/// y[t, :] = W x[t, :] for the T rows already quantized into pf->xq (one MMQ group: the whole chunk).
void DenseModel::Impl::pf_gemv(const QMat& w, float* y, int64_t T, void* s) {
    PScope ps(&prof, w.type == 12 ? Prof::GEMV_Q4K : w.type == 14 ? Prof::GEMV_Q6K
                                : w.type == 8 ? Prof::GEMV_Q80 : Prof::GEMV_OTHER, s);
    strata::prefill::mmq::Product p;
    p.w = w.dev;
    p.type = w.type;
    p.w_rows = w.n_out;
    p.w_cols = w.n_in;
    p.expert_bytes = strata::prefill::mmq::matrix_bytes(w.type, w.n_out, w.n_in);   // one group: never stepped over
    p.n = 1;
    p.xq = pf->xq;
    p.bounds = pf->bounds;                    // {0, T}
    p.ids = pf->ids;                          // iota: each row goes back to its own index
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = w.n_out;
    pf->ctx.run(p, s);
}

void DenseModel::Impl::pf_gdn_block(const DenseConfig& c, Layer& L, int64_t T, void* s) {
    const int C = c.conv_channels(), V = c.value_dim(), E = c.n_embd;
    const int S = c.ssm_state, KH = c.ssm_k_heads, VH = c.ssm_v_heads;
    pf_quantize(pf->xn, L.qkv.type, E, T, s);
    pf_gemv(L.qkv, pf->qkv, T, s);
    if (L.z.type != L.qkv.type) pf_quantize(pf->xn, L.z.type, E, T, s);
    pf_gemv(L.z, pf->z, T, s);
    if (L.ab_quant) {                         // the two tiny projections, quantized in the file
        if (L.alpha_q.type != L.z.type) pf_quantize(pf->xn, L.alpha_q.type, E, T, s);
        pf_gemv(L.alpha_q, pf->alpha, T, s);
        if (L.beta_q.type != L.alpha_q.type) pf_quantize(pf->xn, L.beta_q.type, E, T, s);
        pf_gemv(L.beta_q, pf->beta, T, s);
    } else {
        dense_gemv_f32_rows(L.alpha_w, pf->xn, pf->alpha, E, VH, (int) T, s);
        dense_gemv_f32_rows(L.beta_w, pf->xn, pf->beta, E, VH, (int) T, s);
    }
    {
        PScope ps_gdn(&prof, Prof::GDN_LOOP, s);
        // conv + SiLU over the chunk (the 3-value history carries over), the L2 norm of the q and k heads of every
        // token, the per-head gates, then the delta-rule recurrence walking the chunk with the output norm folded in.
        // {} = no rollback snapshots: a prompt chunk is never partially undone.
        dense_gdn_conv_chunk(L.conv_state, pf->qkv, L.conv_w, pf->h, C, (int) T, {}, s);
        dense_gdn_l2_norm(pf->h, (int) T, C, 0, KH, S, c.rms_eps, s);
        dense_gdn_l2_norm(pf->h, (int) T, C, S * KH, KH, S, c.rms_eps, s);
        dense_gdn_gates(pf->alpha, L.dt, L.ssm_a, pf->gate, pf->beta, VH, (int) T, s);
        dense_gdn_rec_chunk(L.state, pf->h, pf->gate, pf->beta, pf->z, L.ssm_norm, c.rms_eps, pf->y, KH, VH, (int) T, {}, s);
    }
    pf_quantize(pf->y, L.ssm_out.type, V, T, s);
    pf_gemv(L.ssm_out, pf->mix, T, s);
}

void DenseModel::Impl::pf_attn_block(const DenseConfig& c, const QMat& wq, const QMat& wk, const QMat& wv,
                                     const QMat& wo, const float* qn, const float* kn, uint16_t* kc, uint16_t* vc,
                                     int64_t T, int pos0, void* s) {
    const int E = c.n_embd, H = c.n_head, HK = c.n_head_kv, D = c.head_dim;
    pf_quantize(pf->xn, wq.type, E, T, s);
    pf_gemv(wq, pf->q_full, T, s);
    if (wk.type != wq.type) pf_quantize(pf->xn, wk.type, E, T, s);
    pf_gemv(wk, pf->kcur, T, s);
    if (wv.type != wk.type) pf_quantize(pf->xn, wv.type, E, T, s);
    pf_gemv(wv, pf->vcur, T, s);
    const float scale = 1.0f / std::sqrt((float) D);
    {
        PScope ps_attn(&prof, Prof::ATTN_LOOP, s);
        // q is the first head_dim of every head's [q | gate] block; the norms and RoPE are per row, so they batch.
        dense_split_q(pf->q_full, pf->qcur, (int) (T * H), D, s);
        rms_norm_weighted(pf->qcur, qn, T * H, D, c.rms_eps, s);
        rms_norm_weighted(pf->kcur, kn, T * HK, D, c.rms_eps, s);
        dense_positions_i32(pf->pos_dev, (int) (T * H), H, (int32_t) pos0, s);
        native_rope_apply(pf->qcur, pf->qcur, (int) (T * H), D, c.n_rot, rope, pf->pos_dev, s);
        dense_positions_i32(pf->pos_dev, (int) (T * HK), HK, (int32_t) pos0, s);
        native_rope_apply(pf->kcur, pf->kcur, (int) (T * HK), D, c.n_rot, rope, pf->pos_dev, s);
        dense_kv_append_rows_fmt(kc, vc, kfmt, vfmt, pf->kcur, pf->vcur, (int) T, pos0, HK, D, (int) max_context, s);
        dense_attn_chunk_fmt(pf->qcur, kc, vc, kfmt, vfmt, pf->attn, (int) T, pos0, H, HK, D, (int) max_context, scale, s);
        dense_gate_apply(pf->attn, pf->q_full, pf->attn32, (int) (T * H), H, D, s);
    }
    pf_quantize(pf->attn32, wo.type, H * D, T, s);
    pf_gemv(wo, pf->mix, T, s);
}

void DenseModel::Impl::pf_ffn_block(const DenseConfig& c, const QMat& g, const QMat& u, const QMat& d,
                                    const float* post_norm, int64_t T, void* s) {
    const int E = c.n_embd;
    dense_rms_norm(pf->x, post_norm, pf->xn, (int) T, E, c.rms_eps, s);
    pf_quantize(pf->xn, g.type, E, T, s);
    pf_gemv(g, pf->ffn_g, T, s);
    if (u.type != g.type) pf_quantize(pf->xn, u.type, E, T, s);
    pf_gemv(u, pf->ffn_u, T, s);
    dense_swiglu(pf->ffn_g, pf->ffn_u, pf->ffn_h, (int64_t) T * c.n_ff, s);
    pf_quantize(pf->ffn_h, d.type, c.n_ff, T, s);
    pf_gemv(d, pf->mix, T, s);
    add_inplace(pf->x, pf->mix, (int64_t) T * E, s);
}
#endif  // STRATA_DENSE_MMQ

// The MTP block over n columns whose [e_norm | h_norm] inputs are in `cat`.  Leaves the head's logits for the
// last column in mtp_logits when asked.
bool DenseModel::Impl::mtp_block(const DenseConfig& c, int n, int pos0, bool want_logits, void* s) {
    const Mtp& M = mtp;
    const int E = c.n_embd;
    quantize(cat, 2 * E, n, s);
    gemv(M.eh_proj, x, n, s);                       // x = the block's input and its attention residual
    dense_rms_norm(x, M.attn_norm, xn, n, E, c.rms_eps, s);
    if (!want_logits && mtp_kv_only) {
        // An ingest keeps nothing of the block but its K/V rows: the block is one layer, so a position's K and V come
        // from this input alone, and the block's output is read only by a draft step (want_logits).  The rows below are
        // attn_block()'s own launches in its order (same quantize, same projections, same norm, RoPE and append), so
        // the cache gets the same bits; the q projection, the attention over the cache, the output projection and the
        // whole FFN are not run.  Measured 2026-11-11 (RTX 4090, Swift-1.5 27B IQ3_S, k8v4, draft 3): prompt read-in
        // +19 % at 2K and +29 % at 30K, greedy answers and draft cycles identical.  STRATA_DENSE_MTP_KV_ONLY=0 = the old path.
        const int HK = c.n_head_kv, D = c.head_dim;
        quantize(xn, E, n, s);
        gemv(M.k, kcur, n, s);
        gemv(M.v, vcur, n, s);
        rms_norm_weighted(kcur, M.k_norm, n * HK, D, c.rms_eps, s);
        dense_positions_i32(pos_dev, n * HK, HK, pos0, s);
        native_rope_apply(kcur, kcur, n * HK, D, c.n_rot, rope, pos_dev, s);
        dense_kv_append_rows_fmt(M.kc, M.vc, kfmt, vfmt, kcur, vcur, n, pos0, HK, D, (int) max_context, s);
        return true;
    }
    attn_block(c, M.q, M.k, M.v, M.o, M.q_norm, M.k_norm, M.kc, M.vc, n, pos0, s);
    add_inplace(x, mix, (int64_t) n * E, s);
    ffn_block(c, M.ffn_gate, M.ffn_up, M.ffn_down, M.post_norm, n, s);
    if (want_logits) {
        dense_rms_norm(x + (size_t) (n - 1) * E, M.head_norm, xn, 1, E, c.rms_eps, s);
        cudaMemcpyAsync(mtp_h, xn, (size_t) E * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
        quantize(xn, E, 1, s);
        gemv(draft_head.dev ? draft_head : M.head, mtp_logits, 1, s);
    }
    return true;
}

DenseModel::DenseModel() : impl_(new Impl) {}

DenseModel::~DenseModel() {
    if (!impl_) return;
    if (impl_->stream) cudaStreamSynchronize(impl_->stream);
    for (void* p : impl_->allocations) cudaFree(p);
    for (void* p : impl_->host_allocs) cudaFreeHost(p);
    for (auto& c : impl_->ckpt) if (c.host) cudaFreeHost(c.host);
#ifdef STRATA_DENSE_MMQ
    if (impl_->pf && impl_->pf->stage) cudaFreeHost(impl_->pf->stage);
#endif
    if (impl_->embd_stage) cudaFreeHost(impl_->embd_stage);
    if (impl_->embed_done) cudaEventDestroy(impl_->embed_done);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

float* DenseModel::logits_col(int j) const { return impl_->logits + (size_t) j * cfg_.n_vocab; }
int DenseModel::n_taps() const { return impl_->n_taps; }
const float* DenseModel::run_taps() const { return impl_->taps_run; }
const float* DenseModel::ring_taps() const { return impl_->taps_ring; }
const void* DenseModel::head_weights() const { return impl_->head.dev; }
int DenseModel::head_type() const { return impl_->head.type; }
int64_t DenseModel::head_rows() const { return impl_->head.n_out; }
void DenseModel::set_embd_rows(const float* const* rows, int64_t n) {
    impl_->img_rows = rows;
    impl_->img_n = rows != nullptr ? n : 0;
}

bool DenseModel::embed_host(int32_t token, float* out, std::string& err) const {
    const Impl& I = *impl_;
    if (token < 0 || token >= cfg_.n_vocab) { err = "dense model: token id outside the vocabulary"; return false; }
    const uint8_t* row = I.embd.f.file->tensor_data(*I.embd.f.tensor) + (size_t) token * I.embd.row_bytes;
    if (!dequant_row(I.embd.type, row, cfg_.n_embd, out)) { err = "dense model: embedding dequantization"; return false; }
    return true;
}
void* DenseModel::stream() const { return impl_->stream; }

bool DenseModel::load(const std::string& path, int64_t max_context, std::string& err, const DenseOptions& opt) {
    Impl& I = *impl_;
    bool with_mtp = opt.mtp;
    const int draft_max = std::max(1, std::min(opt.draft_max, NC - 1));
    try {
        // ---- 1. the files and the tensor index
        for (const std::string& p : shard_paths(path)) I.files.emplace_back(new strata::GgufFile(p));
        for (const auto& f : I.files)
            for (const auto& t : f->tensors()) I.index[t.name] = Found{f.get(), &t};
        const strata::GgufFile& g0 = *I.files.front();

        // ---- 2. the configuration, from the model's own metadata
        const strata::MetaValue* arch = g0.get("general.architecture");
        if (!arch) { err = "dense model: missing general.architecture"; return false; }
        cfg_.arch = arch->s;
        const std::string a = cfg_.arch + ".";
        if (g0.get(a + "expert_count") && g0.get(a + "expert_count")->u > 0) {
            err = "dense model: this GGUF is a MoE model; run it with `strata`, not `xolotl-dense`";
            return false;
        }
        {
            // Qwen4-generation dense files carry the Flash-Next additions.  Hyper-connections are served (below);
            // a QSA indexer and PLE are not, and an architecture this engine was never validated on is opt-in.
            strata::core::DenseArchProbe probe;
            const strata::MetaValue* hc = g0.get(a + "hyper_connection.count");
            probe.hyper_connections = hc != nullptr && hc->is_num() && hc->num() > 0;
            const strata::MetaValue* ix = g0.get(a + "attention.indexer.head_count");
            probe.qsa_indexer = ix != nullptr && ix->is_num() && ix->num() > 0;
            const strata::MetaValue* pl = g0.get(a + "ple.layers");
            probe.ple = pl != nullptr && pl->type == strata::MetaType::ARRAY && pl->count > 0;
            if (!probe.ple)
                for (const auto& entry : I.index)
                    if (entry.first.ends_with(".ple_key.weight")) { probe.ple = true; break; }
            const char* experimental = std::getenv("STRATA_DENSE_EXPERIMENTAL_ARCH");
            const std::string refusal = strata::core::dense_arch_refusal(
                cfg_.arch, probe, experimental != nullptr && experimental[0] == '1');
            if (!refusal.empty()) { err = refusal; return false; }
        }
        int64_t v = 0;
        auto need = [&](const char* key, int& dst) {
            if (!meta_int(g0, a + key, v)) { err = std::string("dense model: missing ") + a + key; return false; }
            dst = (int) v;
            return true;
        };
        int block_count = 0;
        if (!need("block_count", block_count) || !need("embedding_length", cfg_.n_embd) ||
            !need("feed_forward_length", cfg_.n_ff) || !need("attention.head_count", cfg_.n_head) ||
            !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", cfg_.head_dim) ||
            !need("rope.dimension_count", cfg_.n_rot) || !need("ssm.conv_kernel", cfg_.ssm_d_conv) ||
            !need("ssm.state_size", cfg_.ssm_state) || !need("ssm.group_count", cfg_.ssm_k_heads) ||
            !need("ssm.time_step_rank", cfg_.ssm_v_heads))
            return false;
        // block_count counts the MTP block(s) stored after the trunk; llama.cpp runs only the trunk
        cfg_.n_nextn = meta_int(g0, a + "nextn_predict_layers", v) ? (int) v : 0;
        cfg_.n_layer = block_count - cfg_.n_nextn;
        if (cfg_.n_layer <= 0) { err = "dense model: no trunk layers"; return false; }
        // Hyper-connections (the Qwen4 gated residual).  Absent keys mean the plain pre-norm residual.
        cfg_.hc = meta_int(g0, a + "hyper_connection.count", v) && v > 0 ? (int) v : 0;
        cfg_.hc_lr = meta_int(g0, a + "hyper_connection.low_rank", v) ? (int) v : 0;
        if (!cfg_.has_hc()) cfg_.hc_lr = 0;
        if (const std::string why = strata::core::dense_hc_refusal(cfg_.hc, cfg_.hc_lr, cfg_.n_embd, with_mtp);
            !why.empty()) {
            err = why;
            return false;
        }
        if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
        if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
        // The rope the analytic kernel applies: the process scaling (rope_scaling.hpp, `none` unless a caller
        // has set one) re-based on what this file declares.  The base is the file's, never the caller's: the
        // angles must match the weights.
        I.rope = rope_scaling();
        I.rope.freq_base = (double) cfg_.rope_base;
        if (const char* why = rope_scaling_invalid(I.rope)) {
            err = std::string("dense model: invalid rope scaling configuration: ") + why;
            return false;
        }
        if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
        if (meta_int(g0, a + "ssm.inner_size", v) && v != cfg_.value_dim()) {
            err = "dense model: ssm.inner_size " + std::to_string(v) + " != state_size * time_step_rank";
            return false;
        }
        if (cfg_.ssm_state != 128 || cfg_.ssm_d_conv != 4 || cfg_.ssm_v_heads % cfg_.ssm_k_heads != 0) {
            err = "dense model: the GDN kernels are written for state_size 128 and conv_kernel 4";
            return false;
        }
        if (cfg_.head_dim % 32 != 0 || cfg_.head_dim > 256 || cfg_.n_head % cfg_.n_head_kv != 0 ||
            cfg_.n_rot > cfg_.head_dim || cfg_.n_rot % 2 != 0) {
            err = "dense model: unsupported attention geometry";
            return false;
        }
        if (cfg_.n_embd % 256 != 0 || cfg_.n_ff % 256 != 0 || cfg_.value_dim() % 256 != 0 ||
            (cfg_.n_head * cfg_.head_dim) % 256 != 0) {
            err = "dense model: every GEMV input width must be a multiple of 256 (K-quant super-block)";
            return false;
        }
        const Found* tok = I.find("token_embd.weight");
        if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) cfg_.n_embd) {
            err = "dense model: token_embd.weight missing or not [n_embd, n_vocab]";
            return false;
        }
        cfg_.n_vocab = (int) tok->tensor->shape[1];
        max_context_ = max_context;
        I.max_context = max_context;
        if (max_context < 16) { err = "dense model: context must be at least 16"; return false; }
        const int E = cfg_.n_embd;
        const int mtp_block_idx = cfg_.n_layer;           // the MTP block follows the trunk
        const std::string mp = "blk." + std::to_string(mtp_block_idx) + ".";
        uint64_t draft_bytes = 0;                         // the --draft-vocab head, 0 = none (section 4 decides)
        if (with_mtp && (cfg_.n_nextn < 1 || !I.find(mp + "nextn.eh_proj.weight"))) {
            err = "dense model: this GGUF has no MTP block (nextn_predict_layers = " + std::to_string(cfg_.n_nextn) +
                  "); start without --mtp";
            return false;
        }

        cudaError_t cs = cudaStreamCreate(&I.stream);
        if (cs == cudaSuccess) cs = cudaEventCreateWithFlags(&I.embed_done, cudaEventDisableTiming);
        if (cs == cudaSuccess) cs = cudaHostAlloc((void**) &I.embd_stage, (size_t) NC * E * 4, cudaHostAllocDefault);
        if (cs != cudaSuccess) { err = std::string("dense model: CUDA setup: ") + cudaGetErrorString(cs); return false; }
        if (const char* e = std::getenv("STRATA_DENSE_PROF")) I.prof.enabled = e[0] && e[0] != '0';
        if (const char* e = std::getenv("STRATA_DENSE_MTP_KV_ONLY")) I.mtp_kv_only = e[0] && e[0] != '0';
        if (const char* e = std::getenv("STRATA_DENSE_MMVQ_FAST"))      // the multi-column layout that is faster but not bitwise equal to 1 column
            if (e[0] && e[0] != '0') { native_mmvq_set_multi_exact(false); std::fprintf(stderr, "xolotl-dense: multi-column GEMV: upstream layout\n"); }
        if (const char* e = std::getenv("STRATA_DENSE_DEBUG")) {
            long long a0 = 0, b0 = 0;
            if (std::sscanf(e, "%lld:%lld", &a0, &b0) == 2) { I.dbg_from = a0; I.dbg_to = b0; }
        }

        // ---- 3. the embedding tables stay in the (mapped) file; a row is dequantized per token
        if (!I.setup_embd(I.embd, tok, E, err)) return false;
        I.embd_mtp = I.embd;
        const Found* mtp_embd = with_mtp ? I.find(mp + "nextn.embed_tokens.weight") : nullptr;
        if (mtp_embd && !I.setup_embd(I.embd_mtp, mtp_embd, E, err)) return false;

        // ---- 3b. the KV cache format (strata's --kv)
        {
            std::string kv = opt.kv;
            for (char& ch : kv) ch = (char) std::tolower((unsigned char) ch);
            if (kv.empty() || kv == "fp16" || kv == "f16") { I.kfmt = DENSE_KV_F16; I.vfmt = DENSE_KV_F16; kv_name_ = "fp16"; }
            else if (kv == "int8" || kv == "q8_0" || kv == "q8") { I.kfmt = DENSE_KV_Q8; I.vfmt = DENSE_KV_Q8; kv_name_ = "int8"; }
            else if (kv == "q4_0" || kv == "q4") { I.kfmt = DENSE_KV_Q4; I.vfmt = DENSE_KV_Q4; kv_name_ = "q4_0"; }
            else if (kv == "k8v4") { I.kfmt = DENSE_KV_Q8; I.vfmt = DENSE_KV_Q4; kv_name_ = "k8v4"; }
            else { err = "dense model: --kv must be fp16, int8, q4_0 or k8v4 (got '" + opt.kv + "')"; return false; }
            if ((I.kfmt == DENSE_KV_Q4 || I.vfmt == DENSE_KV_Q4) && cfg_.head_dim != 256) {
                err = "dense model: --kv " + kv_name_ + " rotates 256 values per head, but this model's head_dim is " +
                      std::to_string(cfg_.head_dim) + "; use fp16 or int8";
                return false;
            }
        }
        const uint64_t kcell = dense_kv_cell_bytes(I.kfmt, cfg_.head_dim), vcell = dense_kv_cell_bytes(I.vfmt, cfg_.head_dim);

        // ---- 4. the VRAM budget for the weights
        uint64_t total = 0;
        const int C = cfg_.conv_channels(), V = cfg_.value_dim(), H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
        {   // What is free, minus the KV caches, the rollback snapshots and a margin for the CUDA context, the
            // activations and the kernels.  The rest of the weights stays in pinned host memory.
            int n_attn = 0, n_gdn = 0;
            for (int l = 0; l < cfg_.n_layer; ++l) {
                if (I.find("blk." + std::to_string(l) + ".attn_q.weight")) ++n_attn; else ++n_gdn;
            }
            const uint64_t kv_per_layer = (uint64_t) HK * (uint64_t) max_context * (kcell + vcell);
            const uint64_t snap_one = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4 +
                                      (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;     // one snapshot of one GDN layer
            const uint64_t tap_bytes = opt.taps.empty() ? 0 :
                (uint64_t) (NC + DenseModel::kTapRing) * opt.taps.size() * (uint64_t) cfg_.n_embd * 4;
            // the decode attention's partials: one region per column of a window (the extra NC - 1, ~19 MB each at 200K)
            const uint64_t scratch_extra = (uint64_t) (NC - 1) * dense_attn_scratch_bytes(cfg_.n_head, cfg_.head_dim, (int) max_context);
            auto reserved_for = [&](bool mtp) {
                return (uint64_t) (n_attn + (mtp ? 1 : 0)) * kv_per_layer +
                       ((mtp || opt.spec_snapshots) ? (uint64_t) n_gdn * snap_one * (uint64_t) draft_max : 0) + tap_bytes + scratch_extra;
            };
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t margin = 1024ull << 20;
            auto budget_for = [&](bool mtp) -> int64_t { return (int64_t) free_b - (int64_t) reserved_for(mtp) - (int64_t) margin; };
            // the bytes of every quantized matrix the trunk (and the MTP block) will upload
            auto bytes_of_prefix = [&](const std::string& pre) {
                uint64_t sum = 0;
                for (auto it = I.index.lower_bound(pre); it != I.index.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
                    const strata::TensorInfo& t = *it->second.tensor;
                    if (t.shape.size() != 2 || !native_mmvq_supported((int) t.type)) continue;
                    try { sum += native_mmvq_weight_bytes((int) t.type, (int) t.shape[0], (int) t.shape[1]); } catch (const std::exception&) {}
                }
                return sum;
            };
            uint64_t trunk_bytes = 0, mtp_bytes = 0;
            for (int l = 0; l < cfg_.n_layer; ++l) trunk_bytes += bytes_of_prefix("blk." + std::to_string(l) + ".");
            if (I.find("output.weight")) trunk_bytes += bytes_of_prefix("output.weight");
            else if (tok) trunk_bytes += native_mmvq_supported((int) tok->tensor->type)
                    ? native_mmvq_weight_bytes((int) tok->tensor->type, E, cfg_.n_vocab) : 0;
            mtp_bytes = bytes_of_prefix(mp);
            if (with_mtp && !opt.draft_vocab.empty()) {     // the shortlist head is a second, smaller copy of head rows
                const Found* h = I.find("output.weight") ? I.find("output.weight") : tok;
                if (h && native_mmvq_supported((int) h->tensor->type))
                    draft_bytes = native_mmvq_weight_bytes((int) h->tensor->type, E,
                                                           (int) opt.draft_vocab.size() + std::max(0, opt.draft_vocab_extra));
            }
            const bool user_cap = std::getenv("STRATA_DENSE_GPU_MIB") != nullptr;
            if (with_mtp && !user_cap && !opt.mtp_force && budget_for(true) < (int64_t) (trunk_bytes + mtp_bytes)) {
                // the longest context at which everything, MTP included, stays in VRAM
                const int64_t per_token = (int64_t) (n_attn + 1) * HK * (int64_t) (kcell + vcell);   // KV bytes per token, MTP layer too
                const int64_t room = (int64_t) free_b - (int64_t) margin - (int64_t) (trunk_bytes + mtp_bytes) -
                                     (int64_t) ((uint64_t) n_gdn * snap_one * (uint64_t) draft_max);
                const int64_t fit_ctx = room > 0 ? (room / per_token) / 1024 * 1024 : 0;
                std::fprintf(stderr,
                             "xolotl-dense: MTP is OFF: the weights (%.2f GiB with the MTP block), the KV cache for %lld tokens and "
                             "the %d rollback snapshot(s) do not all fit in VRAM, so some weights would be read over PCIe on every "
                             "token - slower, not faster.%s\n",
                             (trunk_bytes + mtp_bytes) / 1073741824.0, (long long) max_context, draft_max,
                             fit_ctx >= 2048 ? "" : " Even a short context would not fit: use a smaller quant.");
                if (fit_ctx >= 2048)
                    std::fprintf(stderr, "xolotl-dense:   with MTP on, a --context up to about %lld tokens fits in this GPU "
                                 "(or use --draft-max 1); --mtp-force keeps MTP at the current context anyway.\n",
                                 (long long) fit_ctx);
                with_mtp = false;
            }
            // the shortlist head only if it fits next to everything else: without the room it is left out, MTP stays on
            if (draft_bytes && (!with_mtp || (!user_cap && budget_for(true) < (int64_t) (trunk_bytes + mtp_bytes + draft_bytes)))) {
                if (with_mtp)
                    std::fprintf(stderr, "xolotl-dense: --draft-vocab left out: no VRAM for its %.0f MiB (MTP stays on, drafts use the full head)\n",
                                 draft_bytes / 1048576.0);
                draft_bytes = 0;
            }
            const uint64_t reserved = reserved_for(with_mtp);
            const uint64_t kv_need = (uint64_t) (n_attn + (with_mtp ? 1 : 0)) * kv_per_layer;
            if (user_cap) {                                                  // explicit cap for the weights, in MiB
                I.budget = (uint64_t) std::atoll(std::getenv("STRATA_DENSE_GPU_MIB")) << 20;
            } else if ((int64_t) free_b > (int64_t) (reserved + margin)) {
                I.budget = (uint64_t) budget_for(with_mtp) - draft_bytes;   // the weights leave the shortlist head its room
            } else {
                err = "dense model: the KV cache for a context of " + std::to_string(max_context) + " needs " +
                      std::to_string(kv_need >> 20) + " MiB, but only " + std::to_string(free_b >> 20) +
                      " MiB of VRAM are free (1 GiB is kept as margin). Use a shorter --context.";
                return false;
            }
            std::fprintf(stderr, "xolotl-dense: VRAM %zu MiB free of %zu; KV cache %llu MiB (%s, context %lld)%s; "
                         "%llu MiB of weights may go to the GPU (the model needs %.0f MiB), the rest stays in host memory\n",
                         free_b >> 20, total_b >> 20, (unsigned long long) (kv_need >> 20), kv_name_.c_str(), (long long) max_context,
                         with_mtp ? " + MTP snapshots" : "", (unsigned long long) (I.budget >> 20),
                         (double) (trunk_bytes + (with_mtp ? mtp_bytes : 0)) / 1048576.0);
        }

        // ---- 5. the MTP block first: it is small, and a draft that has to cross PCIe is not worth making
        if (with_mtp) {
            Mtp& M = I.mtp;
            auto mat = [&](const char* nm, int n_in, int n_out, QMat& out) {
                return I.upload_mat(I.find(mp + nm), mp + nm, n_in, n_out, out, total, err);
            };
            auto flt = [&](const char* nm, uint64_t n, float** out) {
                return I.upload_floats(I.find(mp + nm), mp + nm, n, out, err);
            };
            if (!mat("nextn.eh_proj.weight", 2 * E, E, M.eh_proj) || !flt("nextn.enorm.weight", E, &M.enorm) ||
                !flt("nextn.hnorm.weight", E, &M.hnorm) || !flt("attn_norm.weight", E, &M.attn_norm) ||
                !flt("post_attention_norm.weight", E, &M.post_norm) || !flt("attn_q_norm.weight", D, &M.q_norm) ||
                !flt("attn_k_norm.weight", D, &M.k_norm) || !mat("attn_q.weight", E, H * 2 * D, M.q) ||
                !mat("attn_k.weight", E, HK * D, M.k) || !mat("attn_v.weight", E, HK * D, M.v) ||
                !mat("attn_output.weight", H * D, E, M.o) || !mat("ffn_gate.weight", E, cfg_.n_ff, M.ffn_gate) ||
                !mat("ffn_up.weight", E, cfg_.n_ff, M.ffn_up) || !mat("ffn_down.weight", cfg_.n_ff, E, M.ffn_down))
                return false;
            if (I.find(mp + "nextn.shared_head_norm.weight")) {
                if (!flt("nextn.shared_head_norm.weight", E, &M.head_norm)) return false;
            } else if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &M.head_norm, err)) {
                return false;
            }
            if (I.find(mp + "nextn.shared_head_head.weight") &&
                !mat("nextn.shared_head_head.weight", E, cfg_.n_vocab, M.head))
                return false;                              // otherwise the trunk's head is shared (set below)
            const uint64_t kb = (uint64_t) HK * (uint64_t) max_context * kcell, vb = (uint64_t) HK * (uint64_t) max_context * vcell;
            if (!I.alloc(&M.kc, kb, err, "MTP K cache") || !I.alloc(&M.vc, vb, err, "MTP V cache")) return false;
            kv_bytes_ += kb + vb;
        }

        // ---- 6. the trunk layers
        I.layers.assign((size_t) cfg_.n_layer, Layer{});
        for (int l = 0; l < cfg_.n_layer; ++l) {
            Layer& L = I.layers[(size_t) l];
            const std::string p = "blk." + std::to_string(l) + ".";
            const std::string at = "layer " + std::to_string(l) + ": ";
            const bool has_gdn = I.find(p + "attn_qkv.weight") != nullptr;
            const bool has_attn = I.find(p + "attn_q.weight") != nullptr;
            if (has_gdn == has_attn) { err = "dense model: " + at + "needs exactly one of attn_qkv.weight / attn_q.weight"; return false; }
            L.attn = has_attn;
            if (cfg_.has_hc()) {
                // The gated residual replaces BOTH layer norms: the mixer's two reads are the equivalent of
                // rms_norm(attn_norm) and rms_norm(post_attention_norm), with the streams mixed in.
                const int HD = cfg_.hc_dim();
                if (!I.upload_floats(I.find(p + "hc_attn_norm.weight"), p + "hc_attn_norm.weight", HD, &L.hc_attn_norm, err) ||
                    !I.upload_floats(I.find(p + "hc_ffn_norm.weight"), p + "hc_ffn_norm.weight", HD, &L.hc_ffn_norm, err) ||
                    !I.upload_bf16(I.find(p + "hc_attn_down.weight"), p + "hc_attn_down.weight", (uint64_t) HD * cfg_.hc_lr, &L.hc_attn_down, err) ||
                    !I.upload_bf16(I.find(p + "hc_attn_up.weight"), p + "hc_attn_up.weight", (uint64_t) HD * cfg_.hc_lr, &L.hc_attn_up, err) ||
                    !I.upload_bf16(I.find(p + "hc_attn_inject.weight"), p + "hc_attn_inject.weight", (uint64_t) HD * cfg_.hc, &L.hc_attn_inject, err) ||
                    !I.upload_bf16(I.find(p + "hc_ffn_down.weight"), p + "hc_ffn_down.weight", (uint64_t) HD * cfg_.hc_lr, &L.hc_ffn_down, err) ||
                    !I.upload_bf16(I.find(p + "hc_ffn_up.weight"), p + "hc_ffn_up.weight", (uint64_t) HD * cfg_.hc_lr, &L.hc_ffn_up, err) ||
                    !I.upload_bf16(I.find(p + "hc_ffn_inject.weight"), p + "hc_ffn_inject.weight", (uint64_t) HD * cfg_.hc, &L.hc_ffn_inject, err))
                    return false;
            } else {
                if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", E, &L.attn_norm, err)) return false;
                if (!I.upload_floats(I.find_any({"post_attention_norm.weight", "ffn_norm.weight"}, p), p + "post_attention_norm.weight",
                                     E, &L.post_norm, err)) return false;
            }
            if (!I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, L.ffn_gate, total, err) ||
                !I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, L.ffn_up, total, err) ||
                !I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, L.ffn_down, total, err))
                return false;
            if (!L.attn) {
                if (!I.upload_mat(I.find(p + "attn_qkv.weight"), p + "attn_qkv.weight", E, C, L.qkv, total, err) ||
                    !I.upload_mat(I.find(p + "attn_gate.weight"), p + "attn_gate.weight", E, V, L.z, total, err) ||
                    !I.upload_mat(I.find(p + "ssm_out.weight"), p + "ssm_out.weight", V, E, L.ssm_out, total, err))
                    return false;
                if (!I.upload_floats(I.find(p + "ssm_conv1d.weight"), p + "ssm_conv1d.weight", (uint64_t) C * cfg_.ssm_d_conv, &L.conv_w, err) ||
                    !I.upload_floats(I.find_any({"ssm_dt.bias", "ssm_dt"}, p), p + "ssm_dt.bias", cfg_.ssm_v_heads, &L.dt, err) ||
                    !I.upload_floats(I.find_any({"ssm_a", "ssm_a.weight"}, p), p + "ssm_a", cfg_.ssm_v_heads, &L.ssm_a, err) ||
                    !I.upload_floats(I.find(p + "ssm_norm.weight"), p + "ssm_norm.weight", cfg_.ssm_state, &L.ssm_norm, err))
                    return false;
                {   // ssm_alpha / ssm_beta: floats in some files, Q8_0 (or another quant) in others - both are served
                    const Found* fa = I.find(p + "ssm_alpha.weight");
                    const Found* fb = I.find(p + "ssm_beta.weight");
                    if (!fa || !fb) { err = "dense model: " + at + "missing ssm_alpha.weight / ssm_beta.weight"; return false; }
                    auto is_float = [](uint32_t t) { return t == kF32 || t == kF16 || t == kBF16; };
                    if (is_float(fa->tensor->type) != is_float(fb->tensor->type)) {
                        err = "dense model: " + at + "ssm_alpha and ssm_beta are stored in different kinds of type";
                        return false;
                    }
                    L.ab_quant = !is_float(fa->tensor->type);
                    if (L.ab_quant) {
                        if (!I.upload_mat(fa, p + "ssm_alpha.weight", E, cfg_.ssm_v_heads, L.alpha_q, total, err) ||
                            !I.upload_mat(fb, p + "ssm_beta.weight", E, cfg_.ssm_v_heads, L.beta_q, total, err))
                            return false;
                    } else if (!I.upload_floats(fa, p + "ssm_alpha.weight", (uint64_t) E * cfg_.ssm_v_heads, &L.alpha_w, err) ||
                               !I.upload_floats(fb, p + "ssm_beta.weight", (uint64_t) E * cfg_.ssm_v_heads, &L.beta_w, err))
                        return false;
                }
                const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
                const uint64_t cbytes = (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;
                if (!I.alloc(&L.state, sbytes, err, "GDN state") || !I.alloc(&L.conv_state, cbytes, err, "conv state")) return false;
                if (with_mtp || opt.spec_snapshots) {
                    L.st_slots.assign((size_t) draft_max, nullptr);
                    L.cv_slots.assign((size_t) draft_max, nullptr);
                    for (int j = 0; j < draft_max; ++j)
                        if (!I.alloc(&L.st_slots[(size_t) j], sbytes, err, "GDN state snapshot") ||
                            !I.alloc(&L.cv_slots[(size_t) j], cbytes, err, "conv snapshot"))
                            return false;
                }
            } else {
                if (!I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", E, H * 2 * D, L.q, total, err) ||
                    !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", E, HK * D, L.k, total, err) ||
                    !I.upload_mat(I.find(p + "attn_v.weight"), p + "attn_v.weight", E, HK * D, L.v, total, err) ||
                    !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, E, L.o, total, err))
                    return false;
                if (!I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
                    !I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err))
                    return false;
                const uint64_t kb = (uint64_t) HK * (uint64_t) max_context * kcell, vb = (uint64_t) HK * (uint64_t) max_context * vcell;
                if (!I.alloc(&L.kc, kb, err, "K cache") || !I.alloc(&L.vc, vb, err, "V cache")) return false;
                kv_bytes_ += kb + vb;
            }
        }

        // ---- 7. the output side
        if (cfg_.has_hc()) {
            // The final mixer replaces output_norm: it collapses the residual stack to the one vector the head takes.
            const int HD = cfg_.hc_dim();
            I.gr_shape = strata::kernels::GrShapes{cfg_.n_embd, cfg_.hc, cfg_.hc_lr};
            if (!I.upload_floats(I.find("output_hc_norm.weight"), "output_hc_norm.weight", HD, &I.out_hc_norm, err) ||
                !I.upload_bf16(I.find("output_hc_down.weight"), "output_hc_down.weight",
                               (uint64_t) HD * cfg_.hc_lr, &I.out_hc_down, err) ||
                !I.upload_bf16(I.find("output_hc_up.weight"), "output_hc_up.weight",
                               (uint64_t) HD * cfg_.hc_lr, &I.out_hc_up, err))
                return false;
        } else if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &I.output_norm, err)) {
            return false;
        }
        const Found* head = I.find("output.weight");
        if (!head) head = tok;                                   // tied embeddings
        if (!I.upload_mat(head, head == tok ? "token_embd.weight (tied head)" : "output.weight", E, cfg_.n_vocab,
                          I.head, total, err))
            return false;
        if (with_mtp && !I.mtp.head.dev) I.mtp.head = I.head;    // the MTP block shares the trunk's head
        if (with_mtp && !opt.draft_vocab.empty() && I.mtp.head.dev != I.head.dev)
            std::fprintf(stderr, "xolotl-dense: --draft-vocab ignored: this MTP block has its own head\n");
        else if (with_mtp && draft_bytes) {
            // Copy the shortlist's rows of the head (rows are whole blocks, contiguous) into one smaller matrix.
            const QMat& src = I.mtp.head;
            const Found* hf = I.find("output.weight") ? I.find("output.weight") : tok;
            const int n = (int) opt.draft_vocab.size(), cap = std::max(0, opt.draft_vocab_extra);
            const uint64_t row = native_mmvq_weight_bytes(src.type, E, 1);
            const uint8_t* base = (const uint8_t*) hf->file->tensor_data(*hf->tensor);
            std::vector<uint8_t> rows((size_t) row * (n + cap));
            for (int i = 0; i < n; ++i) {
                const int32_t id = opt.draft_vocab[i];
                if (id < 0 || id >= cfg_.n_vocab) { err = "dense model: --draft-vocab id " + std::to_string(id) + " is outside the vocabulary"; return false; }
                std::memcpy(rows.data() + (size_t) row * i, base + (size_t) row * id, row);
            }
            void* dev = nullptr;
            if (cudaMalloc(&dev, rows.size()) != cudaSuccess || cudaMemcpy(dev, rows.data(), rows.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
                if (dev) cudaFree(dev);
                cudaGetLastError();
                std::fprintf(stderr, "xolotl-dense: --draft-vocab: no VRAM for the %.0f MiB shortlist head, drafts use the full head\n",
                             rows.size() / 1048576.0);
            } else {
                I.allocations.push_back(dev);
                total += rows.size();
                I.draft_head = QMat{dev, src.type, E, n};
                I.draft_ids = opt.draft_vocab;
                I.draft_base = n;
                I.draft_cap = cap;
                I.draft_in.assign((size_t) cfg_.n_vocab, 0);
                for (int32_t id : opt.draft_vocab) I.draft_in[(size_t) id] = 1;
                I.draft_src = base;
                I.draft_row = row;
                std::fprintf(stderr, "xolotl-dense: MTP drafts score a %d-token shortlist + up to %d of each prompt's own tokens "
                             "(%.0f MiB; verification keeps all %d)\n", n, cap, rows.size() / 1048576.0, cfg_.n_vocab);
            }
        }
        weight_bytes_ = total;
        host_bytes_ = I.host_bytes;
        if (I.host_tensors)
            std::fprintf(stderr, "xolotl-dense: %d tensors (%.2f GiB of %.2f) live in host memory and are read over PCIe "
                         "on every token - a shorter --context or a smaller quant puts more on the GPU\n",
                         I.host_tensors, I.host_bytes / 1073741824.0, total / 1073741824.0);

        // ---- 8. activations and scratch (NC columns each)
        const int max_in = std::max({2 * E, cfg_.n_ff, V, H * D});
        const uint64_t f = 4, K = NC;
        bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(max_in, NC), err, "q8_1 scratch") &&
                  I.alloc(&I.x, K * E * f, err, "x") && I.alloc(&I.xn, K * E * f, err, "xn") &&
                  I.alloc(&I.mix, K * E * f, err, "mix") && I.alloc(&I.qkv, K * C * f, err, "qkv") &&
                  I.alloc(&I.h, K * C * f, err, "h") &&
                  I.alloc(&I.alpha, K * cfg_.ssm_v_heads * f, err, "alpha") && I.alloc(&I.beta, K * cfg_.ssm_v_heads * f, err, "beta") &&
                  I.alloc(&I.gate, K * cfg_.ssm_v_heads * f, err, "gate") &&
                  I.alloc(&I.z, K * V * f, err, "z") && I.alloc(&I.y, K * V * f, err, "y") &&
                  I.alloc(&I.q_full, K * H * 2 * D * f, err, "q_full") && I.alloc(&I.qcur, K * H * D * f, err, "qcur") &&
                  I.alloc(&I.kcur, K * HK * D * f, err, "kcur") && I.alloc(&I.vcur, K * HK * D * f, err, "vcur") &&
                  I.alloc(&I.attn, K * H * D * f, err, "attn") && I.alloc(&I.attn32, K * H * D * f, err, "attn32") &&
                  I.alloc(&I.ffn_g, K * cfg_.n_ff * f, err, "ffn_g") && I.alloc(&I.ffn_u, K * cfg_.n_ff * f, err, "ffn_u") &&
                  I.alloc(&I.ffn_h, K * cfg_.n_ff * f, err, "ffn_h") && I.alloc(&I.logits, K * cfg_.n_vocab * f, err, "logits") &&
                  I.alloc(&I.attn_scratch, (uint64_t) NC * dense_attn_scratch_bytes(H, D, (int) max_context), err, "attention scratch") &&
                  // one position per row of a full NC-column pass: the attention's RoPE is batched over the columns
                  I.alloc(&I.pos_dev, K * (uint64_t) std::max(H, HK) * 4, err, "positions") &&
                  I.alloc(&I.hid, K * E * f, err, "hidden") && I.alloc(&I.h_last, (uint64_t) E * f, err, "last hidden") &&
                  I.alloc(&I.d_tok, 16, err, "sampler output") && I.alloc(&I.d_prob, 16, err, "draft probability");
        if (ok && cfg_.has_hc()) {
            // The residual stack and gr_read's explicit scratch, NC columns each (gr_read_multi writes these
            // per token; the single-token path carves the same numbers out of gr_ws_mem).
            const uint64_t HD = (uint64_t) cfg_.hc_dim();
            ok = I.alloc(&I.R, K * HD * f, err, "residual stack") &&
                 I.alloc(&I.hc_xn, K * HD * f, err, "hyper-connection streams") &&
                 I.alloc(&I.hc_gated, K * HD * f, err, "hyper-connection gate") &&
                 I.alloc(&I.hc_lo, K * (uint64_t) cfg_.hc_lr * f, err, "hyper-connection bottleneck") &&
                 I.alloc(&I.hc_inject, K * (uint64_t) cfg_.hc * f, err, "hyper-connection injection") &&
                 I.alloc(&I.gr_ws_mem, strata::kernels::gr_workspace_bytes(I.gr_shape), err, "gr_read workspace");
            if (ok) {
                strata::kernels::gr_workspace_init(I.gr_shape, I.gr_ws_mem, I.gr_ws);
                std::fprintf(stderr, "xolotl-dense: hyper-connections: %d streams of %d, rank %d\n", cfg_.hc, cfg_.n_embd,
                             cfg_.hc_lr);
            }
        }
        if (ok && with_mtp)
            ok = I.alloc(&I.m_e, K * E * f, err, "MTP embeddings") && I.alloc(&I.hin, K * E * f, err, "MTP hidden input") &&
                 I.alloc(&I.cat, K * 2 * E * f, err, "MTP concat") && I.alloc(&I.mtp_logits, (uint64_t) cfg_.n_vocab * f, err, "MTP logits") &&
                 I.alloc(&I.mtp_h, (uint64_t) E * f, err, "MTP output hidden");
        if (ok && !opt.taps.empty() && cfg_.has_hc()) {
            // A tap is one n_embd vector per layer; a hyper-connection layer holds hc of them.  The drafter would
            // have to be trained on the stack, so refuse rather than feed it a shape it was not built for.
            err = "dense model: drafter taps (--dflash2) are not wired for a hyper-connection model";
            return false;
        }
        if (ok && !opt.taps.empty()) {
            I.tap_slot.assign((size_t) cfg_.n_layer, -1);
            for (size_t t = 0; t < opt.taps.size(); ++t) {
                const int l = opt.taps[t];
                if (l < 0 || l >= cfg_.n_layer || I.tap_slot[(size_t) l] >= 0) {
                    err = "dense model: tap layer " + std::to_string(l) + " is outside the model's " + std::to_string(cfg_.n_layer) +
                          " layers or repeated";
                    return false;
                }
                I.tap_slot[(size_t) l] = (int) t;
            }
            I.n_taps = (int) opt.taps.size();
            const uint64_t row = (uint64_t) I.n_taps * E * f;
            ok = I.alloc(&I.taps_run, (uint64_t) NC * row, err, "drafter taps") &&
                 I.alloc(&I.taps_ring, (uint64_t) DenseModel::kTapRing * row, err, "drafter tap ring");
        }
        if (!ok) return false;

        native_gdn_set_enabled(true);
        has_mtp_ = with_mtp;
        draft_max_ = (with_mtp || opt.spec_snapshots) ? draft_max : 0;
        reset();
        std::string serr;
        if (!sync(serr)) { err = serr; return false; }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

void DenseModel::reset() {
    Impl& I = *impl_;
    const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cbytes = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers) {
        if (L.state) cudaMemsetAsync(L.state, 0, sbytes, I.stream);
        if (L.conv_state) cudaMemsetAsync(L.conv_state, 0, cbytes, I.stream);
    }
    if (I.h_last) cudaMemsetAsync(I.h_last, 0, (size_t) cfg_.n_embd * 4, I.stream);
    pos_ = 0;
    snap_valid_ = false;
}

bool DenseModel::sync(std::string& err) {
    const cudaError_t s = cudaStreamSynchronize(impl_->stream);
    if (s != cudaSuccess) { err = std::string("dense model: ") + cudaGetErrorString(s); return false; }
    return true;
}
int64_t DenseModel::ckpt_pos(int slot) const { return slot >= 0 && slot < ckpt_n_ ? impl_->ckpt[(size_t) slot].pos : -1; }
void DenseModel::ckpt_drop(int slot) { if (slot >= 0 && slot < ckpt_n_) impl_->ckpt[(size_t) slot].pos = -1; }

bool DenseModel::ckpt_init(int slots, std::string& err) {
    Impl& I = *impl_;
    for (auto& c : I.ckpt) if (c.host) cudaFreeHost(c.host);
    I.ckpt.clear();
    ckpt_n_ = 0;
    if (slots <= 0) return true;
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    uint64_t n_gdn = 0;
    for (const Layer& L : I.layers) if (!L.attn) ++n_gdn;
    I.ckpt_bytes = n_gdn * (sb + cb) + (uint64_t) cfg_.n_embd * 4;
    I.ckpt.resize((size_t) slots);
    for (int k = 0; k < slots; ++k) {
        void* p = nullptr;
        const cudaError_t s = cudaHostAlloc(&p, I.ckpt_bytes, cudaHostAllocDefault);
        if (s != cudaSuccess) {
            cudaGetLastError();
            err = "dense model: cannot pin " + std::to_string(I.ckpt_bytes >> 20) + " MiB for checkpoint " +
                  std::to_string(k + 1) + ": " + cudaGetErrorString(s);
            for (auto& c : I.ckpt) if (c.host) cudaFreeHost(c.host);
            I.ckpt.clear();
            return false;
        }
        I.ckpt[(size_t) k].host = (uint8_t*) p;
    }
    ckpt_n_ = slots;
    return true;
}

bool DenseModel::ckpt_save(int slot, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || slot >= ckpt_n_) { err = "dense model: no such checkpoint slot"; return false; }
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    uint8_t* at = I.ckpt[(size_t) slot].host;
    for (const Layer& L : I.layers) {
        if (L.attn) continue;
        cudaMemcpyAsync(at, L.state, sb, cudaMemcpyDeviceToHost, I.stream); at += sb;
        cudaMemcpyAsync(at, L.conv_state, cb, cudaMemcpyDeviceToHost, I.stream); at += cb;
    }
    cudaMemcpyAsync(at, I.h_last, (size_t) cfg_.n_embd * 4, cudaMemcpyDeviceToHost, I.stream);
    const cudaError_t s = cudaGetLastError();
    if (s != cudaSuccess) { err = std::string("dense model: checkpoint save: ") + cudaGetErrorString(s); return false; }
    I.ckpt[(size_t) slot].pos = pos_;
    return true;
}

bool DenseModel::ckpt_restore(int slot, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || slot >= ckpt_n_ || I.ckpt[(size_t) slot].pos < 0) { err = "dense model: that checkpoint is empty"; return false; }
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    const uint8_t* at = I.ckpt[(size_t) slot].host;
    for (Layer& L : I.layers) {
        if (L.attn) continue;
        cudaMemcpyAsync(L.state, at, sb, cudaMemcpyHostToDevice, I.stream); at += sb;
        cudaMemcpyAsync(L.conv_state, at, cb, cudaMemcpyHostToDevice, I.stream); at += cb;
    }
    cudaMemcpyAsync(I.h_last, at, (size_t) cfg_.n_embd * 4, cudaMemcpyHostToDevice, I.stream);
    const cudaError_t s = cudaGetLastError();
    if (s != cudaSuccess) { err = std::string("dense model: checkpoint restore: ") + cudaGetErrorString(s); return false; }
    pos_ = I.ckpt[(size_t) slot].pos;
    snap_valid_ = false;
    return true;
}
void DenseModel::set_last_hidden(int col) {
    cudaMemcpyAsync(impl_->h_last, impl_->hid + (size_t) col * cfg_.n_embd, (size_t) cfg_.n_embd * 4,
                    cudaMemcpyDeviceToDevice, impl_->stream);
}

bool DenseModel::rollback(int keep_col, std::string& err) {
    if (!snap_valid_) { err = "dense model: rollback without a snapshot"; return false; }
    if (keep_col < 0 || keep_col >= snap_cols_ - 1) { err = "dense model: rollback column outside the snapshots"; return false; }
    for (Layer& L : impl_->layers) {
        if (L.attn) continue;
        std::swap(L.state, L.st_slots[(size_t) keep_col]);
        std::swap(L.conv_state, L.cv_slots[(size_t) keep_col]);
    }
    pos_ = snap_pos0_ + keep_col + 1;
    snap_valid_ = false;
    return true;
}

bool DenseModel::run(const int32_t* tokens, int n, Logits lg, bool keep_hidden, bool snapshot, std::string& err) {
    Impl& I = *impl_;
    if (n < 1 || n > NC) { err = "dense model: run() takes 1 to " + std::to_string(NC) + " tokens"; return false; }
    if (pos_ + n > max_context_) { err = "dense model: the context is full"; return false; }
    if (snapshot && (draft_max_ == 0 || n - 1 > draft_max_)) { err = "dense model: bad snapshot request (more columns than --draft-max + 1, or no drafter)"; return false; }
    void* s = I.stream;
    const int E = cfg_.n_embd;
    const int64_t pos0 = pos_;
    const bool dbg = I.dbg_from >= 0 && pos0 < I.dbg_to && pos0 + n > I.dbg_from;
    try {
        if (I.prof.enabled) I.prof.begin(s);
        if (!I.upload_embeddings(I.embd, tokens, n, E, cfg_.n_vocab, I.x, err, nullptr, pos0)) { I.prof.on = false; return false; }
        if (dbg) trace_vec("embed", pos0, -1, "", I.x, n * E, s);
        // A hyper-connection model starts as the embedding broadcast to every stream (the reference's own initial
        // condition); a plain one keeps the single vector in `x`.
        if (cfg_.has_hc()) strata::kernels::broadcast_streams(I.x, I.R, E, cfg_.hc, n, s);

        for (int l = 0; l < cfg_.n_layer; ++l) {
            Layer& L = I.layers[(size_t) l];
            if (cfg_.has_hc()) I.hc_read(L, /*attn_mixer=*/true, cfg_.rms_eps, n, s);   // -> xn (the mixer's input)
            else dense_rms_norm(I.x, L.attn_norm, I.xn, n, E, cfg_.rms_eps, s);
            if (!L.attn) I.gdn_block(cfg_, L, n, snapshot, s);
            else I.attn_block(cfg_, L.q, L.k, L.v, L.o, L.q_norm, L.k_norm, L.kc, L.vc, n, (int) pos0, s);
            if (cfg_.has_hc()) {
                strata::kernels::gr_write_multi(I.R, I.mix, I.hc_inject, I.gr_shape, I.R, n, s);
                I.hc_read(L, /*attn_mixer=*/false, cfg_.rms_eps, n, s);                // -> xn again
                I.ffn_matmul(cfg_, L.ffn_gate, L.ffn_up, L.ffn_down, n, s);            // -> mix
                strata::kernels::gr_write_multi(I.R, I.mix, I.hc_inject, I.gr_shape, I.R, n, s);
            } else {
                add_inplace(I.x, I.mix, (int64_t) n * E, s);
                I.ffn_block(cfg_, L.ffn_gate, L.ffn_up, L.ffn_down, L.post_norm, n, s);
            }
            if (I.n_taps && I.tap_slot[(size_t) l] >= 0)   // this layer's output rows for the drafter
                cudaMemcpy2DAsync(I.taps_run + (size_t) I.tap_slot[(size_t) l] * E, (size_t) I.n_taps * E * 4, I.x, (size_t) E * 4,
                                  (size_t) E * 4, (size_t) n, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
            if (dbg) trace_vec("x", pos0, l, L.attn ? "attn" : "gdn", I.x, n * E, s);
        }
        pos_ += n;
        snap_valid_ = snapshot && n > 1;
        snap_pos0_ = pos0;
        snap_cols_ = n;

        if (lg != Logits::None || keep_hidden) {
            if (cfg_.has_hc()) {
                // The final mixer collapses the stack to the one vector the head takes (no output_norm: the
                // mixer's own norm plays that role, exactly as lm_head_mix does on the MoE path).
                strata::kernels::gr_read_multi(I.R, I.out_hc_norm, I.out_hc_down, I.out_hc_up, nullptr, cfg_.rms_eps,
                                               I.gr_shape, I.gr_ws, I.hc_xn, I.hc_lo, I.hc_gated, I.hid, nullptr, n, s);
            } else {
                dense_rms_norm(I.x, I.output_norm, I.hid, n, E, cfg_.rms_eps, s);     // h_nextn: the post-norm hidden
            }
            if (lg == Logits::Last) {
                I.quantize(I.hid + (size_t) (n - 1) * E, E, 1, s);
                I.gemv(I.head, I.logits, 1, s);
            } else if (lg == Logits::All) {
                I.quantize(I.hid, E, n, s);
                I.gemv(I.head, I.logits, n, s);
            }
            if (dbg && lg != Logits::None) trace_logits(pos0 + n - 1, I.logits, cfg_.n_vocab, s);
        }
        if (I.prof.on) I.prof.finish(s, n);
        return true;
    } catch (const std::exception& e) {
        I.prof.on = false;
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

// ============================================================================ batched prompt processing
//
// run() serves at most kMaxCols columns, so a prompt reads every weight once per eight tokens.  prefill() walks the
// prompt in chunks of prefill_chunk() tokens instead: the projections go through MMQ (the weights stay in their GGUF
// blocks, the activations are rounded to q8_1, the products run on int8 tensor cores) and the order-dependent steps
// walk the chunk's tokens inside one launch.  The recurrent state, the KV cache and the position advance exactly as
// they do through run(), so the logits match.
bool DenseModel::prefill_ready() const { return pf_ready_; }
int64_t DenseModel::prefill_chunk() const { return pf_chunk_; }

bool DenseModel::prefill_setup(int64_t chunk, std::string& err) {
    pf_ready_ = false;
    pf_chunk_ = 0;
    if (cfg_.has_hc()) {
        // The chunked path runs rms_norm over a single vector; a hyper-connection file mixes hc streams first.
        // run() serves the prompt instead - correct, NC tokens per call, and nothing else changes.
        err = "dense model: hyper-connections are served on the run() path; the prompt is fed NC tokens per call";
        return false;
    }
#ifdef STRATA_DENSE_MMQ
    Impl& I = *impl_;
    if (chunk <= 0) chunk = kPrefillDefaultCols;
    chunk = std::max<int64_t>(NC, std::min<int64_t>(chunk, kPrefillMaxCols));
    const int E = cfg_.n_embd, C = cfg_.conv_channels(), V = cfg_.value_dim();
    const int H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;

    // Every projection the prompt path runs has to be a type MMQ covers; if one is not, the caller feeds the prompt
    // through run() and loses nothing.
    auto mat_ok = [&](const QMat& w, const char* what) {
        if (w.dev && !strata::prefill::mmq::supported(w.type)) {
            err = std::string("dense model: prefill cannot multiply ") + what + " of GGUF type " +
                  std::to_string(w.type) + "; feed the prompt with run() instead";
            return false;
        }
        return true;
    };
    for (size_t l = 0; l < I.layers.size(); ++l) {
        const Layer& L = I.layers[l];
        const std::string p = "layer " + std::to_string(l);
        if (!L.attn) {
            if (!mat_ok(L.qkv, (p + " attn_qkv").c_str()) || !mat_ok(L.z, (p + " attn_gate").c_str()) ||
                !mat_ok(L.ssm_out, (p + " ssm_out").c_str()))
                return false;
            if (L.ab_quant && (!mat_ok(L.alpha_q, (p + " ssm_alpha").c_str()) ||
                               !mat_ok(L.beta_q, (p + " ssm_beta").c_str())))
                return false;
        } else if (!mat_ok(L.q, (p + " attn_q").c_str()) || !mat_ok(L.k, (p + " attn_k").c_str()) ||
                   !mat_ok(L.v, (p + " attn_v").c_str()) || !mat_ok(L.o, (p + " attn_output").c_str()))
            return false;
        if (!mat_ok(L.ffn_gate, (p + " ffn_gate").c_str()) || !mat_ok(L.ffn_up, (p + " ffn_up").c_str()) ||
            !mat_ok(L.ffn_down, (p + " ffn_down").c_str()))
            return false;
    }

    // Size the scratch for one chunk.  A bigger chunk reads every weight fewer times, so the caller halves the
    // chunk when the scratch does not fit rather than switching the path off: even 64 tokens at a time is many
    // times faster than run().
    auto size_scratch = [&](int64_t t) -> bool {
    I.pf.reset(new Impl::Pf);
    Impl::Pf& P = *I.pf;
    P.T = t;
    const uint64_t f = 4, T = (uint64_t) t;
    const int max_in = std::max({E, V, cfg_.n_ff, H * D});
    auto pf_alloc = [&](void** out, uint64_t bytes, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate the prefill scratch ") + what + " (" +
                  std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
            return false;
        }
        P.allocs.push_back(p);
        *out = p;
        return true;
    };
    bool ok = pf_alloc((void**) &P.x, T * E * f, "x") && pf_alloc((void**) &P.xn, T * E * f, "xn") &&
              pf_alloc((void**) &P.mix, T * E * f, "mix") && pf_alloc((void**) &P.qkv, T * C * f, "qkv") &&
              pf_alloc((void**) &P.h, T * C * f, "h") && pf_alloc((void**) &P.alpha, T * cfg_.ssm_v_heads * f, "alpha") &&
              pf_alloc((void**) &P.beta, T * cfg_.ssm_v_heads * f, "beta") &&
              pf_alloc((void**) &P.gate, T * cfg_.ssm_v_heads * f, "gate") && pf_alloc((void**) &P.z, T * V * f, "z") &&
              pf_alloc((void**) &P.y, T * V * f, "y") && pf_alloc((void**) &P.q_full, T * H * 2 * D * f, "q_full") &&
              pf_alloc((void**) &P.qcur, T * H * D * f, "qcur") && pf_alloc((void**) &P.kcur, T * HK * D * f, "kcur") &&
              pf_alloc((void**) &P.vcur, T * HK * D * f, "vcur") && pf_alloc((void**) &P.attn, T * H * D * f, "attn") &&
              pf_alloc((void**) &P.attn32, T * H * D * f, "attn32") &&
              pf_alloc((void**) &P.ffn_g, T * cfg_.n_ff * f, "ffn_g") &&
              pf_alloc((void**) &P.ffn_u, T * cfg_.n_ff * f, "ffn_u") &&
              pf_alloc((void**) &P.ffn_h, T * cfg_.n_ff * f, "ffn_h") &&
              pf_alloc((void**) &P.xq, strata::prefill::mmq::q8_bytes(t, max_in), "q8_1") &&
              pf_alloc((void**) &P.ids, T * 4, "row ids") && pf_alloc((void**) &P.bounds, 2 * 4, "group bounds") &&
              pf_alloc((void**) &P.pos_dev, T * (uint64_t) std::max(H, HK) * 4, "positions");
    if (ok) {
        cudaError_t s = cudaHostAlloc((void**) &P.stage, T * E * 4, cudaHostAllocDefault);
        if (s != cudaSuccess)
            err = std::string("dense model: cannot pin the prefill embedding stage: ") + cudaGetErrorString(s);
        else s = cudaHostAlloc((void**) &P.bounds_host, Impl::Pf::kBoundsSlots * 2 * sizeof(int32_t),
                               cudaHostAllocDefault);
        if (s != cudaSuccess)
            err = std::string("dense model: cannot pin the prefill group bounds: ") + cudaGetErrorString(s);
        ok = s == cudaSuccess;
    }
    if (ok) {
        strata::prefill::mmq::iota(P.ids, t, I.stream);
        for (int k = 0; k < Impl::Pf::kBoundsSlots; ++k) {
            P.bounds_host[2 * k] = 0;
            P.bounds_host[2 * k + 1] = (int32_t) t;
        }
        const cudaError_t s = cudaMemcpyAsync(P.bounds, P.bounds_host, 2 * sizeof(int32_t), cudaMemcpyHostToDevice,
                                              I.stream);
        if (s != cudaSuccess) err = std::string("dense model: prefill tables: ") + cudaGetErrorString(s);
        ok = s == cudaSuccess;
    }
    if (!ok) I.pf.reset();   // pf_alloc already named the buffer that did not fit
    return ok;
    };
    for (;;) {
        if (size_scratch(chunk)) break;
        if (chunk <= NC) return false;
        std::fprintf(stderr, "xolotl-dense: the prefill scratch for %lld tokens does not fit; trying %lld\n",
                     (long long) chunk, (long long) (chunk / 2));
        chunk /= 2;
    }
    pf_chunk_ = chunk;
    pf_ready_ = true;
    std::fprintf(stderr, "xolotl-dense: prompt prefill through MMQ, %lld tokens at a time\n", (long long) chunk);
    return true;
#else
    (void) chunk;
    err = "dense model: this build has no MMQ, so prefill() is unavailable; feed the prompt with run()";
    return false;
#endif
}

bool DenseModel::prefill(const int32_t* tokens, int64_t n, Logits lg, std::string& err) {
#ifdef STRATA_DENSE_MMQ
    Impl& I = *impl_;
    if (!pf_ready_) { err = "dense model: prefill() without prefill_setup()"; return false; }
    if (n < 1) { err = "dense model: prefill() takes at least one token"; return false; }
    if (pos_ + n > max_context_) { err = "dense model: the context is full"; return false; }
    Impl::Pf& P = *I.pf;
    void* s = I.stream;
    const int E = cfg_.n_embd;
    int64_t last_rows = 0;
    try {
        if (I.prof.enabled) I.prof.begin(s);
        for (int64_t i = 0; i < n;) {
            const int64_t T = std::min(pf_chunk_, n - i);
            const int64_t pos0 = pos_;
            if (!I.upload_embeddings(I.embd, tokens + i, (int) T, E, cfg_.n_vocab, P.x, err, P.stage, pos0)) {
                I.prof.on = false;
                return false;
            }
            // One MMQ group: the whole chunk.  The bounds go into a fresh pinned slot every time, so the write can
            // never race a copy the stream has not made yet, and a short final chunk cannot leave stale bounds
            // behind for the next call.
            int32_t* slot = P.bounds_host + 2 * (P.bounds_slot++ % Impl::Pf::kBoundsSlots);
            slot[0] = 0;
            slot[1] = (int32_t) T;
            cudaMemcpyAsync(P.bounds, slot, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) s);
            for (int l = 0; l < cfg_.n_layer; ++l) {
                Layer& L = I.layers[(size_t) l];
                dense_rms_norm(P.x, L.attn_norm, P.xn, (int) T, E, cfg_.rms_eps, s);
                if (!L.attn) I.pf_gdn_block(cfg_, L, T, s);
                else I.pf_attn_block(cfg_, L.q, L.k, L.v, L.o, L.q_norm, L.k_norm, L.kc, L.vc, T, (int) pos0, s);
                add_inplace(P.x, P.mix, T * E, s);
                I.pf_ffn_block(cfg_, L.ffn_gate, L.ffn_up, L.ffn_down, L.post_norm, T, s);
                if (I.n_taps && I.tap_slot[(size_t) l] >= 0) {   // the drafter's rows: the chunk's last kTapRing, at position % ring
                    const int64_t R = DenseModel::kTapRing;
                    for (int64_t r = std::max<int64_t>(0, T - R); r < T;) {
                        const int64_t slot = (pos0 + r) % R, len = std::min<int64_t>(T - r, R - slot);
                        cudaMemcpy2DAsync(I.taps_ring + ((size_t) slot * I.n_taps + I.tap_slot[(size_t) l]) * E, (size_t) I.n_taps * E * 4,
                                          P.x + (size_t) r * E, (size_t) E * 4, (size_t) E * 4, (size_t) len,
                                          cudaMemcpyDeviceToDevice, (cudaStream_t) s);
                        r += len;
                    }
                }
            }
            // The MTP block is fed in kMaxCols groups, exactly as the prompt loop does after a run(): each group's
            // column 0 pairs with the previous group's last hidden (h_last, which mtp_ingest leaves behind).
            if (has_mtp_) {
                for (int64_t off = 0; off < T; off += NC) {
                    const int m = (int) std::min<int64_t>(NC, T - off);
                    dense_rms_norm(P.x + (size_t) off * E, I.output_norm, I.hid, m, E, cfg_.rms_eps, s);
                    if (!mtp_ingest(tokens + i + off, m, (int) (pos0 + off), 0, err)) {
                        I.prof.on = false;
                        return false;
                    }
                }
            }
            last_rows = T;
            pos_ += T;
            i += T;
        }
        snap_valid_ = false;
        snap_cols_ = 0;
        if (lg == Logits::Last) {                       // the head runs once, over the last chunk's last row
            dense_rms_norm(P.x + (size_t) (last_rows - 1) * E, I.output_norm, I.xn, 1, E, cfg_.rms_eps, s);
            I.quantize(I.xn, E, 1, s);
            I.gemv(I.head, I.logits, 1, s);
        }
        if (I.prof.on) I.prof.finish(s, (int) std::min<int64_t>(n, INT_MAX));
        return true;
    } catch (const std::exception& e) {
        I.prof.on = false;
        err = std::string("dense model: prefill: ") + e.what();
        return false;
    }
#else
    (void) tokens; (void) n; (void) lg;
    err = "dense model: prefill() is not available in this build";
    return false;
#endif
}

bool DenseModel::mtp_ingest(const int32_t* tokens, int n, int pos0, int start, std::string& err) {
    Impl& I = *impl_;
    if (!has_mtp_) { err = "dense model: no MTP block loaded"; return false; }
    const int m = n - start;
    if (start < 0 || m < 1 || n > NC) { err = "dense model: bad MTP ingest range"; return false; }
    if ((int64_t) pos0 + n > max_context_) { err = "dense model: the context is full"; return false; }
    void* s = I.stream;
    const int E = cfg_.n_embd;
    try {
        if (!I.upload_embeddings(I.embd_mtp, tokens + start, m, E, cfg_.n_vocab, I.m_e, err)) return false;
        for (int j = start; j < n; ++j) {            // hidden[j-1]; column 0 pairs with the hidden before this run
            const float* src = (j == 0) ? I.h_last : I.hid + (size_t) (j - 1) * E;
            cudaMemcpyAsync(I.hin + (size_t) (j - start) * E, src, (size_t) E * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
        }
        for (int jj = 0; jj < m; ++jj) {
            dense_rms_norm(I.m_e + (size_t) jj * E, I.mtp.enorm, I.cat + (size_t) jj * 2 * E, 1, E, cfg_.rms_eps, s);       // e first,
            dense_rms_norm(I.hin + (size_t) jj * E, I.mtp.hnorm, I.cat + (size_t) jj * 2 * E + E, 1, E, cfg_.rms_eps, s);   // then h
        }
        I.mtp_block(cfg_, m, pos0 + start, false, s);
        cudaMemcpyAsync(I.h_last, I.hid + (size_t) (n - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: MTP: ") + e.what();
        return false;
    }
}

void DenseModel::draft_vocab_extend(const int32_t* tokens, int64_t n) {
    Impl& I = *impl_;
    if (!I.draft_head.dev || I.draft_cap == 0) return;
    std::vector<int32_t> extra;
    std::vector<uint8_t> seen = I.draft_in;
    for (int64_t i = n - 1; i >= 0 && (int) extra.size() < I.draft_cap; --i) {     // newest first
        const int32_t t = tokens[i];
        if (t < 0 || t >= cfg_.n_vocab || seen[(size_t) t]) continue;
        seen[(size_t) t] = 1;
        extra.push_back(t);
    }
    // pad to a multiple of 64 rows with copies of a shortlist token (a duplicate row has the same logit, so the
    // argmax maps to the same id) - the GEMV launchers then see the same row-count shapes as the shortlist itself
    while (!extra.empty() && (I.draft_base + extra.size()) % 64 && (int) extra.size() < I.draft_cap) extra.push_back(I.draft_ids[0]);
    if (extra == I.draft_extra) return;             // the same conversation: the rows are already there
    I.draft_stage.resize((size_t) I.draft_row * extra.size());
    for (size_t k = 0; k < extra.size(); ++k)
        std::memcpy(I.draft_stage.data() + I.draft_row * k, I.draft_src + I.draft_row * (size_t) extra[k], I.draft_row);
    if (!extra.empty())
        cudaMemcpyAsync((uint8_t*) I.draft_head.dev + I.draft_row * (size_t) I.draft_base, I.draft_stage.data(),
                        I.draft_stage.size(), cudaMemcpyHostToDevice, (cudaStream_t) I.stream);
    cudaStreamSynchronize((cudaStream_t) I.stream);
    I.draft_ids.resize((size_t) I.draft_base);
    I.draft_ids.insert(I.draft_ids.end(), extra.begin(), extra.end());
    I.draft_head.n_out = I.draft_base + (int) extra.size();
    I.draft_extra = std::move(extra);
}

bool DenseModel::mtp_draft(int32_t token, int pos, int max_n, float p_min, int32_t* out, int* n_out, std::string& err) {
    Impl& I = *impl_;
    *n_out = 0;
    if (!has_mtp_) { err = "dense model: no MTP block loaded"; return false; }
    void* s = I.stream;
    const int E = cfg_.n_embd;
    try {
        int32_t tok = token;
        for (int i = 0; i < max_n; ++i) {
            if ((int64_t) pos + i >= max_context_) break;
            // step 0 pairs the token with the trunk's last hidden; later steps chain on the block's own output
            if (!I.upload_embeddings(I.embd_mtp, &tok, 1, E, cfg_.n_vocab, I.m_e, err)) return false;
            dense_rms_norm(I.m_e, I.mtp.enorm, I.cat, 1, E, cfg_.rms_eps, s);
            dense_rms_norm(i == 0 ? I.h_last : I.mtp_h, I.mtp.hnorm, I.cat + E, 1, E, cfg_.rms_eps, s);
            I.mtp_block(cfg_, 1, pos + i, true, s);
            dense_argmax_prob(I.mtp_logits, I.draft_head.dev ? I.draft_head.n_out : cfg_.n_vocab, I.d_tok, I.d_prob, s);
            int id = -1;
            float prob = 0.0f;
            cudaMemcpyAsync(&id, I.d_tok, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t) s);
            cudaMemcpyAsync(&prob, I.d_prob, sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t) s);
            if (!sync(err)) return false;
            if (prob < p_min) break;                    // not sure enough: this token is not offered
            if (I.draft_head.dev) id = I.draft_ids[id]; // shortlist row -> token id (prob is over the shortlist)
            out[(*n_out)++] = id;
            tok = id;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: MTP: ") + e.what();
        return false;
    }
}

}  // namespace strata::core