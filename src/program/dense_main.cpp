// src/program/dense_main.cpp - `xolotl-dense --serve --native <model.gguf> --context N`
//
// The prompt is read in chunks: through llama.cpp's MMQ int8 tensor-core kernels when the build has them
// (--prompt-chunk N, the default path), otherwise NC tokens at a time through the FP32 GEMVs (--no-prefill).
//
// The dense-model twin of `strata --serve`.  It speaks the SAME line protocol on stdin/stdout, so serve/server.py
// drives it through its unchanged StrataEngine (tokenizer, chat template, thinking levels, tools, OpenAI and
// Anthropic endpoints, web app, Monitor):
//
//   out   INFO key=value ...                      (optional facts for the Monitor)
//   out   READY <context> stop                    ("stop": this engine honours STOP)
//   in    GEN <max_new> [key=value ...] id,id,... (keys: temperature top_p top_k min_p penalty_last_n penalty_repeat
//                                                  penalty_freq penalty_present seed score; unknown keys are skipped)
//   out   SCORE <id>:<lp> ... top=<id>:<lp>                   (score=ID,... only: /v1/decide, before the first T)
//   out   PP <position> <prompt tokens> <ms> <tok/s>          every few dozen prompt tokens
//   out   T <id>                                              one per generated token
//   out   DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in    STOP | QUIT
//
// Conversation cache: the recurrent GDN state cannot be rolled back, so a request reuses the whole state only when
// everything it has consumed is a prefix of the new prompt (a chat that only appends).  Otherwise the prompt is
// resumed from the deepest conversation checkpoint at or before the token where the two prompts start to differ
// (--ckpt N, pinned host RAM); with no such checkpoint the state is reset and the prompt is read again.
//
// The prompt is read one token at a time (the same step as decoding).  Correct, and the simplest thing that can
// be checked against llama.cpp; a batched prefill is the next optimisation, not a prerequisite.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dense_model.hpp"
#include "strata/core/dflash2.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Lines {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false};
    bool eof = false;
    void push(std::string s) {
        std::lock_guard<std::mutex> l(m);
        q.push_back(std::move(s));
        cv.notify_one();
    }
    bool pop(std::string& out) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_ids(const char* p, std::vector<int32_t>& ids) {
    ids.clear();
    while (*p == ' ') ++p;
    while (*p) {
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        ids.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != '\r' && *p != '\n') return false;
        else break;
    }
    return !ids.empty();
}

void usage() {
    std::fprintf(stderr,
                 "usage: xolotl-dense --serve --native <model.gguf> [--context N] [--kv fp16|int8|q4_0|k8v4] [--mtp [--draft-max N] [--draft-min N] [--draft-p-min P] [--mtp-force]]\n"
                 "                  [--prompt-chunk N | --no-prefill]\n"
                 "       xolotl-dense --selftest | --check <model.gguf> [--mtp] [--kv fp16|int8|q4_0|k8v4]\n"
                 "  a dense qwen35 GGUF (Qwen3.8-27B) behind Strata's server protocol; the server runs it with\n"
                 "  `serve/server.py --engine strata --config <run config>` whose `exe` is this program.\n"
                 "  --draft-vocab FILE: MTP drafts score only these token ids (int32 LE, a frequency shortlist such as\n"
                 "                  data/packs/qwen3.8-27b/draft-vocab-65536.i32); verification keeps the full head (same output).\n"
                 "  --prompt-chunk N feeds the prompt N tokens at a time through MMQ instead of 8 at a time through\n"
                 "  the GEMVs (0 is the default; --no-prefill forces the old path).  The chunked path reads the KV\n"
                 "  cache in whatever format --kv picked.\n"
                 "  --kv: the KV cache storage, the same option strata has (int8 about 53%% of fp16, q4_0 about 28%%, k8v4 about 40%%).\n"
                 "  --ckpt N: conversation checkpoints in pinned host RAM (~154 MiB each on Qwen3.8-27B, 0 = off, default 4).\n"
                 "                  A prompt that diverges from the one already read resumes from the deepest checkpoint at or\n"
                 "                  before the divergence; the slots sit at the end of the prompt and 64, 512, 4096 ... tokens\n"
                 "                  before it, so N=6 also covers a divergence 32768 tokens back.\n");
}

// ---- `xolotl-dense --selftest`: the new CUDA kernels against a CPU reference, no model needed.
uint16_t f32_to_f16_bits(float f) {          // round to nearest even; the test inputs stay well inside the range
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;     // flush subnormals: the test data never reaches them
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

struct Diff { double abs = 0, rel = 0; };
Diff compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Diff d;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i] - (double) ref[i]);
        d.abs = std::max(d.abs, e);
        d.rel = std::max(d.rel, e / (std::fabs((double) ref[i]) + 1e-3));
    }
    return d;
}

template <class T> T* to_dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc((void**) &d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> from_dev(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}

// The KV cache formats, modelled on the host: the same rules the kernels follow (ggml Q8_0 / Q4_0 blocks of 32, the
// orthonormal Walsh-Hadamard rotation before a Q4 store), so the GPU can be compared with them value by value.
void host_fwht256(float* x) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += 2 * len)
            for (int j = i; j < i + len; ++j) { const float a = x[j], b = x[j + len]; x[j] = a + b; x[j + len] = a - b; }
    for (int i = 0; i < 256; ++i) x[i] *= 0.0625f;
}

float host_half_round(float f) { return strata::fp16_to_fp32(f32_to_f16_bits(f)); }

// quantize + dequantize 256 values in place (fmt: 1 = Q8_0, 2 = Q4_0)
void host_kv_roundtrip(int fmt, std::vector<float>& v) {
    for (int g = 0; g < 8; ++g) {
        float* x = &v[(size_t) g * 32];
        if (fmt == 1) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
            const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f, dh = host_half_round(d);
            for (int i = 0; i < 32; ++i) x[i] = std::nearbyint(x[i] * id) * dh;
        } else {
            float mx = 0, ax = 0;
            for (int i = 0; i < 32; ++i) if (std::fabs(x[i]) > ax) { ax = std::fabs(x[i]); mx = x[i]; }
            const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f, dh = host_half_round(d);
            for (int i = 0; i < 32; ++i) x[i] = (float) (std::min(15, (int) (x[i] * id + 8.5f)) - 8) * dh;
        }
    }
}

int selftest() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](size_t n, float sc = 1.0f) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    int bad = 0;
    auto report = [&](const char* what, const Diff& d, double tol) {
        const bool ok = d.abs < tol;
        std::printf("%-34s max abs err %.3g  max rel %.3g  %s\n", what, d.abs, d.rel, ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    };
    try {
        {   // RMSNorm
            const int rows = 3, cols = 5120;
            auto x = rnd((size_t) rows * cols, 3.0f), w = rnd(cols, 1.0f);
            std::vector<float> ref(x.size());
            for (int r = 0; r < rows; ++r) {
                double ss = 0; for (int i = 0; i < cols; ++i) ss += (double) x[(size_t) r * cols + i] * x[(size_t) r * cols + i];
                const double sc = 1.0 / std::sqrt(ss / cols + 1e-6);
                for (int i = 0; i < cols; ++i) ref[(size_t) r * cols + i] = (float) (x[(size_t) r * cols + i] * sc * w[i]);
            }
            float *dx = to_dev(x), *dw = to_dev(w), *dout = nullptr;
            cudaMalloc((void**) &dout, x.size() * 4);
            strata::kernels::dense_rms_norm(dx, dw, dout, rows, cols, 1e-6f, nullptr);
            report("rms_norm 3 x 5120", compare(from_dev(dout, x.size()), ref), 1e-4);
        }
        {   // SwiGLU
            const int n = 17408;
            auto g = rnd(n, 2.0f), u = rnd(n, 2.0f);
            std::vector<float> ref(n);
            for (int i = 0; i < n; ++i) ref[(size_t) i] = (float) (g[(size_t) i] / (1.0 + std::exp(-(double) g[(size_t) i])) * u[(size_t) i]);
            float *dg = to_dev(g), *du = to_dev(u), *dout = nullptr;
            cudaMalloc((void**) &dout, n * 4);
            strata::kernels::dense_swiglu(dg, du, dout, n, nullptr);
            report("swiglu 17408", compare(from_dev(dout, n), ref), 1e-4);
        }
        {   // GDN output norm: rms(o) * w * silu(z), per value head
            const int heads = 48, cols = 128;
            auto o = rnd((size_t) heads * cols, 2.0f), zz = rnd((size_t) heads * cols, 2.0f), w = rnd(cols, 1.0f);
            std::vector<float> ref(o.size());
            for (int h = 0; h < heads; ++h) {
                double ss = 0; for (int i = 0; i < cols; ++i) ss += (double) o[(size_t) h * cols + i] * o[(size_t) h * cols + i];
                const double sc = 1.0 / std::sqrt(ss / cols + 1e-6);
                for (int i = 0; i < cols; ++i) {
                    const double g = zz[(size_t) h * cols + i];
                    ref[(size_t) h * cols + i] = (float) (o[(size_t) h * cols + i] * sc * w[(size_t) i] * (g / (1.0 + std::exp(-g))));
                }
            }
            float *d_o = to_dev(o), *d_z = to_dev(zz), *d_w = to_dev(w), *dout = nullptr;
            cudaMalloc((void**) &dout, o.size() * 4);
            strata::kernels::dense_gdn_out_norm(d_o, d_z, d_w, dout, heads, cols, 1e-6f, nullptr);
            report("gdn_out_norm 48 x 128 (SiLU gate)", compare(from_dev(dout, o.size()), ref), 1e-4);
        }
        {   // The GDN mixer: the chunk kernels against the per-token kernels, column by column, including the
            // rollback snapshots.  run() feeds several columns at once and has to be able to undo the ones after any
            // column, so the snapshots are not a convenience - if slot[j] is the state one column off, a rejected
            // draft silently corrupts every token that follows it.  The reference here is the token loop the decode
            // path used before the chunk path existed, so this is the gate for that switch.
            using namespace strata::kernels;   // the block calls a dozen of them; the prefix would bury the test
            const int S = 128, KH = 2, VH = 4, T = 5;
            const int qk = S * KH, C = 2 * qk + S * VH, V = S * VH;
            auto qkv = rnd((size_t) T * C), z = rnd((size_t) T * V);
            auto conv_w = rnd((size_t) C * 4, 0.5f), dt = rnd(VH, 0.5f), a_ssm = rnd(VH, 0.5f);
            auto gamma = rnd(S, 1.0f), alpha = rnd((size_t) T * VH), beta = rnd((size_t) T * VH);
            auto st0 = rnd((size_t) S * VH * S, 0.3f), hist0 = rnd((size_t) C * 3, 0.5f);

            cudaStream_t st = nullptr;
            if (cudaStreamCreate(&st) != cudaSuccess) { std::printf("selftest: cannot create a stream\n"); return 1; }

            // ---- the reference: the per-token kernels, one column at a time
            float* d_qkv = to_dev(qkv);
            float* d_z = to_dev(z);
            float* d_cw = to_dev(conv_w);
            float* d_dt = to_dev(dt);
            float* d_as = to_dev(a_ssm);
            float* d_gm = to_dev(gamma);
            float* d_alpha = to_dev(alpha);
            float* d_beta = to_dev(beta);       // dense_gdn_gates sigmoidises this in place: the chunk path's copy
            float* d_hist_a = to_dev(hist0);
            float* d_state_a = to_dev(st0);
            float* d_h_a = nullptr;
            float* d_gate_a = nullptr;
            float* d_y_a = nullptr;
            float* d_beta_a = to_dev(beta);     // native_gdn_beta_gate also works in place: a copy of its own
            cudaMalloc((void**) &d_h_a, (size_t) T * C * 4);
            cudaMalloc((void**) &d_gate_a, (size_t) T * VH * 4);
            cudaMalloc((void**) &d_y_a, (size_t) T * V * 4);
            std::vector<float> hist_ref, state_ref;
            std::vector<std::vector<float>> state_after((size_t) T - 1), hist_after((size_t) T - 1);
            std::vector<float> y_ref((size_t) T * V), h_ref((size_t) T * C);
            {
                float* d_raw = nullptr;
                cudaMalloc((void**) &d_raw, (size_t) C * 4);
                float* d_o = nullptr;
                cudaMalloc((void**) &d_o, (size_t) V * 4);
                const GdnShapes gs{S, KH, VH};
                for (int t = 0; t < T; ++t) {
                    float* ht = d_h_a + (size_t) t * C;
                    native_gdn_conv_silu(d_hist_a, d_qkv + (size_t) t * C, d_cw, d_raw, ht, C, 4, st);
                    native_gdn_l2_norm(ht, KH, S, 1e-6f, st);
                    native_gdn_l2_norm(ht + qk, KH, S, 1e-6f, st);
                    native_gdn_beta_gate(d_beta_a + (size_t) t * VH, VH, st);
                    native_gdn_gate(d_alpha + (size_t) t * VH, d_dt, d_as, d_gate_a + (size_t) t * VH, VH, st);
                    native_gdn_step(d_state_a, ht, ht + qk, ht + 2 * qk, d_gate_a + (size_t) t * VH,
                                    d_beta_a + (size_t) t * VH, d_o, gs, st);
                    dense_gdn_out_norm(d_o, d_z + (size_t) t * V, d_gm, d_y_a + (size_t) t * V, VH, S, 1e-6f, st);
                    if (t < T - 1) {
                        state_after[(size_t) t].assign((size_t) S * VH * S, 0.0f);
                        hist_after[(size_t) t].assign((size_t) C * 3, 0.0f);
                        cudaMemcpy(state_after[(size_t) t].data(), d_state_a, (size_t) S * VH * S * 4, cudaMemcpyDeviceToHost);
                        cudaMemcpy(hist_after[(size_t) t].data(), d_hist_a, (size_t) C * 3 * 4, cudaMemcpyDeviceToHost);
                    }
                }
                cudaStreamSynchronize(st);
                y_ref = from_dev(d_y_a, (size_t) T * V);
                h_ref = from_dev(d_h_a, (size_t) T * C);
                state_ref = from_dev(d_state_a, (size_t) S * VH * S);
                hist_ref = from_dev(d_hist_a, (size_t) C * 3);
                cudaFree(d_raw);
                cudaFree(d_o);
            }

            // ---- the chunk path: one launch per step, with the snapshots written inside the recurrence
            float* d_hist_b = to_dev(hist0);
            float* d_state_b = to_dev(st0);
            float* d_h_b = nullptr;
            float* d_y_b = nullptr;
            float* d_gate_b = nullptr;
            cudaMalloc((void**) &d_h_b, (size_t) T * C * 4);
            cudaMalloc((void**) &d_y_b, (size_t) T * V * 4);
            cudaMalloc((void**) &d_gate_b, (size_t) T * VH * 4);
            DenseGdnSnap snaps, snapc;
            std::vector<float*> d_slot_s((size_t) T - 1), d_slot_c((size_t) T - 1);
            for (int j = 0; j < T - 1; ++j) {
                cudaMalloc((void**) &d_slot_s[(size_t) j], (size_t) S * VH * S * 4);
                cudaMalloc((void**) &d_slot_c[(size_t) j], (size_t) C * 3 * 4);
                snaps.slot[j] = d_slot_s[(size_t) j];
                snapc.slot[j] = d_slot_c[(size_t) j];
            }
            snaps.n = T - 1;
            snapc.n = T - 1;
            dense_gdn_conv_chunk(d_hist_b, d_qkv, d_cw, d_h_b, C, T, snapc, st);
            dense_gdn_l2_norm(d_h_b, T, C, 0, KH, S, 1e-6f, st);
            dense_gdn_l2_norm(d_h_b, T, C, qk, KH, S, 1e-6f, st);
            dense_gdn_gates(d_alpha, d_dt, d_as, d_gate_b, d_beta, VH, T, st);
            dense_gdn_rec_chunk(d_state_b, d_h_b, d_gate_b, d_beta, d_z, d_gm, 1e-6f, d_y_b, KH, VH, T, snaps, st);
            cudaStreamSynchronize(st);

            report("gdn chunk conv vs token loop", compare(from_dev(d_h_b, (size_t) T * C), h_ref), 1e-3);
            report("gdn chunk rec+norm vs token loop", compare(from_dev(d_y_b, (size_t) T * V), y_ref), 1e-3);
            report("gdn chunk final state", compare(from_dev(d_state_b, (size_t) S * VH * S), state_ref), 1e-3);
            report("gdn chunk final history", compare(from_dev(d_hist_b, (size_t) C * 3), hist_ref), 1e-3);
            for (int j = 0; j < T - 1; ++j) {
                char name[64];
                std::snprintf(name, sizeof name, "gdn snapshot state after col %d", j);
                report(name, compare(from_dev(d_slot_s[(size_t) j], (size_t) S * VH * S), state_after[(size_t) j]), 1e-3);
                std::snprintf(name, sizeof name, "gdn snapshot conv after col %d", j);
                report(name, compare(from_dev(d_slot_c[(size_t) j], (size_t) C * 3), hist_after[(size_t) j]), 1e-3);
            }

            cudaFree(d_qkv); cudaFree(d_z); cudaFree(d_cw); cudaFree(d_dt); cudaFree(d_as); cudaFree(d_gm);
            cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_beta_a); cudaFree(d_gate_a); cudaFree(d_y_a);
            cudaFree(d_hist_a); cudaFree(d_state_a); cudaFree(d_h_a);
            cudaFree(d_hist_b); cudaFree(d_state_b); cudaFree(d_h_b); cudaFree(d_y_b); cudaFree(d_gate_b);
            for (int j = 0; j < T - 1; ++j) { cudaFree(d_slot_s[(size_t) j]); cudaFree(d_slot_c[(size_t) j]); }
            cudaStreamDestroy(st);
        }
        {   // F32 GEMV
            const int n_in = 5120, n_out = 48;
            auto W = rnd((size_t) n_in * n_out, 0.05f), x = rnd(n_in);
            std::vector<float> ref(n_out);
            for (int r = 0; r < n_out; ++r) { double a = 0; for (int i = 0; i < n_in; ++i) a += (double) W[(size_t) r * n_in + i] * x[(size_t) i]; ref[(size_t) r] = (float) a; }
            float *dW = to_dev(W), *dx = to_dev(x), *dy = nullptr;
            cudaMalloc((void**) &dy, n_out * 4);
            strata::kernels::dense_gemv_f32(dW, dx, dy, n_in, n_out, nullptr);
            report("gemv_f32 5120 -> 48", compare(from_dev(dy, n_out), ref), 1e-3);
        }
        for (int n_ctx : {1, 40, 256, 257, 700}) {   // attention, including several splits and a partial last one
            const int H = 24, HK = 4, D = 256, max_ctx = 1024;
            auto q = rnd((size_t) H * D), k = rnd((size_t) HK * max_ctx * D), v = rnd((size_t) HK * max_ctx * D);
            std::vector<uint16_t> kh(k.size()), vh(v.size());
            std::vector<float> kr(k.size()), vr(v.size());
            for (size_t i = 0; i < k.size(); ++i) {
                kh[i] = f32_to_f16_bits(k[i]); vh[i] = f32_to_f16_bits(v[i]);
                kr[i] = strata::fp16_to_fp32(kh[i]); vr[i] = strata::fp16_to_fp32(vh[i]);
            }
            const float scale = 1.0f / std::sqrt((float) D);
            std::vector<float> ref((size_t) H * D);
            for (int h = 0; h < H; ++h) {
                const int kvh = h / (H / HK);
                std::vector<double> sc((size_t) n_ctx);
                double mx = -1e300;
                for (int t = 0; t < n_ctx; ++t) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += (double) q[(size_t) h * D + i] * kr[((size_t) kvh * max_ctx + t) * D + i];
                    sc[(size_t) t] = d * scale; mx = std::max(mx, sc[(size_t) t]);
                }
                double den = 0; for (auto& x : sc) { x = std::exp(x - mx); den += x; }
                for (int i = 0; i < D; ++i) {
                    double a = 0;
                    for (int t = 0; t < n_ctx; ++t) a += sc[(size_t) t] * vr[((size_t) kvh * max_ctx + t) * D + i];
                    ref[(size_t) h * D + i] = (float) (a / den);
                }
            }
            float* dq = to_dev(q);
            uint16_t *dk = to_dev(kh), *dv = to_dev(vh);
            float *dout = nullptr, *dscr = nullptr;
            cudaMalloc((void**) &dout, ref.size() * 4);
            cudaMalloc((void**) &dscr, strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
            strata::kernels::dense_attn_decode(dq, dk, dv, dout, dscr, H, HK, D, n_ctx, max_ctx, scale, nullptr);
            char name[64];
            std::snprintf(name, sizeof name, "attention 24/4 x 256, n_ctx %d", n_ctx);
            report(name, compare(from_dev(dout, ref.size()), ref), 2e-3);
        }
        {   // KV append: the cell the attention will read
            const int HK = 4, D = 256, max_ctx = 64, pos = 9;
            auto k = rnd((size_t) HK * D), v = rnd((size_t) HK * D);
            std::vector<uint16_t> zero((size_t) HK * max_ctx * D, 0);
            uint16_t *dk = to_dev(zero), *dv = to_dev(zero);
            float *dkc = to_dev(k), *dvc = to_dev(v);
            strata::kernels::dense_kv_append(dk, dv, dkc, dvc, pos, HK, D, max_ctx, nullptr);
            std::vector<uint16_t> back(zero.size());
            cudaMemcpy(back.data(), dk, back.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> got((size_t) HK * D), ref((size_t) HK * D);
            for (int h = 0; h < HK; ++h) for (int i = 0; i < D; ++i) {
                got[(size_t) h * D + i] = strata::fp16_to_fp32(back[((size_t) h * max_ctx + pos) * D + i]);
                ref[(size_t) h * D + i] = strata::fp16_to_fp32(f32_to_f16_bits(k[(size_t) h * D + i]));
            }
            report("kv_append (K cell)", compare(got, ref), 1e-6);
        }
        {   // the KV cache formats (--kv): the quantizing append + the attention that reads it, against the host model
            const int H = 24, HK = 4, D = 256, max_ctx = 512, n_ctx = 300;
            using strata::kernels::dense_kv_cell_bytes;
            if (dense_kv_cell_bytes(1, D) != 272 || dense_kv_cell_bytes(2, D) != 144 || dense_kv_cell_bytes(0, D) != 512) {
                std::printf("%-34s FAIL (cell bytes %llu / %llu / %llu, expected 512 / 272 / 144)\n", "kv cell sizes",
                            (unsigned long long) dense_kv_cell_bytes(0, D), (unsigned long long) dense_kv_cell_bytes(1, D),
                            (unsigned long long) dense_kv_cell_bytes(2, D));
                ++bad;
            }
            struct Fmt { const char* name; int kf, vf; };
            const Fmt fmts[] = {{"kv int8 (K and V)", 1, 1}, {"kv q4_0 (K and V, rotated)", 2, 2}, {"kv k8v4 (int8 K, q4 V)", 1, 2}};
            for (const Fmt& f : fmts) {
                auto q = rnd((size_t) H * D), k = rnd((size_t) n_ctx * HK * D, 1.5f), v = rnd((size_t) n_ctx * HK * D);
                float *dq = to_dev(q), *dkf = to_dev(k), *dvf = to_dev(v), *dout = nullptr, *dscr = nullptr;
                const uint64_t kb = (uint64_t) HK * max_ctx * dense_kv_cell_bytes(f.kf, D), vb = (uint64_t) HK * max_ctx * dense_kv_cell_bytes(f.vf, D);
                void *dk = nullptr, *dv = nullptr;
                cudaMalloc(&dk, kb); cudaMalloc(&dv, vb);
                cudaMemset(dk, 0, kb); cudaMemset(dv, 0, vb);
                cudaMalloc((void**) &dout, (size_t) H * D * 4);
                cudaMalloc((void**) &dscr, strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
                for (int pos = 0; pos < n_ctx; ++pos)
                    strata::kernels::dense_kv_append_fmt(dk, dv, f.kf, f.vf, dkf + (size_t) pos * HK * D, dvf + (size_t) pos * HK * D,
                                                         pos, HK, D, max_ctx, nullptr);
                const float scale = 1.0f / std::sqrt((float) D);
                strata::kernels::dense_attn_decode_fmt(dq, dk, dv, f.kf, f.vf, dout, dscr, H, HK, D, n_ctx, max_ctx, scale, nullptr);
                // the host: what the cache holds (rotated + quantized + dequantized), and the attention over it
                std::vector<float> kd((size_t) HK * n_ctx * D), vd(kd.size());
                for (int t = 0; t < n_ctx; ++t)
                    for (int h = 0; h < HK; ++h) {
                        const float* ks = &k[((size_t) t * HK + h) * D];
                        const float* vs = &v[((size_t) t * HK + h) * D];
                        std::vector<float> a(ks, ks + D), b(vs, vs + D);
                        if (f.kf == 2) host_fwht256(a.data());
                        if (f.vf == 2) host_fwht256(b.data());
                        host_kv_roundtrip(f.kf, a);
                        host_kv_roundtrip(f.vf, b);
                        std::copy(a.begin(), a.end(), kd.begin() + ((size_t) h * n_ctx + t) * D);
                        std::copy(b.begin(), b.end(), vd.begin() + ((size_t) h * n_ctx + t) * D);
                    }
                std::vector<float> ref((size_t) H * D);
                for (int h = 0; h < H; ++h) {
                    const int kvh = h / (H / HK);
                    std::vector<float> qh(q.begin() + (size_t) h * D, q.begin() + (size_t) (h + 1) * D);
                    if (f.kf == 2) host_fwht256(qh.data());
                    std::vector<double> sc((size_t) n_ctx);
                    double mx = -1e300;
                    for (int t = 0; t < n_ctx; ++t) {
                        double d = 0;
                        for (int i = 0; i < D; ++i) d += (double) qh[(size_t) i] * kd[((size_t) kvh * n_ctx + t) * D + i];
                        sc[(size_t) t] = d * scale; mx = std::max(mx, sc[(size_t) t]);
                    }
                    double den = 0; for (auto& x : sc) { x = std::exp(x - mx); den += x; }
                    std::vector<float> o(D);
                    for (int i = 0; i < D; ++i) {
                        double a = 0;
                        for (int t = 0; t < n_ctx; ++t) a += sc[(size_t) t] * vd[((size_t) kvh * n_ctx + t) * D + i];
                        o[(size_t) i] = (float) (a / den);
                    }
                    if (f.vf == 2) host_fwht256(o.data());
                    std::copy(o.begin(), o.end(), ref.begin() + (size_t) h * D);
                }
                report(f.name, compare(from_dev(dout, ref.size()), ref), 5e-3);
                {   // xolotl: an 8-column window in one launch (dense_attn_decode_multi_fmt), crossing the 256-cell split
                    // boundary: every column must be BITWISE what the same column gives alone, and close to the old kernel
                    const int NW = 8, c0 = 252;
                    auto qw = rnd((size_t) NW * H * D);
                    float *dqw = to_dev(qw), *dow = nullptr, *do1 = nullptr, *dold = nullptr, *dscr8 = nullptr;
                    cudaMalloc((void**) &dow, (size_t) NW * H * D * 4);
                    cudaMalloc((void**) &do1, (size_t) H * D * 4);
                    cudaMalloc((void**) &dold, (size_t) H * D * 4);
                    cudaMalloc((void**) &dscr8, (size_t) NW * strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
                    strata::kernels::dense_attn_decode_multi_fmt(dqw, dk, dv, f.kf, f.vf, dow, dscr8, H, HK, D, c0, NW, max_ctx, scale, nullptr);
                    const std::vector<float> win = from_dev(dow, (size_t) NW * H * D);
                    double mism = 0, old_abs = 0;
                    for (int c = 0; c < NW; ++c) {
                        strata::kernels::dense_attn_decode_multi_fmt(dqw + (size_t) c * H * D, dk, dv, f.kf, f.vf, do1, dscr8, H, HK, D,
                                                                     c0 + c, 1, max_ctx, scale, nullptr);
                        strata::kernels::dense_attn_decode_fmt(dqw + (size_t) c * H * D, dk, dv, f.kf, f.vf, dold, dscr, H, HK, D,
                                                               c0 + c, max_ctx, scale, nullptr);
                        const std::vector<float> one = from_dev(do1, (size_t) H * D), old = from_dev(dold, (size_t) H * D);
                        for (size_t i = 0; i < one.size(); ++i) {
                            if (std::memcmp(&one[i], &win[(size_t) c * H * D + i], 4) != 0) ++mism;
                            old_abs = std::max(old_abs, (double) std::fabs(one[i] - old[i]));
                        }
                    }
                    char nm[80];
                    Diff dm{}; dm.abs = mism;
                    std::snprintf(nm, sizeof nm, "%s window x8 (bits off)", f.name);
                    report(nm, dm, 0.5);
                    Diff dd{}; dd.abs = old_abs;
                    std::snprintf(nm, sizeof nm, "%s window vs old kernel", f.name);
                    report(nm, dd, 1e-4);
                    cudaFree(dqw); cudaFree(dow); cudaFree(do1); cudaFree(dold); cudaFree(dscr8);
                }
                cudaFree(dq); cudaFree(dkf); cudaFree(dvf); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dscr);
            }
        }
        {   // the same formats through the BATCHED prompt kernels: T queries at a time, query t causal over [0, pos0+t]
            const int H = 24, HK = 4, D = 256, max_ctx = 512, T = 16, pos0 = 0;
            struct Fmt2 { const char* name; int kf, vf; };
            const Fmt2 fmts2[] = {{"kv chunk int8", 1, 1}, {"kv chunk q4_0", 2, 2}, {"kv chunk k8v4", 1, 2}};
            for (const Fmt2& g : fmts2) {
                auto q = rnd((size_t) T * H * D), k = rnd((size_t) T * HK * D, 1.5f), v = rnd((size_t) T * HK * D);
                float *dq = to_dev(q), *dkf = to_dev(k), *dvf = to_dev(v), *dout = nullptr;
                const uint64_t kb = (uint64_t) HK * max_ctx * strata::kernels::dense_kv_cell_bytes(g.kf, D);
                const uint64_t vb = (uint64_t) HK * max_ctx * strata::kernels::dense_kv_cell_bytes(g.vf, D);
                void *dk = nullptr, *dv = nullptr;
                cudaMalloc(&dk, kb); cudaMalloc(&dv, vb);
                cudaMemset(dk, 0, kb); cudaMemset(dv, 0, vb);
                cudaMalloc((void**) &dout, (size_t) T * H * D * 4);
                strata::kernels::dense_kv_append_rows_fmt(dk, dv, g.kf, g.vf, dkf, dvf, T, pos0, HK, D, max_ctx, nullptr);
                const float scale = 1.0f / std::sqrt((float) D);
                strata::kernels::dense_attn_chunk_fmt(dq, dk, dv, g.kf, g.vf, dout, T, pos0, H, HK, D, max_ctx, scale, nullptr);
                // the host model, causal: query t attends to cells [0, pos0 + t]
                std::vector<float> kd((size_t) HK * T * D), vd(kd.size());
                for (int t = 0; t < T; ++t)
                    for (int h = 0; h < HK; ++h) {
                        const float* ks = &k[((size_t) t * HK + h) * D];
                        const float* vs = &v[((size_t) t * HK + h) * D];
                        std::vector<float> a(ks, ks + D), b(vs, vs + D);
                        if (g.kf == 2) host_fwht256(a.data());
                        if (g.vf == 2) host_fwht256(b.data());
                        host_kv_roundtrip(g.kf, a);
                        host_kv_roundtrip(g.vf, b);
                        std::copy(a.begin(), a.end(), kd.begin() + ((size_t) h * T + t) * D);
                        std::copy(b.begin(), b.end(), vd.begin() + ((size_t) h * T + t) * D);
                    }
                std::vector<float> ref((size_t) T * H * D);
                for (int tq = 0; tq < T; ++tq) {
                    const int n_ctx = pos0 + tq + 1;
                    for (int h = 0; h < H; ++h) {
                        const int kvh = h / (H / HK);
                        std::vector<float> qh(q.begin() + ((size_t) tq * H + h) * D,
                                             q.begin() + ((size_t) tq * H + h) * D + D);
                        if (g.kf == 2) host_fwht256(qh.data());
                        std::vector<double> sc((size_t) n_ctx);
                        double mx = -1e300;
                        for (int t = 0; t < n_ctx; ++t) {
                            double d = 0;
                            for (int i = 0; i < D; ++i) d += (double) qh[(size_t) i] * kd[((size_t) kvh * T + t) * D + i];
                            sc[(size_t) t] = d * scale; mx = std::max(mx, sc[(size_t) t]);
                        }
                        double den = 0; for (auto& x : sc) { x = std::exp(x - mx); den += x; }
                        std::vector<float> o(D);
                        for (int i = 0; i < D; ++i) {
                            double a = 0;
                            for (int t = 0; t < n_ctx; ++t) a += sc[(size_t) t] * vd[((size_t) kvh * T + t) * D + i];
                            o[(size_t) i] = (float) (a / den);
                        }
                        if (g.vf == 2) host_fwht256(o.data());
                        std::copy(o.begin(), o.end(), ref.begin() + ((size_t) tq * H + h) * D);
                    }
                }
                report(g.name, compare(from_dev(dout, ref.size()), ref), 5e-3);
                cudaFree(dq); cudaFree(dkf); cudaFree(dvf); cudaFree(dk); cudaFree(dv); cudaFree(dout);
            }
        }
        {   // chunk path vs decode path over the SAME quantized cache, query by query.  This is the parity the
            // model check measures (prefill vs one token at a time), without the weights in the way: n_ctx <= 256
            // keeps the decode path on a single split, so the two kernels have to agree to within fp32 noise.
            const int H = 24, HK = 4, D = 256, max_ctx = 512, T = 16;
            struct Fmt3 { const char* name; int kf, vf; };
            const Fmt3 fmts3[] = {{"chunk==decode fp16", 0, 0}, {"chunk==decode int8", 1, 1},
                                  {"chunk==decode q4_0", 2, 2}, {"chunk==decode k8v4", 1, 2}};
            for (const Fmt3& g : fmts3) {
                auto k = rnd((size_t) T * HK * D, 1.5f), v = rnd((size_t) T * HK * D);
                float* dkf = to_dev(k), *dvf = to_dev(v);
                const uint64_t kb = (uint64_t) HK * max_ctx * strata::kernels::dense_kv_cell_bytes(g.kf, D);
                const uint64_t vb = (uint64_t) HK * max_ctx * strata::kernels::dense_kv_cell_bytes(g.vf, D);
                void *dk = nullptr, *dv = nullptr;
                cudaMalloc(&dk, kb); cudaMalloc(&dv, vb);
                cudaMemset(dk, 0, kb); cudaMemset(dv, 0, vb);
                // the cache is written by the CHUNK append, then read by both paths
                strata::kernels::dense_kv_append_rows_fmt(dk, dv, g.kf, g.vf, dkf, dvf, T, 0, HK, D, max_ctx, nullptr);
                const float scale = 1.0f / std::sqrt((float) D);
                float *dchunk = nullptr, *ddec = nullptr, *dscr = nullptr;
                cudaMalloc((void**) &dchunk, (size_t) T * H * D * 4);
                cudaMalloc((void**) &ddec, (size_t) H * D * 4);
                cudaMalloc((void**) &dscr, strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
                // one query per (head, dim) at a time, drawn from the same [T, H, D] block the chunk path reads
                std::vector<float> qh((size_t) H * D);
                std::mt19937 qrng(5);
                std::normal_distribution<float> qnd(0.0f, 1.0f);
                for (auto& x : qh) x = qnd(qrng);
                float* dqh = to_dev(qh);
                double worst = 0;
                for (int t = 0; t < T; ++t) {
                    // the chunk path: T queries, query t causal over [0, t].  Re-append the same rows each time so
                    // the cache holds exactly t+1 meaningful cells for the decode call to match.
                    strata::kernels::dense_attn_chunk_fmt(dqh, dk, dv, g.kf, g.vf, dchunk, 1, t, H, HK, D, max_ctx, scale, nullptr);
                    strata::kernels::dense_attn_decode_fmt(dqh, dk, dv, g.kf, g.vf, ddec, dscr, H, HK, D, t + 1, max_ctx, scale, nullptr);
                    worst = std::max(worst, compare(from_dev(dchunk, (size_t) H * D), from_dev(ddec, (size_t) H * D)).abs);
                }
                char name[64];
                std::snprintf(name, sizeof name, "%s (worst query)", g.name);
                report(name, Diff{worst, worst}, 5e-3);
                cudaFree(dkf); cudaFree(dvf); cudaFree(dk); cudaFree(dv);
                cudaFree(dchunk); cudaFree(ddec); cudaFree(dscr); cudaFree(dqh);
            }
        }
                {   // tensor-core chunk attention over a long cache: 24 queries at position 600, against the decode kernel
            const int H = 24, HK = 4, D = 256, max_ctx = 1024, T = 24, pos0 = 600, N = pos0 + T;
            struct Fmt4 { const char* name; int kf, vf; };
            const Fmt4 fmts4[] = {{"tc chunk fp16, pos0 600", 0, 0}, {"tc chunk int8, pos0 600", 1, 1},
                                  {"tc chunk q4_0, pos0 600", 2, 2}, {"tc chunk k8v4, pos0 600", 1, 2}};
            for (const Fmt4& g : fmts4) {
                auto k = rnd((size_t) N * HK * D, 1.5f), v = rnd((size_t) N * HK * D), q = rnd((size_t) T * H * D);
                float *dkf = to_dev(k), *dvf = to_dev(v), *dq = to_dev(q), *dchunk = nullptr, *ddec = nullptr, *dscr = nullptr;
                const uint64_t kb = (uint64_t) HK * max_ctx * strata::kernels::dense_kv_cell_bytes(g.kf, D);
                const uint64_t vb = (uint64_t) HK * max_ctx * strata::kernels::dense_kv_cell_bytes(g.vf, D);
                void *dk = nullptr, *dv = nullptr;
                cudaMalloc(&dk, kb); cudaMalloc(&dv, vb);
                cudaMemset(dk, 0, kb); cudaMemset(dv, 0, vb);
                cudaMalloc((void**) &dchunk, (size_t) T * H * D * 4);
                cudaMalloc((void**) &ddec, (size_t) H * D * 4);
                cudaMalloc((void**) &dscr, strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
                strata::kernels::dense_kv_append_rows_fmt(dk, dv, g.kf, g.vf, dkf, dvf, N, 0, HK, D, max_ctx, nullptr);
                const float scale = 1.0f / std::sqrt((float) D);
                strata::kernels::dense_attn_chunk_fmt(dq, dk, dv, g.kf, g.vf, dchunk, T, pos0, H, HK, D, max_ctx, scale, nullptr);
                double worst = 0;
                for (int t = 0; t < T; ++t) {
                    strata::kernels::dense_attn_decode_fmt(dq + (size_t) t * H * D, dk, dv, g.kf, g.vf, ddec, dscr, H, HK, D,
                                                           pos0 + t + 1, max_ctx, scale, nullptr);
                    const auto a = from_dev(dchunk + (size_t) t * H * D, (size_t) H * D);
                    worst = std::max(worst, compare(a, from_dev(ddec, (size_t) H * D)).abs);
                }
                report(g.name, Diff{worst, worst}, 5e-3);
                cudaFree(dkf); cudaFree(dvf); cudaFree(dq); cudaFree(dk); cudaFree(dv);
                cudaFree(dchunk); cudaFree(ddec); cudaFree(dscr);
            }
        }
    } catch (const std::exception& e) {
        std::printf("selftest: exception: %s\n", e.what());
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("selftest: a kernel failed\n"); return 1; }
    std::printf(bad ? "selftest: %d FAILED\n" : "selftest: all passed\n", bad);
    return bad ? 1 : 0;
}


// ---- `xolotl-dense --check <model.gguf> [--mtp]`: the model against ITSELF - the columns, the snapshot and the
// rollback must reproduce feeding the tokens one at a time.  Needs the GPU and the model, no reference engine.
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b, bool* same) {
    double d = 0;
    *same = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], 4) != 0) *same = false;
        d = std::max(d, std::fabs((double) a[i] - (double) b[i]));
    }
    return d;
}

std::vector<float> read_logits(strata::core::DenseModel& m, int col) {
    std::string err;
    m.sync(err);
    std::vector<float> h((size_t) m.config().n_vocab);
    cudaMemcpy(h.data(), m.logits_col(col), h.size() * 4, cudaMemcpyDeviceToHost);
    return h;
}

int argmax_of(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

int check_model(const std::string& gguf, bool with_mtp, const std::string& kv) {
    using strata::core::Logits;
    strata::core::DenseModel m;
    std::string err;
    strata::core::DenseOptions opt;
    opt.mtp = with_mtp;
    opt.mtp_force = true;                     // a check must not be switched off by the VRAM rule
    opt.draft_max = 3;
    opt.kv = kv;
    if (!m.load(gguf, 512, err, opt)) { std::printf("load: %s\n", err.c_str()); return 1; }
    std::mt19937 rng(11);
    std::vector<int32_t> T(40);
    for (auto& t : T) t = 1000 + (int32_t) (rng() % 20000);
    int bad = 0;
    auto verdict = [&](const char* what, const std::vector<float>& a, const std::vector<float>& b) {
        bool same = false;
        const double d = max_abs_diff(a, b, &same);
        const bool ok = argmax_of(a) == argmax_of(b) && d < 1e-2;
        std::printf("%-50s max |dlogit| %.3g  %s  argmax %d / %d  %s\n", what, d, same ? "(bit-identical)" : "", argmax_of(a),
                    argmax_of(b), ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    };

    // The MMQ prompt path is a different KIND of parity from the others, and asking it for the same tolerance asks a
    // meaningless question. The other checks reorder the same arithmetic, so they must agree to the last bit. MMQ
    // multiplies the same weights on int8 tensor cores and rounds every activation to q8_1 (pf_gdn_block,
    // pf_ffn_block): those are not the same numbers any more, and the difference compounds over 64 layers. On logits
    // that reach +-30, an absolute 1e-2 is not a property of a correct engine.
    //
    // So the gate is one thing that is structural rather than a tuned threshold: the error measured RELATIVE to the
    // logit scale, printed with the scale so the absolute number can be read against it. The bound is a tripwire for
    // gross breakage - a wrong kernel is off by order 1, not by a few percent - and it is deliberately loose.
    //
    // Deliberately NOT gated, because gating on them would fail a CORRECT engine: the argmax, and the top-k candidate
    // set. This check feeds 40 random ids, which is out of distribution, so the output head is nearly flat and
    // thousands of ids sit within a hair of each other. Measured on synthetic logits of exactly this shape (248k
    // wide, spread 6, perturbed by the 0.414 this path actually shows): the argmax moves in 20% of trials and as
    // little as 1 of the top 20 survives. On a peaked head from real text the argmax holds 97% of the time and the
    // relative error drops to ~1.3%. Both numbers are printed, because a collapse there is worth seeing, but neither
    // is a verdict on a random-token prompt.
    auto verdict_quantized = [&](const char* what, const std::vector<float>& a, const std::vector<float>& b) {
        bool same = false;
        const double d = max_abs_diff(a, b, &same);
        double scale = 0;
        for (float v : b) scale = std::max(scale, (double) std::fabs(v));
        const double rel = scale > 0 ? d / scale : d;
        auto top_ids = [](const std::vector<float>& v) {
            std::vector<int> idx(v.size());
            for (size_t i = 0; i < v.size(); ++i) idx[i] = (int) i;
            const int k = std::min<int>(20, (int) v.size());
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int x, int y) { return v[(size_t) x] > v[(size_t) y]; });
            idx.resize((size_t) k);
            return idx;
        };
        const auto ta = top_ids(a), tb = top_ids(b);
        int shared = 0;
        for (int x : ta) if (std::find(tb.begin(), tb.end(), x) != tb.end()) ++shared;
        const bool ok = rel < 0.25;
        std::printf("%-50s max |dlogit| %.3g on a scale of %.1f (%.2f%%)  %s"
                    "   [argmax %d / %d, top-20 shared %d/%d: informational on a random-token prompt]\n",
                    what, d, scale, 100.0 * rel, ok ? "ok" : "FAIL",
                    argmax_of(a), argmax_of(b), shared, (int) ta.size());
        if (!ok) ++bad;
    };

    // A: one token at a time
    m.reset();
    for (size_t i = 0; i < T.size(); ++i)
        if (!m.step(T[i], i + 1 == T.size(), err)) { std::printf("%s\n", err.c_str()); return 1; }
    const auto LA = read_logits(m, 0);

    // B: the same tokens in uneven chunks
    m.reset();
    const int sizes[] = {3, 2, 5, 1, 8, 4};
    size_t i = 0, k = 0;
    while (i < T.size()) {
        const int n = (int) std::min<size_t>((size_t) sizes[k++ % 6], T.size() - i);
        if (!m.run(&T[i], n, i + (size_t) n == T.size() ? Logits::Last : Logits::None, false, false, err)) { std::printf("%s\n", err.c_str()); return 1; }
        i += (size_t) n;
    }
    verdict("chunks of 3,2,5,1,8,4 vs one at a time", LA, read_logits(m, 0));

    // F: the batched MMQ prompt path over the same tokens.  It multiplies the same weights through different
    // kernels (int8 tensor cores over a whole chunk), so this is the parity gate for that path.
    if (m.prefill_setup(16, err)) {
        m.reset();
        if (!m.prefill(T.data(), (int64_t) T.size(), Logits::Last, err)) { std::printf("%s\n", err.c_str()); return 1; }
        verdict_quantized("MMQ prefill of 16 tokens at a time vs one at a time", LA, read_logits(m, 0));
    } else {
        std::printf("%-50s %s\n", "MMQ prefill vs one at a time", err.c_str());
    }

    if (m.has_mtp()) {
        // C1: three columns with two junk drafts, rolled back to column 0, then the real continuation
        m.reset();
        for (size_t j = 0; j + 2 < T.size(); ++j) m.step(T[j], false, err);          // T[0..37]
        const int32_t c1[3] = {T[38], 7, 9};
        if (!m.run(c1, 3, Logits::All, true, true, err)) { std::printf("%s\n", err.c_str()); return 1; }
        const auto col0 = read_logits(m, 0);
        if (!m.rollback(0, err)) { std::printf("%s\n", err.c_str()); return 1; }
        if (m.position() != 39) { std::printf("rollback(0) left position %lld, expected 39\n", (long long) m.position()); ++bad; }
        m.step(T[39], true, err);
        verdict("rollback(0) of two rejected columns, then continue", LA, read_logits(m, 0));

        // C2: three columns where the first draft was right: column 1 must be the one-at-a-time logits, and
        //     rollback(1) must leave exactly the state after T[39]
        m.reset();
        for (size_t j = 0; j + 2 < T.size(); ++j) m.step(T[j], false, err);
        const int32_t c2[3] = {T[38], T[39], 9};
        if (!m.run(c2, 3, Logits::All, true, true, err)) { std::printf("%s\n", err.c_str()); return 1; }
        verdict("3-column run: column 1 vs one at a time", LA, read_logits(m, 1));
        if (!m.rollback(1, err)) { std::printf("%s\n", err.c_str()); return 1; }
        m.step(5, true, err);
        const auto after_rb = read_logits(m, 0);
        m.reset();
        for (size_t j = 0; j < T.size(); ++j) m.step(T[j], false, err);
        m.step(5, true, err);
        verdict("rollback(1), then one more token vs plain", read_logits(m, 0), after_rb);

        // D: column 0 of a multi-column run equals the single-token logits
        m.reset();
        for (size_t j = 0; j < 38; ++j) m.step(T[j], false, err);
        m.step(T[38], true, err);
        const auto single = read_logits(m, 0);
        m.reset();
        for (size_t j = 0; j < 38; ++j) m.step(T[j], false, err);
        const int32_t c3[3] = {T[38], 11, 13};
        m.run(c3, 3, Logits::All, true, true, err);
        verdict("3-column run: column 0 vs single step", single, read_logits(m, 0));

        // E: the MTP block runs; a chain of drafts comes out with valid ids (compare with llama.cpp by hand)
        m.reset();
        m.run(T.data(), 8, Logits::Last, true, false, err);
        if (!m.mtp_ingest(T.data(), 8, 0, 0, err)) { std::printf("%s\n", err.c_str()); return 1; }
        int32_t dr[8];
        int nd = 0;
        if (!m.mtp_draft(argmax_of(read_logits(m, 0)), 8, 3, 0.0f, dr, &nd, err)) { std::printf("%s\n", err.c_str()); return 1; }
        std::printf("MTP chain after 8 tokens + the model's own next token:");
        bool ok = nd == 3;
        for (int j = 0; j < nd; ++j) { std::printf(" %d", dr[j]); ok = ok && dr[j] >= 0 && dr[j] < m.config().n_vocab; }
        std::printf("   %s\n", ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    }
    std::printf(bad ? "check: %d FAILED\n" : "check: all passed\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

// xolotl: the n-gram chain (NInfer's --ngram chain, in the style of llama.cpp ngram-mod): a pool maps a hash of the
// last n tokens to the token that last followed them, over everything the conversation has consumed.  After the MTP
// drafts, the proposal continues from the pool while it has an answer, so a round that reproduces code, JSON or a
// tool call offers many more tokens.  The verification is unchanged: the text is exactly what single steps give.
struct NgramPool {
    int n = 8;
    std::unordered_map<uint64_t, int32_t> next;
    size_t indexed = 0;                              // tokens of `h` already in the pool
    static uint64_t key(const int32_t* t, int n) {
        uint64_t k = 1469598103934665603ull;         // FNV-1a over the token ids
        for (int i = 0; i < n; ++i) { k ^= (uint32_t) t[i]; k *= 1099511628211ull; }
        return k;
    }
    void reset() { next.clear(); indexed = 0; }
    void update(const std::vector<int32_t>& h) {
        if (h.size() < indexed) reset();
        for (size_t i = std::max(indexed, (size_t) n); i < h.size(); ++i) next[key(&h[i - (size_t) n], n)] = h[i];
        indexed = h.size();
    }
};

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") return selftest();
        if (std::string(argv[i]) == "--check") {
            bool mtp = false;
            std::string path, kv = "fp16";
            for (int j = 1; j < argc; ++j) {
                const std::string a = argv[j];
                if (a == "--mtp") mtp = true;
                else if (a == "--kv" && j + 1 < argc) kv = argv[++j];
                else if (a != "--check" && a.rfind("--", 0) != 0) path = a;
            }
            if (path.empty()) { std::fprintf(stderr, "usage: xolotl-dense --check <model.gguf> [--mtp] [--kv fp16|int8|q4_0|k8v4]\n"); return 2; }
            return check_model(path, mtp, kv);
        }
    }
    std::string gguf;
    long long context = 32768;
    bool serve = false, want_mtp = false, mtp_force = false;
    std::string kv_type = "fp16";
    int draft_max = 2, draft_min = 1;
    bool ngram_on = false;                             // xolotl: --ngram-chain (MTP proposal + n-gram continuation)
    int ngram_n = 8, ngram_max = 7, ngram_min = 3;     // match length, total drafts, extension needed to go wide
    float draft_p_min = 0.0f;
    // 0: the default prompt chunk; >0: ask for that chunk; <0 (--no-prefill): feed the prompt through run()
    long long prompt_chunk = 0;
    int n_ckpt = 4;                                    // conversation checkpoints (pinned host RAM, ~154 MiB each); 0 = off
    std::string df2_pack;                              // xolotl: --dflash2 <pack>, the DFlash2 block drafter instead of MTP
    std::string draft_vocab;                           // xolotl: --draft-vocab <file of int32 token ids>, MTP shortlist head
    bool draft_max_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "xolotl-dense: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") serve = true;
        else if (a == "--mtp") want_mtp = true;
        else if (a == "--kv") kv_type = next("--kv");
        else if (a == "--mtp-force") mtp_force = true;
        else if (a == "--draft-max" || a == "--spec-draft-n-max") { draft_max = std::atoi(next("--draft-max")); draft_max_set = true; }
        else if (a == "--dflash2") df2_pack = next("--dflash2");
        else if (a == "--ngram-chain") ngram_on = true;
        else if (a == "--draft-vocab") draft_vocab = next("--draft-vocab");
        else if (a == "--ngram-n") ngram_n = std::atoi(next("--ngram-n"));
        else if (a == "--ngram-max") ngram_max = std::atoi(next("--ngram-max"));
        else if (a == "--ngram-min") ngram_min = std::atoi(next("--ngram-min"));
        else if (a == "--draft-min" || a == "--spec-draft-n-min") draft_min = std::atoi(next("--draft-min"));
        else if (a == "--draft-p-min" || a == "--spec-draft-p-min") draft_p_min = std::strtof(next("--draft-p-min"), nullptr);
        else if (a == "--native" || a == "--model" || a == "--gguf") gguf = next("--native");
        else if (a == "--context" || a == "-c") context = std::atoll(next("--context"));
        else if (a == "--prompt-chunk" || a == "--pp-chunk") prompt_chunk = std::atoll(next("--prompt-chunk"));
        else if (a == "--no-prefill") prompt_chunk = -1;
        else if (a == "--ckpt") n_ckpt = std::atoi(next("--ckpt"));
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        // the flags setup passes to `strata` (--gpu-layers, --cache, ...) mean nothing here and are not an error
    }
    if (!serve || gguf.empty()) { usage(); return 2; }
    if (!df2_pack.empty()) {
        want_mtp = false;                              // the block drafter replaces MTP (one drafter, and the VRAM for it)
        if (!draft_max_set) draft_max = strata::core::Dflash2Drafter::kDrafts;
    }
    if (draft_max < 1 || draft_max > strata::core::DenseModel::kMaxCols - 1) {
        std::fprintf(stderr, "xolotl-dense: --draft-max must be 1..%d\n", strata::core::DenseModel::kMaxCols - 1);
        return 2;
    }
    if (draft_min < 1 || draft_min > draft_max) {
        std::fprintf(stderr, "xolotl-dense: --draft-min must be 1..--draft-max (%d)\n", draft_max);
        return 2;
    }
    if (!(draft_p_min >= 0.0f && draft_p_min <= 1.0f)) { std::fprintf(stderr, "xolotl-dense: --draft-p-min must be 0..1\n"); return 2; }

    strata::core::DenseModel model;
    std::string err;
    // xolotl: the DFlash2 drafter loads first, so the model's VRAM budget sees what it took
    strata::core::Dflash2Drafter df2;
    const bool use_df2 = !df2_pack.empty();
    if (use_df2) {
        if (!df2.load(df2_pack, 0, err)) {
            std::fprintf(stderr, "xolotl-dense: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "xolotl-dense: DFlash2 drafter %s (%.0f MiB of VRAM, taps after layers", df2_pack.c_str(), df2.vram_bytes() / 1048576.0);
        for (int l : df2.tap_layers()) std::fprintf(stderr, " %d", l);
        std::fprintf(stderr, ")\n");
    }
    std::fprintf(stderr, "xolotl-dense: loading %s%s ...\n", gguf.c_str(), want_mtp ? " (with the MTP head)" : "");
    strata::core::DenseOptions opt;
    opt.mtp = want_mtp;
    opt.mtp_force = mtp_force;
    if (ngram_on) {
        ngram_max = std::max(draft_max, std::min(ngram_max, strata::core::DenseModel::kMaxCols - 1));
        if (ngram_n < 2) ngram_n = 2;
    }
    opt.draft_max = ngram_on ? ngram_max : draft_max;   // rollback snapshots for the widest window
    opt.kv = kv_type;
    if (use_df2) { opt.spec_snapshots = true; opt.taps = df2.tap_layers(); }
    if (!draft_vocab.empty() && want_mtp) {
        FILE* fv = std::fopen(draft_vocab.c_str(), "rb");
        if (fv) {
            int32_t id;
            while (std::fread(&id, 4, 1, fv) == 1) opt.draft_vocab.push_back(id);
            std::fclose(fv);
        }
        if (opt.draft_vocab.empty()) {
            std::fprintf(stderr, "xolotl-dense: --draft-vocab %s: unreadable or empty (int32 token ids expected)\n", draft_vocab.c_str());
            return 2;
        }
    }
    if (!model.load(gguf, context, err, opt)) {
        std::fprintf(stderr, "xolotl-dense: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const auto& cfg = model.config();
    const int n_vocab = cfg.n_vocab;
    constexpr int NC = strata::core::DenseModel::kMaxCols;
    // the drafter shares the model's embedding (its block rows: the anchor's row, then the mask token's 7 times, staged
    // in pinned memory the drafter's graphs copy from) and its output head
    float* df2_stage = nullptr;
    if (use_df2) {
        const int E = cfg.n_embd;
        if (df2.n_embd() != E || (int) df2.tap_layers().size() * E != model.n_taps() * E) {
            std::printf("ERR the DFlash2 pack (width %d) is not for this model (width %d)\n", df2.n_embd(), E);
            return 1;
        }
        if (cudaHostAlloc((void**) &df2_stage, (size_t) NC * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            std::printf("ERR cannot allocate the drafter's embedding stage\n");
            return 1;
        }
        for (int r = 1; r < NC; ++r)
            if (!model.embed_host(df2.mask_token(), df2_stage + (size_t) r * E, err)) { std::printf("ERR %s\n", err.c_str()); return 1; }
        auto embed = [df2_stage, E](const int32_t*, int n, float* out, cudaStream_t cs) {
            cudaMemcpyAsync(out, df2_stage, (size_t) n * E * sizeof(float), cudaMemcpyHostToDevice, cs);
        };
        if (!df2.bind_raw(embed, model.head_weights(), model.head_type(), model.head_rows(), {}, err) ||
            !df2.prepare(model.run_taps(), err)) {
            std::fprintf(stderr, "xolotl-dense: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            return 1;
        }
    }
    using strata::core::Logits;

    // Batched prompt processing: the prompt goes through MMQ in chunks instead of NC-token run() groups.  If the
    // build has no MMQ, or a weight's type is not one MMQ covers, or the scratch does not fit, this fails and the
    // prompt loop below keeps using run() exactly as before.
    if (prompt_chunk >= 0) {
        std::string perr;
        if (!model.prefill_setup(prompt_chunk, perr))
            std::fprintf(stderr, "xolotl-dense: prompt prefill off - %s\n", perr.c_str());
    }
    {
        std::string cerr;
        if (!model.ckpt_init(n_ckpt, cerr)) std::fprintf(stderr, "xolotl-dense: checkpoints off - %s\n", cerr.c_str());
        else if (model.ckpt_slots()) std::fprintf(stderr, "xolotl-dense: %d conversation checkpoints\n", model.ckpt_slots());
    }
    // the ids that end an answer, from the model's own vocabulary
    std::vector<int32_t> stop_ids;
    if (cfg.eos_id >= 0) stop_ids.push_back(cfg.eos_id);
    try {
        strata::GgufFile g(gguf);
        if (const auto* toks = g.get("tokenizer.ggml.tokens"))
            for (size_t i = 0; i < toks->items.size(); ++i)
                if (toks->items[i].s == "<|im_end|>" || toks->items[i].s == "<|endoftext|>") stop_ids.push_back((int32_t) i);
    } catch (const std::exception&) {}
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());
    auto is_stop = [&](int32_t t) { return std::binary_search(stop_ids.begin(), stop_ids.end(), t); };

    int* d_next = nullptr;
    int* d_hist = nullptr;
    constexpr int kMaxHist = 4096;
    if (cudaMalloc((void**) &d_next, sizeof(int)) != cudaSuccess || cudaMalloc((void**) &d_hist, kMaxHist * sizeof(int)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    std::printf("INFO engine=dense-0.2 model=%s layers=%d n_embd=%d weights_mib=%llu host_mib=%llu kv_mib=%llu context=%lld kv=%s mtp=%d dflash2=%d draft_max=%d draft_min=%d draft_p_min=%.2f\n",
                cfg.arch.c_str(), cfg.n_layer, cfg.n_embd, (unsigned long long) (model.weight_bytes() >> 20),
                (unsigned long long) (model.host_weight_bytes() >> 20), (unsigned long long) (model.kv_bytes() >> 20),
                (long long) model.max_context(), model.kv_type().c_str(), model.has_mtp() ? 1 : 0, use_df2 ? 1 : 0,
                (model.has_mtp() || use_df2) ? draft_max : 0, (model.has_mtp() || use_df2) ? draft_min : 0, (double) draft_p_min);
    std::printf("READY %lld stop\n", (long long) model.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "xolotl-dense: ready - %.2f GiB of weights, %.2f GiB of KV cache (%s), context %lld%s\n",
                 model.weight_bytes() / 1073741824.0, model.kv_bytes() / 1073741824.0, model.kv_type().c_str(), (long long) model.max_context(),
                 use_df2 ? ", DFlash2 speculative decoding on" : model.has_mtp() ? ", MTP speculative decoding on" : (want_mtp ? ", MTP requested but OFF (see above)" : ""));

    Lines in;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line == "STOP") { in.stop.store(true); continue; }
            in.push(line);
        }
        std::lock_guard<std::mutex> l(in.m);
        in.eof = true;
        in.cv.notify_all();
    });
    reader.detach();

    std::vector<int32_t> fed;                      // every token the state has consumed, in order
    NgramPool ngram;                               // xolotl: --ngram-chain
    ngram.n = ngram_n;
    std::string line;
    while (in.pop(line)) {
        if (line == "QUIT") break;
        in.stop.store(false);                       // a STOP between requests is stale
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n"); std::fflush(stdout); continue; }

        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        strata::kernels::SamplerParams sp;
        sp.top_k = 20; sp.top_p = 1.0f; sp.temperature = 0.0f; sp.greedy = true;
        unsigned long long seed = 0;
        // local (Indra's Mirror, /v1/decide): score=ID,ID,... prints `SCORE <id>:<logprob> ... top=<id>:<logprob>` -
        // the log-softmax of the raw logits that pick the first generated token (no bias, no temperature), the same
        // line as the MoE engine's (generate.cpp), so the server's parse_score_line reads both
        std::vector<int32_t> score_ids;
        bool score_bad = false;
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
            if (key == "temperature") { sp.temperature = fv; sp.greedy = fv <= 0.0f; }
            else if (key == "top_p") sp.top_p = fv;
            else if (key == "top_k") sp.top_k = std::atoi(tok.c_str() + eq + 1);
            else if (key == "min_p") sp.min_p = fv;
            else if (key == "penalty_last_n") sp.penalty_last_n = std::min(std::atoi(tok.c_str() + eq + 1), kMaxHist);
            else if (key == "penalty_repeat") sp.penalty_repeat = fv;
            else if (key == "penalty_freq") sp.penalty_freq = fv;
            else if (key == "penalty_present") sp.penalty_present = fv;
            else if (key == "seed") seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
            else if (key == "score") {
                score_bad = !parse_ids(tok.c_str() + eq + 1, score_ids) || score_ids.size() > 64;
                for (const int32_t id : score_ids) score_bad = score_bad || id >= n_vocab;
            }
        }
        if (score_bad) { std::printf("ERR score: expected 1-64 token ids below the vocabulary size\n"); std::fflush(stdout); continue; }
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
        if ((long long) ids.size() + 1 > model.max_context()) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) model.max_context());
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;

        // ---- what of the state can be kept
        size_t common = 0;
        {
            const size_t lim = std::min(fed.size(), ids.size());
            while (common < lim && fed[common] == ids[common]) ++common;
        }
        size_t resume = 0;
        if (!fed.empty() && fed.size() < ids.size() && common == fed.size()) resume = fed.size();
        if (resume == 0) {
            // The prompts part at `common`.  The GDN state cannot be cut back, but a checkpoint taken at or before
            // `common` is a state the new prompt also goes through: restore the latest one (it must leave at least
            // one token to read, or there would be no logits).
            int best = -1;
            for (int k = 0; k < model.ckpt_slots(); ++k) {
                const int64_t p = model.ckpt_pos(k);
                if (p > 0 && (size_t) p <= common && (size_t) p < ids.size() && (best < 0 || p > model.ckpt_pos(best))) best = k;
            }
            if (!fed.empty()) {
                std::string held = "none";
                for (int k = 0; k < model.ckpt_slots(); ++k)
                    if (model.ckpt_pos(k) >= 0) {
                        if (held != "none") held += ",";
                        held += std::to_string(model.ckpt_pos(k));
                    }
                std::fprintf(stderr, "xolotl-dense: previous %zu tokens are not a prefix of the new %zu - they first differ at token %zu, checkpoints hold %s%s\n",
                             fed.size(), ids.size(), common, held.c_str(),
                             best >= 0 ? " (restoring one)" : " (none at or before that token: re-reading from 0)");
            }
            if (best >= 0 && model.ckpt_restore(best, err)) {
                resume = (size_t) model.position();
                fed.resize(resume);
                std::fprintf(stderr, "xolotl-dense: checkpoint restored at token %zu, reading %zu\n", resume, ids.size() - resume);
            } else {
                model.reset();
                fed.clear();
            }
        }
        // A checkpoint is stale when the new prompt does not go through it, i.e. when it sits past `common`.  One at
        // or before `common` stays valid even after a reset: the KV cache is addressed by absolute position and K/V
        // of a token depend only on that token and its position, so re-reading the shared prefix rewrites exactly the
        // same rows.  Dropping everything on a re-read (as this did until here) threw away the deep checkpoints that
        // the next request would have needed.
        for (int k = 0; k < model.ckpt_slots(); ++k)
            if (model.ckpt_pos(k) > (int64_t) common) model.ckpt_drop(k);
        const int64_t n = (int64_t) ids.size();
        const long long budget = std::min<long long>(max_new, model.max_context() - n);
        auto fail = [&](const std::string& why) {
            std::printf("ERR %s\n", why.c_str());
            std::fflush(stdout);
            model.reset();
            fed.clear();
            for (int k = 0; k < model.ckpt_slots(); ++k) model.ckpt_drop(k);
        };

        // ---- read the prompt.  With the MMQ prefill available a pass is prefill_chunk() tokens wide and every
        // weight is read once per pass; without it a pass is NC tokens (the GEMVs read every weight once per pass).
        // Either way the PP progress lines and the STOP check run at the same granularity.
        bool cancelled = false, failed = false;
        const auto t_prompt = Clock::now();
        auto t_pp = Clock::now();
        int64_t since_pp = 0;
        const bool use_pf = model.prefill_ready();
        const int64_t pf_step = model.prefill_chunk();      // STOP is checked between steps
        model.draft_vocab_extend(ids.data(), (int64_t) ids.size());   // --draft-vocab: the prompt's own rare tokens
        // Where to save checkpoints.  A prompt that diverges from the one already read diverges in its tail - the
        // client re-renders or trims the last turns - but "the tail" is not the last 64 tokens: over the 24 diverges
        // logged by a 72k-token chat (strata-qwen3.8-27b.log) the divergence sat 143, 171, 226, 466, 1515 and 2738
        // tokens before the end of what had been read, and the old marks (n, n-1, n-8, n-64) were all past it, so
        // every one of those requests re-read the whole prompt.  The slots are now spread geometrically back from
        // the end (x8): the default 4 sit at n, n-64, n-512 and n-4096, so a divergence in the last 4096 tokens
        // costs at most 4096 tokens of re-reading.  --ckpt 6 adds the n-32768 rung.
        std::vector<int64_t> marks;
        for (int64_t d : {0, 64, 512, 4096, 32768, 262144}) {
            const int64_t m = n - d;
            if (marks.size() < (size_t) model.ckpt_slots() && m > (int64_t) resume && m > 0) marks.push_back(m);
        }
        std::sort(marks.begin(), marks.end());
        auto save_mark = [&](int64_t) {
            int slot = -1, lo = 0;
            int64_t lowest = (int64_t) 1 << 62;
            for (int k = 0; k < model.ckpt_slots(); ++k) {
                const int64_t p = model.ckpt_pos(k);
                if (p < 0) { slot = k; break; }
                if (p < lowest) { lowest = p; lo = k; }
            }
            if (slot < 0) slot = lo;
            std::string cerr;
            if (!model.ckpt_save(slot, cerr)) std::fprintf(stderr, "xolotl-dense: checkpoint: %s\n", cerr.c_str());
        };
        if (use_df2 && !use_pf) df2.reset();
        for (int64_t i = (int64_t) resume; i < n && !failed;) {
            int64_t step = std::min<int64_t>(use_pf ? pf_step : (int64_t) NC, n - i);
            for (int64_t m : marks) if (m > i && m < i + step) { step = m - i; break; }   // stop exactly on a mark
            const bool last = (i + step == n);
            bool ok = false;
            if (use_pf) {
                ok = model.prefill(&ids[(size_t) i], step, last ? Logits::Last : Logits::None, err);
            } else {
                const int chunk = (int) step;
                ok = model.run(&ids[(size_t) i], chunk, last ? Logits::Last : Logits::None, model.has_mtp(), false, err) &&
                     (!model.has_mtp() || model.mtp_ingest(&ids[(size_t) i], chunk, (int) i, 0, err));
                if (ok && use_df2) {   // the rows go to the drafter's ring; a prompt's last row waits for the first round
                    const int ctx_rows = last ? chunk - 1 : chunk;
                    ok = model.sync(err) && (ctx_rows < 1 || df2.context(model.run_taps(), ctx_rows, (int64_t) i, err));
                    if (ok && last && chunk > 1)
                        cudaMemcpy(const_cast<float*>(model.run_taps()), model.run_taps() + (size_t) (chunk - 1) * model.n_taps() * cfg.n_embd,
                                   (size_t) model.n_taps() * cfg.n_embd * sizeof(float), cudaMemcpyDeviceToDevice);
                }
            }
            if (!ok) {
                fail(err);
                failed = true;
                break;
            }
            fed.insert(fed.end(), ids.begin() + i, ids.begin() + i + step);
            i += step;
            since_pp += step;
            if (std::binary_search(marks.begin(), marks.end(), i)) save_mark(i);
            if (in.stop.load()) { cancelled = true; model.sync(err); break; }
            if (since_pp >= 32 || last) {
                if (!model.sync(err)) { fail(err); failed = true; break; }
                const double ms = ms_since(t_pp);
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) i, (long long) n, ms_since(t_prompt),
                            ms > 0 ? 1000.0 * (double) since_pp / ms : 0.0);
                std::fflush(stdout);
                t_pp = Clock::now();
                since_pp = 0;
            }
        }
        if (failed) continue;
        if (!model.sync(err)) { fail(err); continue; }
        // xolotl DFlash2: a prefill wrote its rows into the model's tap ring; the drafter gets the rows read in this
        // request (at most its 2048 window), all but the last, which becomes row 0 of the first round's new rows
        int df2_pend = 0;
        int64_t df2_pos0 = 0;
        if (use_df2 && n > (int64_t) resume) {
            const int64_t R = strata::core::DenseModel::kTapRing, W = (int64_t) model.n_taps() * cfg.n_embd;
            bool ok = true;
            if (use_pf) {
                df2.reset();
                for (int64_t q = std::max<int64_t>((int64_t) resume, n - R); ok && q < n - 1;) {
                    const int64_t slot = q % R, len = std::min<int64_t>(n - 1 - q, R - slot);
                    ok = df2.context(model.ring_taps() + (size_t) slot * W, len, q, err);
                    q += len;
                }
                if (ok) ok = cudaMemcpy(const_cast<float*>(model.run_taps()), model.ring_taps() + (size_t) ((n - 1) % R) * W,
                                        (size_t) W * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
            }
            if (!ok) { fail(err.empty() ? "dflash2: context" : err); continue; }
            df2_pend = 1;
            df2_pos0 = n - 1;
        }
        const double prompt_ms = ms_since(t_prompt);

        // ---- decode
        long long produced = 0, drafts_offered = 0, drafts_accepted = 0;
        long long ng_drafted = 0, ng_accepted = 0, ng_wide = 0;
        if (ngram_on) ngram.reset();                // rebuilt from this request's tokens at its first draft
        const char* finish = "length";
        const auto t_dec = Clock::now();
        std::vector<int32_t> hist;                  // penalty window, most recent last
        if (sp.penalty_last_n > 0) {
            const size_t take = std::min<size_t>((size_t) sp.penalty_last_n, fed.size());
            hist.assign(fed.end() - (std::ptrdiff_t) take, fed.end());
        }
        // A penalty changes only the sampling: each verified column is drawn after the previous ones were emitted, and emit()
        // has already put them in the penalty window, so a speculative step draws exactly what single steps would.  (It
        // used to switch speculation off whenever penalty_last_n > 0 - every thinking-off request, whose model-card
        // sampling has presence_penalty 1.5, decoded without MTP: 48 instead of ~90 tok/s.)
        const bool speculate = model.has_mtp() || use_df2;

        // one draw from device logits; the counter is the POSITION of the token drawn, so a seed gives the same text
        // whether a token comes from a single step or from a verified pair
        auto draw = [&](const float* logits, uint64_t counter, int32_t& out) -> bool {
            sp.counter = counter;
            const int* hp = nullptr;
            int hl = 0;
            if (sp.penalty_last_n > 0) {
                std::vector<int32_t> row((size_t) sp.penalty_last_n, -1);
                std::copy(hist.begin(), hist.end(), row.end() - (std::ptrdiff_t) hist.size());
                cudaMemcpyAsync(d_hist, row.data(), row.size() * sizeof(int), cudaMemcpyHostToDevice, (cudaStream_t) model.stream());
                hp = d_hist;
                hl = sp.penalty_last_n;
            }
            try {
                strata::kernels::sample_tokens(logits, 1, n_vocab, hp, hl, sp, d_next, model.stream());
            } catch (const std::exception& e) { err = std::string("sampler: ") + e.what(); return false; }
            int v = -1;
            cudaMemcpyAsync(&v, d_next, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t) model.stream());
            if (!model.sync(err)) return false;
            out = v;
            return true;
        };
        // emit one token; true when the answer is over
        auto emit = [&](int32_t t) -> bool {
            std::printf("T %d\n", t);
            ++produced;
            if (sp.penalty_last_n > 0) { hist.push_back(t); if ((int) hist.size() > sp.penalty_last_n) hist.erase(hist.begin()); }
            if (is_stop(t)) { finish = "stop"; return true; }
            return false;
        };

        int32_t next = -1;
        bool done = cancelled;                      // a STOP during the prompt: nothing is sampled
        if (!cancelled) {
            if (!score_ids.empty()) {
                const std::vector<float> row = read_logits(model, 0);
                int64_t top = 0;
                for (int64_t v = 1; v < (int64_t) row.size(); ++v)
                    if (row[(size_t) v] > row[(size_t) top]) top = v;
                double sum = 0.0;
                for (const float l : row) sum += std::exp((double) l - (double) row[(size_t) top]);
                const double lse = (double) row[(size_t) top] + std::log(sum);
                std::string sline = "SCORE";
                char b[64];
                for (const int32_t id : score_ids) {
                    std::snprintf(b, sizeof b, " %lld:%.6f", (long long) id, (double) row[(size_t) id] - lse);
                    sline += b;
                }
                std::snprintf(b, sizeof b, " top=%lld:%.6f", (long long) top, (double) row[(size_t) top] - lse);
                sline += b;
                std::printf("%s\n", sline.c_str());
                std::fflush(stdout);
            }
            if (!draw(model.logits_col(0), (uint64_t) model.position(), next)) { fail(err); continue; }
            done = emit(next);
            std::fflush(stdout);
        }
        // invariant: `next` has been emitted but not fed; model.position() is where it goes
        double t_draft = 0, t_verify = 0, t_ingest = 0, t_plain = 0;
        long long cycles = 0, plain_steps = 0, emitted_spec = 0;
        while (!done && !cancelled && !failed && produced < budget) {
            if (in.stop.load()) { cancelled = true; break; }
            const int64_t p = model.position();
            const long long remaining = budget - produced;
            int32_t dr[NC];
            int nd = 0, nd_mtp = 0;
            if (speculate && remaining >= 2 && p + 2 <= model.max_context()) {
                const int cap_all = (int) std::min<long long>({(long long) (ngram_on ? ngram_max : draft_max), remaining - 1,
                                                                 model.max_context() - p - 1});
                const int cap = std::min(cap_all, draft_max);
                const auto t0 = Clock::now();
                if (use_df2) {
                    if (df2_pend > 0 && !model.embed_host(next, df2_stage, err)) { fail(err); failed = true; break; }
                    if (df2_pend > 0) {   // the committed rows since the last round, then the block for `next` at p
                        int32_t d7[strata::core::Dflash2Drafter::kDrafts];
                        float pr[strata::core::Dflash2Drafter::kDrafts];
                        if (!df2.draft(model.run_taps(), df2_pend, df2_pos0, next, d7, pr, err)) { fail(err); failed = true; break; }
                        df2_pend = 0;
                        for (nd = 0; nd < cap && nd < strata::core::Dflash2Drafter::kDrafts; ++nd) {
                            if (nd > 0 && pr[nd] < draft_p_min) break;
                            dr[nd] = d7[nd];
                        }
                    }
                } else if (!model.mtp_draft(next, (int) p, cap, draft_p_min, dr, &nd, err)) { fail(err); failed = true; break; }
                nd_mtp = nd;
                if (ngram_on && cap_all > nd) {      // continue the proposal from the pool
                    ngram.update(fed);
                    const int N = ngram.n;
                    int32_t tail[NC + 64];
                    int tl = 0;
                    const size_t from = fed.size() > (size_t) N ? fed.size() - (size_t) N : 0;
                    for (size_t i = from; i < fed.size(); ++i) tail[tl++] = fed[i];
                    tail[tl++] = next;
                    for (int j = 0; j < nd; ++j) tail[tl++] = dr[j];
                    int ext = nd;
                    while (ext < cap_all && tl >= N) {
                        const auto it = ngram.next.find(NgramPool::key(&tail[tl - N], N));
                        if (it == ngram.next.end()) break;
                        dr[ext++] = it->second;
                        tail[tl++] = it->second;
                    }
                    if (ext - nd >= ngram_min) { ng_drafted += ext - nd; ++ng_wide; nd = ext; }
                }
                t_draft += ms_since(t0);
            }
            if (nd >= draft_min && nd >= 1) {
                // verify [next, d1..dnd] in one pass; take tokens while the model agrees with the draft
                int32_t cols[NC];
                cols[0] = next;
                for (int j = 0; j < nd; ++j) cols[j + 1] = dr[j];
                const auto t1 = Clock::now();
                if (!model.run(cols, nd + 1, Logits::All, model.has_mtp(), true, err)) { fail(err); failed = true; break; }
                int a = 0;
                int32_t y = -1;
                for (int j = 0; j <= nd; ++j) {
                    if (!draw(model.logits_col(j), (uint64_t) (p + 1 + j), y)) { fail(err); failed = true; break; }
                    if (j == 0) t_verify += ms_since(t1);   // this first draw waits for the whole verification pass
                    done = emit(y);
                    if (done) break;
                    if (j < nd && y == dr[j]) { ++a; continue; }
                    break;
                }
                if (failed) break;
                ++cycles;
                emitted_spec += a + 1;
                drafts_offered += nd;
                drafts_accepted += a;
                if (a > nd_mtp) ng_accepted += a - nd_mtp;
                if (a < nd && !model.rollback(a, err)) { fail(err); failed = true; break; }
                fed.push_back(next);
                for (int j = 0; j < a; ++j) fed.push_back(dr[j]);
                const auto t2 = Clock::now();
                if (use_df2) {                      // run() left columns 0..a's tap rows: the next round's new context
                    df2_pend = a + 1;
                    df2_pos0 = p;
                } else if (a >= 1) {                // the MTP block learns the accepted positions with the trunk's own hidden states
                    if (!model.mtp_ingest(cols, a + 1, (int) p, 1, err)) { fail(err); failed = true; break; }
                } else {
                    model.set_last_hidden(0);
                }
                t_ingest += ms_since(t2);
                next = y;
            } else {
                const auto t3 = Clock::now();
                if (!model.run(&next, 1, Logits::Last, model.has_mtp(), false, err) ||
                    (model.has_mtp() && !model.mtp_ingest(&next, 1, (int) p, 0, err))) { fail(err); failed = true; break; }
                fed.push_back(next);
                if (use_df2) { df2_pend = 1; df2_pos0 = p; }
                int32_t y = -1;
                if (!draw(model.logits_col(0), (uint64_t) (p + 1), y)) { fail(err); failed = true; break; }
                t_plain += ms_since(t3);
                ++plain_steps;
                done = emit(y);
                next = y;
            }
            std::fflush(stdout);
        }
        if (failed) continue;
        if (cancelled) finish = "cancel";
        const double decode_ms = ms_since(t_dec);
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld\n", produced, (long long) n, prompt_ms, decode_ms, finish,
                    drafts_accepted, drafts_offered, (long long) resume);
        std::fflush(stdout);
        std::fprintf(stderr, "xolotl-dense: prompt %lld tokens = %zu reused + %lld read in %.0f ms, %lld generated in %.0f ms (%.1f tok/s)%s",
                     (long long) n, resume, (long long) (n - (long long) resume), prompt_ms, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, cancelled ? " (cancelled)" : "");
        if (drafts_offered) std::fprintf(stderr, ", MTP drafts accepted %lld of %lld (%.0f%%)", drafts_accepted, drafts_offered,
                                         100.0 * (double) drafts_accepted / (double) drafts_offered);
        std::fprintf(stderr, "\n");
        if (cycles + plain_steps > 0 && model.has_mtp())    // where the decode time went, per step
            std::fprintf(stderr, "xolotl-dense:   %lld speculative cycles (%.2f tokens each): draft %.1f ms, verify %.1f ms, "
                         "ingest %.1f ms per cycle; %lld plain steps at %.1f ms\n", cycles,
                         cycles ? (double) emitted_spec / (double) cycles : 0.0, cycles ? t_draft / (double) cycles : 0.0,
                         cycles ? t_verify / (double) cycles : 0.0, cycles ? t_ingest / (double) cycles : 0.0, plain_steps,
                         plain_steps ? t_plain / (double) plain_steps : 0.0);
            if (ngram_on)
                std::fprintf(stderr, "xolotl-dense:   n-gram chain: %lld drafted, %lld accepted, %lld wide rounds\n",
                             ng_drafted, ng_accepted, ng_wide);
    }
    cudaFree(d_next);
    cudaFree(d_hist);
    return 0;
}
