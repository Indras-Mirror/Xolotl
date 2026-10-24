// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

// PTQ1_0 (ternary, id 143) is Strata's own: the vendored ggml stops at GGML_TYPE_COUNT 43, so it must never see 143.
const ggml_type_traits_cpu kPtq1Traits = {nullptr, ptq1_0_vec_dot_q8_0, GGML_TYPE_Q8_0, 1};
const ggml_type_traits_cpu* traits(int type) {
    if (type == kPtq1Type) return &kPtq1Traits;
    return type >= 0 && type < GGML_TYPE_COUNT ? ggml_get_type_traits_cpu((ggml_type) type) : nullptr;
}
int64_t blck(int type) { return type == kPtq1Type ? kPtq1Block : ggml_blck_size((ggml_type) type); }
size_t row_size(int type, int64_t n) { return type == kPtq1Type ? ptq1_0_row_bytes(n) : ggml_row_size((ggml_type) type, n); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp / kq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool kq256_supported(int type) noexcept { return type == 12 || type == 7 || type == 8; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % blck(gu_type) || n_ff % blck(d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = row_size(gu_type, n_embd);
    f.d_row = row_size(d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    // NVFP4 in any projection: one scale tail per blob; a mixed expert (e.g. NVFP4 gate/up + Q8_0 down) has 1.0 for
    // the other projection's scale (tools/iq_pack.py), so every path below multiplies by the tail unconditionally
    if (gu_type == kNvfp4Type || d_type == kNvfp4Type) {
        f.tail_off = f.bytes;
        f.bytes += kNvfp4Tail;
    }
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

bool native_fmt3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f,
                 std::string& err) {
    if (!native_fmt(gate_type, d_type, n_embd, n_ff, f, err)) return false;
    if (up_type == gate_type) return true;
    const ggml_type_traits_cpu* tu = traits(up_type);
    if (tu == nullptr || tu->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(up_type);
        return false;
    }
    if ((int) tu->vec_dot_type != f.gu_act || n_embd % blck(up_type)) {
        err = "native experts: up type " + std::to_string(up_type) + " needs a different activation than gate type " +
              std::to_string(gate_type);
        return false;
    }
    f.up_type = up_type;
    f.up_row = row_size(up_type, n_embd);
    f.down_off = f.up_off + f.up_row * (size_t) n_ff;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
// the older CPUs' copy (iq_avx1.cpp): byte-identical like the AVX-2 one, and only where AVX2 is absent (same
// policy as STRATA_NO_IQ128 above; STRATA_NO_Q8K_AVX1=1 keeps ggml's scalar reference on those CPUs).
bool q8k_avx1(int type) {
    static const bool on = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_Q8K_AVX1") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
}  // namespace

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    if (q8k_avx1(f.gu_act)) { q8k_quant_avx1(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    if (q8k_avx1(f.d_act)) { q8k_quant_avx1(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

int native_gu_mt_min(int gu_type) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min = env ? std::atoi(env) : 2;
    // IQ3_S where the AVX-2 kernel gathers its grid (cpu_gather_fast, unless STRATA_IQ256_GATHER=0) on a CPU without
    // AVX-512: there that kernel beats ggml's dot for ONE token too (14900KF P-core, an expert's gate/up rows: 0.34 ->
    // 0.19 ms; its E-cores, which keep the scalar decode: 0.71 -> 0.68), and in decode most CPU experts serve one token
    // of the window.  So IQ3_S takes it for every group, and those rows no longer depend on the drafting.  A machine-
    // wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in expert_layout.cpp
    // and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  STRATA_IQ_MT_MIN set keeps its rule.
    // Opt-in (STRATA_IQ3S_MT1=1): it changes a lone token's IQ3_S rounding on those CPUs, so the default stays 0.1.39's.
    static const bool iq3s_one = env == nullptr && std::getenv("STRATA_IQ3S_MT1") != nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    return iq3s_one && gu_type == 21 ? 1 : mt_min;
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    // STRATA_NO_IQ256 drops the AVX-2 kernel.  cpu_avx2_ok() is ALSO required: iq_avx2.cpp and kq_avx2.cpp
    // are compiled /arch:AVX2 and use AVX2 and FMA3, so calling them on a CPU without either is an
    // illegal instruction, not a slow path.  The gate sat next to `avx512` above, which does test
    // cpu_avx512_ok(), and that asymmetry is what let a pre-Haswell CPU reach iq256_gu_rows.
    // Without it this falls through to ggml-cpu's vec_dot below, which is compiled for whatever
    // baseline the build selected (AVX1 here) and covers the same types - IQ2_XXS and IQ2_S among
    // them.  That path loops over tokens itself, so it is correct for any `nt`, not just one.
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    // The older-CPU tier (STRATA_ISA_FLOOR=avx builds): without it every expert on such a CPU runs on ggml-cpu's
    // scalar _generic dots.  Only where AVX2 is absent, so a modern CPU's rounding never moves (and
    // STRATA_NO_IQ256=1 there still means ggml, as before).  iq_avx1.cpp is compiled for AVX and gated on
    // cpu_avx1_ok().  Measured on an E5-2470 v2 (iq_avx1_parity --bench): 1.2-3.1x ggml's dot from two tokens on,
    // at one token ggml's auto-vectorized generic loop matches the 128-bit kernel - so the same #152 mt_min rule
    // as the AVX-2 kernels, and STRATA_IQ_MT_MIN=1 takes the kernel for one token too (its rows are then
    // independent of the drafting: iq_avx1_parity checks every width bit for bit).
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    const int mt_min = native_gu_mt_min(f.gu_type);   // #152
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return cpu_avx2_ok() && v != nullptr && std::atoi(v) != 0; }();
    if (kq && f.gu_type == 12 && f.up_type < 0 && nt >= 2 && std::isinf(f.swiglu_limit)) {   // kq256 has no clamp   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    const bool mixed = f.up_type >= 0 && f.up_type != f.gu_type;   // the multi-token kernels take one type
    if (!mixed && nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
            return;
        }
    }
    // PTQ1_0 (ternary) gate/up: ptq1_0_rows_q8_0 decodes each block once for every token (3.9x the per-token dot at one
    // token, 6-8x at 2-3; s11 2026-10-15).  STRATA_NO_PTQ1_ROWS=1 = the per-token dot below.
    static const bool ptq_rows = std::getenv("STRATA_NO_PTQ1_ROWS") == nullptr;
    if (ptq_rows && f.gu_type == kPtq1Type && !mixed && r1 > r0) {
        const int R = r1 - r0, n = (int) f.n_embd;
        thread_local std::vector<float> gb, ub;
        gb.resize((size_t) nt * (size_t) R);
        ub.resize((size_t) nt * (size_t) R);
        float* gp[MAXT];
        float* upp[MAXT];
        for (int t = 0; t < nt; ++t) { gp[t] = gb.data() + (size_t) t * R; upp[t] = ub.data() + (size_t) t * R; }
        ptq1_0_rows_q8_0(blob + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, gp, 0, R);
        ptq1_0_rows_q8_0(blob + f.up_off + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, upp, 0, R);
        const float lim = f.swiglu_limit;
        for (int t = 0; t < nt; ++t)
            for (int i = 0; i < R; ++i) {
                const float g = std::min(gp[t][i], lim);       // DeepSeek-V4's clamp, as below
                const float u = std::min(std::max(upp[t][i], -lim), lim);
                ff[t][r0 + i] = (g / (1.f + std::exp(-g))) * u;
            }
        return;
    }
    // Q2_0 gate/up (ggml type 42, the V4.1 XQ middle tier): ggml-cpu's Q2_0 dot is the scalar _generic one on x86
    // (~0.5 GB/s a core); q2_0_rows_q8_0 decodes each block once for every token in AVX2.  STRATA_NO_Q2_ROWS=1 = ggml.
    static const bool q2_rows = cpu_avx2_ok() && std::getenv("STRATA_NO_Q2_ROWS") == nullptr;
    if (q2_rows && f.gu_type == kQ2_0Type && !mixed && r1 > r0) {
        const int R = r1 - r0, n = (int) f.n_embd;
        thread_local std::vector<float> gb, ub;
        gb.resize((size_t) nt * (size_t) R);
        ub.resize((size_t) nt * (size_t) R);
        float* gp[MAXT];
        float* upp[MAXT];
        for (int t = 0; t < nt; ++t) { gp[t] = gb.data() + (size_t) t * R; upp[t] = ub.data() + (size_t) t * R; }
        q2_0_rows_q8_0(blob + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, gp, 0, R);
        q2_0_rows_q8_0(blob + f.up_off + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, upp, 0, R);
        const float lim = f.swiglu_limit;
        for (int t = 0; t < nt; ++t)
            for (int i = 0; i < R; ++i) {
                const float g = std::min(gp[t][i], lim);       // DeepSeek-V4's clamp, as below
                const float u = std::min(std::max(upp[t][i], -lim), lim);
                ff[t][r0 + i] = (g / (1.f + std::exp(-g))) * u;
            }
        return;
    }
    // UNION (xolotl): upstream's older-CPU 128-bit tier - independent of the PTQ1_0 path above (disjoint quant types)
    if (avx1 && nt >= mt_min && iq128_supported(f.gu_type)) {
        iq128_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    const int n = (int) f.n_embd;
    // NVFP4: the per-expert global scales from the blob's tail (1 and 1 otherwise). s_down goes on the down output:
    // folded into up it left the hidden near 1e-5, whose Q8_0 block scale is an FP16 subnormal (expert error 2-12%)
    float sg = 1.f, su = 1.f;
    if (f.tail_off) {
        float tail[4];
        std::memcpy(tail, blob + f.tail_off, sizeof tail);
        sg = tail[0];
        su = tail[1];
    }
    // NVFP4 in 512-bit lanes: ggml's arithmetic, blocks decoded once per window; 1.8x at one token, 3.7x at seven
    // (nvfp4_avx512_parity). STRATA_NO_NVFP4_512 falls back to ggml-cpu.
    static const bool nvfp4_512 = cpu_avx512_ok() && std::getenv("STRATA_NO_NVFP4_512") == nullptr;
    if (f.gu_type == kNvfp4Type && nvfp4_512 && nvfp4_512_fits(n)) {
        nvfp4_512_gu_rows(blob, f.gu_row, f.up_off, n, act, nt, ff, r0, r1, sg, su);
        return;
    }
    // Without AVX-512: the same rows in 256-bit lanes, bit-equal to ggml-cpu's AVX2 dot below; 2.0-2.4x it at 4-8
    // tokens, 2x under DRAM load with 6 threads (nvfp4_avx512_parity). STRATA_NO_NVFP4_256 falls back to ggml-cpu.
    static const bool nvfp4_256 = cpu_avx2_ok() && std::getenv("STRATA_NO_NVFP4_256") == nullptr;
    if (f.gu_type == kNvfp4Type && nvfp4_256 && nvfp4_256_fits(n)) {
        nvfp4_256_gu_rows(blob, f.gu_row, f.up_off, n, act, nt, ff, r0, r1, sg, su);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const ggml_vec_dot_t udot = mixed ? traits(f.up_type)->vec_dot : dot;
    const size_t urow = mixed ? f.up_row : f.gu_row;
    const float lim = f.swiglu_limit;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * urow;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            udot(n, &u, 0, ur, 0, act[t], 0, 1);
            g *= sg;                                     // NVFP4's per-expert scales (1 otherwise)
            u *= su;
            g = std::min(g, lim);                        // DeepSeek-V4's clamp; lim = +inf: no-op
            u = std::min(std::max(u, -lim), lim);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("STRATA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    // Both multi-token kernels below are /arch:AVX2 translation units (kq_avx2.cpp and iq_avx2.cpp),
    // so a CPU without AVX2 has to reach ggml-cpu's vec_dot instead - same reasoning as the gate/up
    // rows above, where `avx512` tested cpu_avx512_ok() and `avx2` did not.
    if (cpu_avx2_ok() && kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8)) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (cpu_avx2_ok() && nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    static const bool ptq_rows = std::getenv("STRATA_NO_PTQ1_ROWS") == nullptr;
    if (ptq_rows && f.d_type == kPtq1Type) {   // PTQ1_0 down: as the gate/up rows above
        ptq1_0_rows_q8_0(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    static const bool q2_rows = cpu_avx2_ok() && std::getenv("STRATA_NO_Q2_ROWS") == nullptr;
    if (q2_rows && f.d_type == kQ2_0Type) {   // Q2_0 down: as the gate/up rows (ggml's x86 dot is scalar)
        q2_0_rows_q8_0(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // UNION (xolotl): the older-CPU tier - IQ4_NL and Q2_0 down rows in 128-bit (ggml-cpu's dots for both are scalar here)
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    if (avx1 && iq128_down_supported(f.d_type)) {
        iq128_down_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const int n = (int) f.n_ff;
    float sd = 1.f;                                   // NVFP4: the expert's s_down (see native_gu_rows)
    if (f.tail_off) std::memcpy(&sd, blob + f.tail_off + 2 * sizeof(float), sizeof sd);
    static const bool nvfp4_512 = cpu_avx512_ok() && std::getenv("STRATA_NO_NVFP4_512") == nullptr;
    if (f.d_type == kNvfp4Type && nvfp4_512 && nvfp4_512_fits(n)) {
        nvfp4_512_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1, sd);
        return;
    }
    static const bool nvfp4_256 = cpu_avx2_ok() && std::getenv("STRATA_NO_NVFP4_256") == nullptr;   // see above
    if (f.d_type == kNvfp4Type && nvfp4_256 && nvfp4_256_fits(n)) {
        nvfp4_256_rows(blob + f.down_off, f.d_row, n, hq, nt, out, r0, r1, sd);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s * sd;
        }
    }
}

}  // namespace strata::kernels::cpu
