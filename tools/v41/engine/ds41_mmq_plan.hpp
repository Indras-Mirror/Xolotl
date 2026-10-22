// tools/v41/engine/ds41_mmq_plan.hpp - run_chunk's MMQ plan for one format class (ds41_moe.cpp), shared with
// ds41_helper's prompt chunks so both cards lay out and multiply an expert the same way.  CUDA + MMQ builds only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "ds41_transcode.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

namespace strata::v41 {

// run_chunk MMQ on PTQ1_0 (type 143, not an MMQ type): the run's ternary matrices are transcoded on the card into
// Q2_0 (ds41_transcode: same d, code = trit + 1 - every weight the same number) and multiplied as Q2_0.
// STRATA_MMQ_PTQ1=0: classes with a PTQ1_0 matrix stay on MMVQ (the old behaviour).
constexpr int kPtq1 = 143, kQ20 = 42;
inline bool mmq_ptq1() {
    static const bool on = [] { const char* e = std::getenv("STRATA_MMQ_PTQ1"); return !(e && e[0] == '0'); }();
    return on;
}
inline int mmq_eff(int t) { return t == kPtq1 && mmq_ptq1() ? kQ20 : t; }
/// One class's MMQ plan: covered or not, the staged-blob stride SL (BL rounded to whole blocks of every type MMQ reads
/// in place), and the transcode layout (offsets of the Q2_0 copies in one expert's slot, SL2 = the slot).
struct MmqPlan {
    bool ok = false;
    int64_t SL = 0;
    int gt = -1, ut = -1, dt = -1;         // the types MMQ multiplies (PTQ1_0 -> Q2_0)
    bool tg = false, tu = false, td = false;   // that matrix is a transcoded PTQ1_0 one
    size_t g_bytes = 0, u_bytes = 0, d_bytes = 0;   // PTQ1_0 bytes of each transcoded matrix in the staged blob
    size_t g2 = 0, u2 = 0, d2 = 0, SL2 = 0;         // its Q2_0 copy's offset in the slot; the slot size
};
inline MmqPlan mmq_plan(const strata::kernels::NativeExpertLayout& GL, int64_t BL) {
    namespace mmq = strata::prefill::mmq;
    MmqPlan P;
    P.SL = BL;
    const int ogt = GL.gu_type, out_ = GL.up_type >= 0 ? GL.up_type : GL.gu_type, odt = GL.d_type;
    P.gt = mmq_eff(ogt);
    P.ut = mmq_eff(out_);
    P.dt = mmq_eff(odt);
    P.tg = P.gt != ogt;
    P.tu = P.ut != out_;
    P.td = P.dt != odt;
    if (!(mmq::fits(P.gt, GL.n_ff) && mmq::fits(P.ut, GL.n_ff) && mmq::fits(P.dt, GL.n_embd) && GL.up_off % 2 == 0 &&
          GL.down_off % 2 == 0))   // a finite swiglu_limit: mmq::swiglu_clamp
        return P;
    if ((P.tg || P.tu) && GL.n_embd % 128 != 0) return P;
    if (P.td && GL.n_ff % 128 != 0) return P;
    int64_t lcm = 16;   // whole blocks of every type read IN PLACE (a transcoded matrix is read by the transcode)
    auto gcd = [](int64_t a, int64_t b) { while (b) { const int64_t t = a % b; a = b; b = t; } return a; };
    const int in_place[3] = { P.tg ? -1 : P.gt, P.tu ? -1 : P.ut, P.td ? -1 : P.dt };
    for (int t : in_place) {
        if (t < 0) continue;
        const int64_t tsz = (int64_t) mmq::matrix_bytes(t, 1, 256);
        lcm = lcm / gcd(lcm, tsz) * tsz;
    }
    const int64_t SL = (BL + lcm - 1) / lcm * lcm;
    if (SL - BL > 1048576 - 4096) return P;
    P.SL = SL;
    const size_t ptq1_row_gu = (size_t) (GL.n_embd / 128) * 28, ptq1_row_d = (size_t) (GL.n_ff / 128) * 28;
    size_t at = 0;
    if (P.tg) { P.g_bytes = (size_t) GL.n_ff * ptq1_row_gu; P.g2 = at; at += ds41_q2_0_bytes_from_ptq1(P.g_bytes); }
    if (P.tu) { P.u_bytes = (size_t) GL.n_ff * ptq1_row_gu; P.u2 = at; at += ds41_q2_0_bytes_from_ptq1(P.u_bytes); }
    if (P.td) { P.d_bytes = (size_t) GL.n_embd * ptq1_row_d; P.d2 = at; at += ds41_q2_0_bytes_from_ptq1(P.d_bytes); }
    P.SL2 = at;   // a multiple of 18 (whole Q2_0 blocks): MMQ's expert stride is counted in blocks
    P.ok = true;
    return P;
}

}  // namespace strata::v41
