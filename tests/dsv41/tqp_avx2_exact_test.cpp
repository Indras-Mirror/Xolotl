// Strata. The TQP (147-149) AVX2 row kernel against the scalar reference, bit for bit: random blocks (every state,
// every wrap window), random int16 activations and scales, KB 5/6/7, 1..8 tokens, odd row splits.  Also checks the
// 8 KB point table against the first pair of every state-table row.  CPU only; skips (exit 0) without AVX2.
//   tqp_avx2_exact_test
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace kc = strata::kernels::cpu;
namespace td = strata::kernels::cpu::tq_detail;

int main() {
    if (!kc::tq_isa_ok(kc::TqIsa::avx2)) { std::printf("no AVX2: skipped\n"); return 0; }
    int fails = 0;
    {
        const int16_t* tab = kc::tq_state_table();
        const uint32_t* pts = kc::tq_point_table();
        int bad = 0;
        for (uint32_t s = 0; s < 32768; ++s) {
            const uint32_t p = (s * 0x9e3779b1u) >> 21;
            const uint32_t pr = (uint32_t) (uint16_t) tab[4 * s] | ((uint32_t) (uint16_t) tab[4 * s + 1] << 16);
            bad += pr != pts[p];
        }
        std::printf("point table vs state table: %d mismatches\n", bad);
        fails += bad != 0;
    }
    std::mt19937 rng(7);
    const int n = 2048, rows = 37;
    std::vector<std::unique_ptr<td::TqAct>> act;
    for (int t = 0; t < kc::kTqMaxTokens; ++t) {
        act.emplace_back(new td::TqAct());
        for (int i = 0; i < n; ++i) act[t]->q[i] = (int16_t) ((int) (rng() % 255) - 127);
        for (int i = 0; i < n / 32; ++i) act[t]->d[i] = std::ldexp((float) (rng() % 1000 + 1), -14);
    }
    const td::TqAct* ap[kc::kTqMaxTokens];
    for (int t = 0; t < kc::kTqMaxTokens; ++t) ap[t] = act[t].get();
    for (int kb = 5; kb <= 7; ++kb) {
        const size_t bsz = 2 + 8 * kb, rb = (size_t) (n / 128) * bsz;
        std::vector<uint8_t> w(rb * rows);
        for (auto& b : w) b = (uint8_t) rng();
        for (size_t i = 0; i < w.size(); i += bsz) { w[i] = (uint8_t) rng(); w[i + 1] = (uint8_t) (0x20 + rng() % 0x10); }
        for (int nt = 1; nt <= kc::kTqMaxTokens; ++nt) {
            std::vector<float> ref((size_t) nt * rows), got((size_t) nt * rows, -1.0f);
            float* pr[kc::kTqMaxTokens];
            float* pg[kc::kTqMaxTokens];
            for (int t = 0; t < nt; ++t) { pr[t] = ref.data() + (size_t) t * rows; pg[t] = got.data() + (size_t) t * rows; }
            td::tq_rows_scalar_tqp(kb, w.data(), rb, n, ap, nt, pr, 0, rows);
            td::tq_rows_avx2_tqp(kb, w.data(), rb, n, ap, nt, pg, 0, 13);
            td::tq_rows_avx2_tqp(kb, w.data(), rb, n, ap, nt, pg, 13, rows);
            const bool ok = std::memcmp(ref.data(), got.data(), ref.size() * 4) == 0;
            if (!ok) { std::printf("KB %d nt %d: MISMATCH (ref %.9g got %.9g)\n", kb, nt, ref[0], got[0]); ++fails; }
        }
        std::printf("KB %d: 1..%d tokens checked\n", kb, kc::kTqMaxTokens);
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
