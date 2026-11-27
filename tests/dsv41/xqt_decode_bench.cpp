// Strata. The engine's CPU expert lane timed on one ERNIE-shaped expert (gate_up 2048 -> 2048x2, down 2048 -> 2048):
// the pair trellis TQP5 / TQP6 through ds4_iq_cpu's dispatcher (iq_cpu_* -> ds4_tq_cpu.cpp) against the ggml block
// types the lane already reads (iq_cpu_* -> ds4_gg_cpu.cpp: ggml-cpu's row dots), 1 and 8 tokens.
//   xqt_decode_bench [--n 2048] [--reps 3] [--only TQP5] [--pool THREADS EXPERTS]
// --pool: the ExpertPool's shape instead - EXPERTS distinct copies of the expert (more bytes than L3, so the weights
// stream from DRAM as in decode), THREADS threads each taking every THREADS-th expert, 1 token; prints the wall time
// of one pass over all EXPERTS (median of reps) and experts/s.
// Prints a CSV: type,bpw,tokens,gu_us,down_us,expert_us,gb_per_s,expert_bytes.  CPU only, no device.
#include "../../src/dsv41/ds4_gguf_cpu.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace gg = strata::ds4::gguf;

namespace {

struct Cand {
    const char* name;
    int type;
    int kb;
};
const Cand kCands[] = {
    {"TQP5", 147, 5}, {"TQP6", 148, 6},
    {"IQ2_XXS", 16, 0}, {"IQ2_XS", 17, 0}, {"Q2_K", 10, 0}, {"IQ3_XXS", 18, 0},
};

double bpw(int type, int kb) {
    if (kb) return 8.0 * (2 + 8 * kb) / 128.0;
    return 8.0 * (double) ggml_type_size((ggml_type) type) / (double) ggml_blck_size((ggml_type) type);
}

// rows x cols weights -> the type's packed rows.  TQP: blocks of the right size with a normal fp16 scale and a
// deterministic byte pattern - a SPEED bench, and the decoder's work (the state extraction and the table lookups) is
// the same whatever the bits say; a real encode of 12.6 M weights would cost hours (tools/trellis/tqenc_pair.hpp is
// ~2.6 k weights/s a thread) and change nothing here.  The other types: ggml's own quantizer over the real weights.
std::vector<uint8_t> pack(const Cand& c, const std::vector<float>& W, int rows, int cols) {
    if (c.kb) {
        const int bsz = 2 + 8 * c.kb;
        std::vector<uint8_t> out((size_t) rows * (cols / 128) * bsz);
        uint32_t x = 12345;
        for (size_t i = 0; i < out.size(); i += (size_t) bsz) {
            out[i] = 0x00;              // fp16 d = 0x2c00 (~0.0615): a normal value, no denormal arithmetic
            out[i + 1] = 0x2c;
            for (int j = 2; j < bsz; ++j) { x = x * 1664525u + 1013904223u; out[i + (size_t) j] = (uint8_t) (x >> 24); }
        }
        return out;
    }
    const size_t rb = ggml_row_size((ggml_type) c.type, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    std::vector<float> im((size_t) cols, 1.0f);
    ggml_quantize_init((ggml_type) c.type);
    const size_t got = ggml_quantize_chunk((ggml_type) c.type, W.data(), out.data(), 0, rows, cols, im.data());
    if (got != (size_t) rows * rb) { std::fprintf(stderr, "%s: quantize wrote %zu\n", c.name, got); std::abort(); }
    return out;
}

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

// one pass over `ne` expert copies by `nth` threads (thread k: experts k, k + nth, ...), 1 token each; wall ms
double pool_pass(const gg::IqCpu& f, const std::vector<std::vector<uint8_t>>& G, const std::vector<std::vector<uint8_t>>& U,
                 const std::vector<std::vector<uint8_t>>& D, int nth, int dim, int ff, const std::vector<float>& x0) {
    const double t0 = now_ms();
    std::vector<std::thread> th;
    for (int k = 0; k < nth; ++k)
        th.emplace_back([&, k] {
            std::vector<uint8_t> act(f.act_bytes), hq(f.h_bytes);
            std::vector<float> h((size_t) ff), o((size_t) dim);
            gg::iq_cpu_quant(f.gu_act, x0.data(), dim, act.data());
            const void* ap = act.data();
            const void* hp = hq.data();
            float* hpo = h.data();
            float* op = o.data();
            for (size_t e = (size_t) k; e < G.size(); e += (size_t) nth) {
                gg::iq_cpu_gate_up(f, G[e].data(), U[e].data(), &ap, 1, &hpo, 0, ff, 0.0f);
                gg::iq_cpu_quant(f.d_act, h.data(), ff, hq.data());
                gg::iq_cpu_down(f, D[e].data(), &hp, 1, &op, 0, dim);
            }
        });
    for (auto& t : th) t.join();
    return now_ms() - t0;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 2048, reps = 3;
    std::string only;
    int pool_t = 0, pool_e = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--n") n = std::atoi(argv[++i]);
        else if (a == "--reps") reps = std::atoi(argv[++i]);
        else if (a == "--only") only = argv[++i];
        else if (a == "--pool") { pool_t = std::atoi(argv[++i]); pool_e = std::atoi(argv[++i]); }   // one type by name (perf stat on one kernel)
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    const int dim = n, ff = n;
    std::mt19937_64 rng(5);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> Wg((size_t) ff * dim), Wu((size_t) ff * dim), Wd((size_t) dim * ff);
    for (auto* m : {&Wg, &Wu, &Wd})
        for (auto& v : *m) v = nd(rng) * (rng() % 16 == 0 ? 4.0f : 1.0f);

    std::printf("type,bpw,tokens,gu_us,down_us,expert_us,expert_us_min,expert_cpu_us,gb_per_s,expert_cpu_gb_s,expert_bytes\n");
    for (const Cand& c : kCands) {
        if (!only.empty() && only != c.name) continue;
        gg::IqCpu f;
        std::string err;
        if (!gg::iq_cpu_fmt(c.type, c.type, dim, ff, f, err)) {
            std::fprintf(stderr, "%s: iq_cpu_fmt: %s\n", c.name, err.c_str());
            return 1;
        }
        std::vector<uint8_t> gate = pack(c, Wg, ff, dim);
        std::vector<uint8_t> up = pack(c, Wu, ff, dim);
        std::vector<uint8_t> down = pack(c, Wd, dim, ff);
        const double bytes = (double) gate.size() + up.size() + down.size();
        if (pool_t > 0) {
            std::vector<std::vector<uint8_t>> G((size_t) pool_e, gate), U((size_t) pool_e, up), D((size_t) pool_e, down);
            std::vector<float> x0((size_t) dim);
            for (auto& v : x0) v = nd(rng);
            std::vector<double> ts;
            for (int r = 0; r <= reps; ++r) {
                const double ms = pool_pass(f, G, U, D, pool_t, dim, ff, x0);
                if (r > 0) ts.push_back(ms);
            }
            std::sort(ts.begin(), ts.end());
            const double med = ts[ts.size() / 2];
            std::printf("POOL %s,%.4f,threads %d,experts %d,%.1f MB,pass %.2f ms (min %.2f),%.0f experts/s,%.1f GB/s\n",
                        c.name, bpw(c.type, c.kb), pool_t, pool_e, bytes * pool_e / 1e6, med, ts.front(),
                        pool_e / (med * 1e-3), bytes * pool_e / (med * 1e-3) / 1e9);
            std::fflush(stdout);
            continue;
        }

        for (int nt : {1, 8}) {
            // activations: x (n_embd) per token, and h (n_ff) the gate_up writes (then quantized for down)
            std::vector<std::vector<uint8_t>> act((size_t) nt), hq((size_t) nt);
            std::vector<const void*> ap((size_t) nt), hp((size_t) nt);
            std::vector<std::vector<float>> ff_out((size_t) nt), out((size_t) nt);
            std::vector<float*> ffp((size_t) nt), outp((size_t) nt);
            std::vector<float> x0((size_t) dim);
            for (auto& v : x0) v = nd(rng);
            for (int t = 0; t < nt; ++t) {
                act[(size_t) t].resize(f.act_bytes);
                gg::iq_cpu_quant(f.gu_act, x0.data(), dim, act[(size_t) t].data());
                ap[(size_t) t] = act[(size_t) t].data();
                hq[(size_t) t].resize(f.h_bytes);
                ff_out[(size_t) t].assign((size_t) ff, 0.f);
                ffp[(size_t) t] = ff_out[(size_t) t].data();
                out[(size_t) t].assign((size_t) dim, 0.f);
                outp[(size_t) t] = out[(size_t) t].data();
            }
            std::vector<double> tgu, tdn, tcpu;
            for (int r = 0; r < reps; ++r) {
                const std::clock_t c0 = std::clock();   // CPU time: load-independent
                const double t0 = now_ms();
                gg::iq_cpu_gate_up(f, gate.data(), up.data(), ap.data(), nt, ffp.data(), 0, ff, 0.0f);
                const double t1 = now_ms();
                for (int t = 0; t < nt; ++t) {
                    gg::iq_cpu_quant(f.d_act, ff_out[(size_t) t].data(), ff, hq[(size_t) t].data());
                    hp[(size_t) t] = hq[(size_t) t].data();
                }
                gg::iq_cpu_down(f, down.data(), hp.data(), nt, outp.data(), 0, dim);
                const double t2 = now_ms();
                const double cpu = (double) (std::clock() - c0) / CLOCKS_PER_SEC * 1e3;
                if (r == 0) continue;   // (the first pass warms the tables / page cache)
                tgu.push_back(t1 - t0);
                tdn.push_back(t2 - t1);
                tcpu.push_back(cpu);
            }
            std::sort(tgu.begin(), tgu.end());
            std::sort(tdn.begin(), tdn.end());
            std::sort(tcpu.begin(), tcpu.end());
            const double gu = tgu.empty() ? 0.0 : tgu[tgu.size() / 2];
            const double dn = tdn.empty() ? 0.0 : tdn[tdn.size() / 2];
            const double tot = gu + dn;
            // the min: the least-preempted run (other work on the machine hits the median, not the minimum)
            const double mn = (tgu.empty() || tdn.empty()) ? 0.0 : tgu.front() + tdn.front();
            const double cpu = tcpu.empty() ? 0.0 : tcpu[tcpu.size() / 2];
            std::printf("%s,%.4f,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%.0f\n", c.name, bpw(c.type, c.kb), nt, gu * 1e3,
                        dn * 1e3, tot * 1e3, mn * 1e3, cpu * 1e3, tot > 0 ? bytes / (tot * 1e-3) / 1e9 : 0.0,
                        cpu > 0 ? bytes / (cpu * 1e-3) / 1e9 : 0.0, bytes);
            std::fflush(stdout);
        }
    }
    return 0;
}
