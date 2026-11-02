// tools/lfm3/lfm3_speech.cpp - `xolotl-lfm3-speech`: LFM3-Omni's speech generator for xolotl-lfm3 (local, Indra's Mirror).
//
// LFM3-Omni speaks by sampling one frame of 12 audio codes per step from the text model's last hidden state (a small
// "depthformer" in the mmproj, MaskGIT: three passes reveal 4 + 4 + 4 codebooks in a random order), feeding the frame's
// embedding back to the text model as its next input row, and turning the codes into 24 kHz audio with the Liquid
// codec (+ ISTFT).  This program is that half: the depthformer, the sampling and the codec, through llama.cpp's mtmd
// (tools/mtmd/mtmd-helper-gen.cpp, class lfm3a_gen_audio_pipeline, upstream f3586e6, MIT - the frame sampler below is a
// port of its sample_frame / apply_rep_penalty / sample / decode_wav).  The engine (src/program/lfm3_main.cpp) runs the
// text model and starts this program the first time it is asked to speak; a separate process keeps libmtmd's shared
// ggml away from the engine's own vendored ggml.
//
//   xolotl-lfm3-speech --mmproj <mmproj.gguf> --model <lfm3.gguf> [--gpu] [--threads N]
//
// Binary protocol (little-endian; stdout carries only the protocol, every log line goes to stderr):
//   out at start:  int32 'LSP1', n_embd, sample_rate, n_codebooks
//   in  1 RESET:   uint32 seed (UINT32_MAX = random), float temp, int32 top_k, float rep_penalty, int32 rep_window
//                  (temp < 0 / top_k < 0 / rep_penalty < 0 / rep_window < 0 = the checkpoint's default)
//       out:       int32 0
//   in  2 FRAME:   float h[n_embd]  (the text model's output-normed hidden state at the last position)
//       out:       int32 status (0 = a frame, 1 = end of audio, -1 = error), int32 codes[n_codebooks],
//                  status 0: float feedback[n_embd], int32 n_samples, float pcm[n_samples]
//   in  3 FLUSH:   out int32 n_samples, float pcm[n_samples]   (the codec's held-back tail; resets the codec)
//   in  4 QUIT
#include "gguf_meta.h"
#include "llama.h"
#include "mtmd.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

FILE* proto = nullptr;   // the protocol stream (the real stdout; fd 1 is pointed at stderr)

bool rd(void* p, size_t n) { return std::fread(p, 1, n, stdin) == n; }
void wr(const void* p, size_t n) { std::fwrite(p, 1, n, proto); }
void wr_i(int32_t v) { wr(&v, sizeof v); }

constexpr int kCb = 12;                       // codebooks (checked by llama.cpp's converter)
constexpr int kSchedule[] = {4, 4, 4};        // codebooks revealed per depthformer pass

struct Speech {
    mtmd_context* mctx = nullptr;
    mtmd_gen_state* gs = nullptr;
    int n_embd = 0;
    float temp = 0.0f, rep_penalty = 1.0f;
    int top_k = 0, rep_window = 0;
    std::mt19937 rng;
    std::vector<int32_t> hist;                // frame-major codes of the frames so far
    std::vector<float> logits, fb;
    bool pending_tail = false;

    bool reset(uint32_t seed, float t, int k, float rp, int rw) {
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        temp = t >= 0 ? t : def.temp;
        top_k = k >= 0 ? k : def.top_k;
        rep_penalty = rp >= 0 ? rp : def.penalty_repeat;
        rep_window = rw >= 0 ? rw : def.penalty_last_n;
        rng.seed(seed == UINT32_MAX ? std::random_device{}() : seed);
        hist.clear();
        pending_tail = false;
        if (gs) mtmd_gen_state_free(gs);
        mtmd_gen_state_params p = mtmd_gen_state_params_default();
        p.seed = seed;
        gs = mtmd_gen_state_init(mctx, p);
        return gs != nullptr;
    }

    void apply_rep_penalty(float* l, size_t n_vocab, int cb) const {
        if (rep_penalty <= 1.0f || rep_window <= 0) return;
        const size_t n_hist = hist.size() / kCb;
        const size_t n_win = std::min(n_hist, (size_t) rep_window);
        std::vector<bool> seen(n_vocab, false);
        for (size_t f = n_hist - n_win; f < n_hist; f++) {
            const int32_t c = hist[f * kCb + (size_t) cb];
            if (c >= 0 && (size_t) c < n_vocab && !seen[(size_t) c]) {
                seen[(size_t) c] = true;
                l[c] = l[c] > 0.0f ? l[c] / rep_penalty : l[c] * rep_penalty;
            }
        }
    }

    int32_t sample(const float* l, size_t n_vocab) {
        if (temp <= 0.0f) return (int32_t) (std::max_element(l, l + n_vocab) - l);
        float lmin = -INFINITY;                // top-k keeps the codes scoring at least the k-th best, ties included
        if (top_k > 0 && (size_t) top_k < n_vocab) {
            std::vector<float> sorted(l, l + n_vocab);
            std::nth_element(sorted.begin(), sorted.begin() + (top_k - 1), sorted.end(), std::greater<float>());
            lmin = sorted[(size_t) top_k - 1];
        }
        const float lmax = *std::max_element(l, l + n_vocab);
        std::vector<double> prob(n_vocab);
        double sum = 0.0;
        for (size_t v = 0; v < n_vocab; v++) {
            prob[v] = l[v] < lmin ? 0.0 : std::exp((double) (l[v] - lmax) / temp);
            sum += prob[v];
        }
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
        for (size_t v = 0; v < n_vocab; v++) {
            r -= prob[v];
            if (r <= 0.0) return (int32_t) v;
        }
        return (int32_t) n_vocab - 1;
    }

    // one frame: codes, and (not at the end of audio) the feedback row; -1 on a failed pass
    int frame(const float* h, std::vector<int32_t>& codes) {
        std::vector<int> order(kCb);
        for (int i = 0; i < kCb; i++) order[(size_t) i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        codes.assign(kCb, -1);
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.embd = const_cast<float*>(h);
        inp.codes = codes.data();
        inp.n_codes = codes.size();
        mtmd_gen_out out{};
        int k = 0;
        for (int n_pick : kSchedule) {
            if (mtmd_gen_audio_process(mctx, gs, &inp, &out) != 0 || out.n_feats == 0 || out.n_feats % kCb != 0) return -1;
            const size_t n_vocab = out.n_feats / kCb;
            logits.assign(out.feats, out.feats + out.n_feats);
            for (int j = 0; j < n_pick; j++, k++) {
                const int i = order[(size_t) k];
                float* l = logits.data() + (size_t) i * n_vocab;
                apply_rep_penalty(l, n_vocab, i);
                codes[(size_t) i] = sample(l, n_vocab);
            }
        }
        const int32_t eos = (int32_t) (logits.size() / kCb) - 1;   // the last code of every codebook ends the audio
        if (std::find(codes.begin(), codes.end(), eos) != codes.end()) {
            codes.assign(kCb, eos);
            return 1;
        }
        if (mtmd_gen_audio_process(mctx, gs, &inp, &out) != 0 || !out.embd) return -1;   // all codes known: the feedback row
        fb.assign(out.embd, out.embd + n_embd);
        hist.insert(hist.end(), codes.begin(), codes.end());
        return 0;
    }

    // the codec: PCM for these codes (n == 0 flushes the ISTFT tail and resets the codec)
    bool wav(const int32_t* codes, size_t n, std::vector<float>& pcm) {
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes = const_cast<int32_t*>(codes);
        inp.n_codes = n;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, gs, &inp, &out) != 0) return false;
        pcm.assign(out.audio, out.audio + out.n_samples);
        pending_tail = n > 0;
        return true;
    }
};

void usage() {
    std::fprintf(stderr, "usage: xolotl-lfm3-speech --mmproj <mmproj.gguf> --model <lfm3.gguf> [--gpu] [--threads N]\n");
}

}  // namespace

int main(int argc, char** argv) {
    // the protocol keeps the real stdout; anything a library prints goes to stderr
    const int pfd = dup(1);
    if (pfd < 0 || dup2(2, 1) < 0) { std::fprintf(stderr, "xolotl-lfm3-speech: cannot set up stdout\n"); return 1; }
    proto = fdopen(pfd, "wb");
    std::string mmproj, model;
    bool gpu = false;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
        else if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--gpu") gpu = true;
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else { usage(); return 2; }
    }
    if (mmproj.empty() || model.empty()) { usage(); return 2; }

    llama_backend_init();
    std::string how;
    llama_model* text = strata_open_text_model(model, &how);   // the vocabulary only, no weights
    if (!text) { std::fprintf(stderr, "xolotl-lfm3-speech: cannot open the text model %s (%s)\n", model.c_str(), how.c_str()); return 1; }
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.print_timings = false;
    cp.warmup = false;
    if (threads > 0) cp.n_threads = threads;
    Speech s;
    s.mctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!s.mctx) { std::fprintf(stderr, "xolotl-lfm3-speech: cannot load %s\n", mmproj.c_str()); return 1; }
    const mtmd_gen_audio_info info = mtmd_gen_audio_get_info(s.mctx);
    if (info.type != MTMD_GEN_AUDIO_TYPE_LFM3A) {
        std::fprintf(stderr, "xolotl-lfm3-speech: %s has no LFM3 speech generator\n", mmproj.c_str());
        return 1;
    }
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(model.c_str(), gp);
        const int64_t k = gg ? gguf_find_key(gg, "lfm3.embedding_length") : -1;
        if (k >= 0) s.n_embd = (int) gguf_get_val_u32(gg, k);
        if (gg) gguf_free(gg);
    }
    if (s.n_embd <= 0) { std::fprintf(stderr, "xolotl-lfm3-speech: no lfm3.embedding_length in %s\n", model.c_str()); return 1; }
    if (!s.reset(UINT32_MAX, -1, -1, -1, -1)) { std::fprintf(stderr, "xolotl-lfm3-speech: the speech state did not start\n"); return 1; }
    std::fprintf(stderr, "xolotl-lfm3-speech: ready on the %s - %d Hz, temp %.2f, top_k %d, repetition penalty %.2f over %d frames\n",
                 gpu ? "GPU" : "CPU", info.sample_rate, s.temp, s.top_k, s.rep_penalty, s.rep_window);
    const int32_t hdr[4] = {0x3150534c /* 'LSP1' */, s.n_embd, info.sample_rate, kCb};
    wr(hdr, sizeof hdr);
    std::fflush(proto);

    std::vector<float> h(s.n_embd), pcm;
    std::vector<int32_t> codes;
    for (;;) {
        int32_t cmd = 0;
        if (!rd(&cmd, sizeof cmd) || cmd == 4) break;
        if (cmd == 1) {
            uint32_t seed; float t, rp; int32_t k, rw;
            if (!rd(&seed, 4) || !rd(&t, 4) || !rd(&k, 4) || !rd(&rp, 4) || !rd(&rw, 4)) break;
            wr_i(s.reset(seed, t, k, rp, rw) ? 0 : -1);
        } else if (cmd == 2) {
            if (!rd(h.data(), h.size() * sizeof(float))) break;
            int st = s.frame(h.data(), codes);
            if (st == 0 && !s.wav(codes.data(), codes.size(), pcm)) st = -1;
            if (st < 0) codes.assign(kCb, -1);
            wr_i(st);
            wr(codes.data(), kCb * sizeof(int32_t));
            if (st == 0) {
                wr(s.fb.data(), s.fb.size() * sizeof(float));
                wr_i((int32_t) pcm.size());
                wr(pcm.data(), pcm.size() * sizeof(float));
            }
        } else if (cmd == 3) {
            pcm.clear();
            if (s.pending_tail && !s.wav(nullptr, 0, pcm)) pcm.clear();
            s.pending_tail = false;
            wr_i((int32_t) pcm.size());
            wr(pcm.data(), pcm.size() * sizeof(float));
        } else {
            std::fprintf(stderr, "xolotl-lfm3-speech: unknown command %d\n", cmd);
            break;
        }
        std::fflush(proto);
    }
    if (s.gs) mtmd_gen_state_free(s.gs);
    mtmd_free(s.mctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
