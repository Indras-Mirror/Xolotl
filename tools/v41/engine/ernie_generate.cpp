// ernie_generate <ernie.gguf> --ids 1,2,3 [-n 32] [--layers 0:3] [--cuda] - ERNIE-5.5-Flash end to end on our engine.
//
// The ERNIE analogue of ds41_generate, deliberately a separate driver (ds41_generate's serve/DSpark/split paths are not
// this slice): the dense half is ErnieGraph (ernie-g1: per-layer ggml graphs on any ggml backend), the routed experts
// are the DS4.1 tier with ERNIE's own geometry and SiTU (ernie-g2), and this file is the loop that joins them:
//
//     begin(tids, n, pos0)
//     for il in [0, n_layer):  attn_router(il) -> tier.run_multi(il) -> finish(il)
//     logits(row)             -> the next token
//
// The prompt is a token id list (`--ids` or `--ids-file` one id per line / comma separated) or TEXT (`-p`, encoded
// with the GGUF's own tokenizer: ernie_tokenize, gated id-for-id against llama-tokenize).  Tokens are DECODED to text
// through the GGUF's own byte-level vocabulary (tokenizer.ggml.tokens).
//
// Gates this binary serves:
//   a. per-layer residuals vs the CPU runner (test_ernie_dense) and the PR dump: --dump-taps DIR writes the same
//      named tensors test_ernie_graph does, for the same compare_ernie.py.
//   b. chunked prefill == one pass (`--prefill-chunk N` vs 0) and decode token == teacher-forced position
//      (`--dump-logits FILE` on both paths, then compare).
//   c. CUDA vs CPU backend (`--cuda`), same file.
#include "ernie_graph.hpp"
#include "ernie_tokenize.hpp"

#include "ds41_moe.hpp"

#include "strata/artifact/gguf_reader.hpp"

#ifdef ERNIE_HAVE_CUDA
#include "ggml-cuda.h"   // ggml_backend_cuda_init (--cuda)
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

struct Args {
    std::string model, ids_csv, ids_file, out_text, dump_logits, dump_logits_rows, dump_taps, prompt, decode_ids;
    bool have_prompt = false, encode_only = false, bos = false, no_bos = false, have_decode = false;
    bool parse_special = false;
    int n_predict = 32;
    int64_t ctx = 4096;
    int threads = 8;
    int prefill_chunk = 0;              // 0 = one pass (up to cfg.max_pass)
    int layer_lo = 0, layer_hi = -1;    // --layers lo:hi (a dev file that holds only those layers)
    float temp = 0.0f;
    bool cuda = false, quiet = false, no_tier = false;
    // the expert tier's switches (Ds41MoeConfig; the same flags as ds41_generate)
    float skip_miss = 0.0f, skip_file = 0.0f;
    bool vram_lru = false, arena_adapt = false, arena_skip_resident = false, chunk_mmq = false, chunk_prestage = false;
    bool chunk_prefetch = true;
    int remote_device = -1;
    double remote_gib = 0.0;
    double pcie = -1.0;                 // --pcie F: share of the single-token decode misses computed on the card (-1 = the tier's 0.25)
    std::string kv_mla = "f16";         // --kv-mla f16|q8_0|q4_0 (ErnieGraphConfig::mla_kv_type)
    int kv_hadamard = -1;               // --kv-hadamard 0|1 (-1 = auto: on for q4_0)
    bool serve = false;                 // --serve: the server's engine line protocol on stdin/stdout (serve_loop)
    std::string experts = "cpu";        // --experts cpu|gpu (the tier's backend)
    double arena_gib = 8.0;
    int64_t slots = 0;
    std::string count_routes, rank_counts;   // --count-routes / --rank-counts (int64 [n_layers][n_experts])
    bool stats = false;
    std::string abl_file, batch, batch_out, batch_logits;
    float abl_scale = 1.0f;
    int ds41_decode = 0;   // --ds41-decode N: the DS4.1 decode stack (see usage), N = spec_read top (0 = no spec reads)
    bool pf = false;
    // --dspark SIDECAR: speculative decoding with ERNIE's DSpark drafter (docs/ernie/ERNIE_DSPARK_SPEC.md)
    std::string dspark, dspark_experts = "cpu";
    int dspark_block = 0;              // 0 = the sidecar's block_size (7)
    int dspark_noise = 1024;           // the block's mask token (config dspark_noise_token_id)
    double dspark_arena_gib = 11.0, dspark_slot_gib = 0.0;
};

void usage(std::FILE* out) {
    std::fprintf(out,
        "usage: ernie_generate [-m] MODEL --ids 1,2,3 | --ids-file F | -p TEXT [options]\n"
        "  -p, --prompt TEXT     encode TEXT with the GGUF's own tokenizer (mutually exclusive with --ids/--ids-file)\n"
        "  --bos                 prepend the BOS id to -p TEXT (the default; it is the file's add_bos_token)\n"
        "  --no-bos              do not prepend BOS to -p TEXT\n"
        "  --parse-special       -p TEXT: also match type-3 CONTROL literals (<|ERNIE_BEGIN|>), as llama-tokenize does\n"
        "  --encode-only         print the -p ids (comma separated) and exit without loading any weights\n"
        "  --decode IDS          print the text of the comma-separated ids (the encoder's round-trip check)\n"
        "  -n N                  tokens to generate (default 32; 0 = prefill only)\n"
        "  --ctx N               context budget (default 4096; window ring / compressor caches)\n"
        "  --threads N           CPU threads for the dense graphs (default 8)\n"
        "  --prefill-chunk N     prompt passes of at most N tokens (0 = one pass)\n"
        "  --layers lo:hi        run only layers [lo, hi) (a dev GGUF that holds just them)\n"
        "  --cuda                dense graphs on CUDA device 0 (default CPU)\n"
        "  --experts cpu|gpu     the routed-expert tier's backend (default cpu)\n"
        "  --arena-gib G         host arena budget for the tier (default 8)\n"
        "  --slots N             VRAM expert slots (0 = auto)\n"
        "  --no-tier             dense block only: skip the routed experts (layers must be dense)\n"
        "  --temp T              sampling temperature (0 = greedy, default)\n"
        "  --dump-logits FILE    logits of the LAST prefill row, float32, n_vocab values\n"
        "  --dump-logits-rows FILE  every prefill row's logits, float32, n_rows x n_vocab (one prefill pass;\n"
        "                        the teacher-forced NLL input for llama-perplexity corroboration)\n"
        "  --dump-taps DIR       per-layer taps as <name>.f32 (compare_ernie.py's format)\n"
        "  --out-text FILE       decoded text of the prompt + generated tokens\n"
        "  --count-routes FILE   ADD this run's routed-token count per (layer, expert) to FILE (int64\n"
        "                        [block_count][n_experts], created if missing): a calibration profile built over runs\n"
        "  --rank-counts FILE    seed the VRAM cache, then the RAM arena, hottest (layer, expert) of FILE first\n"
        "                        (default: index order - layer 0's experts take the whole cache)\n"
        "  --ds41-decode N       decode through the DS4.1 path: tier.run (PCIe share + prefetch staging) for 1-token\n"
        "                        passes, the next layer's router guessed on the residual (ErnieGraph::predict) ->\n"
        "                        spec_read of its top N file-tier experts (N=0: none); prints the guess's recall\n"
        "  --pf                  with --ds41-decode: also tier.prefetch (arena -> VRAM staging) of the guess\n"
        "  --abl FILE            directional ablation at load (ErnieGraphConfig::abl_file); --abl-scale S (default 1)\n"
        "  --batch FILE          many prompts in one load: FILE = rendered chat texts separated by 0x1E (special\n"
        "                        tokens parsed, BOS added); each runs from position 0 (snapshot/restore)\n"
        "  --batch-out FILE      with --batch and -n N: one line per prompt, the greedy continuation's ids then a TAB\n"
        "                        and its text with \\n escaped\n"
        "  --batch-logits FILE   with --batch: each prompt's last-row logits appended, float32 n_vocab\n"
        "  --skip-miss W         decode: drop a VRAM-miss expert weighing < W x the token's weight sum (0 = off; measure ppl)\n"
        "  --skip-file W         the same rule for file-tier (NVMe) experts only\n"
        "  --pcie F              share of the decode misses (arena-resident) computed on the card instead of the CPU pool\n"
        "                        (default 0.25)\n"
        "  --vram-lru            a PCIe-share miss takes its layer's least-recently-used VRAM slot (the cache follows the chat)\n"
        "  --arena-adapt         a file-tier read replaces its layer's least-recently-used RAM arena expert\n"
        "  --arena-skip-resident the VRAM seed's experts stay out of the RAM arena (room for the next-ranked ones)\n"
        "  --chunk-mmq / --chunk-prestage / --no-chunk-prefetch   prompt-chunk expert paths (see Ds41MoeConfig)\n"
        "  --remote-device N [--remote-gib G]   a second CUDA card holds and computes the next-ranked experts\n"
        "  --dspark FILE         speculative decoding with the DSpark sidecar (ERNIE-5.5-Flash-Preview-DSpark-*.gguf):\n"
        "                        each step drafts a block of 7 (Markov-refined) and the trunk verifies 8 in one pass;\n"
        "                        greedy only, the output is the trunk's own greedy text\n"
        "  --dspark-block N      draft block (default: the sidecar's block_size)\n"
        "  --dspark-experts cpu|gpu  the draft stages' routed experts (default cpu: read per call; gpu: own tier)\n"
        "  --dspark-arena-gib G / --dspark-slot-gib G   the gpu draft tier's RAM arena (default 11) / VRAM cache (0)\n"
        "  --kv-mla T            the MLA layers' KV rows (~96 %% of the KV at long context): f16 (default), q8_0 (half the\n"
        "                        VRAM), q4_0 (a quarter; Hadamard-rotated by default)\n"
        "  --kv-hadamard 0|1     rotate the MLA KV rows (default: on for q4_0 only)\n"
        "  --serve               the server's engine protocol on stdin/stdout (GEN / STOP / QUIT in; READY, RESUME, PP,\n"
        "                        T, DONE, ERR out); conversation prefixes are reused between requests\n"
        "  --stats               print the tier's counters (hits / cpu / pcie / file-tier, ms) for prefill and decode\n");
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--serve") a.serve = true;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (f == "-m" || f == "--model") { if (!(v = next())) return false; a.model = v; }
        else if (f == "-n") { if (!(v = next())) return false; a.n_predict = std::atoi(v); }
        else if (f == "--ids") { if (!(v = next())) return false; a.ids_csv = v; }
        else if (f == "--ids-file") { if (!(v = next())) return false; a.ids_file = v; }
        else if (f == "-p" || f == "--prompt") { if (!(v = next())) return false; a.prompt = v; a.have_prompt = true; }
        else if (f == "--encode-only") a.encode_only = true;
        else if (f == "--bos") a.bos = true;
        else if (f == "--no-bos") a.no_bos = true;
        else if (f == "--decode") { if (!(v = next())) return false; a.decode_ids = v; a.have_decode = true; }
        else if (f == "--parse-special") a.parse_special = true;
        else if (f == "--ctx") { if (!(v = next())) return false; a.ctx = std::atoll(v); }
        else if (f == "--threads") { if (!(v = next())) return false; a.threads = std::atoi(v); }
        else if (f == "--prefill-chunk") { if (!(v = next())) return false; a.prefill_chunk = std::atoi(v); }
        else if (f == "--layers") {
            if (!(v = next())) return false;
            if (std::sscanf(v, "%d:%d", &a.layer_lo, &a.layer_hi) != 2) return false;
        } else if (f == "--cuda") a.cuda = true;
        else if (f == "--experts") { if (!(v = next())) return false; a.experts = v; }
        else if (f == "--arena-gib") { if (!(v = next())) return false; a.arena_gib = std::atof(v); }
        else if (f == "--slots") { if (!(v = next())) return false; a.slots = std::atoll(v); }
        else if (f == "--no-tier") a.no_tier = true;
        else if (f == "--temp") { if (!(v = next())) return false; a.temp = (float) std::atof(v); }
        else if (f == "--dump-logits") { if (!(v = next())) return false; a.dump_logits = v; }
        else if (f == "--dump-logits-rows") { if (!(v = next())) return false; a.dump_logits_rows = v; }
        else if (f == "--dump-taps") { if (!(v = next())) return false; a.dump_taps = v; }
        else if (f == "--out-text") { if (!(v = next())) return false; a.out_text = v; }
        else if (f == "--quiet") a.quiet = true;
        else if (f == "--count-routes") { if (!(v = next())) return false; a.count_routes = v; }
        else if (f == "--rank-counts") { if (!(v = next())) return false; a.rank_counts = v; }
        else if (f == "--stats") a.stats = true;
        else if (f == "--ds41-decode") { if (!(v = next())) return false; a.ds41_decode = std::max(0, std::atoi(v)) + 1; }
        else if (f == "--pf") a.pf = true;
        else if (f == "--dspark") { if (!(v = next())) return false; a.dspark = v; }
        else if (f == "--dspark-block") { if (!(v = next())) return false; a.dspark_block = std::atoi(v); }
        else if (f == "--dspark-noise") { if (!(v = next())) return false; a.dspark_noise = std::atoi(v); }
        else if (f == "--dspark-experts") { if (!(v = next())) return false; a.dspark_experts = v; }
        else if (f == "--dspark-arena-gib") { if (!(v = next())) return false; a.dspark_arena_gib = std::atof(v); }
        else if (f == "--dspark-slot-gib") { if (!(v = next())) return false; a.dspark_slot_gib = std::atof(v); }
        else if (f == "--abl") { if (!(v = next())) return false; a.abl_file = v; }
        else if (f == "--abl-scale") { if (!(v = next())) return false; a.abl_scale = (float) std::atof(v); }
        else if (f == "--batch") { if (!(v = next())) return false; a.batch = v; }
        else if (f == "--batch-out") { if (!(v = next())) return false; a.batch_out = v; }
        else if (f == "--batch-logits") { if (!(v = next())) return false; a.batch_logits = v; }
        else if (f == "-h" || f == "--help") { usage(stdout); return false; }
        else if (f == "--serve") a.serve = true;
        else if (f == "--kv-mla") { if (!(v = next())) return false; a.kv_mla = v; }
        else if (f == "--kv-hadamard") { if (!(v = next())) return false; a.kv_hadamard = std::atoi(v); }
        else if (f == "--skip-miss") { if (!(v = next())) return false; a.skip_miss = (float) std::atof(v); }
        else if (f == "--skip-file") { if (!(v = next())) return false; a.skip_file = (float) std::atof(v); }
        else if (f == "--vram-lru") a.vram_lru = true;
        else if (f == "--pcie") { if (!(v = next())) return false; a.pcie = std::atof(v); }
        else if (f == "--no-vram-lru") a.vram_lru = false;
        else if (f == "--arena-adapt") a.arena_adapt = true;
        else if (f == "--arena-skip-resident") a.arena_skip_resident = true;
        else if (f == "--chunk-mmq") a.chunk_mmq = true;
        else if (f == "--no-chunk-mmq") a.chunk_mmq = false;
        else if (f == "--chunk-prestage") a.chunk_prestage = true;
        else if (f == "--no-chunk-prefetch") a.chunk_prefetch = false;
        else if (f == "--remote-device") { if (!(v = next())) return false; a.remote_device = std::atoi(v); }
        else if (f == "--remote-gib") { if (!(v = next())) return false; a.remote_gib = std::atof(v); }
        else if (!f.empty() && f[0] != '-') {
            if (!a.model.empty()) { std::fprintf(stderr, "the model is given twice (%s, %s)\n", a.model.c_str(), f.c_str()); return false; }
            a.model = f;
        }
        else if (a.serve && f.size() > 2 && f[0] == '-' && f[1] == '-') {
            // the server adds options meant for other engines (profiles, drafting, layer splits): say so and go on,
            // taking a following non-flag word as its value
            std::fprintf(stderr, "serve: ignoring %s", f.c_str());
            if (i + 1 < argc && argv[i + 1][0] != '-') std::fprintf(stderr, " %s", argv[++i]);
            std::fprintf(stderr, " (not an ernie_generate option)\n");
        }
        else { std::fprintf(stderr, "unknown argument %s\n", f.c_str()); return false; }
    }
    if (a.model.empty()) return false;
    if (a.have_prompt && (!a.ids_csv.empty() || !a.ids_file.empty())) {
        std::fprintf(stderr, "-p/--prompt is mutually exclusive with --ids/--ids-file\n");
        return false;
    }
    if (a.encode_only && !a.have_prompt) {
        std::fprintf(stderr, "--encode-only needs -p/--prompt TEXT\n");
        return false;
    }
    if (a.have_decode && (a.have_prompt || !a.ids_csv.empty() || !a.ids_file.empty())) {
        std::fprintf(stderr, "--decode takes the ids alone (no -p / --ids / --ids-file)\n");
        return false;
    }
    if (a.bos && a.no_bos) {
        std::fprintf(stderr, "--bos and --no-bos are mutually exclusive\n");
        return false;
    }
    return !a.ids_csv.empty() || !a.ids_file.empty() || a.have_prompt || a.have_decode || !a.batch.empty() || a.serve;
}

std::vector<int> parse_ids(const std::string& csv) {
    std::vector<int> ids;
    size_t p = 0;
    while (p < csv.size()) {
        size_t q = csv.find_first_of(", \n\t\r", p);
        if (q == std::string::npos) q = csv.size();
        if (q > p) ids.push_back(std::atoi(csv.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return ids;
}

bool read_ids_file(const std::string& path, std::vector<int>& ids) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string s;
    char buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) s.append(buf, n);
    std::fclose(f);
    ids = parse_ids(s);
    return !ids.empty();
}

/// The GPT-2 byte-level vocabulary, reversed: a token's printable unicode text -> its bytes.  The mapping is the one
/// the converter's vocabulary was built with (bytes_to_unicode in tools/strata_tokenizer.py): 33..126, 161..172 and
/// 174..255 map to themselves, the remaining 68 bytes to U+0100+n in their own order.
std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp = 0;
        size_t n = 0;
        if (c < 0x80) { cp = c; n = 0; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; n = 1; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; n = 2; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; n = 3; }
        else { ++i; continue; }
        if (n > 0 && i + n >= s.size()) { ++i; continue; }   // truncated sequence: skip the lead byte
        for (size_t k = 1; k <= n; ++k) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        out.push_back(cp);
        i += n + 1;
    }
    return out;
}

std::string detok_bytes(const std::string& token) {
    // codepoint -> byte: the inverse of GPT-2's bytes_to_unicode
    static std::vector<std::pair<uint32_t, uint8_t>> map;
    if (map.empty()) {
        bool in[256] = {false};
        for (int b = 33; b <= 126; ++b) in[b] = true;
        for (int b = 161; b <= 172; ++b) in[b] = true;
        for (int b = 174; b <= 255; ++b) in[b] = true;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (in[b]) map.emplace_back((uint32_t) b, (uint8_t) b);
            else map.emplace_back((uint32_t) (256 + n++), (uint8_t) b);
        }
    }
    std::string out;
    for (uint32_t cp : utf8_codepoints(token)) {
        bool hit = false;
        for (const auto& e : map) {
            if (e.first == cp) { out.push_back((char) e.second); hit = true; break; }
        }
        if (!hit) {   // not a byte-level char (a special token's text): pass the codepoint's utf-8 through
            std::string t;
            if (cp < 0x80) t.push_back((char) cp);
            else if (cp < 0x800) { t.push_back((char) (0xC0 | (cp >> 6))); t.push_back((char) (0x80 | (cp & 0x3F))); }
            else if (cp < 0x10000) {
                t.push_back((char) (0xE0 | (cp >> 12)));
                t.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                t.push_back((char) (0x80 | (cp & 0x3F)));
            } else {
                t.push_back((char) (0xF0 | (cp >> 18)));
                t.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
                t.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
                t.push_back((char) (0x80 | (cp & 0x3F)));
            }
            out += t;
        }
    }
    return out;
}

int argmax(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return best;
}

/// One prompt or decode step through every layer: the G3 loop.  out_logits: n_vocab for the last row when non-null.
/// all_taps (optional): every named intermediate of every layer, appended - the dump format test_ernie_graph writes,
/// so tools/v41/engine/compare_ernie.py diffs the driver's per-layer numbers against the CPU runner and the PR dump.
/// FILE as int64[n]; a missing file is all zeros only when `missing_ok`.
bool read_counts(const std::string& path, size_t n, std::vector<int64_t>& c, std::string& err, bool missing_ok = false) {
    c.assign(n, 0);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (missing_ok) return true;
        err = "cannot open " + path;
        return false;
    }
    const size_t got = std::fread(c.data(), sizeof(int64_t), n, f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != n || extra) { err = path + ": not " + std::to_string(n) + " int64 counts (another model's file?)"; return false; }
    return true;
}

/// Routed-token counts per (layer, expert), row-major [n_layers][n_experts]; empty = not counting.
std::vector<int64_t> g_route_counts;
int64_t g_route_experts = 0;
int g_ds41 = 0;           // --ds41-decode N  ->  N + 1 (0 = off)
bool g_pf = false;
int64_t g_pred_n = 0, g_pred_in16 = 0, g_pred_in8 = 0;   // routed experts guessed / found in the top 16 / top 8
/// DSpark main_hidden capture: when g_mh is set, run_pass writes the pass's LAST g_mh_rows columns' outputs of the
/// target layers (mHC-stream means) as [rows][n_targets][n_embd] (token-major: one main_hidden row per position).
std::vector<float>* g_mh = nullptr;
int g_mh_rows = 0;
std::vector<int> g_mh_layers;

bool run_pass(ErnieGraph& eg, Ds41MoeTier* tier, int il0, int il1, const int* tids, int n, int64_t pos0,
              std::vector<ErnieGraphTap>* all_taps, std::vector<float>* out_logits, std::string& err) {
    if (!eg.begin(tids, n, (int) pos0)) { err = eg.last_error(); return false; }
    const size_t nk = (size_t) eg.geom().n_used;
    std::vector<int32_t> ids((size_t) n * nk);
    std::vector<float> wts((size_t) n * nk);
    std::vector<float> routed;
    const bool want_taps = all_taps != nullptr;
    for (int il = il0; il < il1; ++il) {
        const float* fn_host = nullptr;
        std::vector<ErnieGraphTap> taps;
        const bool dense = il < eg.geom().dense_lead;
        int guess[ErnieGraph::kPredK];
        const bool guessed = g_ds41 && tier && n == 1 && !dense && eg.predict(il, guess);
        if (guessed) {
            if (g_ds41 > 1) tier->spec_read(il, guess, std::min(g_ds41 - 1, (int) ErnieGraph::kPredK));
            if (g_pf) tier->prefetch(il, guess, ErnieGraph::kPredK);
        }
        if (!eg.attn_router(il, ids.data(), wts.data(), &fn_host, want_taps ? &taps : nullptr)) {
            err = eg.last_error();
            return false;
        }
        if (guessed)
            for (size_t k = 0; k < nk; ++k) {
                ++g_pred_n;
                for (int j = 0; j < ErnieGraph::kPredK; ++j)
                    if (guess[j] == ids[k]) { ++g_pred_in16; g_pred_in8 += j < 8; break; }
            }
        if (dense || !tier) {
            if (!eg.finish(il, nullptr, want_taps ? &taps : nullptr)) { err = eg.last_error(); return false; }
        } else {
            const int64_t L = eg.geom().n_latent;
            routed.assign((size_t) n * (size_t) L, 0.0f);
            if (!g_route_counts.empty())
                for (size_t k = 0; k < (size_t) n * nk; ++k)
                    if (ids[k] >= 0 && ids[k] < g_route_experts &&
                        (size_t) il * g_route_experts + ids[k] < g_route_counts.size())
                        ++g_route_counts[(size_t) il * g_route_experts + ids[k]];
            // run_multi takes at most kMaxMulti (16) rows - the verify-pass width; a prompt pass goes through run_chunk
            // (the batched prompt path: MMQ on the GPU tier, token by token on the CPU tier).
            const bool ok = (g_ds41 && n == 1) ? tier->run(il, ids.data(), wts.data(), fn_host, routed.data())
                          : n <= Ds41MoeTier::kMaxMulti
                                ? tier->run_multi(il, n, ids.data(), wts.data(), fn_host, routed.data())
                                : tier->run_chunk(il, n, ids.data(), wts.data(), fn_host, routed.data());
            if (!ok) {
                err = "tier " + std::string(n <= Ds41MoeTier::kMaxMulti ? "run_multi" : "run_chunk") +
                      " failed at layer " + std::to_string(il);
                return false;
            }
            if (!eg.finish(il, routed.data(), want_taps ? &taps : nullptr)) { err = eg.last_error(); return false; }
        }
        if (want_taps) all_taps->insert(all_taps->end(), taps.begin(), taps.end());
        if (g_mh)
            for (size_t j = 0; j < g_mh_layers.size(); ++j) {
                static const int mh_shift = [] { const char * e = std::getenv("ERNIE_DSPARK_MHSHIFT"); return e ? std::atoi(e) : 0; }();
                if (g_mh_layers[j] != il + mh_shift) continue;   // probe: 1 = the residual entering the target layer
                const int rows = std::min(n, g_mh_rows);
                const int64_t D = eg.geom().n_embd, NT = (int64_t) g_mh_layers.size();
                std::vector<float> m((size_t) rows * D);
                if (!eg.read_state_mean(n - rows, rows, m.data())) { err = eg.last_error(); return false; }
                g_mh->resize((size_t) rows * NT * D);
                for (int r = 0; r < rows; ++r)
                    std::memcpy(g_mh->data() + ((size_t) r * NT + j) * D, m.data() + (size_t) r * D, (size_t) D * 4);
            }
    }
    if (out_logits) {
        out_logits->assign((size_t) eg.geom().vocab, 0.0f);
        if (!eg.logits(n - 1, 1, out_logits->data(), nullptr)) { err = eg.last_error(); return false; }
    }
    return true;
}

// ================================ DSpark speculative decoding ================================
//
// The drafter = the sidecar's 3 stages as an ErnieGraph in draft mode (same backend as the trunk) + its own routed-
// expert tier over the sidecar's dspark.blk.N experts.  A step at P (P positions committed, `anchor` = the trunk's
// newest token, not yet run):
//   draft:   block [anchor, noise x 6] at P..P+6 through the stages -> 7 greedy drafts (base logits + Markov chain)
//   verify:  the trunk runs [anchor, d0..d6] at P..P+7 in one pass; a = the longest prefix with argmax(row j) == d_j;
//            the bonus = argmax(row a).  Emitted: anchor, d0..d(a-1); the next anchor = the bonus.
//   commit:  the trunk's ring / pending slots of the rejected positions are put back (spec_rollback(P + a + 1)); the
//            drafter's window gets the main_hidden rows of P..P+a (captured in the verify pass).
// Greedy, so the emitted text is the trunk's own greedy text (up to the verify pass's width-dependent arithmetic).
struct Drafter {
    ErnieGraph dg;
    Ds41MoeGeom gm;
    std::unique_ptr<Ds41GgufBlobs> src;
    std::unique_ptr<Ds41MoeTier> tier;
    int block = 7, noise = 1024;
    std::vector<int> targets;
    int64_t steps = 0, drafted = 0, accepted = 0;
    double draft_ms = 0, verify_ms = 0, commit_ms = 0;
    std::vector<int64_t> hist;   // accepted-drafts histogram, 0..block
};

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

bool drafter_init(Drafter& d, const Args& a, const ErnieGraph& trunk, const Ds41MoeGeom& trunk_gm, const ErnieGraphConfig& tcfg_graph,
                  std::string& err) {
    {
        std::unique_ptr<GgufModel> m(new GgufModel(GgufModel::open(a.dspark)));
        const MetaValue* bs = m->meta().get("ernie5_5.dspark.block_size");
        d.block = a.dspark_block > 0 ? a.dspark_block : (bs && bs->is_num() ? (int) bs->num() : 7);
        const MetaValue* tl = m->meta().get("ernie5_5.dspark.target_layers");
        if (tl && tl->type == MetaType::ARRAY) for (const MetaValue& e : tl->items) d.targets.push_back((int) e.num());
        if (d.targets.empty()) d.targets = {40, 41, 42};
        const TensorInfo* g0 = m->find("dspark.blk.0.ffn_gate_exps.weight");
        const TensorInfo* d0 = m->find("dspark.blk.0.ffn_down_exps.weight");
        if (!g0 || !d0) { err = a.dspark + ": no dspark.blk.0 routed experts"; return false; }
        d.gm = trunk_gm;
        d.gm.n_layers = 0;
        while (m->find("dspark.blk." + std::to_string(d.gm.n_layers) + ".ffn_gate_exps.weight")) ++d.gm.n_layers;
        d.gm.gu_type = (int) g0->type;
        d.gm.d_type = (int) d0->type;
        d.gm.gate_types.clear(); d.gm.up_types.clear(); d.gm.down_types.clear();
        d.gm.eclass.clear(); d.gm.cls_gu.clear(); d.gm.cls_d.clear();
        d.gm.tensor_prefix = "dspark.";
    }
    d.noise = a.dspark_noise;
    if (d.block < 1 || d.block > 15) { err = "--dspark-block must be 1..15"; return false; }
    for (int t : d.targets)
        if (t < 0 || t >= trunk.n_layer()) { err = "DSpark target layer " + std::to_string(t) + " outside the trunk"; return false; }
    ErnieGraphConfig dc = tcfg_graph;
    dc.dspark_trunk = &trunk;
    dc.max_pass = d.block;
    dc.abl_file.clear();
    if (!d.dg.init(a.dspark, dc, err)) { err = "draft graph: " + err; return false; }
    if (d.dg.n_layer() != d.gm.n_layers) { err = "draft stages and expert layers disagree"; return false; }
    d.src.reset(new Ds41GgufBlobs(d.gm, a.dspark));
    if (!d.src->open(err)) { err = "draft blobs: " + err; return false; }
    Ds41MoeConfig c;
    c.cpu_only = a.dspark_experts != "gpu";
    c.threads = a.threads;
    c.arena_gib = a.dspark_arena_gib;
    c.max_arena_gib = a.dspark_arena_gib;
    d.tier.reset(new Ds41MoeTier());
    if (!d.tier->init(d.gm, d.src.get(), c, err)) { err = "draft tier: " + err; return false; }
    if (!c.cpu_only) {
        if (a.dspark_slot_gib > 0.0) d.tier->set_slot_gib(a.dspark_slot_gib);
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int64_t l = 0; l < d.gm.n_layers; ++l)
            for (int64_t e = 0; e < d.gm.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        if (!d.tier->seed_from_ranked(ranked, err)) { err = "draft tier seed: " + err; return false; }
    }
    d.hist.assign((size_t) d.block + 1, 0);
    if (!a.quiet)
        std::printf("dspark: %lld stages, block %d, targets %d/%d/%d, draft experts %s (%s)\n", (long long) d.dg.n_layer(),
                    d.block, d.targets[0], d.targets.size() > 1 ? d.targets[1] : -1, d.targets.size() > 2 ? d.targets[2] : -1,
                    d.tier->mode(), c.cpu_only ? "per-call reads" : "own tier");
    return true;
}

/// The window rows of positions [pos0, pos0 + rows) from main_hidden rows ([rows][targets][n_embd]); only the last
/// 128 matter (the ring), so a longer span commits its tail.
bool drafter_commit(Drafter& d, const std::vector<float>& mh, int64_t pos0, int rows, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t W = (int64_t) d.targets.size() * d.dg.geom().n_embd;
    int r0 = std::max(0, rows - 128);
    // a gap (nothing committed for a while, or a long span): one whole new ring
    if (pos0 + r0 != d.dg.dspark_committed() && rows - r0 < 128) {
        err = "drafter_commit: positions " + std::to_string(pos0 + r0) + ".. do not follow the committed " +
              std::to_string(d.dg.dspark_committed());
        return false;
    }
    if (!d.dg.dspark_commit(mh.data() + (size_t) r0 * W, pos0 + r0, rows - r0)) { err = d.dg.last_error(); return false; }
    d.commit_ms += ms_since(t0);
    return true;
}

/// One draft block at P from `anchor`: d.block greedy drafts into out.
bool drafter_draft(Drafter& d, int anchor, int64_t P, int* out, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> blk((size_t) d.block, d.noise);
    blk[0] = anchor;
    if (!d.dg.begin(blk.data(), d.block, (int) P)) { err = "draft: " + d.dg.last_error(); return false; }
    const size_t nk = (size_t) d.dg.geom().n_used;
    const int64_t L = d.dg.geom().n_latent;
    std::vector<int32_t> ids((size_t) d.block * nk);
    std::vector<float> wts((size_t) d.block * nk), routed((size_t) d.block * L);
    for (int s = 0; s < (int) d.dg.n_layer(); ++s) {
        const float* fn = nullptr;
        if (!d.dg.attn_router(s, ids.data(), wts.data(), &fn)) { err = "draft: " + d.dg.last_error(); return false; }
        std::fill(routed.begin(), routed.end(), 0.0f);
        if (!d.tier->run_multi(s, d.block, ids.data(), wts.data(), fn, routed.data())) {
            err = "draft tier run_multi failed at stage " + std::to_string(s);
            return false;
        }
        if (!d.dg.finish(s, routed.data())) { err = "draft: " + d.dg.last_error(); return false; }
    }
    if (!d.dg.dspark_draft(anchor, out)) { err = "draft: " + d.dg.last_error(); return false; }
    d.draft_ms += ms_since(t0);
    return true;
}

/// One speculative step at P: draft, verify, accept, rollback, commit.  Appends the emitted tokens (anchor first) to
/// `emit` and returns the next anchor in *next.
bool dspark_step(Drafter& d, ErnieGraph& eg, Ds41MoeTier* tier, int il0, int il1, int anchor, int64_t& P,
                 std::vector<int>& emit, int* next, std::string& err) {
    std::vector<int> dr((size_t) d.block);
    if (!drafter_draft(d, anchor, P, dr.data(), err)) return false;
    const int nv = d.block + 1;
    std::vector<int> v((size_t) nv);
    v[0] = anchor;
    for (int j = 0; j < d.block; ++j) v[(size_t) j + 1] = dr[(size_t) j];
    const auto t0 = std::chrono::steady_clock::now();
    if (!eg.spec_save(P, nv)) { err = eg.last_error(); return false; }
    std::vector<float> mh;
    g_mh = &mh;
    g_mh_rows = nv;
    g_mh_layers = d.targets;
    const bool ok = run_pass(eg, tier, il0, il1, v.data(), nv, P, nullptr, nullptr, err);
    g_mh = nullptr;
    if (!ok) return false;
    const int64_t V = eg.geom().vocab;
    std::vector<float> lg((size_t) nv * V);
    if (!eg.logits(0, nv, lg.data(), nullptr)) { err = eg.last_error(); return false; }
    int acc = 0;
    while (acc < d.block && argmax(lg.data() + (size_t) acc * V, (int) V) == dr[(size_t) acc]) ++acc;
    *next = argmax(lg.data() + (size_t) acc * V, (int) V);
    static const bool trace = [] { const char * e = std::getenv("ERNIE_DSPARK_TRACE"); return e && *e == '1'; }();
    if (trace) {
        std::fprintf(stderr, "dsp P=%lld anchor=%d acc=%d next=%d | dr", (long long) P, anchor, acc, *next);
        for (int j = 0; j < d.block; ++j) std::fprintf(stderr, " %d", dr[(size_t) j]);
        std::fprintf(stderr, " | argmax");
        for (int j = 0; j < nv; ++j) std::fprintf(stderr, " %d", argmax(lg.data() + (size_t) j * V, (int) V));
        std::fprintf(stderr, "\n");
    }
    if (!eg.spec_rollback(P + acc + 1)) { err = eg.last_error(); return false; }
    d.verify_ms += ms_since(t0);
    if (!drafter_commit(d, mh, P, acc + 1, err)) return false;
    emit.push_back(anchor);
    for (int j = 0; j < acc; ++j) emit.push_back(dr[(size_t) j]);
    P += acc + 1;
    ++d.steps;
    d.drafted += d.block;
    d.accepted += acc;
    ++d.hist[(size_t) acc];
    return true;
}

void drafter_report(const Drafter& d) {
    if (d.steps == 0) return;
    std::printf("dspark: %lld steps, %.2f drafts accepted per step (%.1f %%), %.2f tokens per trunk pass | draft %.1f "
                "verify %.1f commit %.1f ms/step | accepted:",
                (long long) d.steps, (double) d.accepted / d.steps, 100.0 * d.accepted / std::max<int64_t>(1, d.drafted),
                (double) (d.accepted + d.steps) / d.steps, d.draft_ms / d.steps, d.verify_ms / d.steps, d.commit_ms / d.steps);
    for (size_t i = 0; i < d.hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) d.hist[i]);
    std::printf("\n");
}

/// test_ernie_graph's dump layout: <sanitized name>.f32 plus manifest.json, which compare_ernie.py reads for the ne.
bool write_dump(const std::string& dir, const std::vector<ErnieGraphTap>& taps, std::string& err) {
    std::string cmd = "mkdir -p '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) { err = "cannot create " + dir; return false; }
    std::FILE* man = std::fopen((dir + "/manifest.json").c_str(), "wb");
    if (!man) { err = "cannot write " + dir + "/manifest.json"; return false; }
    std::fprintf(man, "{\"tensors\": [\n");
    int written = 0;
    for (const ErnieGraphTap& t : taps) {
        std::string file = t.name;
        for (char& c : file)
            if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-')) c = '_';
        file += ".f32";
        std::FILE* f = std::fopen((dir + "/" + file).c_str(), "wb");
        if (!f) { std::fclose(man); err = "cannot write " + dir + "/" + file; return false; }
        std::fwrite(t.data.data(), sizeof(float), t.data.size(), f);
        std::fclose(f);
        int64_t ne[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < t.ne.size() && i < 4; ++i) ne[i] = t.ne[i] > 0 ? t.ne[i] : 1;
        std::fprintf(man,
                     "%s    {\"name\": \"%s\", \"file\": \"%s\", \"seq\": 1, \"ne\": [%lld, %lld, %lld, %lld], "
                     "\"ggml_type\": \"F32\", \"n_elements\": %zu}",
                     written ? ",\n" : "", t.name.c_str(), file.c_str(), (long long) ne[0], (long long) ne[1],
                     (long long) ne[2], (long long) ne[3], t.data.size());
        ++written;
    }
    std::fprintf(man, "\n]}\n");
    std::fclose(man);
    return true;
}

/// --count-routes: ADD the counts gathered since the last flush to FILE (atomic rename), then zero them.
bool flush_route_counts(const std::string& path, std::string& err) {
    std::vector<int64_t> c;
    if (!read_counts(path, g_route_counts.size(), c, err, true)) return false;
    for (size_t i = 0; i < c.size(); ++i) c[i] += g_route_counts[i];
    const std::string tmpf = path + ".tmp";
    std::FILE* f = std::fopen(tmpf.c_str(), "wb");
    if (!f || std::fwrite(c.data(), sizeof(int64_t), c.size(), f) != c.size() || std::fclose(f) != 0 ||
        std::rename(tmpf.c_str(), path.c_str()) != 0) {
        err = "cannot write " + path;
        return false;
    }
    std::fill(g_route_counts.begin(), g_route_counts.end(), 0);
    return true;
}

// ================================================================== --serve
// The server's engine line protocol (serve/server.py StrataEngine; the same lines ds41_generate speaks):
//   in:  GEN <max_new> [temperature=F top_p=F top_k=N min_p=F seed=N prefill=1 ...] <ids csv> | STOP | QUIT
//   out: INFO ctx=.. , READY <ctx> stop   once, after loading
//        RESUME <reused>                  per request: prompt tokens not read again
//        PP <read> <total> <ms> <tok/s>   after every prompt chunk (also the server's heartbeat)
//        T <id>                           per generated token (a stop token is sent too, then the request ends)
//        DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish> 0 0 <reused>     finish: stop | length | cancel | error
//        ERR <why>                        then DONE ... error
// Unknown keys on a GEN line are ignored (penalties, grammar: not in this engine).  One request at a time.
//
// Conversation reuse: `hist` is exactly the tokens the caches hold.  Snapshot slot 0 = position 0, slot 1 = the end of
// the last prompt (with that row's logits).  A request whose prompt extends `hist` reads only the new tokens in place;
// one that shares the last prompt (an agent turn whose history dropped the thinking, a regenerate) restores slot 1; any
// other rewinds to slot 0 and reads everything.

struct Sampling {
    float temp = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    bool has_seed = false;
    uint64_t seed = 0;
    // repetition penalties over the last `penalty_last_n` tokens (llama.cpp's rules)
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int penalty_last_n = 64;
    // DRY ("don't repeat yourself", llama.cpp's parameter names): a token that would extend a sequence already seen
    // in the context by n >= allowed_length tokens loses multiplier * base^(n - allowed_length) from its logit.
    // A match never crosses a sequence breaker (a token whose text holds a newline, ':', '"' or '*').
    float dry_multiplier = 0.0f, dry_base = 1.75f;
    int dry_allowed_length = 2, dry_penalty_last_n = -1;   // -1 = the whole context
    bool active() const {
        return penalty_repeat != 1.0f || penalty_freq != 0.0f || penalty_present != 0.0f || dry_multiplier > 0.0f;
    }
};

/// The penalties and DRY on `lg` in place, from `hist` (everything the context holds, newest last).
void apply_penalties(std::vector<float>& lg, const std::vector<int>& hist, const Sampling& s, const std::vector<char>& breaker) {
    const int64_t N = (int64_t) hist.size(), V = (int64_t) lg.size();
    if (N == 0) return;
    if (s.penalty_repeat != 1.0f || s.penalty_freq != 0.0f || s.penalty_present != 0.0f) {
        const int64_t w = s.penalty_last_n < 0 ? N : std::min<int64_t>(N, s.penalty_last_n);
        std::vector<std::pair<int, int>> cnt;   // (token, count), small: at most w entries
        for (int64_t i = N - w; i < N; ++i) {
            const int t = hist[(size_t) i];
            auto it = std::find_if(cnt.begin(), cnt.end(), [&](const auto& c) { return c.first == t; });
            if (it == cnt.end()) cnt.emplace_back(t, 1); else ++it->second;
        }
        for (const auto& c : cnt) {
            if (c.first < 0 || c.first >= V) continue;
            float& l = lg[(size_t) c.first];
            if (s.penalty_repeat != 1.0f) l = l > 0.0f ? l / s.penalty_repeat : l * s.penalty_repeat;
            l -= (float) c.second * s.penalty_freq + s.penalty_present;
        }
    }
    if (s.dry_multiplier > 0.0f && N >= 2) {
        const int64_t start = s.dry_penalty_last_n > 0 ? std::max<int64_t>(0, N - s.dry_penalty_last_n) : 0;
        auto brk = [&](int t) { return t >= 0 && (size_t) t < breaker.size() && breaker[(size_t) t]; };
        constexpr int kMaxMatch = 64;   // base^(n - allowed) past this would only overflow
        std::vector<std::pair<int, int>> best;   // (token, longest match it would extend)
        const int64_t last = N - 1;
        for (int64_t j = start + 1; j < N; ++j) {   // hist[j] followed the context ending at j - 1
            int k = 0;
            while (k < kMaxMatch && j - 1 - k >= start && hist[(size_t) (j - 1 - k)] == hist[(size_t) (last - k)] &&
                   !brk(hist[(size_t) (last - k)]))
                ++k;
            if (k < s.dry_allowed_length) continue;
            const int t = hist[(size_t) j];
            if (brk(t)) continue;
            auto it = std::find_if(best.begin(), best.end(), [&](const auto& b) { return b.first == t; });
            if (it == best.end()) best.emplace_back(t, k); else it->second = std::max(it->second, k);
        }
        for (const auto& b : best)
            if (b.first >= 0 && b.first < V)
                lg[(size_t) b.first] -= s.dry_multiplier * std::pow(s.dry_base, (float) (b.second - s.dry_allowed_length));
    }
}

/// Greedy at temperature 0; otherwise the top_k (<= 64) candidates, softmax at `temp`, then min_p and top_p.
int sample(const std::vector<float>& lg, const Sampling& s, std::mt19937_64& rng) {
    const int n = (int) lg.size();
    if (s.temp <= 0.0f || n == 0) return argmax(lg.data(), n);
    const int k = std::max(1, std::min({ s.top_k > 0 ? s.top_k : 64, 64, n }));
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int x, int y) { return lg[(size_t) x] > lg[(size_t) y]; });
    idx.resize((size_t) k);
    std::vector<double> p((size_t) k);
    const double mx = lg[(size_t) idx[0]];
    double sum = 0.0;
    for (int i = 0; i < k; ++i) sum += p[(size_t) i] = std::exp(((double) lg[(size_t) idx[(size_t) i]] - mx) / s.temp);
    for (double& v : p) v /= sum;
    int keep = k;
    if (s.min_p > 0.0f)
        for (int i = 1; i < keep; ++i) if (p[(size_t) i] < s.min_p * p[0]) { keep = i; break; }
    if (s.top_p < 1.0f) {
        double c = 0.0;
        for (int i = 0; i < keep; ++i) { c += p[(size_t) i]; if (c >= s.top_p) { keep = i + 1; break; } }
    }
    double tot = 0.0;
    for (int i = 0; i < keep; ++i) tot += p[(size_t) i];
    double r = std::uniform_real_distribution<double>(0.0, tot)(rng);
    for (int i = 0; i < keep; ++i) { r -= p[(size_t) i]; if (r <= 0.0) return idx[(size_t) i]; }
    return idx[(size_t) (keep - 1)];
}

int serve_loop(const Args& a, ErnieGraph& eg, Ds41MoeTier* tier, int il0, int il1, const std::vector<int>& stop_ids,
               const std::vector<char>& breaker) {
    std::mutex qm;
    std::condition_variable qc;
    std::deque<std::string> q;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            std::lock_guard<std::mutex> lk(qm);
            q.push_back(line);
            qc.notify_one();
        }
        std::lock_guard<std::mutex> lk(qm);
        q.push_back("QUIT");   // stdin closed: the server is gone
        qc.notify_one();
    });
    reader.detach();
    auto pop = [&]() -> std::string {
        std::unique_lock<std::mutex> lk(qm);
        qc.wait(lk, [&] { return !q.empty(); });
        std::string l = q.front();
        q.pop_front();
        return l;
    };
    auto stop_pending = [&]() -> bool {
        std::lock_guard<std::mutex> lk(qm);
        for (auto it = q.begin(); it != q.end(); ++it)
            if (*it == "STOP") { q.erase(it); return true; }
        return false;
    };
    auto is_stop = [&](int t) { return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end(); };
    auto now_ms = [] {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };

    const int ctxv = (int) a.ctx;
    const int chunk_max = a.prefill_chunk > 0 ? a.prefill_chunk : 128;   // > 128 runs out of VRAM beside an auto cache
    std::string err;
    if (!eg.snapshot(0, 0)) { std::fprintf(stderr, "serve: snapshot: %s\n", eg.last_error().c_str()); return 1; }
    std::vector<int> hist;
    bool s1_valid = false;
    std::vector<float> s1_logits;
    std::random_device rd;

    std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\n", ctxv, (long long) (tier ? tier->resident() : 0),
                (long long) (tier ? tier->arena_experts() : 0));
    std::printf("READY %d stop\n", ctxv);
    std::fflush(stdout);
    std::fprintf(stderr, "serve: ready (ctx %d, prefill chunk %d, %zu stop ids)\n", ctxv, chunk_max, stop_ids.size());

    for (;;) {
        const std::string line = pop();
        if (line == "QUIT") break;
        if (line.empty() || line == "STOP") continue;   // a STOP after the request already ended
        if (line.rfind("GEN ", 0) != 0) {
            const std::string cmd = line.substr(0, line.find(' '));
            std::printf("ERR %s is not supported by ernie_generate\n", cmd.c_str());
            std::fflush(stdout);
            continue;
        }
        // ---- parse: GEN <max_new> [k=v ...] <ids> ----
        std::vector<std::string> f;
        for (size_t p = 0; p < line.size();) {
            const size_t e = line.find(' ', p);
            const size_t end = e == std::string::npos ? line.size() : e;
            if (end > p) f.push_back(line.substr(p, end - p));
            p = end + 1;
        }
        Sampling sp;
        int max_new = f.size() > 1 ? std::atoi(f[1].c_str()) : 0;
        std::vector<int> ids;
        for (size_t i = 2; i < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) { ids = parse_ids(f[i]); continue; }
            const std::string k = f[i].substr(0, eq), v = f[i].substr(eq + 1);
            if (k == "temperature") sp.temp = (float) std::atof(v.c_str());
            else if (k == "top_p") sp.top_p = (float) std::atof(v.c_str());
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = (float) std::atof(v.c_str());
            else if (k == "seed") { sp.has_seed = true; sp.seed = std::strtoull(v.c_str(), nullptr, 10); }
            else if (k == "prefill" && v == "1") max_new = 0;
            else if (k == "penalty_repeat") sp.penalty_repeat = (float) std::atof(v.c_str());
            else if (k == "penalty_freq") sp.penalty_freq = (float) std::atof(v.c_str());
            else if (k == "penalty_present") sp.penalty_present = (float) std::atof(v.c_str());
            else if (k == "penalty_last_n") sp.penalty_last_n = std::atoi(v.c_str());
            else if (k == "dry_multiplier") sp.dry_multiplier = (float) std::atof(v.c_str());
            else if (k == "dry_base") sp.dry_base = (float) std::atof(v.c_str());
            else if (k == "dry_allowed_length") sp.dry_allowed_length = std::atoi(v.c_str());
            else if (k == "dry_penalty_last_n") sp.dry_penalty_last_n = std::atoi(v.c_str());
        }
        const int total = (int) ids.size();
        auto fail = [&](const std::string& why, int reused) {
            std::printf("ERR %s\n", why.c_str());
            std::printf("DONE 0 %d 0.0 0.0 error 0 0 %d\n", total, reused);
            std::fflush(stdout);
        };
        if (ids.empty()) { fail("empty prompt", 0); continue; }
        if (total + std::max(0, max_new) + 8 > ctxv) {
            fail("prompt " + std::to_string(total) + " + max_new " + std::to_string(max_new) + " exceeds the context " +
                 std::to_string(ctxv) + " (start the engine with a larger --ctx)", 0);
            continue;
        }
        const double tp0 = now_ms();
        // ---- reuse ----
        size_t L = 0;
        while (L < hist.size() && L < ids.size() && hist[L] == ids[L]) ++L;
        const int s1_pos = eg.snapshot_pos(1);
        std::vector<float> logits;
        int reused = 0;
        bool ok = true;
        if (s1_valid && s1_pos == total && L >= (size_t) s1_pos && !s1_logits.empty()) {
            // the same prompt again (regenerate / a retry): its last row's logits are kept, nothing to read
            if (hist.size() != (size_t) s1_pos) ok = eg.restore(1);
            hist.resize((size_t) s1_pos);
            reused = total;
            logits = s1_logits;
        } else {
            const size_t Lc = std::min(L, ids.size() - 1);   // the last prompt token is always read (its logits)
            if (Lc == hist.size()) {
                reused = (int) Lc;                          // the prompt extends the caches: read on in place
            } else if (s1_valid && s1_pos >= 0 && (size_t) s1_pos <= Lc) {
                ok = eg.restore(1);
                hist.resize((size_t) s1_pos);
                reused = s1_pos;
            } else {
                ok = eg.restore(0);
                hist.clear();
                s1_valid = false;                           // the positions under slot 1 are about to be rewritten
                reused = 0;
            }
        }
        if (!ok) { fail("restore: " + eg.last_error(), 0); hist.clear(); s1_valid = false; continue; }
        std::printf("RESUME %d\n", reused);
        std::fflush(stdout);
        // ---- prompt ----
        int64_t pos = reused;
        bool cancelled = false;
        while (pos < total) {
            const int n = (int) std::min<int64_t>(chunk_max, total - pos);
            const bool last = pos + n == total;
            std::vector<float> lg;
            if (!run_pass(eg, tier, il0, il1, ids.data() + pos, n, pos, nullptr, last ? &lg : nullptr, err)) {
                ok = false;
                break;
            }
            hist.insert(hist.end(), ids.begin() + pos, ids.begin() + pos + n);
            pos += n;
            if (last) logits = std::move(lg);
            const double ms = now_ms() - tp0;
            std::printf("PP %lld %d %.0f %.1f\n", (long long) pos, total, ms, 1000.0 * (double) (pos - reused) / std::max(ms, 1e-9));
            std::fflush(stdout);
            if (!last && stop_pending()) { cancelled = true; break; }
        }
        if (!ok) { fail("prompt: " + err, reused); hist.clear(); s1_valid = false; continue; }
        const double prompt_ms = now_ms() - tp0;
        if (cancelled) {
            std::printf("DONE 0 %d %.1f 0.0 cancel 0 0 %d\n", total, prompt_ms, reused);
            std::fflush(stdout);
            continue;
        }
        if (reused < total || !s1_valid || s1_pos != total) {   // the next turn's restore point
            if (eg.snapshot((int) pos, 1)) { s1_valid = true; s1_logits = logits; }
            else { s1_valid = false; std::fprintf(stderr, "serve: snapshot: %s\n", eg.last_error().c_str()); }
        }
        if (max_new <= 0) {
            std::printf("DONE 0 %d %.1f 0.0 length 0 0 %d\n", total, prompt_ms, reused);
            std::fflush(stdout);
            continue;
        }
        // ---- decode ----
        std::mt19937_64 rng(sp.has_seed ? sp.seed : (((uint64_t) rd() << 32) ^ rd()));
        const double td0 = now_ms();
        int gen = 0;
        const char* finish = "length";
        if (sp.active()) apply_penalties(logits, hist, sp, breaker);
        int next = sample(logits, sp, rng);
        for (;;) {
            std::printf("T %d\n", next);
            std::fflush(stdout);
            ++gen;
            if (is_stop(next)) { finish = "stop"; break; }
            if (gen >= max_new || pos + 1 >= ctxv) break;
            if (stop_pending()) { finish = "cancel"; break; }
            std::vector<float> lg;
            if (!run_pass(eg, tier, il0, il1, &next, 1, pos, nullptr, &lg, err)) { ok = false; break; }
            hist.push_back(next);
            ++pos;
            if (sp.active()) apply_penalties(lg, hist, sp, breaker);
            next = sample(lg, sp, rng);
        }
        if (!ok) {
            std::printf("ERR decode: %s\n", err.c_str());
            finish = "error";
            hist.clear();
            s1_valid = false;
        }
        std::printf("DONE %d %d %.1f %.1f %s 0 0 %d\n", gen, total, prompt_ms, now_ms() - td0, finish, reused);
        std::fflush(stdout);
        // the routing profile a later start seeds from (--rank-counts): kept current after every request
        if (!g_route_counts.empty() && !flush_route_counts(a.count_routes, err))
            std::fprintf(stderr, "serve: --count-routes: %s\n", err.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) { usage(stderr); return 2; }
    if (a.serve) a.quiet = true;   // stdout is the protocol: the run's notes go to stderr (the server's log)
    g_ds41 = a.ds41_decode;
    g_pf = a.pf;

    std::vector<int> ids;
    if (a.have_decode) {
        ErnieTokenizer tk;
        std::string terr;
        if (!ernie_tokenizer_load(a.model, tk, terr)) { std::fprintf(stderr, "tokenizer: %s\n", terr.c_str()); return 1; }
        std::string text;
        try {
            text = ernie_tokenizer_decode(tk, parse_ids(a.decode_ids));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "decode: %s\n", e.what());
            return 1;
        }
        std::fwrite(text.data(), 1, text.size(), stdout);
        return 0;
    }
    if (a.have_prompt) {
        // TEXT -> ids through the GGUF's own tokenizer; --encode-only stops here, before any weights are opened.
        ErnieTokenizer tk;
        std::string terr;
        if (!ernie_tokenizer_load(a.model, tk, terr)) { std::fprintf(stderr, "tokenizer: %s\n", terr.c_str()); return 1; }
        try {
            ids = ernie_tokenizer_encode(tk, a.prompt, a.parse_special);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "encode: %s\n", e.what());
            return 1;
        }
        if (!a.no_bos) ids.insert(ids.begin(), tk.bos_id);   // the files carry add_bos_token = true
        if (a.encode_only) {
            for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? "," : "", ids[i]);
            std::printf("\n");
            return 0;
        }
    } else if (!a.ids_file.empty()) {
        if (!read_ids_file(a.ids_file, ids)) { std::fprintf(stderr, "cannot read ids from %s\n", a.ids_file.c_str()); return 2; }
    } else {
        ids = parse_ids(a.ids_csv);
    }
    std::vector<std::vector<int>> batch_ids;
    if (!a.batch.empty()) {
        std::FILE* f = std::fopen(a.batch.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot read %s\n", a.batch.c_str()); return 2; }
        std::string all;
        char buf[1 << 16];
        size_t k;
        while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, k);
        std::fclose(f);
        ErnieTokenizer tk;
        std::string terr;
        if (!ernie_tokenizer_load(a.model, tk, terr)) { std::fprintf(stderr, "tokenizer: %s\n", terr.c_str()); return 1; }
        size_t b = 0;
        while (b <= all.size()) {
            size_t e = all.find('\x1e', b);
            if (e == std::string::npos) e = all.size();
            if (e > b) {
                std::vector<int> v = ernie_tokenizer_encode(tk, all.substr(b, e - b), true);
                if (!a.no_bos) v.insert(v.begin(), tk.bos_id);
                batch_ids.push_back(std::move(v));
            }
            b = e + 1;
        }
        if (batch_ids.empty()) { std::fprintf(stderr, "--batch: no prompts in %s\n", a.batch.c_str()); return 2; }
        if (ids.empty()) ids = batch_ids[0];
    }
    if (ids.empty() && !a.serve) { std::fprintf(stderr, "empty prompt\n"); return 2; }

    // ---- tokenizer (decode only) from the GGUF's own vocabulary ----
    std::vector<std::string> tokens;
    {
        std::unique_ptr<GgufModel> m(new GgufModel(GgufModel::open(a.model)));
        const MetaValue* t = m->meta().get("tokenizer.ggml.tokens");
        if (t && t->type == MetaType::ARRAY) {
            tokens.reserve(t->items.size());
            for (const MetaValue& e : t->items) tokens.push_back(e.s);
        }
    }
    int eog_id = -1;   // <|ernie:im_end|>: a --batch continuation stops there
    for (size_t t = 0; t < tokens.size(); ++t) if (tokens[t] == "<|ernie:im_end|>") { eog_id = (int) t; break; }
    auto text_of = [&](int id) -> std::string {
        if (id >= 0 && (size_t) id < tokens.size()) return detok_bytes(tokens[(size_t) id]);
        return "<" + std::to_string(id) + ">";
    };

    // ---- the expert tier (same GGUF) ----
    std::unique_ptr<Ds41MoeTier> tier;
    std::unique_ptr<Ds41GgufBlobs> src;
    Ds41MoeGeom gm;
    Ds41MoeConfig tcfg;
    std::string err;
    if (!a.no_tier) {
        if (!ds41_moe_geom_from_gguf(a.model, gm, err)) { std::fprintf(stderr, "moe geom: %s\n", err.c_str()); return 1; }
        if (gm.act != cpu::NativeAct::SiTU) { std::fprintf(stderr, "moe geom is not SiTU (arch not ernie5_5?)\n"); return 1; }
        src.reset(new Ds41GgufBlobs(gm, a.model));
        if (!src->open(err)) { std::fprintf(stderr, "moe blobs: %s\n", err.c_str()); return 1; }
        tcfg.cpu_only = a.experts != "gpu";
        tcfg.threads = a.threads;
        tcfg.arena_gib = a.arena_gib;
        tcfg.max_arena_gib = a.arena_gib;
        tcfg.slots = a.slots > 0 ? a.slots : 0;   // 0 = auto (sized at the seed from the free VRAM)
        tcfg.skip_miss = a.skip_miss;
        tcfg.skip_file = a.skip_file;
        tcfg.vram_lru = a.vram_lru;
        tcfg.arena_adapt = a.arena_adapt;
        tcfg.arena_skip_resident = a.arena_skip_resident;
        tcfg.chunk_mmq = a.chunk_mmq;
        tcfg.chunk_prestage = a.chunk_prestage;
        tcfg.chunk_prefetch = a.chunk_prefetch;
        tcfg.remote_device = a.remote_device;
        tcfg.remote_gib = a.remote_gib;
        if (a.pcie >= 0.0) tcfg.pcie_frac = a.pcie;
        tier.reset(new Ds41MoeTier());
        if (!tier->init(gm, src.get(), tcfg, err)) { std::fprintf(stderr, "tier init: %s\n", err.c_str()); return 1; }
        if (!a.count_routes.empty()) {
            g_route_experts = gm.n_experts;
            g_route_counts.assign((size_t) (gm.n_layers * gm.n_experts), 0);
        }
    }

    // ---- the dense graphs ----
    ErnieGraphConfig cfg;
    cfg.abl_file = a.abl_file;
    cfg.abl_scale = a.abl_scale;
    cfg.n_threads = a.threads;
    cfg.ctx_max = a.ctx;
    if (a.kv_mla == "q8_0") cfg.mla_kv_type = GGML_TYPE_Q8_0;
    else if (a.kv_mla == "q4_0") cfg.mla_kv_type = GGML_TYPE_Q4_0;
    else if (a.kv_mla != "f16") { std::fprintf(stderr, "--kv-mla: f16, q8_0 or q4_0 (got %s)\n", a.kv_mla.c_str()); return 2; }
    cfg.mla_hadamard = a.kv_hadamard;
    if (a.cuda) {
#ifdef ERNIE_HAVE_CUDA
        cfg.backend = ggml_backend_cuda_init(0);
        if (!cfg.backend) { std::fprintf(stderr, "--cuda: ggml_backend_cuda_init(0) failed\n"); return 1; }
#else
        std::fprintf(stderr, "--cuda: this build has no CUDA backend\n");
        return 1;
#endif
    }
    ErnieGraph eg;
    if (!eg.init(a.model, cfg, err)) { std::fprintf(stderr, "graph init: %s\n", err.c_str()); return 1; }
    const int il1 = a.layer_hi > 0 ? std::min<int>(a.layer_hi, (int) eg.n_layer()) : (int) eg.n_layer();
    const int il0 = a.layer_lo;
    // the backend the graphs really run on (not the flag): a --cuda build must NOT report "cpu" here
    const char* be = ggml_backend_dev_name(ggml_backend_get_device(eg.backend()));
    if (!a.quiet)
        std::printf("ernie_generate: %lld layers (running %d..%d), ctx %lld, dense backend %s, prompt %zu ids\n",
                    (long long) eg.n_layer(), il0, il1, (long long) a.ctx, be, ids.size());

    // ---- the DSpark drafter: its dense half takes VRAM BEFORE the tier's auto cache budget reads the free memory ----
    std::unique_ptr<Drafter> drafter;
    if (!a.dspark.empty()) {
        if (a.serve || !batch_ids.empty()) { std::fprintf(stderr, "--dspark: one-shot decode only (not --serve / --batch yet)\n"); return 2; }
        if (!tier) { std::fprintf(stderr, "--dspark needs the expert tier\n"); return 2; }
        if (a.temp != 0.0f) { std::fprintf(stderr, "--dspark is greedy only (--temp 0)\n"); return 2; }
        if (il0 != 0 || il1 != (int) eg.n_layer()) { std::fprintf(stderr, "--dspark needs every layer\n"); return 2; }
        drafter.reset(new Drafter());
        if (!drafter_init(*drafter, a, eg, gm, cfg, err)) { std::fprintf(stderr, "dspark: %s\n", err.c_str()); return 1; }
    }

    // ---- the expert tier's arena and VRAM cache, after the dense graphs own their VRAM ----
    // (an auto budget measures the FREE VRAM, so it must be read once the dense half has allocated what it needs)
    if (tier && !tcfg.cpu_only) {
        double slot_gib = 0.0;
#ifdef ERNIE_HAVE_CUDA
        // --slots 0 = auto: size the VRAM cache from the card's free memory (ds41_generate's serve seed does the
        // same).  Without a budget the tier opens NO cache and every expert miss goes to the arena/file tier.
        if (a.slots <= 0) {
            ggml_backend_t gbe = ggml_backend_cuda_init(0);
            if (gbe) {
                size_t fr = 0, tot = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(gbe), &fr, &tot);
                ggml_backend_free(gbe);
                slot_gib = std::max(0.0, ((double) fr - 1.0 * 1073741824.0) / 1073741824.0);
            }
        }
#endif
        if (slot_gib > 0.0) {
            tier->set_slot_gib(slot_gib);
            if (!a.quiet)
                std::printf("tier: --slots auto -> %.2f GiB of VRAM for the expert cache\n", slot_gib);
        }
        // seed the VRAM cache from the layers this run uses, in order (a dev file: all of them)
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int64_t l = 0; l < gm.n_layers; ++l)
            for (int64_t e = 0; e < gm.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        if (!a.rank_counts.empty()) {
            std::vector<int64_t> c;
            if (!read_counts(a.rank_counts, (size_t) (gm.n_layers * gm.n_experts), c, err)) {
                std::fprintf(stderr, "--rank-counts: %s\n", err.c_str());
                return 1;
            }
            // hottest first; ties keep index order (stable), so never-routed experts trail in layer order
            std::stable_sort(ranked.begin(), ranked.end(), [&](const auto& x, const auto& y) {
                return c[(size_t) x.first * gm.n_experts + x.second] > c[(size_t) y.first * gm.n_experts + y.second];
            });
            int64_t tot = 0, nz = 0;
            for (int64_t v : c) { tot += v; nz += v > 0; }
            if (!a.quiet)
                std::printf("tier: seed order from %s (%lld routed tokens, %lld of %zu experts ever routed)\n",
                            a.rank_counts.c_str(), (long long) tot, (long long) nz, c.size());
        }
        if (!tier->seed_from_ranked(ranked, err)) { std::fprintf(stderr, "tier seed: %s\n", err.c_str()); return 1; }
        if (!a.quiet)
            std::printf("tier %s: %lld resident, %lld arena\n", tier->mode(), (long long) tier->resident(),
                        (long long) tier->arena_experts());
    }

    auto print_stats = [&](const char* phase, int toks) {
        if (!a.stats || !tier) return;
        const Ds41MoeStats& s = tier->stats();
        std::printf("stats %s (%d tok): lookups %lld | hits %lld cpu %lld pcie %lld file %lld skipped %lld | "
                    "hit %.0f cpu %.0f pcie %.0f file %.0f fwait %.0f gap %.0f wall %.0f ms\n",
                    phase, toks, (long long) s.lookups(), (long long) s.hits, (long long) s.cpu, (long long) s.pcie,
                    (long long) s.file_tier, (long long) s.skipped, s.hit_ms, s.cpu_ms, s.pcie_ms, s.file_ms,
                    s.fwait_ms, s.gap_ms, s.wall_ms);
        tier->reset_stats();
    };
    if (a.serve) {
        // stop ids: <|ernie:im_end|> (the end of a turn) and the GGUF's eos
        std::vector<int> stop_ids;
        if (eog_id >= 0) stop_ids.push_back(eog_id);
        {
            std::unique_ptr<GgufModel> m(new GgufModel(GgufModel::open(a.model)));
            const MetaValue* e = m->meta().get("tokenizer.ggml.eos_token_id");
            if (e) {
                const int id = (int) e->u;
                if (std::find(stop_ids.begin(), stop_ids.end(), id) == stop_ids.end()) stop_ids.push_back(id);
            }
        }
        // DRY's sequence breakers: every token whose text holds a newline, ':', '"' or '*' (llama.cpp's defaults)
        // (through the tokenizer: the GGUF vocabulary list read above can come back short)
        std::vector<char> breaker((size_t) eg.geom().vocab, 0);
        {
            ErnieTokenizer btk;
            std::string bterr;
            if (!ernie_tokenizer_load(a.model, btk, bterr)) { std::fprintf(stderr, "tokenizer: %s\n", bterr.c_str()); return 1; }
            for (size_t t = 0; t < breaker.size(); ++t) {
                std::string tx;
                try { tx = ernie_tokenizer_decode(btk, std::vector<int>{ (int) t }); } catch (const std::exception&) { continue; }
                breaker[t] = tx.find_first_of("\n:\"*") != std::string::npos;
            }
            const std::vector<int> e = ernie_tokenizer_encode(btk, "<|ernie:im_end|>", true);
            if (e.size() == 1 && std::find(stop_ids.begin(), stop_ids.end(), e[0]) == stop_ids.end()) stop_ids.push_back(e[0]);
        }
        return serve_loop(a, eg, tier.get(), il0, il1, stop_ids, breaker);
    }
    // ---- --batch: every prompt from position 0 in this one load ----
    if (!batch_ids.empty()) {
        if (!eg.snapshot(0)) { std::fprintf(stderr, "snapshot: %s\n", eg.last_error().c_str()); return 1; }
        std::FILE* fo = a.batch_out.empty() ? nullptr : std::fopen(a.batch_out.c_str(), "wb");
        std::FILE* fl = a.batch_logits.empty() ? nullptr : std::fopen(a.batch_logits.c_str(), "wb");
        if ((!a.batch_out.empty() && !fo) || (!a.batch_logits.empty() && !fl)) {
            std::fprintf(stderr, "--batch: cannot open an output file\n");
            return 1;
        }
        // the GGUF vocabulary list above can come back short (metadata array cap): text + the stop id via the tokenizer
        ErnieTokenizer btk;
        std::string bterr;
        if (!ernie_tokenizer_load(a.model, btk, bterr)) { std::fprintf(stderr, "tokenizer: %s\n", bterr.c_str()); return 1; }
        {
            const std::vector<int> e = ernie_tokenizer_encode(btk, "<|ernie:im_end|>", true);
            if (e.size() == 1) eog_id = e[0];
        }
        const auto tb0 = std::chrono::steady_clock::now();
        int64_t ptoks = 0, dtoks = 0;
        for (size_t pi = 0; pi < batch_ids.size(); ++pi) {
            const std::vector<int>& pid = batch_ids[pi];
            if (!eg.restore()) { std::fprintf(stderr, "restore: %s\n", eg.last_error().c_str()); return 1; }
            const int chunk = a.prefill_chunk > 0 ? a.prefill_chunk : (int) pid.size();
            std::vector<float> lg;
            int64_t pos = 0;
            while (pos < (int64_t) pid.size()) {
                const int n = (int) std::min<int64_t>(chunk, (int64_t) pid.size() - pos);
                const bool last = pos + n == (int64_t) pid.size();
                if (!run_pass(eg, tier.get(), il0, il1, pid.data() + pos, n, pos, nullptr, last ? &lg : nullptr, err)) {
                    std::fprintf(stderr, "batch %zu prefill: %s\n", pi, err.c_str());
                    return 1;
                }
                pos += n;
            }
            ptoks += (int64_t) pid.size();
            if (fl) std::fwrite(lg.data(), 4, lg.size(), fl);
            if (fo) {
                std::string out_text, out_ids;
                std::vector<int> gen;
                int next = argmax(lg.data(), (int) lg.size());
                for (int k = 0; k < a.n_predict && pos < a.ctx; ++k) {
                    out_ids += (k ? "," : "") + std::to_string(next);
                    gen.push_back(next);
                    if (next == eog_id) break;
                    std::vector<float> l2;
                    if (!run_pass(eg, tier.get(), il0, il1, &next, 1, pos, nullptr, &l2, err)) {
                        std::fprintf(stderr, "batch %zu decode: %s\n", pi, err.c_str());
                        return 1;
                    }
                    ++pos;
                    ++dtoks;
                    next = argmax(l2.data(), (int) l2.size());
                }
                try { out_text = ernie_tokenizer_decode(btk, gen); } catch (const std::exception&) { out_text = "<decode error>"; }
                std::string esc;
                for (char c : out_text) {
                    if (c == '\n') esc += "\\n";
                    else if (c == '\t') esc += "\\t";
                    else if (c == '\\') esc += "\\\\";
                    else esc += c;
                }
                std::fprintf(fo, "%s\t%s\n", out_ids.c_str(), esc.c_str());
                std::fflush(fo);
            }
            if (!a.quiet && (pi + 1) % 10 == 0) {
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count();
                std::fprintf(stderr, "batch: %zu / %zu prompts, %.0f s, %lld prefill + %lld decode tokens\n", pi + 1,
                             batch_ids.size(), el, (long long) ptoks, (long long) dtoks);
            }
        }
        for (std::FILE* f : { fo, fl }) if (f) std::fclose(f);
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count();
        std::printf("batch: %zu prompts in %.1f s (%lld prefill tokens, %lld decode tokens)\n", batch_ids.size(), el,
                    (long long) ptoks, (long long) dtoks);
        print_stats("batch", (int) (ptoks + dtoks));
        return 0;
    }

    // ---- prefill ----
    if (tier) tier->reset_stats();
    const auto t0 = std::chrono::steady_clock::now();
    // --prefill-chunk applies to a taps dump too: the dump holds the LAST pass's columns (a chunk's taps are only that
    // chunk's columns), which is what the chunk-vs-single-pass gate compares against the matching columns of a
    // one-pass dump.
    const int chunk = a.prefill_chunk > 0 ? a.prefill_chunk : (int) ids.size();
    if (!a.dump_logits_rows.empty() && chunk < (int) ids.size()) {
        std::fprintf(stderr, "--dump-logits-rows needs the whole prompt in one prefill pass (--prefill-chunk %d, %zu ids)\n",
                     a.prefill_chunk, ids.size());
        return 1;
    }
    std::vector<float> logits;
    std::vector<ErnieGraphTap> all_taps;
    int64_t pos = 0;
    while (pos < (int64_t) ids.size()) {
        const int n = (int) std::min<int64_t>(chunk, (int64_t) ids.size() - pos);
        const bool last = pos + n == (int64_t) ids.size();
        std::vector<float> lg, mh;
        if (drafter) { g_mh = &mh; g_mh_rows = 128; g_mh_layers = drafter->targets; }
        const bool pok = run_pass(eg, tier.get(), il0, il1, ids.data() + pos, n, pos,
                                  last && !a.dump_taps.empty() ? &all_taps : nullptr, last ? &lg : nullptr, err);
        g_mh = nullptr;
        if (!pok) {
            std::fprintf(stderr, "prefill: %s\n", err.c_str());
            return 1;
        }
        if (drafter) {   // the pass's last min(n, 128) positions into the drafter's window
            const int rows = std::min(n, 128);
            if (!drafter_commit(*drafter, mh, pos + n - rows, rows, err)) { std::fprintf(stderr, "dspark: %s\n", err.c_str()); return 1; }
        }
        if (last) logits = std::move(lg);
        pos += n;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t1 - t0).count();

    if (!a.dump_taps.empty() && !write_dump(a.dump_taps, all_taps, err)) {
        std::fprintf(stderr, "dump: %s\n", err.c_str());
        return 1;
    }
    // a degenerate head (every logit equal) is not a result: the dev files carry Base PLACEHOLDER MultiMax ranges and
    // clamp the whole vocabulary to one value, so logits from them say nothing about the model - gate on the taps
    {
        const float v0 = logits.empty() ? 0.0f : logits[0];
        bool flat = !logits.empty();
        for (float v : logits) if (v != v0) { flat = false; break; }
        if (flat)
            std::fprintf(stderr, "WARNING: all %zu logits are %.6f - a degenerate head (placeholder MultiMax ranges in a "
                                 "dev file?); compare --dump-taps, not logits\n", logits.size(), (double) v0);
    }
    if (!a.dump_logits.empty()) {
        std::FILE* f = std::fopen(a.dump_logits.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", a.dump_logits.c_str()); return 1; }
        std::fwrite(logits.data(), sizeof(float), logits.size(), f);
        std::fclose(f);
    }
    // every row, row-major [n_rows, vocab]: row i predicts token i+1, so a teacher-forced NLL is
    // -mean(log_softmax(row_i)[ids[i+1]]) over i = 0..n_rows-2 (llama-perplexity's ln(PPL) over one chunk)
    if (!a.dump_logits_rows.empty()) {
        const size_t nrows = ids.size(), nvocab = (size_t) eg.geom().vocab;
        std::vector<float> rows(nrows * nvocab, 0.0f);
        if (!eg.logits(0, (int) nrows, rows.data(), nullptr)) {
            std::fprintf(stderr, "dump-logits-rows: %s\n", eg.last_error().c_str());
            return 1;
        }
        std::FILE* f = std::fopen(a.dump_logits_rows.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", a.dump_logits_rows.c_str()); return 1; }
        std::fwrite(rows.data(), sizeof(float), rows.size(), f);
        std::fclose(f);
        if (!a.quiet)
            std::fprintf(stderr, "logits rows: %zu x %zu (%.1f MiB) -> %s\n", nrows, nvocab,
                         (double) rows.size() * 4.0 / (1024.0 * 1024.0), a.dump_logits_rows.c_str());
    }

    // ---- decode ----
    std::string text;
    for (int id : ids) text += text_of(id);
    std::vector<int> generated;
    int next = argmax(logits.data(), (int) logits.size());
    print_stats("prefill", (int) ids.size());
    const auto t2 = std::chrono::steady_clock::now();
    if (drafter) {
        int64_t P = pos;
        while ((int) generated.size() < a.n_predict) {
            if (P + drafter->block + 1 > a.ctx) { generated.push_back(next); text += text_of(next); break; }
            std::vector<int> emit;
            int nxt = -1;
            if (!dspark_step(*drafter, eg, tier.get(), il0, il1, next, P, emit, &nxt, err)) {
                std::fprintf(stderr, "dspark decode: %s\n", err.c_str());
                return 1;
            }
            for (int id : emit) {
                if ((int) generated.size() >= a.n_predict) break;
                generated.push_back(id);
                text += text_of(id);
            }
            next = nxt;
        }
        pos = P;
    }
    for (int k = drafter ? a.n_predict : 0; k < a.n_predict; ++k) {
        generated.push_back(next);
        text += text_of(next);
        if (pos >= a.ctx) break;
        std::vector<float> lg;
        if (!run_pass(eg, tier.get(), il0, il1, &next, 1, pos, nullptr, &lg, err)) {
            std::fprintf(stderr, "decode: %s\n", err.c_str());
            return 1;
        }
        ++pos;
        logits = std::move(lg);
        next = argmax(logits.data(), (int) logits.size());
    }
    const auto t3 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t3 - t2).count();
    print_stats("decode", a.n_predict);
    if (drafter && !a.quiet) drafter_report(*drafter);
    if (g_pred_n > 0 && !a.quiet)
        std::printf("predict: %lld routed, %.1f %% in the guessed top 8, %.1f %% in the top 16\n", (long long) g_pred_n,
                    100.0 * g_pred_in8 / g_pred_n, 100.0 * g_pred_in16 / g_pred_n);
    if (!g_route_counts.empty() && !flush_route_counts(a.count_routes, err)) {
        std::fprintf(stderr, "--count-routes: %s\n", err.c_str());
        return 1;
    }

    if (!a.out_text.empty()) {
        std::FILE* f = std::fopen(a.out_text.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", a.out_text.c_str()); return 1; }
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
    }
    if (!a.quiet) {
        std::printf("%s\n", text.c_str());
        std::printf("prefill %zu tok in %.2f s (%.2f tok/s) | decode %d tok in %.2f s (%.2f tok/s)\n", ids.size(),
                    prefill_s, (double) ids.size() / std::max(1e-9, prefill_s), (int) generated.size(), decode_s,
                    (double) generated.size() / std::max(1e-9, decode_s));
        std::printf("generated:");
        for (int id : generated) std::printf(" %d", id);
        std::printf("\n");
    }
    return 0;
}
