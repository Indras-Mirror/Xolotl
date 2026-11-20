// tests/core/ernie_geometry_test.cpp - the Phase-1 geometry + tensor-map gate for ERNIE 5.5 Flash.
//
// Two parts:
//
//   * A pure unit test of the conditional tensor map, on a synthesized geometry with the released artifact's
//     numbers.  No file, no GPU: it checks the count (1514 - the real file's own `split.tensors.count`) and that
//     the HCA / MLA+indexer / dense-lead / latent-MoE / MTP tensors appear on exactly the blocks the
//     architecture says, with the shapes llama.cpp's `load_arch_tensors` creates.
//
//   * The real-file gate.  ERNIE ships as 58 shards (~122 GB for the build we have); shard 1 carries the whole
//     metadata and every SMALL tensor (819 of the 1514), and no big projection.  So this part opens that one
//     shard with `GgufFile` (not `GgufModel`, which refuses a partial multi-shard file), reads every
//     `ernie5_5.*` key, and then requires:
//       - zero unknown tensor names among the 819 present,
//       - every present tensor matching its spec's shape and type class,
//       - the missing set to be exactly the 695 big tensors, every one of them `Weight`-class,
//       - present + missing == 1514.
//     That is a real check of the map without the 120 GB download; the full-file gate is the same call with
//     `GgufModel` once all 58 shards are on disk.
//
// The shard path is argv[1] or $ERNIE_GGUF.  When it is absent the real-file part is skipped with a printed
// reason (so a box without the file still builds and runs the unit part).
#include "strata/artifact/ernie_geometry.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// The released artifact's geometry, filled here the way ernie_read_geometry fills it (values read from the real
// 60G/122G/245G GGUF header on 2026-11-16).
strata::ErnieGeometry artifact_geometry() {
    strata::ErnieGeometry g;
    g.n_embd = 4096;
    g.n_layer_all = 44;
    g.n_layer_nextn_requested = 1;
    g.n_layer_nextn = 1;
    g.n_layer_dense_lead = 1;
    g.n_head = 64;
    g.n_head_kv = 1;
    g.d_rope = 64;
    g.r_q = 1024;
    g.o_groups = 8;
    g.o_lora = 1024;
    g.n_swa = 128;
    g.d_head_hca = 512;
    g.d_head_hca_v = 512;
    g.d_head_mla_k = 256;
    g.d_head_mla_v = 256;
    g.kv_lora = 512;
    g.d_kv_row = 576;  // key_length = kv_lora + rope, NOT the HCA head
    g.n_expert = 512;
    g.n_expert_used = 10;
    g.n_ff = 14336;
    g.n_ff_exp = 2048;
    g.n_expert_shared = 1;
    g.n_embd_lat = 2048;
    g.gating_func = 2;
    g.weights_norm = true;
    g.hc = 4;
    g.hc_sinkhorn_iters = 20;
    g.hc_eps = 1e-6;
    g.indexer_n_head = 64;
    g.indexer_key_dim = 128;
    g.indexer_top_k = 2048;
    g.situ_beta = 4.0;
    g.situ_linear_beta = 25.0;
    g.logit_multimax = {14.875, 15.8125, 8.75, 0.3046875, -0.095703125, 0.470703125, -0.15625, -0.01531982421875};
    g.rms_eps = 1e-6;
    g.rope_scaling_type = "yarn";
    g.rope_freq_base = 8000000.0;
    g.rope_factor = 16.0;
    g.rope_orig_ctx = 65536.0;
    g.yarn_beta_fast = 32.0;
    g.yarn_beta_slow = 1.0;
    g.compress_rope_base = 160000.0;
    g.context_length = 1048576;
    g.vocab_size = 201216;
    g.tokenizer_pre = "ernie5";
    // the real array, verbatim: a 0 every 7th block, and the MTP block (43) is also a 0.
    g.compress_ratios = {128, 128, 128, 128, 128, 128, 128, 0,   128, 128, 128, 128, 128, 128, 0,   128,
                         128, 128, 128, 128, 128, 0,   128, 128, 128, 128, 128, 128, 0,   128, 128, 128,
                         128, 128, 128, 0,   128, 128, 128, 128, 128, 128, 0,   0};
    return g;
}

bool has_name(const std::vector<strata::ErnieTensorSpec>& v, const std::string& n) {
    for (const auto& s : v) if (s.name == n) return true;
    return false;
}
const strata::ErnieTensorSpec* find_spec(const std::vector<strata::ErnieTensorSpec>& v, const std::string& n) {
    for (const auto& s : v) if (s.name == n) return &s;
    return nullptr;
}

void unit_tests() {
    const strata::ErnieGeometry g = artifact_geometry();
    const auto t = strata::ernie_expected_tensors(g);
    check(t.size() == 1514, "synthesized map has 1514 tensors (= the file's split.tensors.count)");

    // derived dimensions
    check(g.n_layer() == 43 && g.hc_dim() == 16384 && g.hc_mix_dim() == 24, "43 trunk layers; hc_dim 16384; hc_mix 24");
    check(g.mla_qk_nope() == 192 && g.hca_heads_dim() == 32768 && g.o_group_dim() == 4096,
          "MLA qk_nope 192; HCA heads_dim 32768; o_group_dim 4096");
    check(g.mla_heads_dim() == 16384, "MLA heads_dim 16384");

    // the layer pattern: 0 is the dense lead, 1..6 HCA, 7 MLA, ... 43 the MTP block
    check(has_name(t, "blk.0.ffn_gate.weight") && !has_name(t, "blk.0.ffn_gate_inp.weight"),
          "blk.0 (dense lead): plain FFN, no latent MoE");
    check(has_name(t, "blk.1.ffn_gate_inp.weight") && !has_name(t, "blk.1.ffn_gate.weight"),
          "blk.1 (latent MoE): router present, no dense FFN");
    check(has_name(t, "blk.6.attn_compressor_ape.weight") && !has_name(t, "blk.6.indexer.proj.weight"),
          "blk.6 (ratio 128, HCA): compressor present, no indexer");
    check(has_name(t, "blk.7.indexer.proj.weight") && !has_name(t, "blk.7.attn_compressor_ape.weight"),
          "blk.7 (ratio 0, MLA+DSA): indexer present, no compressor");
    check(has_name(t, "blk.43.nextn.eh_proj.weight") && has_name(t, "blk.43.ffn_gate_inp.weight") &&
              has_name(t, "blk.43.indexer.proj.weight"),
          "blk.43 (MTP): nextn head + latent MoE + MLA indexer");
    check(!has_name(t, "blk.43.attn_compressor_ape.weight"), "blk.43 (MTP) is an MLA block, not HCA");

    // counts by family, from the architecture rather than the total
    int hca = 0, mla = 0;
    for (int64_t i = 0; i < g.n_layer_all; ++i) (g.is_hca(i) ? hca : mla)++;
    check(hca == 37 && mla == 7, "37 HCA blocks, 7 MLA blocks (6 in the trunk + the MTP block)");

    // shape spot checks against llama.cpp's load_arch_tensors (GGUF order)
    auto shape_is = [&](const char* n, std::initializer_list<uint64_t> want, const char* what) {
        const strata::ErnieTensorSpec* s = find_spec(t, n);
        check(s != nullptr && s->shape == std::vector<uint64_t>(want), what);
    };
    shape_is("blk.6.attn_q_b.weight", {1024, 32768}, "HCA attn_q_b is [q_lora, n_head*d_head_hca]");
    shape_is("blk.7.attn_q_b.weight", {1024, 16384}, "MLA attn_q_b is [q_lora, n_head*d_head_mla]");
    shape_is("blk.7.attn_kv_a_mqa.weight", {4096, 576}, "MLA attn_kv_a_mqa is [D, kv_lora+rope]");
    shape_is("blk.7.attn_k_b.weight", {192, 512, 64}, "MLA attn_k_b is [qk_nope, kv_lora, n_head]");
    shape_is("blk.6.attn_output_a.weight", {4096, 1024, 8}, "HCA attn_output_a is [o_group_dim, o_lora, o_groups]");
    shape_is("blk.6.attn_compressor_ape.weight", {512, 128}, "HCA compressor APE is [d_head_hca, 128]");
    shape_is("blk.0.ffn_gate.weight", {4096, 14336}, "dense lead FFN is [D, feed_forward_length]");
    shape_is("blk.1.ffn_gate_exps.weight", {2048, 2048, 512}, "routed experts are [latent, n_ff_exp, n_expert]");
    shape_is("blk.1.ffn_gate_shexp.weight", {4096, 2048}, "shared expert is [D, n_ff_exp*n_shared]");
    shape_is("blk.1.ffn_gate_inp.weight", {4096, 1024}, "the two routers are stacked: [D, 2*n_expert]");
    shape_is("token_embd.weight", {4096, 201216}, "token_embd is [D, vocab]");
}

void real_shard(const std::string& path) {
    strata::GgufFile f(path);
    std::printf("  opened %s: %zu tensors\n", path.c_str(), f.tensors().size());

    std::string err = strata::check_architecture_ernie(f);
    check(err.empty(), err.empty() ? "architecture is ernie5_5 with the expected constants" : err.c_str());
    if (!err.empty()) return;

    strata::ErnieGeometry g;
    const auto finder = [&f](const std::string& n) { return f.find(n); };
    err = strata::ernie_read_geometry_meta(f, g, finder);
    check(err.empty(), err.empty() ? "every ernie5_5.* key read into ErnieGeometry" : err.c_str());
    if (!err.empty()) return;

    check(g.n_embd == 4096 && g.n_layer_all == 44 && g.n_layer() == 43, "D=4096, 44 blocks, 43 trunk layers");
    check(g.n_head == 64 && g.n_head_kv == 1 && g.d_rope == 64, "64 q heads / 1 kv head / rope 64");
    check(g.d_head_hca == 512 && g.d_head_mla_k == 256 && g.d_head_mla_v == 256 && g.kv_lora == 512,
          "HCA head 512 (key_length_swa); MLA head 256/256; kv_lora 512");
    check(g.d_kv_row == 576 && g.d_kv_row == g.kv_lora + g.d_rope,
          "MLA cache row K half is 576 = kv_lora + rope, distinct from the HCA head");
    check(g.r_q == 1024 && g.o_groups == 8 && g.o_lora == 1024, "q_lora 1024, 8 output groups, o_lora 1024");
    check(g.n_expert == 512 && g.n_expert_used == 10 && g.n_ff == 14336 && g.n_ff_exp == 2048,
          "512 experts, top-10, dense FFN 14336, expert width 2048");
    check(g.n_embd_lat == 2048 && g.n_expert_shared == 1, "latent 2048, 1 shared expert");
    check(g.hc == 4 && g.hc_sinkhorn_iters == 20 && g.gating_func == 2, "hc 4, sinkhorn 20, gating func 2");
    check(g.indexer_n_head == 64 && g.indexer_key_dim == 128 && g.indexer_top_k == 2048, "indexer 64/128/2048");
    check(g.situ_beta == 4.0 && g.situ_linear_beta == 25.0, "situ beta 4 / linear beta 25");
    check(g.logit_multimax.size() == 8, "MultiMax head has 8 learned parameters");
    check(g.rope_scaling_type == "yarn" && g.rope_factor == 16 && g.rope_orig_ctx == 65536 && g.rope_freq_base == 8e6,
          "YaRN x16 from 65536, rope base 8e6");
    check(g.compress_rope_base == 160000, "compress rope base 160000");
    check(g.n_swa == 128, "sliding window 128");
    check(g.vocab_size == 201216 && g.tokenizer_pre == "ernie5", "vocab 201216 from the tokenizer array, pre 'ernie5'");
    check(g.n_layer_nextn_requested == 1 && g.n_layer_nextn == 1, "nextn requested 1 and present (blk.43 has nextn.*)");
    check(g.compress_ratios.size() == 44, "compress_ratios covers every block");

    // ---- the map: zero unknowns, every present tensor matching its spec
    const auto want = strata::ernie_expected_tensors(g);
    const auto idx = strata::ernie_spec_index(want);
    const uint64_t payload = f.file_size() - f.data_start();
    std::string first_bad;
    size_t bad = 0;
    for (const auto& t : f.tensors()) {
        const auto it = idx.find(t.name);
        if (it == idx.end()) {
            if (first_bad.empty()) first_bad = "unknown tensor '" + t.name + "'";
            ++bad;
            continue;
        }
        const uint64_t bytes = strata::tensor_payload_bytes(t);
        const bool bounded = bytes != 0 && t.offset <= payload && bytes <= payload - t.offset;
        const std::string e = strata::ernie_check_one(*it->second, t, bounded);
        if (!e.empty()) {
            if (first_bad.empty()) first_bad = e;
            ++bad;
        }
    }
    check(bad == 0, bad == 0 ? "every tensor in the shard is known and matches its shape and type class"
                             : first_bad.c_str());

    const auto missing = strata::ernie_missing_tensors(want, finder);
    check(want.size() == 1514, "the map has 1514 tensors");
    check(missing.size() == 695, "695 tensors are in later shards (the big projections)");
    check(want.size() - missing.size() == f.tensors().size(), "present + missing == 1514");

    size_t missing_not_weight = 0;
    for (const auto& n : missing) {
        const strata::ErnieTensorSpec* s = find_spec(want, n);
        if (s == nullptr || s->cls != strata::ErnieTypeClass::Weight) ++missing_not_weight;
    }
    check(missing_not_weight == 0, "every missing tensor is a Weight-class tensor (no small tensor is unaccounted)");
    std::printf("    e.g. missing: %s, %s, %s\n", missing.empty() ? "-" : missing.front().c_str(),
                missing.size() > 1 ? missing[1].c_str() : "-", missing.size() > 2 ? missing[2].c_str() : "-");
}
}  // namespace

int main(int argc, char** argv) {
    unit_tests();
    const std::string path =
            argc > 1 ? argv[1]
                     : (std::getenv("ERNIE_GGUF")
                                ? std::getenv("ERNIE_GGUF")
                                : "models/ernie55/ERNIE-5.5-Flash-Base-abliterated-Mix-60G/"
                                  "ERNIE-5.5-Flash-Base-abliterated-Mix-60G-00001-of-00058.gguf");
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("  SKIP real-file gate: %s not readable (pass argv[1] or $ERNIE_GGUF)\n", path.c_str());
    } else {
        std::fclose(f);
        real_shard(path);
    }
    std::printf("%s (%d failure%s)\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
