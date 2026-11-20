// LoRA adapters as a llama.cpp-format GGUF (what llama.cpp's --lora and ARA's export use):
//   general.type = "adapter", general.architecture = <base arch>, adapter.type = "lora", adapter.lora.alpha = f32
//   per adapted weight "<base tensor name minus .weight>.weight.lora_a"  ne = [n_in, rank]   (F32)
//                      "<...>.weight.lora_b"                             ne = [rank, n_out]  (F32)
// llama.cpp pairs "<name>.lora_a"/"<name>.lora_b" with the model tensor "<name>" and checks
// W.ne[0] == a.ne[0], W.ne[1] == b.ne[1]; the applied delta is (alpha / rank) * B (A x) times the user scale.
// Header-only; needs ggml + gguf (ggml-base).
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace xolotl::train {

struct LoraTensorPair {
    std::string name;            // base tensor name, e.g. "blk.41.attn_output.weight"
    int64_t n_in = 0, n_out = 0, rank = 0;
    std::vector<float> A;        // [n_in, rank]  (ggml ne order: n_in fastest)
    std::vector<float> B;        // [rank, n_out]
};

inline bool write_lora_gguf(const std::string & path, const std::string & arch, float alpha,
                            const std::vector<LoraTensorPair> & ad, std::string & err) {
    size_t bytes = 0;
    for (const auto & p : ad) bytes += (p.A.size() + p.B.size()) * sizeof(float) + 2 * ggml_tensor_overhead();
    ggml_init_params ip = { bytes + 1024 * 1024, nullptr, false };
    ggml_context * c = ggml_init(ip);
    if (!c) { err = "ggml_init failed"; return false; }
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "adapter");
    gguf_set_val_str(g, "general.architecture", arch.c_str());
    gguf_set_val_str(g, "general.name", "xolotl-lora");
    gguf_set_val_str(g, "adapter.type", "lora");
    gguf_set_val_f32(g, "adapter.lora.alpha", alpha);
    for (const auto & p : ad) {
        if ((int64_t) p.A.size() != p.n_in * p.rank || (int64_t) p.B.size() != p.rank * p.n_out) {
            err = "bad adapter shape for " + p.name;
            gguf_free(g); ggml_free(c);
            return false;
        }
        ggml_tensor * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.n_in, p.rank);
        ggml_tensor * b = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.rank, p.n_out);
        ggml_set_name(a, (p.name + ".lora_a").c_str());
        ggml_set_name(b, (p.name + ".lora_b").c_str());
        std::memcpy(a->data, p.A.data(), p.A.size() * sizeof(float));
        std::memcpy(b->data, p.B.data(), p.B.size() * sizeof(float));
        gguf_add_tensor(g, a);
        gguf_add_tensor(g, b);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    if (!ok) err = "gguf_write_to_file failed: " + path;
    gguf_free(g);
    ggml_free(c);
    return ok;
}

// Reads a llama.cpp-format LoRA GGUF (F32 or F16 lora_a/lora_b). Returns pairs in file order.
inline bool read_lora_gguf(const std::string & path, std::vector<LoraTensorPair> & out, float & alpha,
                           std::string & arch, std::string & err) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { false, &c };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { err = "cannot read GGUF " + path; return false; }
    auto str = [&](const char * k) {
        const int64_t id = gguf_find_key(g, k);
        return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING ? std::string(gguf_get_val_str(g, id)) : std::string();
    };
    bool ok = true;
    if (str("general.type") != "adapter" || str("adapter.type") != "lora") { err = path + " is not a LoRA adapter GGUF"; ok = false; }
    arch = str("general.architecture");
    const int64_t ia = gguf_find_key(g, "adapter.lora.alpha");
    alpha = ia >= 0 ? gguf_get_val_f32(g, ia) : 0.0f;
    auto fetch = [&](ggml_tensor * t, std::vector<float> & v) {
        v.resize((size_t) ggml_nelements(t));
        if (t->type == GGML_TYPE_F32) std::memcpy(v.data(), t->data, v.size() * sizeof(float));
        else if (t->type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, v.data(), (int64_t) v.size());
        else return false;
        return true;
    };
    for (int64_t i = 0; ok && i < gguf_get_n_tensors(g); ++i) {
        std::string nm = gguf_get_tensor_name(g, i);
        const std::string sa = ".lora_a";
        if (nm.size() <= sa.size() || nm.compare(nm.size() - sa.size(), sa.size(), sa) != 0) continue;
        const std::string base = nm.substr(0, nm.size() - sa.size());
        ggml_tensor * a = ggml_get_tensor(c, nm.c_str());
        ggml_tensor * b = ggml_get_tensor(c, (base + ".lora_b").c_str());
        if (!a || !b || a->ne[1] != b->ne[0]) { err = "unpaired or mis-shaped adapter " + base; ok = false; break; }
        LoraTensorPair p;
        p.name = base;
        p.n_in = a->ne[0]; p.rank = a->ne[1]; p.n_out = b->ne[1];
        if (!fetch(a, p.A) || !fetch(b, p.B)) { err = "adapter " + base + ": only F32/F16 supported"; ok = false; break; }
        out.push_back(std::move(p));
    }
    gguf_free(g);
    if (c) ggml_free(c);
    return ok;
}

}  // namespace xolotl::train
