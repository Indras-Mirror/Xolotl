// lora_bin_to_gguf <in.bin> <out.gguf> <arch> <alpha>
// Converts a raw adapter file from hy4_train (HY4_TRAIN_SAVE=*.bin: per adapter u32 name len, name, i64 n_in, n_out,
// rank, F32 A [n_in*rank], F32 B [rank*n_out]) into a llama.cpp-format LoRA GGUF, then reads it back and checks
// every value round-trips bit-exactly.
#include "lora_gguf.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s <in.bin> <out.gguf> <arch> <alpha>\n", argv[0]); return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<xolotl::train::LoraTensorPair> ad;
    for (uint32_t len; in.read((char *) &len, 4);) {
        xolotl::train::LoraTensorPair p;
        p.name.assign(len, '\0');
        in.read(p.name.data(), len);
        in.read((char *) &p.n_in, 8); in.read((char *) &p.n_out, 8); in.read((char *) &p.rank, 8);
        p.A.resize((size_t) (p.n_in * p.rank)); p.B.resize((size_t) (p.rank * p.n_out));
        in.read((char *) p.A.data(), (std::streamsize) (p.A.size() * 4));
        in.read((char *) p.B.data(), (std::streamsize) (p.B.size() * 4));
        if (!in) { std::fprintf(stderr, "truncated %s\n", argv[1]); return 1; }
        ad.push_back(std::move(p));
    }
    std::string err;
    const float alpha = (float) std::atof(argv[4]);
    if (!xolotl::train::write_lora_gguf(argv[2], argv[3], alpha, ad, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<xolotl::train::LoraTensorPair> back;
    float a2 = 0;
    std::string arch2;
    if (!xolotl::train::read_lora_gguf(argv[2], back, a2, arch2, err)) { std::fprintf(stderr, "read back: %s\n", err.c_str()); return 1; }
    bool same = back.size() == ad.size() && a2 == alpha && arch2 == argv[3];
    for (size_t i = 0; same && i < ad.size(); ++i)
        same = back[i].name == ad[i].name && back[i].n_in == ad[i].n_in && back[i].n_out == ad[i].n_out &&
               back[i].rank == ad[i].rank && back[i].A == ad[i].A && back[i].B == ad[i].B;
    std::printf("%zu adapters -> %s (arch %s, alpha %.3g): round trip %s\n", ad.size(), argv[2], argv[3], alpha,
                same ? "EXACT" : "MISMATCH");
    return same ? 0 : 1;
}
