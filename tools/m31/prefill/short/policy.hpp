// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#pragma once
namespace strata::m31::shortprefill {
enum class Path { decode, cpu, hybrid, batch };
constexpr Path select(int rows) {
    return rows <= 1 ? Path::decode : rows <= 31 ? Path::cpu :
        rows <= 1024 ? Path::hybrid : Path::batch;
}
} // namespace strata::m31::shortprefill
