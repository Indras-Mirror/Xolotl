// MiniMax-M3.1 short prefill. Derived from llama.cpp PR #31192 at 6b95a990 (MIT).
#include "prefill/short/shortprefill_cuda_math_cpu.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace strata::m31::cuda_math_cpu;
namespace {
void check(bool value, const char * why) { if (!value) throw std::runtime_error(why); }
std::vector<int32_t> corrections_for(const std::vector<Q8Mmq32> & input) {
    std::vector<int32_t> result(input.size());
    for(size_t group=0;group<input.size();++group) {
        int32_t sum=0;for(const auto value:input[group].qs)sum+=value;
        result[group]=128*sum;
    }
    return result;
}

}
int main() {
    try {
        std::vector<float> activation(256);
        activation[0] = 127.f; activation[1] = -127.f;
        activation[2] = .5f; activation[3] = -.5f;
        std::vector<Q8Mmq32> quant;
        quantize_mmq_d4(activation.data(), activation.size(), quant);
        check(quant.size() == 8 && quant[0].d == 1.f && quant[0].qs[0] == 127 &&
              quant[0].qs[1] == -127 && quant[0].qs[2] == 1 && quant[0].qs[3] == -1,
              "CUDA roundf boundary differs");
        for (size_t group = 1; group < quant.size(); ++group) {
            check(quant[group].d == 0, "zero scale");
            for (auto value : quant[group].qs) check(value == 0, "zero quant");
        }
        std::mt19937 random{7123};
        std::uniform_real_distribution<float> distribution(-.6f, .6f);
        for (size_t sample = 0; sample < 1024; ++sample) {
            int8_t weights[32], activations[32];
            int32_t expected = 0;
            for (size_t k = 0; k < 32; ++k) {
                weights[k] = int8_t(random() & 255);
                activations[k] = int8_t(random() & 255);
                if (k == 0) { weights[k] = -128; activations[k] = -128; }
                if (k == 1) { weights[k] = -127; activations[k] = -128; }
                expected += int32_t(weights[k]) * int32_t(activations[k]);
            }
            check(integer_dot32(weights, activations) == expected, "signed int8 boundary dot differs");
        }
        size_t checked_values = 0;
        for (ggml_type type : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS}) {
            constexpr size_t columns = 768, rows = 16;
            std::vector<float> original(rows * columns), importance(columns, 1.f), dequantized(columns);
            for (auto & value : original) value = distribution(random);
            std::vector<uint8_t> weights(rows * ggml_row_size(type, columns));
            check(ggml_quantize_chunk(type, original.data(), weights.data(), 0, rows, columns, importance.data()) == weights.size(),
                  "quantized weight size");
            for (auto & value : activation) value = distribution(random);
            activation.resize(columns);
            for (auto & value : activation) value = distribution(random);
            quantize_mmq_d4(activation.data(), columns, quant);
            for (size_t row = 0; row < rows; ++row) {
                const void * source = weights.data() + row * ggml_row_size(type, columns);
                ggml_get_type_traits(type)->to_float(source, dequantized.data(), columns);
                float manual = 0;
                for (size_t group = 0; group < columns / 32; ++group) {
                    int8_t values[32]; float scale;
                    unpack_weight32(type, source, group, values, scale);
                    int32_t integer_sum = 0;
                    for (size_t k = 0; k < 32; ++k) {
                        const float reconstructed = float(values[k]) * scale;
                        check(std::memcmp(&reconstructed, &dequantized[group * 32 + k], sizeof(float)) == 0,
                              "CUDA loader integer/scale differs from original weight dequantization");
                        integer_sum += int32_t(values[k]) * quant[group].qs[k];
                        ++checked_values;
                    }
                    const float partial = float(integer_sum) * scale;
                    manual = std::fma(partial, quant[group].d, manual);
                }
                const float actual = dot_mmq(type, source, columns, quant.data());
                check(std::memcmp(&actual, &manual, sizeof(float)) == 0, "integer/fma scalar reference differs");
            }
            std::vector<float> first, second;
            mul_mat_rows(type, weights.data(), ggml_row_size(type, columns), rows, columns,
                         activation.data(), 1, first);
            mul_mat_rows(type, weights.data(), ggml_row_size(type, columns), rows, columns,
                         activation.data(), 1, second);
            check(first == second && first.size() == rows, "matrix reference not deterministic");
            std::array<const Q8Mmq32 *, 16> activations;
            activations.fill(quant.data());
            const auto correction=corrections_for(quant);
            std::array<const int32_t *,16> correction_rows;correction_rows.fill(correction.data());
            for (size_t count = 1; count <= activations.size(); ++count) {
                constexpr float sentinel = 123.75f;
                std::vector<float> tiled((rows + 3) * count + 3, sentinel), avx2(tiled),legacy(tiled),prepared(tiled);
                mul_mat_quantized_tile(type, weights.data(), ggml_row_size(type, columns), rows,
                    columns, activations.data(), count, tiled.data() + 1, rows + 3);
                mul_mat_quantized_tile_avx2_reference(type, weights.data(), ggml_row_size(type, columns), rows,
                    columns, activations.data(), count, avx2.data() + 1, rows + 3);
                mul_mat_quantized_tile_vnni_reference(type,weights.data(),ggml_row_size(type,columns),rows,
                    columns,activations.data(),count,legacy.data()+1,rows+3);
                mul_mat_quantized_tile(type,weights.data(),ggml_row_size(type,columns),rows,
                    columns,activations.data(),count,prepared.data()+1,rows+3,correction_rows.data());
                check(!std::memcmp(tiled.data(), avx2.data(), tiled.size() * sizeof(float)),
                      "VNNI/AVX2 tile bits or masked stores differ");
                check(!std::memcmp(tiled.data(),legacy.data(),tiled.size()*sizeof(float)) &&
                      !std::memcmp(tiled.data(),prepared.data(),tiled.size()*sizeof(float)),
                      "transpose/sidecar changed legacy VNNI bits or output padding");
                for (size_t token = 0; token < count; ++token) {
                    check(!std::memcmp(first.data(), tiled.data() + 1 + token * (rows + 3), rows * sizeof(float)),
                          "SIMD tile changed scalar integer/FMA math");
                    for (size_t pad = rows; pad < rows + 3; ++pad)
                        check(tiled[1 + token * (rows + 3) + pad] == sentinel, "SIMD wrote into padding");
                }
                check(tiled.front() == sentinel && tiled.back() == sentinel, "SIMD wrote outside output");
            }
            // Executor tiles use an uninitialized 16x16 scratch array and read
            // only active cells. Poison every cell to prove complete writes
            // for every row/token tail, with guards around the entire tile.
            uint32_t poison_bits=0x7fc00dadu;float poison;
            std::memcpy(&poison,&poison_bits,sizeof(poison));
            for(size_t active_rows=1;active_rows<=16;++active_rows)for(size_t count=1;count<=16;++count){
                constexpr size_t stride=16;
                std::vector<float> plain(stride*count+2,poison),prepared(plain);
                mul_mat_quantized_tile(type,weights.data(),ggml_row_size(type,columns),active_rows,
                    columns,activations.data(),count,plain.data()+1,stride);
                mul_mat_quantized_tile(type,weights.data(),ggml_row_size(type,columns),active_rows,
                    columns,activations.data(),count,prepared.data()+1,stride,correction_rows.data());
                check(!std::memcmp(plain.data(),prepared.data(),plain.size()*sizeof(float)),
                    "prepared tile differs for row/token tail");
                check(!std::memcmp(plain.data(),&poison,sizeof(float))&&
                    !std::memcmp(plain.data()+plain.size()-1,&poison,sizeof(float)),
                    "tile touched outer canary");
                for(size_t token=0;token<count;++token){
                    check(!std::memcmp(plain.data()+1+token*stride,first.data(),active_rows*sizeof(float)),
                        "tile left an active cell unwritten");
                    for(size_t row=active_rows;row<stride;++row)
                        check(!std::memcmp(plain.data()+1+token*stride+row,&poison,sizeof(float)),
                            "tile touched masked row canary");
                }
            }
            // An external prequantized caller can contain -128, although the
            // CUDA D4 quantizer never emits it. VNNI remains exact; AVX2's
            // reference entry takes a rare scalar fallback before its hot loop.
            auto extreme = quant;
            extreme[0].qs[0] = -128; extreme[0].qs[1] = 127;
            extreme[0].qs[2] = -128; extreme[0].qs[3] = -127;
            activations[3] = extreme.data();
            const auto extreme_correction=corrections_for(extreme);correction_rows[3]=extreme_correction.data();
            std::vector<float> vnni(rows * 16), avx2(rows * 16),prepared(rows*16);
            mul_mat_quantized_tile(type, weights.data(), ggml_row_size(type, columns), rows,
                columns, activations.data(), 16, vnni.data(), rows);
            mul_mat_quantized_tile_avx2_reference(type, weights.data(), ggml_row_size(type, columns), rows,
                columns, activations.data(), 16, avx2.data(), rows);
            mul_mat_quantized_tile(type,weights.data(),ggml_row_size(type,columns),rows,
                columns,activations.data(),16,prepared.data(),rows,correction_rows.data());
            check(!std::memcmp(vnni.data(), avx2.data(), vnni.size() * sizeof(float)),
                  "crafted -128 activation changed VNNI/reference math");
            check(!std::memcmp(vnni.data(),prepared.data(),vnni.size()*sizeof(float)),"crafted -128 correction changed bits");
            correction_rows[0]=nullptr;bool rejected=false;
            try{mul_mat_quantized_tile(type,weights.data(),ggml_row_size(type,columns),rows,
                columns,activations.data(),16,prepared.data(),rows,correction_rows.data());}
            catch(const std::invalid_argument &){rejected=true;}
            check(rejected,"null sidecar row accepted");
        }
        std::cout << "PASS CUDA MMQ CPU scalar math: decoded_values=" << checked_values
                  << " IQ3/IQ4 weight decode exact; Q8_D4 round/zero and ordered integer/FMA exact\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
}
