#include "../../src/qwen35/q36_resident_verify.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using strata::q36::resident::Difference;
static void check(bool value){if(!value)throw std::runtime_error("resident byte comparator check failed");}
int main(){
 std::array<uint8_t,32> gpu{},ram{};for(size_t i=0;i<32;++i)gpu[i]=ram[i]=(uint8_t)i;
 Difference same;same.observe(100,gpu.data(),ram.data(),32,4);check(same.same()&&same.bytes==32&&same.gpu_hash==same.ram_hash);
 gpu[5]^=1;gpu[29]^=2;gpu[30]^=4;
 Difference bad;bad.observe(100,gpu.data(),ram.data(),16,4);bad.observe(104,gpu.data()+16,ram.data()+16,16,4);
 check(!bad.same()&&bad.different_bytes==3&&bad.different_cells==2&&bad.first_cell==101&&bad.first_byte_in_cell==1&&bad.last_cell==107);
 Difference whole;whole.observe(100,gpu.data(),ram.data(),32,4);check(whole.gpu_hash==bad.gpu_hash&&whole.ram_hash==bad.ram_hash&&whole.different_bytes==bad.different_bytes);
 bool refused=false;try{whole.observe(0,gpu.data(),ram.data(),3,4);}catch(const std::invalid_argument&){refused=true;}check(refused);
 refused=false;try{whole.observe(UINT64_MAX,gpu.data(),ram.data(),8,4);}catch(const std::invalid_argument&){refused=true;}check(refused);
 std::cout<<"resident KV byte comparator checks PASS\n";
}
