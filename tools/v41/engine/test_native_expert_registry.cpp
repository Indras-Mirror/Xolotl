// CPU only: the very same macro expansions CUDA uses, plus block/role guards.
#include "strata/kernels/native_expert_registry.hpp"
#include <cstdio>
#include <cstdlib>
using namespace strata::kernels::registry;
int main(){
#define EMIT(T) std::printf("%d,",T);
 std::printf("GU=");STRATA_GU_FMTS(EMIT) STRATA_GU_FMTS_FORK(EMIT) std::puts("");
 std::printf("D=");STRATA_D_FMTS(EMIT) STRATA_D_FMTS_FORK(EMIT) std::puts("");
 std::printf("MMVQ=");STRATA_MMVQ_FMTS(EMIT) STRATA_MMVQ_FMTS_FORK(EMIT) std::puts("");
#undef EMIT
 auto require=[](bool b){if(!b)std::abort();};
 require(supported_triple(22,22,21,5120,2304));require(supported_triple(18,18,21,5120,2304));
 require(!supported_triple(22,22,21,5120,640));require(!supported_triple(22,22,21,5119,2304));
 require(!supported_triple(22,18,21,5120,2304));require(!supported_triple(-1,-1,21,5120,2304));
 require(!supported_triple(22,22,999,5120,2304));require(!supported_triple(22,22,21,0,2304));
 require(supported_triple(10,11,21,5120,2304));require(supported_triple(11,10,21,5120,2304));
#ifdef STRATA_Q6K_EXPERTS
 require(gate_block(14)==256 && down_block(14)==256);
#else
 require(gate_block(14)==0 && down_block(14)==0);
#endif
 std::puts("CPU registry gates PASS");return 0;
}
