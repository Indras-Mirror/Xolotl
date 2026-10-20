#include "strata/platform/cuda_cc.hpp"

#include <cstdio>

int main() {
    int checks = 0;
    int failures = 0;
    auto check = [&](bool ok, const char* msg) {
        ++checks;
        if (!ok) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s\n", msg);
        }
    };
    check(strata::cuda_cc_encode(7, 0) == 70, "sm_70 encodes to 70");
    check(strata::cuda_cc_below_mmq_tensor_cores(70), "sm_70 is below MMQ tensor-core path");
    check(!strata::cuda_cc_below_mmq_tensor_cores(75), "sm_75 is not below MMQ gate");
    check(!strata::cuda_cc_below_mmq_tensor_cores(89), "sm_89 is not below MMQ gate");
    check(strata::cuda_min_cc_encoded({89, 70}) == 70, "min cc across split pair");
    check(!strata::cuda_cc_below_mmq_tensor_cores(strata::cuda_min_cc_encoded({89, 86})), "no Volta in list");
    if (failures == 0) std::printf("test_cuda_cc_policy: %d checks passed\n", checks);
    return failures == 0 ? 0 : 1;
}
