// xolp2p_selftest <model file> [offset GiB] [MiB] - the check an engine runs once at startup
#include "xolp2p.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model file> [offset GiB] [MiB]\n", argv[0]); return 1; }
    const unsigned long long off = (argc > 2 ? strtoull(argv[2], 0, 10) : 1) << 30;
    const size_t mb = argc > 3 ? strtoull(argv[3], 0, 10) : 64;
    cuInit(0); CUdevice d; cuDeviceGet(&d, 0); CUcontext c; CUctxCreateParams cp = {}; cuCtxCreate(&c, &cp, 0, d);
    std::string rep;
    const bool ok = xolp2p::selftest(argv[1], off, mb << 20, &rep);
    printf("%s: %s\n", ok ? "PASS" : "UNAVAILABLE/FAIL", rep.c_str());
    return ok ? 0 : 2;
}
