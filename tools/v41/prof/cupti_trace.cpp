// cupti_trace.dll - CUDA_INJECTION64_PATH tracer: every kernel and memcpy (GPU start/end ns, stream, name/bytes) to
// the CSV named by CUPTI_TRACE_OUT.  CUPTI is loaded at run time from CUPTI_DLL (the wheel has no import library).
#include <windows.h>
#include <cupti_activity.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>

static HMODULE g_cupti;
static FILE* g_out;
static std::mutex g_mu;
static decltype(&cuptiActivityGetNextRecord) p_next;
static decltype(&cuptiActivityFlushAll) p_flush;

static void CUPTIAPI buf_req(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 16u << 20;
    *buffer = (uint8_t*) _aligned_malloc(*size, 8);
    *max_records = 0;
}

static void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
    CUpti_Activity* r = nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    while (g_out && p_next(buffer, valid, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const CUpti_ActivityKernel13* k = (const CUpti_ActivityKernel13*) r;
            std::fprintf(g_out, "K,%llu,%llu,%u,%s\n", (unsigned long long) k->start, (unsigned long long) k->end, k->streamId,
                         k->name ? k->name : "?");
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            const CUpti_ActivityMemcpy7* m = (const CUpti_ActivityMemcpy7*) r;
            std::fprintf(g_out, "M,%llu,%llu,%u,%llu,%u\n", (unsigned long long) m->start, (unsigned long long) m->end, m->streamId,
                         (unsigned long long) m->bytes, (unsigned) m->copyKind);
        }
    }
    _aligned_free(buffer);
}

static void at_exit() {
    if (p_flush) p_flush(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_out) { std::fclose(g_out); g_out = nullptr; }
}

extern "C" __declspec(dllexport) int InitializeInjection(void) {
    const char* dll = std::getenv("CUPTI_DLL");
    const char* out = std::getenv("CUPTI_TRACE_OUT");
    if (!dll || !out) return 0;
    g_cupti = LoadLibraryA(dll);
    if (!g_cupti) { std::fprintf(stderr, "cupti_trace: cannot load %s\n", dll); return 0; }
    auto reg = (decltype(&cuptiActivityRegisterCallbacks)) GetProcAddress(g_cupti, "cuptiActivityRegisterCallbacks");
    auto en = (decltype(&cuptiActivityEnable)) GetProcAddress(g_cupti, "cuptiActivityEnable");
    p_next = (decltype(&cuptiActivityGetNextRecord)) GetProcAddress(g_cupti, "cuptiActivityGetNextRecord");
    p_flush = (decltype(&cuptiActivityFlushAll)) GetProcAddress(g_cupti, "cuptiActivityFlushAll");
    if (!reg || !en || !p_next || !p_flush) { std::fprintf(stderr, "cupti_trace: missing CUPTI entry points\n"); return 0; }
    g_out = std::fopen(out, "w");
    if (!g_out) return 0;
    std::setvbuf(g_out, nullptr, _IOFBF, 8u << 20);
    if (reg(buf_req, buf_done) != CUPTI_SUCCESS) { std::fprintf(stderr, "cupti_trace: register failed\n"); return 0; }
    const CUptiResult a = en(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), b = en(CUPTI_ACTIVITY_KIND_MEMCPY);
    std::fprintf(stderr, "cupti_trace: on (kernel %d, memcpy %d) -> %s\n", (int) a, (int) b, out);
    std::atexit(at_exit);
    return 1;
}
