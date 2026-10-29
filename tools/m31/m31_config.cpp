#include "m31_config.hpp"
#include "cpu/row_schedule.hpp"
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace strata::m31 {
namespace {
void require(bool ok, const char * why) { if (!ok) throw std::invalid_argument(std::string("M31 configuration: ") + why); }
std::string mmq_precision_mode() {
    const char * value = std::getenv("GGML_CUDA_MMQ_PREC");
    std::string mode = value ? value : "auto";
    for (char & c : mode) c = char(std::tolower(static_cast<unsigned char>(c)));
    require(mode == "auto" || mode == "q4" || mode == "q8",
            "GGML_CUDA_MMQ_PREC must be auto, q4 or q8; a precision change requires a new numerical gate");
    return mode;
}
void reject_legacy_environment() {
    // Known research switches must not silently select a different numeric path.
    static const char * const names[] = {
        "M31_ARENA_THP",
        "M31_ARENA_THP_FORCE",
        "M31_CACHE_CANDIDATE_BUILD",
        "M31_COMMON_SMOKE_AUDIT",
        "M31_COMMON_SMOKE_TREE",
        "M31_COMPONENT_GU",
        "M31_COMPONENT_TENSOR",
        "M31_CPU_AFFINITY",
        "M31_CPU_BATCH_PAIR",
        "M31_CPU_BENCH_EXCLUSIVE",
        "M31_CPU_DIRECT",
        "M31_CPU_FUSED",
        "M31_CPU_FUSED_PAIRED_QUANT",
        "M31_CPU_POLL",
        "M31_CPU_SHARED_OVERLAP",
        "M31_CPU_SHARED_OVERLAP_FAIL_AFTER_LAUNCH",
        "M31_CPU_SHARED_OVERLAP_TRACE",
        "M31_DECODE_FUSION_UNFENCED",
        "M31_DENSE_ASYNC_READS",
        "M31_DENSE_IO_TRACE",
        "M31_DENSE_MSA_VERSION",
        "M31_DENSE_OPS_VERSION",
        "M31_DENSE_PINNED_IO",
        "M31_DSPARK_BATCH_DENSE",
        "M31_DSPARK_BATCH_MSA",
        "M31_DSPARK_BATCH_OPS",
        "M31_DSPARK_CAPTURE_REUSE",
        "M31_DSPARK_CAPTURE_VIEW",
        "M31_DSPARK_COALESCE_INPUTS",
        "M31_DSPARK_GPU_CACHED",
        "M31_DSPARK_PACKED_ROUTED",
        "M31_DSPARK_PERSISTENT_VERIFY",
        "M31_DSPARK_SHARED_INPUTS",
        "M31_DSPARK_SHARED_OVERLAP",
        "M31_DSPARK_VIEW_INPUTS",
        "M31_GPU_AUDIT_WEIGHTS",
        "M31_GPU_CACHED_DECODE",
        "M31_GPU_CACHED_DECODE_EXACT",
        "M31_GPU_CACHED_DECODE_NATIVE",
        "M31_GPU_CACHED_GATE_UP",
        "M31_GPU_CACHED_GROUPED",
        "M31_GPU_CACHED_IQ3_IMMA",
        "M31_GPU_CACHED_RANK_OUTER",
        "M31_GPU_CAUSAL_FILL",
        "M31_GPU_CPU_QUANTS",
        "M31_GPU_CPU_QUANTS_BATCHED",
        "M31_GPU_CPU_QUANTS_HOST_THREADS",
        "M31_GPU_CPU_QUANTS_PINNED",
        "M31_GPU_CPU_QUANTS_PROFILE",
        "M31_GPU_CPU_QUANTS_STREAM",
        "M31_GPU_CPU_QUANTS_TENSOR",
        "M31_GPU_CPU_QUANTS_TIMING",
        "M31_GPU_DECODE_CHAIN",
        "M31_GPU_DECODE_FUSION",
        "M31_GPU_DECODE_SEED_HOT",
        "M31_GPU_EXACT",
        "M31_GPU_EXPERT_PROFILE",
        "M31_GPU_EXPERT_SORT_ROUTES",
        "M31_GPU_FAST_FFN",
        "M31_GPU_FAST_FFN_COMPENSATED",
        "M31_GPU_FAST_FFN_DEBUG_DIR",
        "M31_GPU_FAST_FFN_DOTS",
        "M31_GPU_FAST_FFN_EXPERT_BATCH",
        "M31_GPU_FAST_FFN_EXP_DOUBLE",
        "M31_GPU_FAST_FFN_FUSED_REDUCE",
        "M31_GPU_FAST_FFN_LIBM_EXPF",
        "M31_GPU_FAST_FFN_MMA256",
        "M31_GPU_FAST_FFN_MMA32",
        "M31_GPU_FAST_FFN_RANK_PHASES",
        "M31_GPU_FAST_FFN_VALIDATE_INPUT",
        "M31_GPU_HOST_GRAIN",
        "M31_GPU_HOST_MIN_ROWS",
        "M31_GPU_HOST_THREADS",
        "M31_GPU_LAYER_BANKS",
        "M31_GPU_NATIVE_GATHER",
        "M31_GPU_NATIVE_GU_CONCAT",
        "M31_GPU_PACK_THREADS",
        "M31_GPU_PARALLEL_REDUCE",
        "M31_GPU_PREFETCH_HISTORY",
        "M31_GPU_PREFETCH_LIMIT",
        "M31_GPU_PREFETCH_NEXT",
        "M31_GPU_PREFETCH_RANK",
        "M31_GPU_PREFILL_ASYNC",
        "M31_GPU_PREFILL_CHAIN",
        "M31_GPU_PREFILL_COMPACT",
        "M31_GPU_PREFILL_TEST_FAIL_AFTER",
        "M31_GPU_PROJECTION_CONCAT",
        "M31_GPU_QUANTS_DEBUG_DIR",
        "M31_GPU_QUANTS_DEBUG_LAYER",
        "M31_GPU_REGISTER_ARENA",
        "M31_GPU_REGISTER_CHUNK_MIB",
        "M31_GPU_REGISTER_LIMIT_GIB",
        "M31_GPU_REGISTER_THREADS",
        "M31_GPU_REUSE_BUFFERS",
        "M31_GPU_RMS_FUSION",
        "M31_GPU_SEED_HOT",
        "M31_GPU_SEED_IDENTITY",
        "M31_GPU_SEED_PATH",
        "M31_GPU_SEED_RECEIPT",
        "M31_GPU_SEED_RECEIPT_SHA256",
        "M31_GPU_SEED_SHA256",
        "M31_GPU_SORT_ROUTES",
        "M31_GPU_UPLOAD_DEPTH",
        "M31_GPU_UPLOAD_STREAMS",
        "M31_MMV_API_VERSION",
        "M31_MSA_CAUSAL_F16",
        "M31_MSA_INDEX_NVIS",
        "M31_MSA_INDEX_TILE",
        "M31_MSA_QUERY_TILE",
        "M31_MSA_SORTED_LOOKUP",
        "M31_MSA_SPARSE_FA",
        "M31_MSA_TOPK_MASK",
        "M31_ORACLE_CAPACITY_EQUIVALENCE",
        "M31_ORACLE_CHUNK_PLAN",
        "M31_PREFILL_ADAPTIVE",
        "M31_PROMPT_CACHE_HASH_PORTABLE",
        "M31_PROMPT_CACHE_MIB",
        "M31_REUSE_HOST_READS",
        "M31_RMS_BACKEND_VERSION",
        "M31_ROUTE_SPEED_WINDOW",
        "M31_SAFE_FUSION",
        "M31_SHARED_OVERLAP_NVTX",
        "M31_SHARED_OVERLAP_TESTING",
        "M31_SHORT_CPU_GBPS",
        "M31_SHORT_CPU_GOPS",
        "M31_SHORT_CPU_UNTILED_FACTOR",
        "M31_SHORT_GPU_GOPS",
        "M31_SHORT_PCIE_GBPS",
        "M31_SHORT_PREFILL_COMPARE_DIR",
        "M31_SHORT_PREFILL_COMPARE_LAYERS",
        "M31_SHORT_PREFILL_CPU_MAX_N",
        "M31_SHORT_PREFILL_DEBUG_HIDDEN",
        "M31_SHORT_PREFILL_DEVICE_GATHER",
        "M31_SHORT_PREFILL_GPU_OAI",
        "M31_SHORT_PREFILL_GPU_QUANT",
        "M31_SHORT_PREFILL_MATH",
        "M31_SHORT_PREFILL_MAX_N",
        "M31_SHORT_PREFILL_MODE",
        "M31_SHORT_PREFILL_NVFP4_CPU",
        "M31_SHORT_PREFILL_PREPARED_CORRECTIONS",
        "M31_SHORT_PREFILL_PROFILE",
        "M31_SHORT_PREFILL_SCHEDULE",
        "M31_SHORT_PREFILL_SCHEDULE_CHUNK",
        "M31_SHORT_PREFILL_TASK_PROFILE",
        "M31_SHORT_PREFILL_UPLOAD_WINDOW",
        "M31_TTFT_TRACE",
    };
    for (const char * name : names) {
        const char * value = std::getenv(name);
        if (value && *value && std::string(value) != "0")
            throw std::invalid_argument(std::string("M31 configuration: removed environment option ") + name +
                "; unset it and use the documented CLI resource options");
    }
}
}
void validate_config(const Config & c, bool initializing_cuda) {
    require(c.threads > 0 && c.threads <= 512, "threads must be 1..512");
    require(c.context > 0 && c.context <= INT32_MAX, "context must be 1..INT32_MAX");
    require(c.max_chunk > 0 && c.max_chunk <= 32768, "chunk must be 1..32768");
    require(std::isfinite(c.arena_gib) && c.arena_gib >= 0 && c.arena_gib <= 4096, "invalid RAM arena budget");
    require(std::isfinite(c.gpu_experts_gib) && c.gpu_experts_gib >= 0 && c.gpu_experts_gib <= 512, "invalid GPU expert budget");
    require(c.gpu_experts_min_tokens == 32, "the accepted prefill policy requires gpu-experts-min-tokens=32");
    require(c.reuse_allocations && c.persistent_cpu_pool && c.cpu_residual && c.cpu_decode_cache && c.dense_decode_cache,
        "the accepted path requires allocation reuse, persistent CPU pool, CPU residual and decode caches");
    require(c.cpu_poll >= -1 && c.cpu_poll <= 100, "CPU poll must be -1 or 0..100");
    require(c.gpu_experts_gib == 0 || c.backend || initializing_cuda, "GPU experts require an explicit GPU backend");
    require(c.gpu_pack_threads > 0 && c.gpu_pack_threads <= 16, "pack threads must be 1..16");
    require(c.upload_depth > 0 && c.upload_depth <= 16 && c.upload_streams > 0 && c.upload_streams <= 4,
        "upload depth must be 1..16 and copy streams 1..4");
    require(c.prefetch_limit <= 128, "prefetch limit must be 0..128");
    require(!c.register_arena || ((c.backend || initializing_cuda) && c.register_limit_bytes > 0),
        "arena registration requires an explicit pin budget and GPU backend");
    require(c.register_threads >= 1 && c.register_threads <= 64, "registration threads must be 1..64");
    require(c.register_chunk_bytes >= 4096 && c.register_chunk_bytes <= (uint64_t(16) << 30), "invalid registration chunk budget");
#ifndef M31_REGISTERED_ARENA_CUDA
    require(!c.register_arena, "arena registration is unavailable in this build");
#endif
    (void)mmq_precision_mode();
    (void)RowSchedule::from_environment();
    reject_legacy_environment();
}
void prepare_engine_environment() {
    const char * fusion = std::getenv("GGML_CUDA_DISABLE_FUSION");
    require(!fusion || std::string(fusion) == "1", "native M31 requires GGML_CUDA_DISABLE_FUSION=1; fusion has not passed the numeric gate");
#ifdef _WIN32
    require(_putenv_s("GGML_CUDA_DISABLE_FUSION", "1") == 0, "cannot set CUDA fusion guard");
#else
    require(setenv("GGML_CUDA_DISABLE_FUSION", "1", 1) == 0, "cannot set CUDA fusion guard");
#endif
    (void)mmq_precision_mode();
    (void)RowSchedule::from_environment();
    reject_legacy_environment();
}
std::string effective_config(const Config & c) {
    std::ostringstream s;
    s << "native batch/adaptive short chunks; CPU fused=" << c.cpu_fused << "; context=" << c.context << " chunk=" << c.max_chunk
      << " threads=" << c.threads << " arena_gib=" << c.arena_gib << " expert_gib=" << c.gpu_experts_gib
      << " registered=" << c.register_arena << " pin_budget=" << c.register_limit_bytes
      << " upload_depth=" << c.upload_depth << " copy_streams=" << c.upload_streams
      << " pack_threads=" << c.gpu_pack_threads << " fusion=off mmq_precision=";
    s << mmq_precision_mode();
    return s.str();
}
} // namespace strata::m31
