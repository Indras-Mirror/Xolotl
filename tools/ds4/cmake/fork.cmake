# tools/ds4/cmake/fork.cmake - every Strata-DS4 fork target, kept out of upstream's CMakeLists.txt (which has one
# include() line for this file at its end; docs/UPSTREAM_SYNC.md).  Included in the root directory scope, so the
# root's variables (_strata_gpu_include_directories, _strata_gpu_runtime_target) and targets are visible.

if(STRATA_NATIVE_EXPERTS AND TARGET strata_kernels_cpu)
  # PTQ1_0 ternary expert rows (DeepSeek-V4.1 mixed tier); native_expert.cpp calls them (function-level AVX2
  # attributes, no per-file flags)
  target_sources(strata_kernels_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/ptq1_0.cpp)
  if(STRATA_BUILD_TESTS)   # layout vs llama-tq's reference, dot parity
    add_executable(ptq1_0_test tests/cpu/ptq1_0_test.cpp)
    target_link_libraries(ptq1_0_test PRIVATE strata_kernels_cpu)
    add_test(NAME ptq1_0_test COMMAND ptq1_0_test)
  endif()
endif()

if(STRATA_BUILD_TESTS AND TARGET strata_artifact AND NOT TARGET ds4_geometry_test)
  # DSV4 geometry + full tensor map (header mmap only; the real-file part skips without $DS4_GGUF/argv[1])
  add_executable(ds4_geometry_test tests/core/ds4_geometry_test.cpp)
  target_link_libraries(ds4_geometry_test PRIVATE strata_artifact)
  add_test(NAME ds4_geometry_test COMMAND ds4_geometry_test $ENV{DS4_GGUF})
endif()

if(STRATA_NATIVE_EXPERTS AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP) AND NOT STRATA_HIP_GFX906)
  # DS4 phase 2 gate: one real DeepSeek-V4 expert (gate/up IQ2_XXS, down Q2_K) through the CPU pool and the
  # GPU native MMVQ path, against ggml's dequantizer + an F32 matmul.  The 0731 artifact comes from
  # argv[1]/$DS4_GGUF; without it the binary prints "skipped" and exits 0.  --cpu-only runs no CUDA.
  add_executable(ds4_expert_parity src/kernels/ds4_expert_parity.cpp)
  target_include_directories(ds4_expert_parity PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ds4_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
  if(STRATA_BUILD_TESTS)
    add_test(NAME ds4_expert_parity COMMAND ds4_expert_parity "$ENV{DS4_GGUF}")
  endif()
  # DS4 phase 4a: the DeepSeek-V4 expert engine replayed on recorded routes (tools/ds4/moe_replay.cpp).
  # It links the real machinery - strata_engine's ExpertCache, the CPU ExpertPool, the native grouped
  # kernels, PinnedArena - and drives them from ds4routes.bin.  --correctness --cpu-only needs no GPU.
  add_executable(ds4_moe_replay tools/ds4/moe_replay.cpp tools/ds4/moe_gap.cu)
  target_include_directories(ds4_moe_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_strata_gpu_include_directories})
  target_link_libraries(ds4_moe_replay PRIVATE strata_engine strata_kernels strata_kernels_cpu strata_artifact
    ${_strata_gpu_runtime_target})
endif()

if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND TARGET strata_kernels)
  # PTQ1_0 ternary on the GPU (iq_mmvq / iq_dequant_f16, type 143) against the CPU kernels; skips without a device
  add_executable(ptq1_0_cuda_test tests/cuda/ptq1_0_cuda_test.cpp)
  target_include_directories(ptq1_0_cuda_test PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(ptq1_0_cuda_test PRIVATE strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  add_test(NAME ptq1_0_cuda_test COMMAND ptq1_0_cuda_test)
endif()

# ---- ds4 reference forward pass (Phase 3) ---------------------------------
# tools/ds4 is not part of the engine, but ds4_ref is the Phase-3 correctness oracle: a plain CPU ggml
# graph that emits golden_dump-format activations for tools/ds4/compare_golden.py.  CPU-only, no CUDA.
if(TARGET ggml-cpu)
  add_executable(ds4_ref tools/ds4/ds4_ref.cpp)
  target_include_directories(ds4_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ds4_ref PRIVATE ggml ggml-cpu ggml-base)
endif()

# ---- ds4 decode engine (Phase 4b proper) ---------------------------------
# Each engine slice owns exactly one of these files (conductor plan 2026-10-06); OPTIONAL so a missing one is fine.
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_dense.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_moe.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/ds4/cmake/ds4_engine.cmake OPTIONAL)

# DeepSeek-V4.1 engine (tools/v41/engine)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine/v41_engine.cmake OPTIONAL)
