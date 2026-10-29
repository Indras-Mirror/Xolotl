# MiniMax-M3.1-Flash text engine. The default build is unchanged.
option(STRATA_M31 "Build the MiniMax-M3.1-Flash text engine" OFF)
if(NOT STRATA_M31)
  return()
endif()
if(STRATA_ENABLE_HIP OR STRATA_HIP_GFX906)
  message(FATAL_ERROR "STRATA_M31 supports CPU and CUDA builds; HIP has not been validated")
endif()
if(NOT TARGET ggml-cpu)
  message(FATAL_ERROR "STRATA_M31 requires STRATA_NATIVE_EXPERTS=ON (ggml-cpu)")
endif()

set(_m31_dir "${CMAKE_CURRENT_SOURCE_DIR}/tools/m31")
set(_m31_main "${_m31_dir}/m31_generate.cpp")
if(NOT EXISTS "${_m31_main}")
  message(STATUS "STRATA_M31=ON but ${_m31_main} does not exist; skipping the engine")
  return()
endif()
if(NOT STRATA_GGML_DIR OR NOT EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-impl.h")
  message(FATAL_ERROR "STRATA_M31 needs the pinned llama.cpp source used to build ggml")
endif()
if(STRATA_ENABLE_CUDA AND NOT TARGET ggml-cuda)
  message(FATAL_ERROR "STRATA_M31 CUDA requires -DSTRATA_GGML_CUDA=ON as well as -DSTRATA_ENABLE_CUDA=ON")
endif()
find_package(Threads REQUIRED)
set(_m31_ggml_src "${STRATA_GGML_DIR}/ggml/src")

# Keep the source list explicit: offline probes and optional helpers have their own main().
add_library(m31_engine STATIC
  "${_m31_dir}/m31_engine.cpp"
  "${_m31_dir}/m31_config.cpp"
  "${_m31_dir}/gpu/experts.cpp"
  "${_m31_dir}/gpu/host_executor.cpp")
target_include_directories(m31_engine PUBLIC "${_m31_dir}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(m31_engine PRIVATE "${_m31_ggml_src}" "${_m31_ggml_src}/ggml-cpu")
target_compile_features(m31_engine PUBLIC cxx_std_20)
target_link_libraries(m31_engine PUBLIC ggml ggml-cpu ggml-base Threads::Threads)
if(STRATA_ENABLE_CUDA AND TARGET ggml-cuda AND NOT STRATA_HIP_GFX906)
  target_sources(m31_engine PRIVATE "${_m31_dir}/gpu/registered_arena.cu")
  target_compile_definitions(m31_engine PUBLIC M31_ENGINE_CUDA=1 PRIVATE M31_REGISTERED_ARENA_CUDA=1)
  target_link_libraries(m31_engine PUBLIC ggml-cuda CUDA::cudart)
  set_target_properties(m31_engine PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

# Tokenization and constrained sampling use this same pinned ggml. No PR oracle is a runtime dependency.
include("${_m31_dir}/cmake/grammar.cmake")
add_executable(m31_generate "${_m31_main}")
target_include_directories(m31_generate PRIVATE "${_m31_ggml_src}")
target_link_libraries(m31_generate PRIVATE m31_engine m31_grammar)
message(STATUS "STRATA_M31: building the text engine and m31_generate --serve")

option(STRATA_M31_VISION "Enable M3.1 mixed prefill with an external mtmd helper" OFF)
if(STRATA_M31_VISION)
  target_compile_definitions(m31_engine PUBLIC M31_ENGINE_VISION=1)
  message(STATUS "STRATA_M31_VISION: mixed prefill enabled; build the optional mtmd helper separately")
endif()

option(STRATA_M31_TESTS "Build small model-free M31 CPU and protocol checks" ON)
if(STRATA_M31_TESTS)
  enable_testing()
  include("${_m31_dir}/tests/tests.cmake")
endif()
