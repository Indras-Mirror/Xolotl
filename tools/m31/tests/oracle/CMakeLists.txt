# Separate build: never link llama.cpp's ggml and Xolotl's ggml into the same process.
cmake_minimum_required(VERSION 3.24)
project(m31_oracle LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(M31_LLAMA_SOURCE "" CACHE PATH "PR #31192 source (read only)")
set(M31_LLAMA_BUILD "" CACHE PATH "Completed PR #31192 build (read only)")
foreach(_header include/llama.h ggml/include/ggml.h)
  if(NOT EXISTS "${M31_LLAMA_SOURCE}/${_header}")
    message(FATAL_ERROR "Missing ${M31_LLAMA_SOURCE}/${_header}")
  endif()
endforeach()
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${M31_LLAMA_SOURCE}"
  OUTPUT_VARIABLE _m31_pr_sha OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT _m31_pr_sha STREQUAL "6b95a9908d2c7b242817a447708aec38e937e2c0")
  message(FATAL_ERROR "Oracle source is not the audited PR SHA: ${_m31_pr_sha}")
endif()
foreach(_name llama ggml ggml-cpu ggml-base)
  string(REPLACE "-" "_" _key "${_name}")
  find_library(M31_${_key}_LIBRARY NAMES ${_name}
    PATHS "${M31_LLAMA_BUILD}/src" "${M31_LLAMA_BUILD}/ggml/src"
          "${M31_LLAMA_BUILD}/ggml/src/ggml-cpu" "${M31_LLAMA_BUILD}/bin"
    NO_DEFAULT_PATH REQUIRED)
  list(APPEND _m31_oracle_libs "${M31_${_key}_LIBRARY}")
endforeach()
add_executable(m31_oracle m31_oracle.cpp)
add_library(m31_ref INTERFACE)
target_include_directories(m31_oracle PRIVATE .. "${M31_LLAMA_SOURCE}/include" "${M31_LLAMA_SOURCE}/ggml/include"
  "${M31_LLAMA_SOURCE}/src")
target_link_libraries(m31_ref INTERFACE ${_m31_oracle_libs})
target_link_libraries(m31_oracle PRIVATE m31_ref)
add_executable(m31_tokenizer m31_tokenizer.cpp)
target_include_directories(m31_tokenizer PRIVATE "${M31_LLAMA_SOURCE}/include" "${M31_LLAMA_SOURCE}/ggml/include"
  "${M31_LLAMA_SOURCE}/common" "${M31_LLAMA_SOURCE}/vendor" "${M31_LLAMA_SOURCE}/vendor/nlohmann")
find_library(M31_COMMON_LIBRARY NAMES llama-common PATHS "${M31_LLAMA_BUILD}/common" NO_DEFAULT_PATH REQUIRED)
find_library(M31_COMMON_BASE_LIBRARY NAMES llama-common-base PATHS "${M31_LLAMA_BUILD}/common" NO_DEFAULT_PATH REQUIRED)
find_library(M31_VENDOR_HASH_LIBRARY NAMES vendor-hash PATHS "${M31_LLAMA_BUILD}/vendor/hash" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(m31_tokenizer PRIVATE ${M31_COMMON_LIBRARY} ${M31_COMMON_BASE_LIBRARY} m31_ref ${M31_VENDOR_HASH_LIBRARY})
if(M31_llama_LIBRARY MATCHES "\\.a$")
  # Match llama-bench dependency order for static reference archives.
  find_package(Threads REQUIRED)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(m31_ref INTERFACE OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS} m rt)
  file(STRINGS "${M31_LLAMA_BUILD}/CMakeCache.txt" _m31_ref_cuda REGEX "^GGML_CUDA:BOOL=ON$")
  if(_m31_ref_cuda)
    find_package(CUDAToolkit REQUIRED)
    find_library(M31_ggml_cuda_LIBRARY NAMES ggml-cuda
      PATHS "${M31_LLAMA_BUILD}/ggml/src/ggml-cuda" NO_DEFAULT_PATH REQUIRED)
    # Repeating ggml-base after CUDA also resolves its references from the CUDA archive.
    target_link_libraries(m31_ref INTERFACE ${M31_ggml_cuda_LIBRARY} ${M31_ggml_base_LIBRARY}
      CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
    find_library(M31_culibos_LIBRARY NAMES culibos PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
    if(M31_culibos_LIBRARY)
      target_link_libraries(m31_ref INTERFACE ${M31_culibos_LIBRARY})
    endif()
  endif()
endif()

# Authoritative live-history cache check: remove KV through the public PR API,
# preserving the old prompt's actual computation before replaying a suffix.
add_executable(m31_cache_oracle m31_cache_oracle.cpp)
target_include_directories(m31_cache_oracle PRIVATE "${M31_LLAMA_SOURCE}/include"
  "${M31_LLAMA_SOURCE}/ggml/include" "${M31_LLAMA_SOURCE}/src"
  "${M31_LLAMA_SOURCE}/vendor" "${M31_LLAMA_SOURCE}/vendor/nlohmann")
target_compile_definitions(m31_cache_oracle PRIVATE M31_CACHE_ORACLE_REFERENCE_SHA="${_m31_pr_sha}")
target_link_libraries(m31_cache_oracle PRIVATE m31_ref)

# This executable consumes prepared mtmd F32 rows; no mtmd ABI enters Xolotl.
option(M31_BUILD_VISION_ORACLE "Build the separate mixed-input vision oracle" OFF)
if(M31_BUILD_VISION_ORACLE)
  add_executable(m31_vision_oracle m31_vision_oracle.cpp)
  target_include_directories(m31_vision_oracle PRIVATE "${M31_LLAMA_SOURCE}/include"
    "${M31_LLAMA_SOURCE}/ggml/include" "${M31_LLAMA_SOURCE}/src"
    "${M31_LLAMA_SOURCE}/vendor/nlohmann")
  target_link_libraries(m31_vision_oracle PRIVATE m31_ref)
endif()
