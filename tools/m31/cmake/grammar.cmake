# Native tokenization and GBNF support from the SAME source tree as ggml.
# Shared by M31 and q36. This adds no vendor patches,
# process-wide compiler flags, external downloads, or llama.cpp server.
if(NOT STRATA_M31 AND NOT STRATA_Q36_GRAMMAR)
  return()
endif()

function(m31_add_grammar)
  if(TARGET m31_grammar)
    return()
  endif()
  set(_src "${STRATA_GGML_DIR}")
  if(NOT _src AND strata_llamacpp_SOURCE_DIR)
    set(_src "${strata_llamacpp_SOURCE_DIR}")
  endif()
  foreach(_required src/llama.cpp common/sampling.cpp common/reasoning-budget.cpp
                    vendor/nlohmann/json.hpp vendor/cpp-httplib/httplib.cpp)
    if(NOT EXISTS "${_src}/${_required}")
      message(FATAL_ERROR "M31 grammar needs the full llama.cpp source used for ggml; missing ${_src}/${_required}")
    endif()
  endforeach()
  find_package(Threads REQUIRED)

  # Lists follow the pinned src/common CMakeLists. Keep them local: including the
  # llama.cpp root would configure ggml again and alter unrelated build options.
  set(_core
    llama.cpp llama-adapter.cpp llama-arch.cpp llama-batch.cpp llama-chat.cpp
    llama-context.cpp llama-cparams.cpp llama-grammar.cpp llama-graph.cpp
    llama-hparams.cpp llama-impl.cpp llama-io.cpp llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp llama-kv-cache-dsa.cpp llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp llama-kv-cache-dsv4.cpp llama-memory.cpp
    llama-moe-cache.cpp llama-memory-hybrid.cpp llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp llama-memory-recurrent.cpp llama-mmap.cpp
    llama-model-loader.cpp llama-model-saver.cpp llama-model.cpp llama-quant.cpp
    llama-sampler.cpp llama-vocab.cpp unicode-data.cpp unicode.cpp)
  list(TRANSFORM _core PREPEND "${_src}/src/")
  file(GLOB _models CONFIGURE_DEPENDS "${_src}/src/models/*.cpp")
  add_library(m31_llama STATIC ${_core} ${_models})
  set_target_properties(m31_llama PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 16)
  set_source_files_properties(${_core} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  target_compile_features(m31_llama PUBLIC cxx_std_17)
  target_link_libraries(m31_llama PUBLIC ggml)

  set(_generated "${CMAKE_CURRENT_BINARY_DIR}/tools/m31/grammar")
  file(MAKE_DIRECTORY "${_generated}")
  set(LLAMA_VERSION "m31-source")
  file(STRINGS "${_src}/CMakeLists.txt" _version_lines
    REGEX "^set\\(LLAMA_VERSION_(MAJOR|MINOR|PATCH) [0-9]+\\)")
  foreach(_line IN LISTS _version_lines)
    if(_line MATCHES "^set\\((LLAMA_VERSION_(MAJOR|MINOR|PATCH)) ([0-9]+)\\)")
      set("${CMAKE_MATCH_1}" "${CMAKE_MATCH_3}")
    endif()
  endforeach()
  if(DEFINED LLAMA_VERSION_MAJOR AND DEFINED LLAMA_VERSION_MINOR AND DEFINED LLAMA_VERSION_PATCH)
    set(LLAMA_VERSION "${LLAMA_VERSION_MAJOR}.${LLAMA_VERSION_MINOR}.${LLAMA_VERSION_PATCH}-m31")
  endif()
  # An explicit source override may be an archive without .git. Do not borrow
  # the containing Xolotl repository's changing HEAD for this build stamp.
  set(LLAMA_BUILD_COMMIT "source-archive")
  if(EXISTS "${_src}/.git")
    execute_process(COMMAND git -C "${_src}" rev-parse --short=12 HEAD
      OUTPUT_VARIABLE _commit OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _git_rc)
    if(_git_rc EQUAL 0)
      set(LLAMA_BUILD_COMMIT "${_commit}")
    endif()
  endif()
  configure_file("${_src}/src/llama-version.h.in" "${_generated}/llama-version.h" @ONLY)
  target_include_directories(m31_llama PRIVATE "${_src}/src" "${_generated}"
    PUBLIC "${_src}/include")

  if(STRATA_Q36_GRAMMAR)
    # q36 already supplies the ggml CUDA host symbols used by its MMQ kernels.
    # Vocab-only llama needs no GPU backend: use the same core sources with a
    # CPU-only registry so it never pulls in a second copy of those symbols.
    add_library(q36_vocab_ggml STATIC "${_src}/ggml/src/ggml-backend-reg.cpp"
      "${_src}/ggml/src/ggml-backend-dl.cpp")
    target_include_directories(q36_vocab_ggml PRIVATE "${_src}/ggml/src")
    target_compile_definitions(q36_vocab_ggml PRIVATE GGML_USE_CPU)
    target_link_libraries(q36_vocab_ggml PUBLIC ggml-base ggml-cpu ${CMAKE_DL_LIBS})
    target_compile_features(q36_vocab_ggml PRIVATE cxx_std_17)
    set(_q36_core ${_core})
    list(REMOVE_ITEM _q36_core "${_src}/src/llama-sampler.cpp")
    set(_q36_sampler "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../src/qwen35/q36_grammar_state.cpp")
    set_source_files_properties("${_q36_sampler}" PROPERTIES
      SKIP_UNITY_BUILD_INCLUSION ON
      COMPILE_DEFINITIONS "STRATA_Q36_LLAMA_SAMPLER_SOURCE=\"${_src}/src/llama-sampler.cpp\"")
    add_library(q36_llama STATIC ${_q36_core} ${_models} "${_q36_sampler}")
    set_target_properties(q36_llama PROPERTIES UNITY_BUILD ON UNITY_BUILD_BATCH_SIZE 16)
    target_compile_features(q36_llama PUBLIC cxx_std_17)
    target_include_directories(q36_llama PRIVATE "${_src}/src" "${_generated}"
      PUBLIC "${_src}/include")
    target_link_libraries(q36_llama PUBLIC q36_vocab_ggml)
  endif()

  set(LLAMA_BUILD_NUMBER 0)
  set(BUILD_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  set(BUILD_TARGET "${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME}")
  configure_file("${_src}/common/build-info.cpp.in" "${_generated}/build-info.cpp" @ONLY)
  set(_common
    arg.cpp chat-auto-parser-generator.cpp chat-auto-parser-helpers.cpp
    chat-diff-analyzer.cpp chat-peg-parser.cpp chat.cpp common.cpp console.cpp
    debug.cpp download.cpp fit.cpp file-lock.cpp hf-cache.cpp imatrix-loader.cpp
    json-schema-to-grammar.cpp json-schema.cpp json.cpp llguidance.cpp log.cpp
    ngram-cache.cpp ngram-map.cpp ngram-mod.cpp peg-parser.cpp preset.cpp
    reasoning-budget.cpp sampling.cpp speculative.cpp subproc.cpp trie.cpp
    unicode.cpp jinja/lexer.cpp jinja/parser.cpp jinja/runtime.cpp jinja/value.cpp
    jinja/string.cpp jinja/caps.cpp)
  list(TRANSFORM _common PREPEND "${_src}/common/")
  include("${_src}/common/parsers/sources.cmake")
  add_library(m31_common STATIC ${_common} ${LLAMA_CHAT_PARSERS_SOURCES}
    "${_generated}/build-info.cpp" "${_src}/vendor/cpp-httplib/httplib.cpp")
  target_compile_features(m31_common PUBLIC cxx_std_17)
  target_include_directories(m31_common PUBLIC "${_src}/common" "${_src}/vendor"
    "${_src}/vendor/nlohmann")
  target_compile_definitions(m31_common PRIVATE
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    CPPHTTPLIB_LISTEN_BACKLOG=512 CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    CPPHTTPLIB_TCP_NODELAY=1)
  target_link_libraries(m31_common PUBLIC m31_llama Threads::Threads)
  if(WIN32)
    target_link_libraries(m31_common PRIVATE ws2_32)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    target_link_libraries(m31_common PRIVATE bsd)
  endif()

  add_library(m31_grammar INTERFACE)
  target_link_libraries(m31_grammar INTERFACE m31_common)
  target_compile_definitions(m31_grammar INTERFACE M31_HAVE_GRAMMAR=1)
  target_include_directories(m31_grammar INTERFACE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
  # The model-specific test option and registration belong to tests/tests.cmake.

endfunction()

m31_add_grammar()
