find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(m31_core_check "${_m31_dir}/tests/core_check.cpp")
target_link_libraries(m31_core_check PRIVATE m31_engine)
set(_m31_allocation_check_args)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  list(APPEND _m31_allocation_check_args --allocation-check $<TARGET_FILE:m31_fused_allocation_check>)
endif()
add_test(NAME m31_core COMMAND "${Python3_EXECUTABLE}" "${_m31_dir}/tests/run_core_check.py"
  --check $<TARGET_FILE:m31_core_check> --cache-check $<TARGET_FILE:m31_cache_model_check>
  ${_m31_allocation_check_args})
set_tests_properties(m31_core PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 120)

add_test(NAME m31_serve COMMAND "${Python3_EXECUTABLE}" "${_m31_dir}/tests/test_serve.py" -v)
set_tests_properties(m31_serve PROPERTIES LABELS "m31;cpu;protocol" TIMEOUT 120)

add_executable(m31_resource_checks "${_m31_dir}/tests/resource_checks.cpp")
target_link_libraries(m31_resource_checks PRIVATE m31_engine)
add_test(NAME m31_resources COMMAND m31_resource_checks)
set_tests_properties(m31_resources PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 30)

add_executable(m31_protocol_check "${_m31_dir}/tests/protocol_check.cpp")
target_include_directories(m31_protocol_check PRIVATE "${_m31_dir}")
add_test(NAME m31_protocol COMMAND m31_protocol_check)
set_tests_properties(m31_protocol PROPERTIES LABELS "m31;cpu;protocol" TIMEOUT 30)

if(NOT TARGET m31_grammar_api_check)
  add_executable(m31_grammar_api_check "${_m31_dir}/tests/grammar_api_check.cpp")
  target_link_libraries(m31_grammar_api_check PRIVATE m31_grammar)
  add_test(NAME m31_grammar_api COMMAND m31_grammar_api_check)
  set_tests_properties(m31_grammar_api PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 30)
endif()

add_test(NAME m31_bench COMMAND "${Python3_EXECUTABLE}" "${_m31_dir}/tests/test_bench.py" -v)
set_tests_properties(m31_bench PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 60)

add_executable(m31_protocol_fixture "${_m31_dir}/tests/protocol_fixture.cpp")
target_include_directories(m31_protocol_fixture PRIVATE "${_m31_dir}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(m31_protocol_fixture PRIVATE ggml-base Threads::Threads)
add_test(NAME m31_protocol_loop COMMAND "${Python3_EXECUTABLE}" "${_m31_dir}/tests/test_protocol.py"
  --fixture $<TARGET_FILE:m31_protocol_fixture>)
set_tests_properties(m31_protocol_loop PROPERTIES LABELS "m31;cpu;protocol" TIMEOUT 90)

add_test(NAME m31_config_environment COMMAND "${Python3_EXECUTABLE}"
  "${_m31_dir}/tests/test_config_environment.py" $<TARGET_FILE:m31_resource_checks>)
set_tests_properties(m31_config_environment PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 30)

# Explicit vocabulary-only hardware check; never registered in default ctest.
add_executable(m31_grammar_model_check "${_m31_dir}/tests/grammar_model_check.cpp")
target_link_libraries(m31_grammar_model_check PRIVATE m31_grammar)

add_test(NAME m31_import_isolation COMMAND "${Python3_EXECUTABLE}" "${_m31_dir}/tests/test_isolation.py")
set_tests_properties(m31_import_isolation PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 30)

add_custom_target(m31_checks DEPENDS m31_core_check m31_resource_checks
  m31_protocol_check m31_protocol_fixture m31_grammar_api_check)

# This executable loads a model only when explicitly invoked; ctest supplies a tiny CPU fixture.
add_executable(m31_cache_model_check "${_m31_dir}/tests/cache_model_check.cpp")
target_link_libraries(m31_cache_model_check PRIVATE m31_engine)

add_dependencies(m31_checks m31_cache_model_check)

# GNU ld's wrap seam affects this test executable only, never production code.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(m31_fused_allocation_check "${_m31_dir}/tests/fused_allocation_check.cpp")
  target_link_libraries(m31_fused_allocation_check PRIVATE m31_engine)
  add_dependencies(m31_checks m31_fused_allocation_check)
  target_link_options(m31_fused_allocation_check PRIVATE "LINKER:--wrap=ggml_gallocr_alloc_graph")
  add_test(NAME m31_fused_allocation COMMAND m31_fused_allocation_check)
  set_tests_properties(m31_fused_allocation PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 30)
endif()

# These checks execute no CUDA operation, including in a CUDA build.
foreach(_m31_short_check policy cost cpu_lane cuda_math_cpu cuda_executor)
  set(_m31_short_target "m31_shortprefill_${_m31_short_check}_test")
  add_executable(${_m31_short_target} "${_m31_dir}/tests/shortprefill_${_m31_short_check}_test.cpp")
  target_include_directories(${_m31_short_target} PRIVATE "${_m31_ggml_src}")
  target_link_libraries(${_m31_short_target} PRIVATE m31_engine)
  if(NOT MSVC)
    target_compile_options(${_m31_short_target} PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_dependencies(m31_checks ${_m31_short_target})
  add_test(NAME "m31_shortprefill_${_m31_short_check}" COMMAND ${_m31_short_target})
  set_tests_properties("m31_shortprefill_${_m31_short_check}" PROPERTIES LABELS "m31;cpu;unit" TIMEOUT 60)
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_definitions(m31_shortprefill_cuda_executor_test PRIVATE M31_SHORT_ALLOC_WRAP=1)
  target_link_options(m31_shortprefill_cuda_executor_test PRIVATE "LINKER:--wrap=ggml_gallocr_alloc_graph")
endif()
