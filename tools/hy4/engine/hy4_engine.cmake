# tools/hy4/engine/hy4_engine.cmake - the Hy4-flash engine (included from tools/ds4/cmake/fork.cmake after the v41 engine,
# whose GGUF key readers it shares).
set(_hy4_dir ${CMAKE_CURRENT_SOURCE_DIR}/tools/hy4/engine)
add_executable(test_hy4_geometry ${_hy4_dir}/test_hy4_geometry.cpp)
target_include_directories(test_hy4_geometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/engine ${_hy4_dir})
if(STRATA_BUILD_TESTS)
    add_test(NAME test_hy4_geometry COMMAND test_hy4_geometry)
    set_tests_properties(test_hy4_geometry PROPERTIES SKIP_RETURN_CODE 77)
endif()
