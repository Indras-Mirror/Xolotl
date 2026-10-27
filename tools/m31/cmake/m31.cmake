# tools/m31/cmake/m31.cmake - the MiniMax-M3.1-Flash engine (tools/m31, community port, issue #62).
# include()d (OPTIONAL) at the end of the root CMakeLists.txt, after tools/ds4/cmake/fork.cmake and
# tools/glm/cmake/glm.cmake - the same place and the same contract as glm.cmake.
#
# Opt-in and OFF by default: a model the default install does not ship must not change the default build's targets
# or their bytes (AGENTS.md: "A new feature is opt-in, and the default path stays byte-identical").
#   -DSTRATA_M31=ON          m31_engine (the port's sources) + m31_generate (its serve entry point)
#   -DSTRATA_M31_VISION=ON   m31_vision too; needs an mtmd target, see the end of this file
#
# With STRATA_M31=OFF, or with no port sources in tools/m31 yet, this file configures nothing and says so - a
# checkout that carries the cmake before the port still builds.  That is the state this scaffold ships in.

option(STRATA_M31 "Build the MiniMax-M3.1-Flash engine (tools/m31; community port, issue #62)" OFF)
if(NOT STRATA_M31)
  return()
endif()
if(NOT TARGET ggml-cpu)
  message(STATUS "STRATA_M31=ON but ggml-cpu is not a target in this configure; skipping the M3.1 engine")
  return()
endif()

set(_m31_dir "${CMAKE_CURRENT_SOURCE_DIR}/tools/m31")
set(_m31_main "${_m31_dir}/m31_generate.cpp")
if(NOT EXISTS "${_m31_main}")
  message(STATUS "STRATA_M31=ON but ${_m31_main} does not exist - the M3.1 port is not in this tree; skipping")
  return()
endif()

# The port's own sources: everything in tools/m31/ except the serve entry point.  CONFIGURE_DEPENDS so a branch that
# adds a file is picked up without a fresh configure.  Exactly one main() may live here (m31_generate.cpp): the
# development probes are expected to be gone before merge (#62, plan item 2), and a second main() would collide at
# link time.
file(GLOB _m31_lib CONFIGURE_DEPENDS "${_m31_dir}/*.cpp")
list(REMOVE_ITEM _m31_lib "${_m31_main}")

# ggml-impl.h (graph uid): honour STRATA_GGML_DIR, as glm.cmake does, so a worktree without third_party/llama.cpp
# still configures.
set(_m31_ggml_src "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/src")
if(STRATA_GGML_DIR)
  set(_m31_ggml_src "${STRATA_GGML_DIR}/ggml/src")
endif()
set(_m31_inc ${CMAKE_CURRENT_SOURCE_DIR}/include ${_m31_dir})

if(_m31_lib)
  add_library(m31_engine STATIC ${_m31_lib})
  target_include_directories(m31_engine PUBLIC ${_m31_inc})
  target_include_directories(m31_engine PRIVATE ${_m31_ggml_src})
  target_link_libraries(m31_engine PUBLIC ggml ggml-cpu ggml-base)
  if(TARGET ggml-cuda)
    target_link_libraries(m31_engine PUBLIC ggml-cuda)
  endif()
  set(_m31_link m31_engine)
else()
  # a single-file port: no library target, the entry point carries everything
  set(_m31_link "")
endif()

add_executable(m31_generate "${_m31_main}")
target_include_directories(m31_generate PRIVATE ${_m31_inc} ${_m31_ggml_src})
target_link_libraries(m31_generate PRIVATE ${_m31_link} ggml ggml-cpu ggml-base)
if(TARGET ggml-cuda)
  target_link_libraries(m31_generate PRIVATE ggml-cuda)
endif()
message(STATUS "STRATA_M31: building m31_generate from ${_m31_main}")

# Vision (#62, PR (b)): the port runs llama.cpp's mtmd encoder in a SEPARATE helper process with its own ggml build.
# It is its own target, built only when the port's vision sources AND an mtmd target both exist - linking an encoder
# this tree does not have would be a claim the build cannot keep.  llama.cpp PR #31192 carries the mtmd code; until it
# lands in the pinned ggml, the documented default is the contributor's helper as an opt-in (docs/MINIMAX_M31.md),
# not this target.
option(STRATA_M31_VISION "Build the M3.1 vision helper (tools/m31/vision/*.cpp; needs an mtmd target)" OFF)
if(STRATA_M31_VISION)
  file(GLOB _m31_vision CONFIGURE_DEPENDS "${_m31_dir}/vision/*.cpp")
  if(NOT _m31_vision)
    message(STATUS "STRATA_M31_VISION=ON but tools/m31/vision/*.cpp is empty; skipping the vision helper")
  elseif(NOT TARGET mtmd)
    message(STATUS "STRATA_M31_VISION=ON but this tree has no mtmd target (llama.cpp PR #31192 is not in the pinned "
                   "ggml); skipping the vision helper")
  else()
    add_executable(m31_vision ${_m31_vision})
    target_include_directories(m31_vision PRIVATE ${_m31_inc} ${_m31_ggml_src})
    target_link_libraries(m31_vision PRIVATE mtmd ggml ggml-base)
    if(TARGET ggml-cuda)
      target_link_libraries(m31_vision PRIVATE ggml-cuda)
    endif()
    message(STATUS "STRATA_M31_VISION: building m31_vision")
  endif()
endif()
