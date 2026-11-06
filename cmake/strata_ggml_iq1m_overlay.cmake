# cmake/strata_ggml_iq1m_overlay.cmake - llama.cpp's MMQ prompt kernels for IQ1_M, on the pinned ggml-cuda.
#
# Upstream MMQ has no IQ1_M (IQ1_S only), so QFN45's IQ1_M gate/up layers (27 of 48) run the prompt's experts on the FP16
# path or pf_wmma.  Port of Basalt 86b1e97f4 (Jesus Gulfo): the MMQ tile holds an IQ1_M weight exactly as int8 with one
# float scale per 16 values, IQ2_XS's tile layout, so IQ1_M is routed through every MMQ table as IQ2_XS is (D4 activation
# layout, Q8_0_16 dot kernels, IQ2_XS's tile configs) with its own loader (src/prefill/mmq_iq1m_tiles.cuh).
#
# Same mechanism as cmake/strata_ggml_tq_overlay.cmake: copies of the ggml-cuda headers in the build tree with the type
# added, first on strata_mmq's include path only; the pinned tree (shared with other builds) is never touched.  Every
# edit is anchored on pinned text; a missing or repeated anchor turns the overlay off (STRATA_IQ1M_OVERLAY_OK=OFF) and
# IQ1_M keeps its current prompt path.  If the trellis overlay already wrote OUT, pass INPLACE and the edits apply to it.
#
#   strata_ggml_iq1m_overlay(<llama.cpp dir> <out dir> [INPLACE])  ->  STRATA_IQ1M_OVERLAY_OK (parent scope)

function(strata_ggml_iq1m_overlay LLAMA OUT)
  set(STRATA_IQ1M_OVERLAY_OK OFF PARENT_SCOPE)
  set(_inplace OFF)
  if("${ARGN}" STREQUAL "INPLACE")
    set(_inplace ON)
  endif()
  set(_cu "${LLAMA}/ggml/src/ggml-cuda")
  set(_inc "${LLAMA}/ggml/include")
  foreach(_f "${_cu}/mmq.cuh" "${_cu}/mmq-load-tiles.cuh" "${_cu}/common.cuh" "${_cu}/quantize.cu")
    if(NOT EXISTS "${_f}")
      message(STATUS "IQ1_M MMQ overlay: ${_f} is missing: off")
      return()
    endif()
  endforeach()

  macro(_q1_edit VAR WHERE ANCHOR TEXT)   # WHERE = BEFORE | AFTER; the anchor must occur exactly once
    string(FIND "${${VAR}}" "${ANCHOR}" _p)
    if(_p LESS 0)
      set(_q1_fail "anchor not found: ${ANCHOR}")
    else()
      string(LENGTH "${ANCHOR}" _alen)
      math(EXPR _after "${_p} + ${_alen}")
      string(SUBSTRING "${${VAR}}" ${_after} -1 _tail)
      string(FIND "${_tail}" "${ANCHOR}" _p2)
      if(NOT _p2 LESS 0)
        set(_q1_fail "anchor is not unique: ${ANCHOR}")
      else()
        string(SUBSTRING "${${VAR}}" 0 ${_p} _head)
        if("${WHERE}" STREQUAL "AFTER")
          set(${VAR} "${_head}${ANCHOR}${TEXT}${_tail}")
        else()
          set(${VAR} "${_head}${TEXT}${ANCHOR}${_tail}")
        endif()
      endif()
    endif()
  endmacro()
  set(_q1_fail "")

  file(MAKE_DIRECTORY "${OUT}/include" "${OUT}/template-instances")
  if(NOT _inplace)
    # the whole header set: quote includes resolve next to the including file, so one left behind would pull the pinned
    # mmq.cuh back in beside the patched one
    file(GLOB _all_cu "${_cu}/*.cuh" "${_cu}/*.h")
    foreach(_f IN LISTS _all_cu)
      get_filename_component(_n "${_f}" NAME)
      configure_file("${_f}" "${OUT}/${_n}" COPYONLY)
    endforeach()
    configure_file("${_cu}/quantize.cu" "${OUT}/quantize.cu" COPYONLY)   # it includes mmq.cuh
    if(IS_DIRECTORY "${_cu}/vendors")
      file(COPY "${_cu}/vendors" DESTINATION "${OUT}")
    endif()
    file(GLOB _all_inc "${_inc}/*.h")
    foreach(_f IN LISTS _all_inc)
      get_filename_component(_n "${_f}" NAME)
      configure_file("${_f}" "${OUT}/include/${_n}" COPYONLY)
    endforeach()
  endif()
  # (no CMAKE_CONFIGURE_DEPENDS here: configure_file already makes every copied file a configure dependency, and a
  # second entry gives ninja two rules for one path)

  # ---- mmq.cuh: IQ1_M wherever IQ2_XS is routed (the pinned signatures: ggml_prec prec_src1, has_fusion) ---------------
  file(READ "${OUT}/mmq.cuh" _m)
  string(REPLACE "\r\n" "\n" _m "${_m}")
  string(FIND "${_m}" "ggml_cuda_mmq_get_nthreads(ggml_type type, int J, bool fallback, ggml_prec prec_src1)" _sig)
  if(_sig LESS 0)
    set(_q1_fail "the MMQ helper signatures of this ggml-cuda are not the ones the overlay knows (ggml_prec)")
  endif()
  # the activation layout: D4, IQ2_XS's group
  _q1_edit(_m BEFORE "        case GGML_TYPE_IQ2_XXS:\n        case GGML_TYPE_IQ2_XS:\n        case GGML_TYPE_IQ2_S:\n"
    "        case GGML_TYPE_IQ1_M:\n")
  # the dp4a tile sizes
  _q1_edit(_m AFTER "        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;\n"
    "        case GGML_TYPE_IQ1_M:   return MMQ_DP4A_TXS_Q8_0_16;\n")
  # dp4a kernels
  _q1_edit(_m BEFORE "            case GGML_TYPE_IQ2_XS:\n                return ggml_cuda_mmq_util_funcs(\n                    VDR_IQ2_XS_Q8_1_MMQ,"
    "            case GGML_TYPE_IQ1_M:\n                return ggml_cuda_mmq_util_funcs(\n                    VDR_IQ2_XS_Q8_1_MMQ,\n                    ggml_cuda_mmq_load_tiles_iq1_m<type, J, fallback>,\n                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,\n                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);\n")
  # MMA kernels
  _q1_edit(_m BEFORE "        case GGML_TYPE_IQ2_XS:\n            return ggml_cuda_mmq_util_funcs(\n                -1,\n                ggml_cuda_mmq_load_tiles_iq2_xs<type, J, fallback>,"
    "        case GGML_TYPE_IQ1_M:   // one scale per 16 values: IQ2_XS's Q8_0_16 dot (Basalt 86b1e97f4)\n            return ggml_cuda_mmq_util_funcs(\n                -1,\n                ggml_cuda_mmq_load_tiles_iq1_m<type, J, fallback>,\n                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback, prec_src1>,\n                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1, false>);\n")
  _q1_edit(_m AFTER "extern DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);\n" "extern DECL_MMQ_CASE(GGML_TYPE_IQ1_M);\n")
  file(WRITE "${OUT}/mmq.cuh" "${_m}")

  # ---- mmq-load-tiles.cuh: the loader at the end ------------------------------------------------------------------------
  file(READ "${OUT}/mmq-load-tiles.cuh" _l)
  string(FIND "${_l}" "ggml_cuda_mmq_load_tiles_iq1_m" _have)
  if(_have LESS 0)
    file(WRITE "${OUT}/mmq-load-tiles.cuh" "${_l}\n#include \"mmq_iq1m_tiles.cuh\"\n")
  endif()
  configure_file("${CMAKE_SOURCE_DIR}/src/prefill/mmq_iq1m_tiles.cuh" "${OUT}/mmq_iq1m_tiles.cuh" COPYONLY)

  # ---- mmq-config-*.cuh: every IQ2_XS CASE line gets an IQ1_M copy (Basalt's IQ1_M configs are IQ2_XS's) ----------------
  file(GLOB _cfgs "${OUT}/mmq-config-*.cuh")
  foreach(_c ${_cfgs})
    file(READ "${_c}" _t)
    string(REPLACE "\r\n" "\n" _t "${_t}")
    string(FIND "${_t}" "GGML_TYPE_IQ1_M," _done)
    if(NOT _done LESS 0)
      continue()
    endif()
    string(REPLACE ";" "<SEMI>" _t "${_t}")
    string(REGEX MATCHALL "[^\n]*CASE\\(GGML_TYPE_IQ2_XS,[^\n]*\n" _lines "${_t}")
    list(REMOVE_DUPLICATES _lines)
    foreach(_ln IN LISTS _lines)
      string(REPLACE "GGML_TYPE_IQ2_XS," "GGML_TYPE_IQ1_M, " _x "${_ln}")
      _q1_edit(_t AFTER "${_ln}" "${_x}")
    endforeach()
    string(REPLACE "<SEMI>" ";" _t "${_t}")
    file(WRITE "${_c}" "${_t}")
  endforeach()

  # ---- the template instance -------------------------------------------------------------------------------------------
  file(WRITE "${OUT}/template-instances/mmq-instance-iq1_m.cu"
       "// Strata IQ1_M overlay (cmake/strata_ggml_iq1m_overlay.cmake)\n\n#include \"../mmq.cuh\"\n\nDECL_MMQ_CASE(GGML_TYPE_IQ1_M);\n")

  if(_q1_fail)
    message(STATUS "IQ1_M MMQ overlay off: ${_q1_fail}")
    return()
  endif()
  set(STRATA_IQ1M_OVERLAY_OK ON PARENT_SCOPE)
endfunction()
