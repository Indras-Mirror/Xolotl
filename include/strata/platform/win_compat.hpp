// strata/platform/win_compat.hpp - the POSIX pieces MSVC does not have, for a program's include site.  Include it where
// `<unistd.h>` stood.  Elsewhere it is that header and changes nothing.
//
// Windows (MSVC):
//   * off_t is 32-bit and there is no fseeko.  A GGUF's offsets run past 2 GiB (Inkling-2-Mini's file is 13.9 GB, Muse
//     Glimmer's 16.1 GB), so a seek cast through (off_t) would cut the offset and read from the wrong place, without a
//     word.  off_t becomes int64_t and fseeko _fseeki64.
//   * __builtin_popcount is a GCC builtin; MSVC's is __popcnt.
//   * M_PI is hidden unless _USE_MATH_DEFINES is defined.
//   * setenv is _putenv_s.  The overwrite flag is not honoured: the value is always set.
// The macros reach every header included after this one, so it goes where <unistd.h> was, after the CRT's own.
#pragma once

#if defined(_WIN32)
#include <cstdint>
#include <cstdio>
#include <intrin.h>
#include <io.h>
#include <stdlib.h>
#include <sys/types.h>     // declares off_t itself: included here, before the macro, it is not read again after it

#define off_t int64_t
#define fseeko(f, o, w) _fseeki64((f), (o), (w))
#define __builtin_popcount(x) __popcnt((unsigned) (x))
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define setenv(name, value, overwrite) _putenv_s((name), (value))
#else
#include <unistd.h>
#endif
