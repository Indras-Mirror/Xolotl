"""Mechanical part of the Windows port of the DeepSeek-V4.1 engine: route POSIX file/memory calls through
tools/v41/engine/ds41_sys.hpp. Idempotent; prints what changed per file. Usage: python port_win_sys.py"""
import re
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "v41" / "engine"
FILES = ["ds41_moe.cpp", "ds41_moe.hpp", "ds41_engram.cpp", "ds41_engram.hpp", "ds41_geometry.hpp",
         "ds41_dense.cpp", "ds41_dense.hpp", "ds41_dspark.cpp", "ds41_generate.cpp",
         "test_ds41_dense.cpp", "test_ds41_geometry.cpp"]
NOT_ID = r"(?<![A-Za-z0-9_])"          # '::open(' but not 'foo::open('
CALLS = ["open", "close", "pread", "pwrite", "fsync", "posix_fadvise", "fstat"]
POSIX_INCLUDES = ["fcntl.h", "unistd.h", "sys/stat.h"]

for name in FILES:
    p = ENGINE / name
    if not p.exists():
        continue
    src = p.read_text(encoding="utf-8")
    orig = src
    counts = {}

    def sub(pattern, repl, key, flags=0):
        global src
        src, n = re.subn(pattern, repl, src, flags=flags)
        if n:
            counts[key] = counts.get(key, 0) + n

    # includes: the first POSIX header becomes ds41_sys.hpp, the rest go
    if "ds41_sys.hpp" not in src:
        first = True
        for inc in POSIX_INCLUDES:
            pat = r"^#include <" + re.escape(inc) + r">[ \t]*\r?\n"
            m = re.search(pat, src, flags=re.M)
            if not m:
                continue
            if first:
                src = src[:m.start()] + '#include "ds41_sys.hpp"\n' + src[m.end():]
                first = False
            else:
                src = src[:m.start()] + src[m.end():]
            counts["include"] = counts.get("include", 0) + 1
        if first and re.search(NOT_ID + r"::(open|pread|close|fstat)\(", src):
            # no POSIX include to replace but POSIX calls: add the shim after the last local include
            m = list(re.finditer(r'^#include "[^"]+"[ \t]*\r?\n', src, flags=re.M))
            at = m[-1].end() if m else 0
            src = src[:at] + '#include "ds41_sys.hpp"\n' + src[at:]
            counts["include"] = counts.get("include", 0) + 1
    for c in CALLS:
        sub(NOT_ID + r"::" + c + r"\(", "ds41_sys::" + c + "(", c)
    sub(r"(?<![A-Za-z0-9_:])pread\(", "ds41_sys::pread(", "pread-bare")          # geometry.hpp calls it unqualified
    sub(r"\bstruct stat\b", "ds41_sys::Stat", "struct stat")
    sub(r"\boff_t\b", "int64_t", "off_t")
    sub(r"(?<![A-Za-z0-9_:])posix_memalign\(", "ds41_sys::posix_memalign(", "posix_memalign")
    sub(r"\bstd::aligned_alloc\(", "ds41_sys::aligned_alloc(", "aligned_alloc")
    sub(r"(?<![A-Za-z0-9_:])setenv\(", "ds41_sys::setenv(", "setenv")
    if src != orig:
        p.write_text(src, encoding="utf-8", newline="\n")
    print(f"{name:24s} {counts if counts else 'unchanged'}")
