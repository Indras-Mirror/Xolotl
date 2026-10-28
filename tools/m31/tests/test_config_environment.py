#!/usr/bin/env python3
"""Validate engine-only fusion isolation and migration errors without a GPU."""
import os
from pathlib import Path
import subprocess
import sys

exe = str(Path(sys.argv[1]).resolve())
base = {key: value for key, value in os.environ.items() if not key.startswith("M31_")}
base.pop("GGML_CUDA_DISABLE_FUSION", None)
base.pop("GGML_CUDA_MMQ_PREC", None)

def run(extra):
    return subprocess.run([exe, "--environment"], env={**base, **extra}, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)

accepted = run({})
assert accepted.returncode == 0, accepted.stderr
assert accepted.stdout.splitlines()[0] == "fusion=1", accepted.stdout
assert "mmq_precision=auto" in accepted.stdout, accepted.stdout
accepted = run({"GGML_CUDA_DISABLE_FUSION": "1"})
assert accepted.returncode == 0, accepted.stderr
for value in ("auto", "Q4", "q8"):
    accepted = run({"GGML_CUDA_MMQ_PREC": value})
    assert accepted.returncode == 0 and f"mmq_precision={value.lower()}" in accepted.stdout, accepted
for value in ("", "garbage", "q16", "nan"):
    rejected = run({"GGML_CUDA_MMQ_PREC": value})
    assert rejected.returncode != 0 and "GGML_CUDA_MMQ_PREC must be" in rejected.stderr, rejected
for value in ("0", "", "true", "2"):
    rejected = run({"GGML_CUDA_DISABLE_FUSION": value})
    assert rejected.returncode != 0 and "requires GGML_CUDA_DISABLE_FUSION=1" in rejected.stderr, rejected
for name in ("M31_GPU_FAST_FFN", "M31_GPU_SORT_ROUTES", "M31_SHORT_PREFILL_MODE", "M31_CPU_POLL"):
    rejected = run({name: "1"})
    assert rejected.returncode != 0 and name in rejected.stderr and "removed environment option" in rejected.stderr, rejected
assert run({"M31_GPU_FAST_FFN": "0"}).returncode == 0
print("M31 subprocess environment checks passed")
