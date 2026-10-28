#!/usr/bin/env python3
"""Check the real helper's startup profile before any model/backend init."""
import os
from pathlib import Path
import subprocess
import sys

helper = str(Path(sys.argv[1]).resolve(strict=True))
base = {key: value for key, value in os.environ.items() if key != "GGML_CUDA_DISABLE_FUSION"}
for value in (None, "1", "0", "", "true", "2"):
    env = dict(base)
    if value is not None:
        env["GGML_CUDA_DISABLE_FUSION"] = value
    result = subprocess.run([helper, "--check-environment"], env=env, capture_output=True,
                            text=True, timeout=5)
    if value in (None, "1"):
        assert result.returncode == 0 and result.stdout.strip() == "fusion=1", result
    else:
        assert result.returncode != 0 and "requires GGML_CUDA_DISABLE_FUSION=1" in result.stderr, result
print("M31 helper startup environment checks passed (no model/backend initialized)")
