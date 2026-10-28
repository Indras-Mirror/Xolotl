#!/usr/bin/env python3
"""CPU fixture and independent NumPy MSA checks; no model download or GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from tiny_fixture import make_fixture
from numpy_reference import forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True, type=Path)
    parser.add_argument("--cache-check", type=Path)
    parser.add_argument("--allocation-check", type=Path)
    args = parser.parse_args()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    with tempfile.TemporaryDirectory(prefix="m31-core-") as temp:
        root = Path(temp)
        model, output = root / "tiny.gguf", root / "logits.f32"
        make_fixture(model)
        subprocess.run([str(args.check), str(model), str(output)], env=env, check=True, timeout=90)
        ids = np.array([(i * 17) % 64 for i in range(29)], dtype=np.int32)
        expected = forward(model, ids)
        actual = np.fromfile(output, dtype="<f4").reshape(29, 64)
        np.testing.assert_allclose(actual, expected, atol=.005, rtol=.005)
        print(f"independent NumPy max_abs={np.max(np.abs(actual - expected)):.7g}")
        if args.cache_check:
            prefix_ids = root / "prefix.i32"
            np.array([(i * 17) % 64 for i in range(263)], dtype="<i4").tofile(prefix_ids)
            subprocess.run([str(args.cache_check), "--model", str(model), "--ids-file", str(prefix_ids),
                            "--out", str(root / "cache-model"), "--length", "263", "--reuse", "256",
                            "--ctx", "768", "--threads", "2", "--chunk", "300", "--cpu-fused", "0",
                            "--checkpoint-mib", "1"], env=env, check=True, timeout=60)
        if args.allocation_check:
            subprocess.run([str(args.allocation_check), str(model)], env=env, check=True, timeout=30)
        for name, kwargs in (("missing-index", {"missing_dense_index": True}),
                             ("unknown-tensor", {"unknown_tensor": True}),
                             ("oversized-block", {"block": 512}),
                             ("unaligned-block", {"block": 3}),
                             ("overflow-rms", {"overflow_float": "attention.layer_norm_rms_epsilon"}),
                             ("overflow-rope", {"overflow_float": "rope.freq_base"}),
                             ("overflow-route", {"overflow_float": "expert_weights_scale"})):
            invalid = root / f"{name}.gguf"
            make_fixture(invalid, **kwargs)
            result = subprocess.run([str(args.check), str(invalid), str(output)], env=env, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 or "M31:" not in result.stderr:
                raise AssertionError(f"invalid geometry accepted: {name}: {result.stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
