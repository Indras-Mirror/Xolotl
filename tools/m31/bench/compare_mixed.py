#!/usr/bin/env python3
"""Require all mixed prompt rows to match the PR #31192 GPU-offload oracle.

Each dump must consume the same immutable prepared input with one actual prefill.
An optional independently prepared input must have identical arrays too. This
checks encoder parity separately from LLM parity; neither replaces the other.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from compare_logits import compare_case, execution_chunks, full_last_layer_before_crop, load_case

FILES = ("text_ids.i32", "visual_rows.i32", "positions.i32", "visual.f32")
PR_SHA = "6b95a9908d2c7b242817a447708aec38e937e2c0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs(manifest):
    manifest = Path(manifest).resolve(strict=True)
    m = json.loads(manifest.read_text())
    if (m.get("format") != "m31-vision-batch-v1" or m.get("source_pr") != PR_SHA
            or m.get("encoded") is not True or m.get("llm_inference") is not False
            or m.get("n_embd") != 6144):
        raise ValueError("unsupported mixed-input provenance")
    rows, visual = m.get("rows"), m.get("visual_rows")
    if type(rows) is not int or rows <= 0 or type(visual) is not int or not 0 <= visual <= rows:
        raise ValueError("invalid mixed-input dimensions")
    for name in FILES:
        path = (manifest.parent / name).resolve(strict=True)
        if path.parent != manifest.parent or path.stat().st_size != (visual * 6144 * 4 if name == "visual.f32" else rows * 4):
            raise ValueError("invalid mixed-input file: " + name)
    def integers(name):
        values = array("i")
        if values.itemsize != 4:
            raise ValueError("32-bit array required")
        values.frombytes((manifest.parent / name).read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        return values.tolist()
    ids = integers("text_ids.i32")
    mapping = integers("visual_rows.i32")
    if integers("positions.i32") != list(range(rows)):
        raise ValueError("decoder positions must be contiguous linear rows")
    expected_visual = 0
    for token, index in zip(ids, mapping):
        if token == -1:
            if index != expected_visual or index >= visual:
                raise ValueError("invalid complete ordered visual mapping")
            expected_visual += 1
        elif token < 0 or index != -1:
            raise ValueError("invalid text/visual mapping")
    if expected_visual != visual:
        raise ValueError("visual row count does not match mapping")
    if visual:
        values = np.memmap(manifest.parent / "visual.f32", dtype="<f4", mode="r")
        if any(not np.isfinite(values[start:start + (1 << 20)]).all()
               for start in range(0, values.size, 1 << 20)):
            raise ValueError("non-finite prepared visual embeddings")
    chunks = m.get("chunks")
    if not isinstance(chunks, list) or not chunks or len(chunks) > rows:
        raise ValueError("invalid input chunk list")
    first = first_visual = 0
    for index, chunk in enumerate(chunks):
        count = chunk.get("rows")
        if (type(count) is not int or not 0 < count <= rows - first or chunk.get("index") != index
                or chunk.get("row") != first or chunk.get("position") != first or chunk.get("n_pos") != count):
            raise ValueError("invalid input chunk continuity")
        if chunk.get("type") == "text":
            if chunk.get("tokens") != ids[first:first + count] or any(t < 0 for t in chunk["tokens"]):
                raise ValueError("input text chunk disagrees with token IDs")
        elif chunk.get("type") == "image":
            if (chunk.get("visual_row") != first_visual or ids[first:first + count] != [-1] * count
                    or chunk.get("decoder_positions") != {"start": first, "step": 1, "axes": 1}):
                raise ValueError("input image chunk disagrees with linear visual rows")
            first_visual += count
        else:
            raise ValueError("unsupported input chunk type")
        first += count
    if first != rows or first_visual != visual:
        raise ValueError("input chunks do not cover rows")
    return {"rows": rows, "visual_rows": visual,
            "sha256": {name: sha(manifest.parent / name) for name in FILES}}, manifest.parent


def check(reference, candidate, mixed_manifest, independent_manifest=None):
    prepared, directory = inputs(mixed_manifest)
    result = {"format": "m31-mixed-comparison-v1", "input": prepared, "pass": False}
    if independent_manifest:
        independent, _ = inputs(independent_manifest)
        result["independent_input"] = independent
        if independent != prepared:
            raise ValueError("independently encoded input arrays differ")
    a, _ = load_case(reference)
    b, _ = load_case(candidate)
    if a.get("engine") != "llama.cpp-PR31192" or a.get("backend") != "cuda" or a.get("oracle_op_offload") is not True:
        raise ValueError("requires PR #31192 CUDA GPU-op-offload oracle")
    if b.get("engine") != "xolotl-m31" or b.get("backend") != "cuda":
        raise ValueError("requires Xolotl M31 CUDA candidate")
    if a.get("cuda_fusion_disabled") is not True or b.get("cuda_fusion_disabled") is not True:
        raise ValueError("GPU mixed acceptance requires fusion disabled in both engines")
    if a.get("context") != b.get("context") or type(a.get("context")) is not int:
        raise ValueError("GPU mixed acceptance requires matching actual context capacity")
    if not full_last_layer_before_crop(a) or not full_last_layer_before_crop(b):
        raise ValueError("head must follow the full final-layer FFN")
    if any(token >= a["vocab"] for token in np.fromfile(directory / "text_ids.i32", dtype="<i4")):
        raise ValueError("prepared text token lies outside the dump vocabulary")
    rows = prepared["rows"]
    for path, manifest in ((reference, a), (candidate, b)):
        declared = manifest.get("mixed_manifest")
        if not declared:
            raise ValueError("dump lacks prepared mixed-input provenance")
        authenticated, _ = inputs(declared)
        if authenticated != prepared:
            raise ValueError("dump declared mixed-input arrays differ from supplied prepared input")
        if execution_chunks(manifest) != [(0, rows)]:
            raise ValueError("mixed acceptance requires one actual complete prefill")
        if manifest.get("positions") != list(range(rows)):
            raise ValueError("mixed acceptance requires every prompt row")
        if (Path(path) / "tokens.i32").read_bytes() != (directory / "text_ids.i32").read_bytes():
            raise ValueError("dump input differs from supplied text/visual row stream")
    metrics = compare_case(Path(reference), Path(candidate))
    result.update(metrics)
    result["pass"] = metrics["bitwise_identical"] and metrics["full_rows"] and metrics["max_abs"] == 0
    result["scope"] = "all text and visual prompt rows; one prefill; strict raw F32 equality"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("mixed_manifest", type=Path)
    p.add_argument("--independent-input", type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    result = check(a.reference, a.candidate, a.mixed_manifest, a.independent_input)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
