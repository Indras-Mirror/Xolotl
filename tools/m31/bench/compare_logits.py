#!/usr/bin/env python3
"""Compare matching raw logits, streaming rows. Acceptance always checks IDs and positions.

--require-suite additionally requires 20 complete 64..3000-token RU/EN/code/math cases,
including MSA block selection. A partial diagnostic must not be reported as complete gate20 acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_case(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != "m31-logits-v1" or manifest.get("kv") != "f16":
        raise ValueError(f"{path}: unsupported dump format/cache type")
    n, vocab = manifest["tokens"], manifest["vocab"]
    positions = manifest["positions"]
    if not isinstance(n, int) or n <= 0 or not isinstance(vocab, int) or vocab <= 0:
        raise ValueError(f"{path}: invalid dimensions")
    if not positions or positions != sorted(set(positions)) or any(type(p) is not int or p < 0 or p >= n for p in positions):
        raise ValueError(f"{path}: invalid positions")
    if (path / "tokens.i32").stat().st_size != n * 4:
        raise ValueError(f"{path}: token count mismatch")
    if (path / "logits.f32").stat().st_size != len(positions) * vocab * 4:
        raise ValueError(f"{path}: truncated/extra logits")
    chunks = manifest.get('actual_chunks', [])
    if chunks:
        next_position = 0
        for chunk in chunks:
            if (not isinstance(chunk, dict) or type(chunk.get('position')) is not int
                    or type(chunk.get('tokens')) is not int or chunk['position'] != next_position
                    or chunk['tokens'] <= 0):
                raise ValueError(f"{path}: invalid actual chunk sequence")
            next_position += chunk['tokens']
        if next_position != n:
            raise ValueError(f"{path}: actual chunks do not cover input")
    values = np.memmap(path / "logits.f32", dtype="<f4", mode="r", shape=(len(positions), vocab))
    return manifest, values


def execution_chunks(manifest):
    if manifest.get('actual_chunks'):
        return [(chunk['position'], chunk['tokens']) for chunk in manifest['actual_chunks']]
    # Legacy frozen references predate actual_chunks; recover their fixed policy.
    if type(manifest.get('chunk')) is not int or manifest['chunk'] <= 0:
        return None
    total, limit, tail = manifest['tokens'], manifest['chunk'], manifest.get('decode_tail', 0)
    prefix = max(0, total - tail)
    chunks, position = [], 0
    while position < total:
        count = 1 if tail and position >= prefix else min(limit, (prefix if tail else total) - position)
        chunks.append((position, count))
        position += count
    return chunks


def full_last_layer_before_crop(manifest):
    value = manifest.get('full_last_layer_before_crop')
    if value is not None and type(value) is not bool:
        raise ValueError('invalid last-layer crop policy flag')
    bounded_oracle = manifest.get('engine') == 'llama.cpp-PR31192' and manifest.get('head_tail', 0) > 0
    if bounded_oracle:
        # Old bounded references selected rows before FFN59. Missing metadata
        # must retain that historical meaning rather than silently mean "full".
        return False if value is None else value
    if value is False:
        raise ValueError('last-layer crop policy contradicts native/default-all execution')
    return True


def compare_case(reference: Path, candidate: Path, batch_rows=8):
    a, av = load_case(reference)
    b, bv = load_case(candidate)
    for key in ("model", "tokens", "vocab", "positions", "kv"):
        if a[key] != b[key]:
            raise ValueError(f"{candidate.name}: {key} differs from reference")
    if a.get('head_tail', 0) != b.get('head_tail', 0):
        raise ValueError(f"{candidate.name}: head_tail differs from reference")
    if a.get('head_tail', 0) and execution_chunks(a) != execution_chunks(b):
        raise ValueError(f"{candidate.name}: bounded head actual chunk geometry differs from reference")
    if full_last_layer_before_crop(a) != full_last_layer_before_crop(b):
        raise ValueError(f"{candidate.name}: full last-layer crop policy differs from reference")
    if (reference / "tokens.i32").read_bytes() != (candidate / "tokens.i32").read_bytes():
        raise ValueError(f"{candidate.name}: token IDs differ")
    total_kl = total_sq = max_kl = max_abs = 0.0
    same = count = 0
    bitwise = True
    worst_position = -1
    for start in range(0, av.shape[0], batch_rows):
        bitwise = bitwise and np.array_equal(av[start:start + batch_rows].view("u4"), bv[start:start + batch_rows].view("u4"))
        x = np.asarray(av[start:start + batch_rows], dtype=np.float64)
        y = np.asarray(bv[start:start + batch_rows], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{candidate.name}: non-finite logits at row chunk {start}")
        same += int(np.count_nonzero(x.argmax(1) == y.argmax(1)))
        diff = x - y
        max_abs = max(max_abs, float(np.abs(diff).max()))
        total_sq += float(np.square(diff).sum())
        x -= x.max(1, keepdims=True)
        y -= y.max(1, keepdims=True)
        x -= np.log(np.exp(x).sum(1, keepdims=True))
        y -= np.log(np.exp(y).sum(1, keepdims=True))
        kl = (np.exp(x) * (x - y)).sum(1)
        if not np.isfinite(kl).all() or float(kl.min()) < -1e-10:
            raise ValueError(f"{candidate.name}: invalid KL computation")
        kl = np.maximum(kl, 0)
        k = int(kl.argmax())
        if float(kl[k]) > max_kl:
            max_kl, worst_position = float(kl[k]), a["positions"][start + k]
        total_kl += float(kl.sum())
        count += len(x)
    return dict(bitwise_identical=bitwise, case=candidate.name, tokens=a["tokens"], rows=count, top1_matches=same,
                top1=same / count, mean_kl=total_kl / count, max_kl=max_kl,
                worst_position=worst_position, max_abs=max_abs,
                logits_rms=(total_sq / (count * a["vocab"])) ** .5,
                full_rows=a["positions"] == list(range(a["tokens"])), head_tail=a.get('head_tail', 0),
                full_last_layer_before_crop={'reference': full_last_layer_before_crop(a),
                                             'candidate': full_last_layer_before_crop(b)})


def compare_suite(reference: Path, candidate: Path, top1=.99, mean_kl=.001,
                  max_kl=.05, require_suite=False, oracle_contract="gpu-offload", require_bitwise=False):
    ref = {p.name: p for p in reference.iterdir() if p.is_dir() and (p / "manifest.json").exists()}
    cand = {p.name: p for p in candidate.iterdir() if p.is_dir() and (p / "manifest.json").exists()}
    # Incomplete case directories must not disappear from the comparison silently.
    for root in (reference, candidate):
        for p in root.iterdir():
            if p.is_dir() and not (p / "manifest.json").exists():
                raise ValueError(f"incomplete case: {p}")
    if not ref or ref.keys() != cand.keys():
        raise ValueError("case sets differ or are empty")
    if require_suite:
        for name in ref:
            rm, _ = load_case(ref[name])
            cm, _ = load_case(cand[name])
            if rm["engine"] != "llama.cpp-PR31192" or cm["engine"] != "xolotl-m31":
                raise ValueError("gate20 requires the PR oracle and the independent Xolotl engine")
            if bool(rm.get('oracle_op_offload', False)) != (oracle_contract == "gpu-offload"):
                raise ValueError("oracle placement differs from the selected contract")
            if oracle_contract == "gpu-offload" and (rm.get('cuda_fusion_disabled') is not True or cm.get('cuda_fusion_disabled') is not True):
                raise ValueError("GPU gate20 requires fusion disabled in both engines")
            if execution_chunks(rm) != execution_chunks(cm):
                raise ValueError('gate20 requires matching actual chunk geometry')
    for name in ref:
        rm, _ = load_case(ref[name])
        cm, _ = load_case(cand[name])
        if rm.get("engine") != "llama.cpp-PR31192" or bool(rm.get("oracle_op_offload", False)) != (oracle_contract == "gpu-offload"):
            raise ValueError("reference engine/placement contradicts oracle contract")
        if cm.get("engine") != "xolotl-m31":
            raise ValueError("candidate is not the independent Xolotl engine")
        if oracle_contract == "gpu-offload" and (rm.get("cuda_fusion_disabled") is not True or cm.get("cuda_fusion_disabled") is not True):
            raise ValueError("GPU comparison requires fusion disabled in both engines")
        if rm.get("context") != cm.get("context"):
            from replay_oracle import validate_capacity
            validate_capacity(reference, candidate, name)
        if execution_chunks(rm) != execution_chunks(cm):
            raise ValueError("actual chunk geometry differs from oracle")
        if not full_last_layer_before_crop(rm) or not full_last_layer_before_crop(cm):
            raise ValueError("selected head must follow full final-layer FFN")
    cases = [compare_case(ref[name], cand[name]) for name in sorted(ref)]
    if require_suite:
        groups = {name.split("_", 1)[0] for name in ref}
        if len(cases) != 20 or not {"ru", "en", "code", "math"} <= groups:
            raise ValueError("gate20 requires exactly 20 RU/EN/code/math cases")
        if sum(c["tokens"] for c in cases) != 28723:
            raise ValueError("gate20 requires all 28,723 positions")
        if min(c["tokens"] for c in cases) != 64 or max(c["tokens"] for c in cases) != 3000:
            raise ValueError("gate20 requires the 64..3000-token range")
        if not all(64 <= c["tokens"] <= 3000 and c["full_rows"] and c['head_tail'] == 0 for c in cases):
            raise ValueError("gate20 acceptance requires every row of every prompt")
        if sum(c["tokens"] > 2048 for c in cases) < 4:
            raise ValueError("not enough sparse-selection cases")
    rows = sum(c["rows"] for c in cases)
    result = dict(rows=rows, top1=sum(c["top1_matches"] for c in cases) / rows,
                  mean_kl=sum(c["mean_kl"] * c["rows"] for c in cases) / rows,
                  max_kl=max(c["max_kl"] for c in cases), cases=cases,
                  thresholds=dict(top1=top1, mean_kl=mean_kl, max_kl=max_kl),
                  scope="gate20-complete-logits" if require_suite else "selected-row-diagnostic")
    result["oracle_contract"] = oracle_contract
    result["acceptance_eligible"] = require_suite and oracle_contract == "gpu-offload"
    result["bitwise_identical"] = all(c["bitwise_identical"] for c in cases)
    result["pass"] = (result["top1"] >= top1 and result["mean_kl"] <= mean_kl and result["max_kl"] <= max_kl
                      and (not require_bitwise or result["bitwise_identical"]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--top1", type=float, default=.99)
    p.add_argument("--mean-kl", type=float, default=.001)
    p.add_argument("--max-kl", type=float, default=.05)
    p.add_argument("--require-suite", action="store_true")
    p.add_argument("--oracle-contract", choices=("gpu-offload", "cpu-strict"), default="gpu-offload")
    p.add_argument("--require-bitwise", action="store_true")
    a = p.parse_args()
    if not (0 < a.top1 <= 1 and a.mean_kl > 0 and a.max_kl > 0):
        p.error("invalid thresholds")
    try:
        result = compare_suite(a.reference, a.candidate, a.top1, a.mean_kl, a.max_kl, a.require_suite, a.oracle_contract, a.require_bitwise)
    except (ValueError, KeyError, TypeError, OSError) as e:
        p.exit(2, f"comparison refused: {e}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if a.out:
        with a.out.open("x") as f:
            f.write(text)
    print(text, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
