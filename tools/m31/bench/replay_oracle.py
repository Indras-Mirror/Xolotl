#!/usr/bin/env python3
"""Run the separate PR #31192 oracle with the candidate's actual chunk geometry.

Run this only in an exclusive GPU measurement window. No system configuration is
changed. The oracle executable must come from tests/oracle and the pinned PR.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def validate_capacity(reference, candidate, case):
    """Prove that a smaller physical oracle KV allocation preserved active geometry."""
    native = json.loads((candidate / case / "manifest.json").read_text())
    oracle_dir = reference / case
    oracle = json.loads((oracle_dir / "manifest.json").read_text())
    if native["context"] == oracle["context"]:
        return
    trace = [json.loads(line) for line in (oracle_dir / "capacity-equivalence.jsonl").read_text().splitlines()]
    chunks = native["actual_chunks"]
    if oracle["context"] != native["tokens"] or len(trace) != len(chunks):
        raise ValueError("invalid bounded-capacity metadata")
    for item, chunk in zip(trace, chunks):
        end = chunk["position"] + chunk["tokens"]
        expected = {"position": chunk["position"], "tokens": chunk["tokens"], "seq_min": 0,
            "seq_max": end - 1, "active_keys_padded_from_verified_positions": ((end + 255) // 256) * 256,
            "physical_context_requested": native["tokens"],
            "physical_context_actual": ((native["tokens"] + 255) // 256) * 256,
            "rope_ext_factor": 0, "rope_orig_context": 1048576, "rope_frequency_scale": 1}
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("bounded oracle changed active KV geometry or trained RoPE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--capacity", choices=("same", "input-length"), default="same",
                   help="input-length reduces physical allocation, verifies active KV and trained RoPE")
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    a.out.mkdir(parents=True, exist_ok=False)
    reference = a.out / "reference"
    reference.mkdir()
    cases = sorted(path for path in a.candidate.iterdir() if path.is_dir())
    if not cases:
        p.error("candidate has no cases")
    for case in cases:
        m = json.loads((case / "manifest.json").read_text())
        chunks = m["actual_chunks"]
        end = 0
        for chunk in chunks:
            if chunk["position"] != end or chunk["tokens"] <= 0:
                raise ValueError("noncontiguous candidate chunks")
            end += chunk["tokens"]
        if end != m["tokens"]:
            raise ValueError("incomplete candidate chunk coverage")
        plan = a.out / (case.name + ".tsv")
        plan.write_text("".join(f"{c['position']}\t{c['tokens']}\n" for c in chunks))
        output = a.out / (case.name + "-oracle")
        command = [str(a.oracle.resolve()), "--model", str(a.model.resolve()), "--ids-file", str((a.ids / (case.name + ".i32")).resolve()),
            "--out", str(output.resolve()), "--backend", "cuda", "--ctx", str(m["tokens"] if a.capacity == "input-length" else m["context"]),
            "--chunk", str(m["chunk"]), "--threads", str(a.threads), "--rows", m.get("row_selection", "all"),
            "--oracle-op-offload", "1", "--taps", "none"]
        for key, flag in (("head_tail", "--head-tail"), ("decode_tail", "--decode-tail")):
            if m.get(key):
                command.extend([flag, str(m[key])])
        env = dict(os.environ, GGML_CUDA_DISABLE_FUSION="1", M31_ORACLE_CHUNK_PLAN=str(plan.resolve()))
        env.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
        env.pop("CUDA_DEVICE_MAX_COPY_CONNECTIONS", None)
        if a.capacity == "input-length":
            env["M31_ORACLE_CAPACITY_EQUIVALENCE"] = "1"
        else:
            env.pop("M31_ORACLE_CAPACITY_EQUIVALENCE", None)
        (a.out / (case.name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
        with (a.out / (case.name + ".log")).open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Retain separate logs/work directories; merge only the completed case.
        (reference / case.name).symlink_to((output / case.name).resolve(), target_is_directory=True)
        validate_capacity(reference, a.candidate, case.name)
    command = [sys.executable, str(Path(__file__).with_name("compare_logits.py")), str(reference), str(a.candidate),
               "--out", str(a.out / "compare.json")]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
