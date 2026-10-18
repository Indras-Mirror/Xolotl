#!/usr/bin/env python3
"""Compare opt-out DS41 text inference against a separately built baseline (CPU, real model)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engram", required=True, help="colon-separated Engram directories")
    parser.add_argument("--out", type=Path, help="retain logs and logits in this directory")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "DS41_"))}
    env.update(CUDA_VISIBLE_DEVICES="-1", HIP_VISIBLE_DEVICES="-1", STRATA_ENGRAM_CACHE_ROWS="65536",
               OMP_NUM_THREADS=str(args.threads))
    common = ["-m", str(args.model.resolve()), "--engram", args.engram, "--backend", "cpu", "--experts", "cpu",
              "--arena-gib", "0", "--ctx", "4096", "--threads", str(args.threads)]
    ids = "0,123,456,789,1024,2048,4096,8192"

    with tempfile.TemporaryDirectory(prefix="ds41-text-regression-") as temporary:
        out = args.out or Path(temporary)
        out.mkdir(parents=True, exist_ok=True)
        binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
        results = {}

        def execute(label, case, flags, request=None, extra_env=None):
            prefix = out / (label + "-" + case)
            command = [str(binaries[label]), *common, *flags]
            start = time.monotonic()
            with prefix.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, input=request, text=True, stdout=subprocess.PIPE, stderr=log,
                                        env={**env, **(extra_env or {})}, timeout=1200)
            prefix.with_suffix(".stdout").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"{label}/{case} exited {result.returncode}; see {prefix}.log")
            print(f"{label}/{case}: {time.monotonic() - start:.2f} s", flush=True)
            return prefix, result.stdout

        def identical(case, left, right):
            a, b = left.read_bytes(), right.read_bytes()
            if not a or a != b:
                raise RuntimeError(f"{case} differs: {left} vs {right}")
            results[case] = {"bytes": len(a), "sha256": hashlib.sha256(a).hexdigest()}
            print(f"PASS {case}: {len(a)} identical bytes", flush=True)

        for case, batch, chunk in (("sequential", 0, 1), ("chunked", 0, 4), ("layer-major", 8, 4)):
            outputs = {}
            for label in binaries:
                prefix = out / (label + "-" + case)
                _, text = execute(label, case, ["--ids", ids, "-n", "3", "--temp", "0.8", "--seed", "17",
                    "--prompt-batch", str(batch), "--prefill-chunk", str(chunk),
                    "--dump-logits", str(prefix.with_suffix(".logits")), "--dump-last", str(prefix.with_suffix(".last"))])
                outputs[label] = prefix
                if not text.strip() or not all(line.isdigit() for line in text.splitlines()):
                    raise RuntimeError(f"unexpected token output: {label}/{case}")
            for suffix in (".logits", ".last", ".stdout"):
                identical(case + suffix, outputs["baseline"].with_suffix(suffix), outputs["candidate"].with_suffix(suffix))

        requests = (f"GEN 2 temperature=0.8 seed=17 {ids}\n" * 2 +
                    f"GEN 2 temperature=0.8 seed=17 {ids},17,19\nQUIT\n")
        protocols = {}
        for label in binaries:
            dumps = out / (label + "-serve-logits")
            dumps.mkdir(exist_ok=True)
            _, text = execute(label, "serve", ["--serve", "--prompt-batch", "16", "--prefill-chunk", "4"],
                              request=requests, extra_env={"STRATA_SERVE_DUMP": str(dumps)})
            if any(line.startswith("ERR") for line in text.splitlines()):
                raise RuntimeError(text)
            protocols[label] = [line for line in text.splitlines() if line.startswith(("T ", "RESUME "))]
            if len(list(dumps.glob("req*.f32"))) != 3:
                raise RuntimeError(f"{label}: expected three prompt logit dumps")
        if not protocols["baseline"] or protocols["baseline"] != protocols["candidate"]:
            raise RuntimeError("serve token/reuse protocol differs")
        results["serve_protocol"] = protocols["candidate"]
        for index in range(3):
            name = f"req{index:03d}.f32"
            identical("serve/" + name, out / "baseline-serve-logits" / name, out / "candidate-serve-logits" / name)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        print("PASS: vision-disabled logits, token IDs, regeneration and continuation match the baseline byte for byte")


if __name__ == "__main__":
    main()
