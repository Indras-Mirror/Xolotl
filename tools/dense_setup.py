#!/usr/bin/env python3
"""tools/dense_setup.py - get Qwen3.8-27B (a DENSE qwen35 GGUF) running on Strata.

    python tools/dense_setup.py --gguf /path/Qwen3.8-27B-UD-Q4_K_M.gguf --build --arch 86 --context 32768
    python tools/dense_setup.py --download UD-Q4_K_M --build --arch 86

No pack step: `xolotl-dense` reads the GGUF blocks as they are (every type the native GEMVs take, IQ3_S included),
so there is nothing to convert.  What this does:

  1. (--download) fetches the GGUF from huggingface.co/unsloth/Qwen3.8-27B-GGUF
  2. exports the tokenizer + chat template from the GGUF (tools/strata_tokenizer.py) into data/packs/<name>/tokenizer
  3. (--build) builds the `xolotl-dense` target with CUDA
  4. writes strata-qwen3.8-27b.json, the run config serve/server.py (and START-HERE) reads

Then:   python serve/server.py --engine strata --config strata-qwen3.8-27b.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == "nt"
REPO = "unsloth/Qwen3.8-27B-GGUF"
EXE = "xolotl-dense.exe" if WIN else "xolotl-dense"


def say(msg: str) -> None:
    print(msg, flush=True)


def run(cmd, **kw) -> None:
    say("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def download(quant: str, dst_dir: Path) -> Path:
    name = f"Qwen3.8-27B-{quant}.gguf"
    dst = dst_dir / name
    if dst.exists():
        say(f"  {name} is already there")
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(REPO, name, local_dir=str(dst_dir)))
    except ImportError:
        pass
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    say(f"  downloading {url}")
    part = dst.with_suffix(".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r, open(part, "ab" if done else "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.rename(dst)
    return dst


def build(arch: str, build_dir: Path) -> Path:
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("cmake not found (pip install cmake ninja, or install it from your package manager)")
    gen = ["-G", "Ninja"] if shutil.which("ninja") else []
    run([cmake, *gen, "-S", ROOT, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}"])
    run([cmake, "--build", build_dir, "--target", "xolotl-dense", "-j", str(max(2, (os.cpu_count() or 4) // 2))])
    for cand in (build_dir / EXE, build_dir / "Release" / EXE):
        if cand.exists():
            return cand
    sys.exit(f"built, but {EXE} is not in {build_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="a Qwen3.8-27B GGUF you already have (first shard if split)")
    src.add_argument("--download", metavar="QUANT", help="fetch Qwen3.8-27B-<QUANT>.gguf, e.g. UD-Q4_K_M")
    ap.add_argument("--models-dir", default=str(ROOT / "data" / "models"))
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--build", action="store_true", help="build xolotl-dense (CUDA)")
    ap.add_argument("--arch", default="native", help="CMAKE_CUDA_ARCHITECTURES, e.g. 86 (RTX 30), 89 (RTX 40), 120 (RTX 50)")
    ap.add_argument("--kv", choices=["fp16", "int8", "q4_0", "k8v4"], default="fp16",
                    help="KV cache storage, the same option as strata's setup: fp16 (default), int8 (~53%% of the memory), "
                         "q4_0 (~28%%, Hadamard-rotated 4-bit) or k8v4 (int8 keys + 4-bit values, ~40%%). Per 1K tokens of "
                         "this model (17 attention layers incl. MTP): fp16 ~ 71 MB, int8 ~ 38, k8v4 ~ 29, q4_0 ~ 20. "
                         "The MMQ prompt prefill reads whichever format you pick")
    ap.add_argument("--mtp", action="store_true",
                    help="speculative decoding with the model's own MTP block (it is inside the GGUF; ~1.4 GB more VRAM)")
    ap.add_argument("--draft-max", type=int, default=2, metavar="N",
                    help="MTP: most tokens proposed per step, 1..7 (default 2). Each unit costs ~156 MB of VRAM for rollback")
    ap.add_argument("--draft-min", type=int, default=1, metavar="N",
                    help="MTP: speculate only when at least N tokens were proposed, 1..draft-max (default 1)")
    ap.add_argument("--draft-p-min", type=float, default=0.0, metavar="P",
                    help="MTP: stop drafting at a token the head gives less than this probability, 0..1 (default 0)")
    ap.add_argument("--mtp-force", action="store_true",
                    help="MTP: keep it on even when the weights do not all fit in VRAM (by default it switches itself off)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--exe", help="use this xolotl-dense instead of building")
    a = ap.parse_args()
    if not 1 <= a.draft_max <= 7:
        sys.exit("--draft-max must be 1..7")
    if not 1 <= a.draft_min <= a.draft_max:
        sys.exit(f"--draft-min must be 1..--draft-max ({a.draft_max})")
    if not 0.0 <= a.draft_p_min <= 1.0:
        sys.exit("--draft-p-min must be 0..1")

    gguf = Path(a.gguf) if a.gguf else download(a.download, Path(a.models_dir))
    if not gguf.exists():
        sys.exit(f"{gguf} does not exist")
    pack = ROOT / "data" / "packs" / "qwen3.8-27b"
    if not (pack / "tokenizer" / "vocab.json").exists():
        say("Exporting the tokenizer and chat template from the GGUF ...")
        run([sys.executable, ROOT / "tools" / "strata_tokenizer.py", "--gguf", gguf, "--out", pack])

    if a.exe:
        exe = Path(a.exe)
    elif a.build:
        exe = build(a.arch, ROOT / "build-dense")
    else:
        exe = ROOT / "build-dense" / EXE
        if not exe.exists():
            say(f"NOTE: {exe} does not exist yet - run again with --build")

    cfg = {
        "exe": str(exe), "args": ["--native", str(gguf), "--context", str(a.context)] +
            (["--kv", a.kv] if a.kv != "fp16" else []) + (
            ["--mtp", "--draft-max", str(a.draft_max), "--draft-min", str(a.draft_min), "--draft-p-min", str(a.draft_p_min)]
            + (["--mtp-force"] if a.mtp_force else []) if a.mtp else []), "cwd": str(ROOT),
        "tokenizer": str(pack / "tokenizer"), "model_name": "qwen3.8-27b", "port": a.port,
        "log": str(ROOT / "strata-qwen3.8-27b.log"), "lib_dirs": [],
        # the model card's recommended sampling for thinking mode; the request's own values win
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    }
    cfg_path = ROOT / "strata-qwen3.8-27b.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    say(f"\nConfig written: {cfg_path}")
    say(f"Start it:  {Path(sys.executable).name} serve/server.py --engine strata --config {cfg_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
