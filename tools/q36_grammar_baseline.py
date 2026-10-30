#!/usr/bin/env python3
"""Build the pre-grammar q36 translation unit with the current build's flags/libs.

Writes only to --out. Run without a concurrent benchmark, before acquiring a GPU.
This avoids relying on a binary from another worktree for the regression check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-ref", "--revision", dest="revision", required=True,
                        help="git revision containing the pre-change q36 engine")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "baseline_q36_engine.cu"
    code = subprocess.check_output(["git", "show", f"{args.revision}:src/qwen35/q36_engine.cu"], cwd=ROOT)
    source.write_bytes(code)
    obj = args.out / "baseline_q36_engine.o"
    exe = args.out / "baseline-strata-q36"

    def command(target):
        return subprocess.check_output(["ninja", "-C", str(args.build), "-t", "commands", target], text=True).splitlines()[-1]

    compile_args = shlex.split(command("CMakeFiles/strata_qwen35.dir/src/qwen35/q36_engine.cu.o"))
    assert str(ROOT / "src/qwen35/q36_engine.cu") in compile_args, "unexpected compile command"
    compile_args = [arg for arg in compile_args if arg != "-DSTRATA_Q36_GRAMMAR=1"]
    for i, arg in enumerate(compile_args):
        if arg == str(ROOT / "src/qwen35/q36_engine.cu"):
            compile_args[i] = str(source)
        if arg in ("-o", "-MF", "-MT"):
            compile_args[i + 1] = str(obj) + (".d" if arg == "-MF" else "")
    subprocess.run(compile_args, cwd=args.build, check=True)
    raw = command("strata-q36")
    if raw.startswith(": && ") and raw.endswith(" && :"):
        raw = raw[5:-5]
    link_args = shlex.split(raw)
    assert "libstrata_qwen35.a" in link_args, "unexpected link command"
    link_args = [str(obj) if arg == "libstrata_qwen35.a" else arg for arg in link_args
                 if not arg.startswith("-Wl,--dependency-file=")]
    link_args[link_args.index("-o") + 1] = str(exe)
    subprocess.run(link_args, cwd=args.build, check=True)
    info = {"revision": args.revision, "source_sha256": hashlib.sha256(code).hexdigest(),
            "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            "compile": compile_args, "link": link_args}
    (args.out / "baseline-build.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Baseline built:", exe, flush=True)


if __name__ == "__main__":
    main()
