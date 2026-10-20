#!/usr/bin/env python3
"""restructure_v2.py - after the 1017 upload is complete (2026-10-19, Mal): move the 1017 build into v2/ under the XQ
name and rename the repo from ...-GSQ-... to ...-XQ-....  Server-side copies (no re-upload), one atomic commit.

    python3 tools/v41/hf-1017/restructure_v2.py            # dry run: prints the plan
    python3 tools/v41/hf-1017/restructure_v2.py --apply    # commit + rename

root keeps what both builds share: README.md, tokenizer/, the two sidecars, the two NEED-TO-DOWNLOAD-ENGRAM notes.
v2/ gets: V41-XQ-1017.gguf.part-NN-of-15, V41-XQ-1017.gguf.experts.json, rank-1017-085.bin, SHA256SUMS (same hashes,
new names), reassemble.sh (out = V41-XQ-1017.gguf).
"""
import argparse, os, re, sys
from huggingface_hub import HfApi, CommitOperationCopy, CommitOperationDelete, CommitOperationAdd

OLD_REPO = "IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored"
NEW_REPO = "IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored"
HERE = os.path.dirname(os.path.abspath(__file__))
SUMS = os.path.expanduser("~/AI/Strata-Backup/v41-dev/hf-1017/SHA256SUMS")
OLD, NEW = "V41-GSQ-1017.gguf", "V41-XQ-1017.gguf"
MOVE = {"rank-1017-085.bin": "v2/rank-1017-085.bin", f"{OLD}.experts.json": f"v2/{NEW}.experts.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    files = {f.path: getattr(f, "size", None) for f in api.list_repo_tree(OLD_REPO, recursive=True)}
    parts = sorted(p for p in files if re.fullmatch(rf"{re.escape(OLD)}\.part-\d\d-of-\d\d", p))
    n = int(parts[0].rsplit("-of-", 1)[1]) if parts else 0
    if not parts or len(parts) != n:
        sys.exit(f"upload not complete: {len(parts)} parts on the hub, expected {n}")
    for side in ("V41-MTP-Vision-1017.gguf", "V41-MTP-Vision-1017.Q2_K.gguf"):
        if side not in files:
            sys.exit(f"upload not complete: {side} missing")
    moves = {p: "v2/" + p.replace(OLD, NEW) for p in parts}
    moves.update({k: v for k, v in MOVE.items() if k in files})
    sums = open(SUMS).read().replace(OLD, NEW)
    reasm = open(os.path.join(HERE, "reassemble.sh")).read().replace(OLD, NEW).replace("V41-GSQ", "V41-XQ")
    ops = [CommitOperationCopy(src_path_in_repo=s, path_in_repo=d) for s, d in moves.items()]
    ops += [CommitOperationAdd(path_in_repo="v2/SHA256SUMS", path_or_fileobj=sums.encode()),
            CommitOperationAdd(path_in_repo="v2/reassemble.sh", path_or_fileobj=reasm.encode())]
    ops += [CommitOperationDelete(path_in_repo=s) for s in moves] + \
           [CommitOperationDelete(path_in_repo=p) for p in ("SHA256SUMS", "reassemble.sh") if p in files]
    for s, d in moves.items():
        print(f"  {s} -> {d}")
    print(f"  + v2/SHA256SUMS ({sums.count(NEW)} names), v2/reassemble.sh;  - root SHA256SUMS, reassemble.sh")
    print(f"  rename {OLD_REPO} -> {NEW_REPO}")
    if not a.apply:
        print("dry run (pass --apply)")
        return
    api.create_commit(OLD_REPO, operations=ops, commit_message="v2/: the 1017 build under the XQ name (server-side move)")
    after = {f.path: getattr(f, "size", None) for f in api.list_repo_tree(OLD_REPO, recursive=True)}
    bad = [d for s, d in moves.items() if after.get(d) != files[s]]
    if bad:
        sys.exit(f"size check failed after the move: {bad[:3]}")
    print("moved + size-checked")
    api.move_repo(from_id=OLD_REPO, to_id=NEW_REPO)
    print("renamed; old URL redirects")


if __name__ == "__main__":
    main()
