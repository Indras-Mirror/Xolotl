"""Fetch named files of a Hugging Face repo at a pinned revision: resumable, one curl attempt per run (the loop
re-reads the size; curl's own --retry with -C - can append bytes twice), checked against the LFS sha256.

  python win_fetch_files.py <repo> <revision> <dest_dir> <path-in-repo> [<path-in-repo> ...]
Files land in <dest_dir>/<basename>.  Uses a generic User-Agent and no credentials.
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

UA = "xolotl-win-fetch/1.0"
repo, rev, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
names = sys.argv[4:]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


url = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
    info = {s["rfilename"]: s for s in json.loads(r.read().decode()).get("siblings") or []}
dest.mkdir(parents=True, exist_ok=True)
for name in names:
    s = info[name]
    size, want = s.get("size") or 0, (s.get("lfs") or {}).get("sha256")
    dst = dest / Path(name).name
    for attempt in range(1, 300):
        have = dst.stat().st_size if dst.exists() else 0
        if have == size:
            break
        if have > size:
            log(f"{dst.name}: {have} bytes is more than {size}: starting over")
            dst.unlink()
            have = 0
        log(f"get {dst.name}: {have / 1e9:.1f}/{size / 1e9:.1f} GB (attempt {attempt})")
        rc = subprocess.run(["curl", "-L", "--fail", "-sS", "-A", UA, "--speed-limit", "1000000", "--speed-time", "60",
                             "-C", "-", "-o", str(dst), f"https://huggingface.co/{repo}/resolve/{rev}/{name}"]).returncode
        if rc != 0:
            time.sleep(min(60, 5 * attempt))
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    if want and h.hexdigest() != want:
        dst.unlink()
        raise SystemExit(f"{dst.name}: sha256 mismatch (removed; run again)")
    log(f"ok  {dst.name} ({size / 1e9:.1f} GB, sha256 verified)")
log("DONE")
