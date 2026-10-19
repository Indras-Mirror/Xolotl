"""Join a byte-split model (<name>.part-NN-of-MM) into <name>, hashing while writing, and check it against SHA256SUMS -
reassemble.sh for Windows (or any OS) without bash.  Writes <name>.partial and renames it only when the hash matches.
The parts are kept (delete them yourself once the joined file is verified).

  python win_join_parts.py <dir> <name>        e.g. python win_join_parts.py D:\\models\\dsv41-1017-gsq V41-GSQ-1017.gguf
"""
import hashlib
import os
import sys
import time
from pathlib import Path

d, name = Path(sys.argv[1]), sys.argv[2]
out = d / name
sums = {}
for ln in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    if ln.strip():
        h, n = ln.split(maxsplit=1)
        sums[n.strip().lstrip("*")] = h
parts = sorted(p for p in os.listdir(d) if p.startswith(name + ".part-"))
if not parts:
    sys.exit(f"no {name}.part-* files in {d}")
total_n = int(parts[0].rsplit("-of-", 1)[1])
if len(parts) != total_n:
    sys.exit(f"found {len(parts)} parts, expected {total_n}")
if out.exists():
    sys.exit(f"{out} already exists - move it away first")
size = sum((d / p).stat().st_size for p in parts)
sha, done, t0 = hashlib.sha256(), 0, time.time()
with open(str(out) + ".partial", "wb") as w:
    for p in parts:
        with open(d / p, "rb") as r:
            while b := r.read(64 << 20):
                sha.update(b)
                w.write(b)
                done += len(b)
        print(f"{p}: {done / 1e9:.1f} / {size / 1e9:.1f} GB", flush=True)
got = sha.hexdigest()
if sums.get(name) and got != sums[name]:
    sys.exit(f"HASH MISMATCH: {got} != {sums[name]} (left as {name}.partial)")
os.replace(str(out) + ".partial", out)
print(f"OK: {name} {'sha256 verified' if sums.get(name) else '(no SHA256SUMS entry to check)'} "
      f"({size / 1e9:.1f} GB, {time.time() - t0:.0f} s)")
