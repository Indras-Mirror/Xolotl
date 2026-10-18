"""Windows (or any OS) download + join of the Xolotl DeepSeek-V4.1 GSQ model without the hf CLI or bash.

Downloads, resumable with curl (-C -, retried until complete), every file below at a pinned revision, checks each
against Hugging Face's LFS sha256, then joins the 16 parts into V41-GSQ.gguf (deleting each part once appended,
like `reassemble.sh --delete-parts`) and checks the joined file against SHA256SUMS.

  python win_fetch_v41.py <dest_dir>            e.g. D:\\models\\dsv41-xolotl
Sends no credentials and no personal data; uses a generic User-Agent.
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

UA = "xolotl-win-fetch/1.0"
GSQ = ("IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored", "6992cf5e7a170c667b5f0b9eb2671b103562694a")
ENGRAM_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
DEST = Path(sys.argv[1])


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def api(repo, rev=None):
    url = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{rev}" if rev else "") + "?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        return json.loads(r.read().decode())


def sha256(p, chunk=16 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def fetch(repo, rev, name, size, want_sha, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok_mark = dst.with_name(dst.name + ".ok")
    if ok_mark.exists() and dst.exists() and dst.stat().st_size == size:
        log(f"have {name}")
        return
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    for attempt in range(1, 200):
        have = dst.stat().st_size if dst.exists() else 0
        if have == size:
            break
        if have > size:
            dst.unlink()
        log(f"get {name}: {have / 1e9:.1f}/{size / 1e9:.1f} GB (attempt {attempt})")
        # one attempt per curl run: curl's own --retry with -C - can resume from the first run's offset and append
        # the same bytes twice (seen on a 47 GB file that came out 54.9 GB); this loop re-reads the size instead
        rc = subprocess.run(["curl", "-L", "--fail", "-sS", "-A", UA, "--speed-limit", "1000000", "--speed-time", "60",
                             "-C", "-", "-o", str(dst), url]).returncode
        if rc != 0:
            time.sleep(min(60, 5 * attempt))
    if not dst.exists() or dst.stat().st_size != size:
        raise SystemExit(f"{name}: download incomplete")
    if want_sha:
        got = sha256(dst)
        if got != want_sha:
            dst.unlink()
            raise SystemExit(f"{name}: sha256 mismatch (file removed; run again)")
    ok_mark.write_text(want_sha or "size-only", encoding="ascii")
    log(f"ok  {name} ({size / 1e9:.1f} GB{', sha256 verified' if want_sha else ''})")


def lfs_files(info):
    out = {}
    for s in info.get("siblings") or []:
        lfs = s.get("lfs") or {}
        out[s["rfilename"]] = (s.get("size") or 0, lfs.get("sha256"))
    return out


gsq = lfs_files(api(*GSQ))
gdir = DEST / "V41-GSQ"
skip = {"V41-MTP-Vision.gguf"}          # the Q2_K sidecar is the recommended DSpark file
for name, (size, sh) in sorted(gsq.items()):
    if name in skip or name.endswith(".NEED-TO-DOWNLOAD-ENGRAM.txt"):
        continue
    fetch(GSQ[0], GSQ[1], name, size, sh, gdir / name)

eng_info = api(ENGRAM_REPO)
eng_rev = eng_info["sha"]
eng = lfs_files(eng_info)
edir = DEST / "FP8-engram"
for name in ["config.json", "model.safetensors.index.json", "model-00047-of-00048.safetensors",
             "model-00048-of-00048.safetensors"]:
    size, sh = eng[name]
    fetch(ENGRAM_REPO, eng_rev, name, size, sh, edir / name)
(edir / "REVISION.txt").write_text(f"{ENGRAM_REPO}@{eng_rev}\n", encoding="ascii")

# join the parts (reassemble.sh --delete-parts), then check SHA256SUMS
joined = gdir / "V41-GSQ.gguf"
parts = sorted(p for p in gdir.glob("V41-GSQ.gguf.part-*-of-*") if not p.name.endswith(".ok"))
sums = {}
for line in (gdir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    if line.strip():
        h, n = line.split(None, 1)
        sums[n.strip().lstrip("*")] = h
if parts:
    log(f"joining {len(parts)} parts into {joined.name}")
    with open(joined, "ab") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(64 << 20)
                    if not b:
                        break
                    out.write(b)
            p.unlink()
            (p.with_name(p.name + ".ok")).unlink(missing_ok=True)
            log(f"appended + deleted {p.name}")
want = sums.get("V41-GSQ.gguf")
if want:
    got = sha256(joined)
    log(f"V41-GSQ.gguf sha256 {'OK' if got == want else 'MISMATCH ' + got}")
    if got != want:
        raise SystemExit(2)
log("DONE")
