"""hf_upload.py - resumable upload of the V4.1 GSQ 1017 release to the Hugging Face Hub (~a day at home upload speeds).

    python3 tools/v41/hf-1017/hf_upload.py --dry-run      # show the plan, touch nothing remote, hash nothing
    bash tools/v41/hf-1017/upload-start.sh                # run it detached (survives closing the terminal / Claude)
    tail -f ~/AI/Strata-Backup/v41-dev/hf-1017/upload.log  # watch it;  bash tools/v41/hf-1017/upload-stop.sh  to pause

Resumable: every run lists what is already on the Hub and skips it.  The unit of work is one ~10 GiB part
(V41-GSQ-1017.gguf.part-NN-of-MM); an interrupted part is uploaded again (hf_xet usually skips the chunks it already has).

SAFETY - the source GGUF is precious and is only ever READ:
  - it is opened O_RDONLY and read with pread(); nothing in SRC_DIR is written, renamed or deleted;
  - parts are cut into STAGE (another drive), which must not be inside SRC_DIR; the only files this script deletes
    are its own staged parts, matched by exact name inside STAGE;
  - if the GGUF's size or mtime differs from the state file's record, the run stops (no mixing of two versions).
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = "/media/mal/SSD NVME/AI/V41-GSQ-1017"
GGUF = os.path.join(SRC_DIR, "V41-GSQ-1017.gguf")
SMALL = [os.path.join(SRC_DIR, "rank-1017-085.bin"), os.path.join(SRC_DIR, "V41-GSQ-1017.gguf.experts.json")]
TOKENIZER = "/media/mal/SSD NVME/AI/V41-GSQ/tokenizer"
SIDECARS = [
    "/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision-1017.gguf",
    "/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision-1017.Q2_K.gguf",
]
STAGE = "/media/mal/NVME1TB/hf-staging-1017"
WORK = os.path.expanduser("~/AI/Strata-Backup/v41-dev/hf-1017")
STATE = os.path.join(WORK, "state.json")
REPO = "IndrasMirror/DeepSeek-V4.1-Flash-1017-GSQ-Uncensored"
PART = 10 << 30
CHUNK = 64 << 20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def part_name(i, n):
    return f"V41-GSQ-1017.gguf.part-{i + 1:02d}-of-{n:02d}"


def load_state():
    try:
        return json.load(open(STATE))
    except FileNotFoundError:
        return {}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def safety_checks():
    src = os.path.realpath(SRC_DIR)
    stage = os.path.realpath(STAGE)
    if stage == src or stage.startswith(src + os.sep):
        raise SystemExit(f"refusing: staging dir {stage} is inside the source dir {src}")
    if not os.path.isfile(GGUF):
        raise SystemExit(f"{GGUF} not found (is 'SSD NVME' mounted?  udisksctl mount -b /dev/nvme0n1p3)")


def source_id():
    s = os.stat(GGUF)
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns}


def hash_pass(size, n):
    """One read-only pass: sha256 of every part and of the whole file."""
    full = hashlib.sha256()
    parts = []
    fd = os.open(GGUF, os.O_RDONLY)
    try:
        t0, off = time.time(), 0
        for i in range(n):
            h = hashlib.sha256()
            end = min(size, (i + 1) * PART)
            while off < end:
                b = os.pread(fd, min(CHUNK, end - off), off)
                if not b:
                    raise OSError("short read")
                h.update(b)
                full.update(b)
                off += len(b)
            parts.append(h.hexdigest())
            log(f"  hashed {part_name(i, n)} ({off / 1e9:.0f}/{size / 1e9:.0f} GB, {off / 1e6 / (time.time() - t0):.0f} MB/s)")
    finally:
        os.close(fd)
    return full.hexdigest(), parts


def stage_part(i, n, size, want_sha):
    """Copy part i (a byte range of the read-only GGUF) into STAGE; verify its sha256 while writing."""
    name = part_name(i, n)
    dst = os.path.join(STAGE, name)
    tmp = dst + ".tmp"
    a, b = i * PART, min(size, (i + 1) * PART)
    h = hashlib.sha256()
    fd = os.open(GGUF, os.O_RDONLY)
    try:
        with open(tmp, "wb") as out:
            off = a
            while off < b:
                buf = os.pread(fd, min(CHUNK, b - off), off)
                if not buf:
                    raise OSError("short read")
                h.update(buf)
                out.write(buf)
                off += len(buf)
    finally:
        os.close(fd)
    if h.hexdigest() != want_sha:
        remove_staged(tmp)
        raise SystemExit(f"{name}: staged copy hash mismatch - the source changed or a read failed; stopping")
    os.replace(tmp, dst)
    return dst


STAGED_RE = re.compile(r"^V41-GSQ-1017\.gguf\.part-\d{2}-of-\d{2}(\.tmp)?$")


def remove_staged(path):
    """Delete a staged part - only ever a part-named file directly inside STAGE."""
    p = os.path.realpath(path)
    if os.path.dirname(p) != os.path.realpath(STAGE) or not STAGED_RE.match(os.path.basename(p)):
        raise SystemExit(f"refusing to delete {p}: not a staged part")
    os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the repo private (default public - Mal 2026-10-12)")
    a = ap.parse_args()
    safety_checks()
    os.makedirs(WORK, exist_ok=True)
    lock = open(os.path.join(WORK, "upload.lock"), "a+")   # not "w": that would wipe a running instance's pid
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another hf_upload.py is running")
    if not a.dry_run:
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import HfApi
    api = HfApi()
    sid = source_id()
    size = sid["size"]
    n = math.ceil(size / PART)
    st = load_state()
    if st.get("source") and st["source"] != sid:
        raise SystemExit(f"the GGUF changed since the state was written ({st['source']} -> {sid}); "
                         f"move {STATE} away to start over")

    remote = {}
    if api.repo_exists(REPO):
        for f in api.list_repo_tree(REPO, recursive=True):
            if getattr(f, "size", None) is not None:
                remote[f.path] = f.size
    log(f"{REPO}: {'exists' if remote or api.repo_exists(REPO) else 'not created yet'}, {len(remote)} files there")
    log(f"source {GGUF}: {size} bytes -> {n} parts of {PART >> 30} GiB")
    todo = [part_name(i, n) for i in range(n) if remote.get(part_name(i, n)) != min(PART, size - i * PART)]
    log(f"parts still to upload: {len(todo)} of {n}")
    if a.dry_run:
        for i in range(n):
            nm = part_name(i, n)
            log(f"  {nm}  bytes {i * PART}..{min(size, (i + 1) * PART)}  {'on hub' if nm not in todo else 'TODO'}")
        log(f"small files: {[os.path.basename(p) for p in SMALL]} + tokenizer/ + README.md + reassemble.sh + SHA256SUMS"
            f" + NEED-TO-DOWNLOAD-ENGRAM notes")
        for s in SIDECARS:
            log(f"sidecar: {s} ({os.path.getsize(s) / 1e9:.2f} GB) {'on hub' if os.path.basename(s) in remote else 'TODO'}")
        log(f"staging dir: {STAGE} (free {os.statvfs(os.path.dirname(STAGE)).f_bavail * os.statvfs(os.path.dirname(STAGE)).f_frsize / 1e9:.0f} GB)")
        return

    if "parts_sha256" not in st:
        log("hashing the source once (read-only) ...")
        full, parts = hash_pass(size, n)
        st = {"source": sid, "part_bytes": PART, "full_sha256": full, "parts_sha256": parts}
        save_state(st)
    sums = "".join(f"{h}  {part_name(i, n)}\n" for i, h in enumerate(st["parts_sha256"])) + \
        f"{st['full_sha256']}  V41-GSQ-1017.gguf\n"
    os.makedirs(STAGE, exist_ok=True)
    with open(os.path.join(WORK, "SHA256SUMS"), "w") as f:
        f.write(sums)

    api.create_repo(REPO, private=a.private, exist_ok=True)   # exist_ok: an existing repo keeps its visibility

    def put(path, name):
        if remote.get(name) == os.path.getsize(path) and name not in ("README.md", "SHA256SUMS", "reassemble.sh"):
            return
        t0 = time.time()
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=REPO, commit_message=f"add {name}")
        dt = time.time() - t0
        log(f"uploaded {name} ({os.path.getsize(path) / 1e6:.1f} MB in {dt:.0f} s, {os.path.getsize(path) / 1e6 / max(dt, 1e-3):.2f} MB/s)")

    put(os.path.join(HERE, "README.md"), "README.md")
    put(os.path.join(HERE, "reassemble.sh"), "reassemble.sh")
    put(os.path.join(WORK, "SHA256SUMS"), "SHA256SUMS")
    for nm in ("model-00047-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt",
               "model-00048-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt"):
        put(os.path.join(HERE, nm), nm)
    for p in SMALL:
        put(p, os.path.basename(p))
    for f in sorted(os.listdir(TOKENIZER)):
        put(os.path.join(TOKENIZER, f), f"tokenizer/{f}")

    for i in range(n):
        name = part_name(i, n)
        if name not in todo:
            continue
        for leftover in (name, name + ".tmp"):   # a staged copy from an interrupted run is re-cut (re-verified)
            if os.path.exists(os.path.join(STAGE, leftover)):
                remove_staged(os.path.join(STAGE, leftover))
        log(f"staging {name} ...")
        path = stage_part(i, n, size, st["parts_sha256"][i])
        try:
            put(path, name)
        finally:
            remove_staged(path)
        info = api.get_paths_info(REPO, [name])
        got = info[0].size if info else None
        want = min(PART, size - i * PART)
        if got != want:
            raise SystemExit(f"{name}: hub reports {got} bytes, expected {want}; stopping")
        log(f"{name} done ({i + 1}/{n})")

    for s in SIDECARS:
        put(s, os.path.basename(s))
    log("ALL DONE - every part, both sidecars and the small files are on the hub")


if __name__ == "__main__":
    main()
