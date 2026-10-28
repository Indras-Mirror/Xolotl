#!/usr/bin/env python3
"""Range backup of GGUF tensor payloads, and the matching restore.

An in-place GGUF edit rewrites only the byte ranges of the tensors it selects.
Copying those ranges to another physical disk is therefore a complete rollback
for the edit, at ~1/4 the bytes of a whole-file copy. `--restore` puts them back
and proves the result by re-reading, so a restore is verified, not assumed.

The manifest is the same JSON the editor writes:
    [{"name": str, "offset": int, "nbytes": int, "type": str}, ...]

  backup:  tensor_backup.py --model M --manifest F --out DIR   [--verify]
  restore: tensor_backup.py --model M --manifest F --from DIR  [--verify]
"""
import argparse
import hashlib
import json
import os
import pathlib


def chunk():
    return 16 << 20


def read_range(f, offset, nbytes):
    f.seek(offset)
    out = bytearray()
    while len(out) < nbytes:
        b = f.read(min(nbytes - len(out), chunk()))
        if not b:
            raise IOError(f"short read at {offset}+{len(out)}")
        out += b
    return bytes(out)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(manifest, model):
    m = json.loads(pathlib.Path(manifest).read_text())
    size = os.path.getsize(model)
    for t in m:
        if t["offset"] < 0 or t["offset"] + t["nbytes"] > size:
            raise SystemExit(f"{t['name']}: range {t['offset']}+{t['nbytes']} "
                             f"is outside {model} ({size} bytes)")
    return m


def backup(model, manifest, out, verify):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob = out / "tensors.bin"
    if blob.exists():
        # A finished backup always has backup.json. A blob without one is an
        # interrupted attempt (power loss, kill) and carries no guarantee, so it
        # must not wedge the next run into refusing forever.
        if (out / "backup.json").exists():
            raise SystemExit(f"{blob} exists; refusing to overwrite a rollback")
        print(f"removing incomplete backup attempt: {blob}")
        blob.unlink()
    m = load(manifest, model)
    total = sum(t["nbytes"] for t in m)
    free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
    if free < total + (2 << 30):
        raise SystemExit(f"{free / 2**30:.1f} GiB free, need {total / 2**30:.1f} GiB + 2 GiB reserve")
    recs, pos = [], 0
    with open(model, "rb") as f, open(blob, "xb") as o:
        for i, t in enumerate(m, 1):
            data = read_range(f, t["offset"], t["nbytes"])
            o.write(data)
            recs.append({**t, "blob_offset": pos, "sha256": sha(data)})
            pos += len(data)
            if i % 20 == 0 or i == len(m):
                print(f"[{i}/{len(m)}] {pos / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
        o.flush()
        os.fsync(o.fileno())
    man = out / "backup.json"
    man.write_text(json.dumps({"schema": 1, "model": str(pathlib.Path(model).resolve()),
                               "blob": str(blob.resolve()), "total_bytes": total,
                               "tensors": recs}, indent=1) + "\n")
    print(f"backup: {total / 2**30:.3f} GiB, {len(recs)} tensors -> {blob}")
    if verify:
        check_blob(blob, recs)
        print("backup verified against its own hashes; comparing to the model...")
        bad = compare_to_model(model, blob, recs)
        if bad:
            raise SystemExit(f"backup differs from the model at {bad[:5]}")
        print("backup is byte-identical to the model for every covered range")
    return man


def check_blob(blob, recs):
    with open(blob, "rb") as b:
        for t in recs:
            got = sha(read_range(b, t["blob_offset"], t["nbytes"]))
            if got != t["sha256"]:
                raise SystemExit(f"blob corrupt at {t['name']}")


def compare_to_model(model, blob, recs):
    bad = []
    with open(model, "rb") as f, open(blob, "rb") as b:
        for t in recs:
            if read_range(f, t["offset"], t["nbytes"]) != read_range(b, t["blob_offset"], t["nbytes"]):
                bad.append(t["name"])
    return bad


def restore(model, manifest, frm, verify):
    frm = pathlib.Path(frm)
    man = json.loads((frm / "backup.json").read_text())
    blob = pathlib.Path(man["blob"])
    if not blob.exists():
        raise SystemExit(f"missing blob {blob}")
    m = load(manifest, model)
    by_name = {t["name"]: t for t in man["tensors"]}
    missing = [t["name"] for t in m if t["name"] not in by_name]
    if missing:
        raise SystemExit(f"backup lacks {len(missing)} tensors, e.g. {missing[:5]}")
    check_blob(blob, man["tensors"])
    with open(model, "r+b", buffering=0) as f, open(blob, "rb") as b:
        for i, t in enumerate(m, 1):
            src = by_name[t["name"]]
            if (src["offset"], src["nbytes"]) != (t["offset"], t["nbytes"]):
                raise SystemExit(f"{t['name']}: backup offsets differ from the manifest")
            f.seek(t["offset"])
            f.write(read_range(b, src["blob_offset"], src["nbytes"]))
            if i % 40 == 0 or i == len(m):
                print(f"[{i}/{len(m)}] restored", flush=True)
        f.flush()
        os.fsync(f.fileno())
    print(f"restored {len(m)} tensors into {model}")
    if verify:
        bad = compare_to_model(model, blob, [by_name[t["name"]] for t in m])
        if bad:
            raise SystemExit(f"RESTORE FAILED for {bad[:5]}")
        print("restore verified: every restored range matches the backup")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", help="backup directory")
    ap.add_argument("--from", dest="frm", help="backup directory to restore from")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if bool(a.out) == bool(a.frm):
        ap.error("give exactly one of --out (backup) or --from (restore)")
    if a.out:
        backup(a.model, a.manifest, a.out, a.verify)
    else:
        restore(a.model, a.manifest, a.frm, a.verify)


if __name__ == "__main__":
    main()
