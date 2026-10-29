#!/usr/bin/env python3
"""Apply the actual abliteration LoRA to an ordinary Q8_0 GGUF, in place.

The source adapter stores W_new = W_base + scale * B @ A.  Rank 1 tensors are
recomputed against the dequantized Q8 trunk (the documented re-export); rank-k
tensors use their actual B @ A, not a rank-1 approximation.  The Q8 file is
both the calibration input and the source for the later quantize pass, so the
edit necessarily precedes imatrix, dumps, screening and allocation.

ONE FILE OR A SHARD SET
  --model FILE        edit a single GGUF (V1 path, unchanged).
  --model-dir DIR     edit a sharded GGUF set (e.g. Swift 1.5 Q8_0, 5 shards).
                      Every tensor is read from, and written back to, the shard
                      that owns it.  The tool refuses to run unless split.count
                      matches the number of *.gguf files found, every shard
                      agrees on split.tensors.count, and the loaded tensor
                      count equals it.  That is what stops it from silently
                      editing shard 1 only and shipping an unabliterated model.

Use --prepare FIRST to record a rollback.  There are three honest kinds:

  --range-backup DIR   a tensor_backup.py range backup of exactly the tensors
                       this edit rewrites.  Complete for the edit, ~1/4 the
                       bytes of a whole-file copy, and verified against the
                       model at prepare time.  This is the supported path.
                       For a shard set the directory holds one backup per shard
                       (shard-NN/), created with --make-range-backup.
  --pristine FILE      an independent full copy on another file (single file).
  --backup FILE        a hard link (single file only).

The old --backup hard link is not a rollback for an in-place edit: it shares
the inode, so writing at data_offset changes both names.  It is kept only for
a model that is a separate file from the one being edited.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V41 = HERE.parent / "v41"
sys.path.insert(0, str(V41))
import v41_formats as F  # noqa: E402

sys.path.insert(0, "/home/mal/AI/llama.cpp-master-rebase/gguf-py")
import gguf  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def pairs(adapter):
    out = {}
    for t in gguf.GGUFReader(adapter).tensors:
        base, kind = t.name.rsplit(".lora_", 1)
        # gguf-py reverses the GGUF dimensions for numpy. Dense factors are
        # [rank,out] / [in,rank]; expert factors are [expert,out,rank] /
        # [expert,rank,in]. Normalize only the dense pair to out-major form.
        value = np.asarray(t.data, dtype=np.float32)
        out.setdefault(base, {})[kind] = value.T if value.ndim == 2 else value
    return out


def q8_rows(t, first=0, count=None):
    """Q8_0 rows in numpy order; no transposition of the GGUF ne0 axis."""
    if t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        raise ValueError(f"{t.name}: {t.tensor_type.name}, expected Q8_0")
    n_in, nb = int(t.shape[0]), int(t.shape[0]) // 32
    assert n_in % 32 == 0
    count = int(t.data.size // t.data.shape[-1]) - first if count is None else count
    raw = np.asarray(t.data).reshape(-1, nb, 34)[first:first + count]
    d = raw[:, :, :2].copy().reshape(-1).view(np.float16).astype(np.float32)
    d = d.reshape(count, nb, 1)
    q = raw[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(tuple(t.data.shape[:-1]) + (n_in,))


def apply_pair(W, B, A, scale, recompute_rank1):
    """Apply one dense matrix or a chunk of expert matrices."""
    W = np.asarray(W)
    original_shape = W.shape
    dense = W.ndim == 2
    if dense:
        W = W[None, ...]
    if W.ndim == 4 and W.shape[0] == 1:
        W = W[0]
        original_shape = W.shape
    if W.ndim != 3:
        raise ValueError(f"unsupported weight shape {W.shape}")
    E, out, inn = W.shape
    if B.ndim == 2:
        B, A = B.T, A.T
    if B.ndim == 2:
        B, A = B[None, ...], A[None, ...]
    if B.shape[0] == 1 and E > 1:
        B = np.broadcast_to(B, (E,) + B.shape[1:])
        A = np.broadcast_to(A, (E,) + A.shape[1:])
    if B.ndim != 3 or A.ndim != 3:
        raise ValueError(f"unsupported factor shapes B={B.shape} A={A.shape}")
    if B.shape[0] != E or B.shape[1] != out or A.shape[0] != E or A.shape[2] != inn or B.shape[2] != A.shape[1]:
        raise ValueError(f"edit shape mismatch W={W.shape} B={B.shape} A={A.shape}")
    rank = A.shape[1]
    if rank == 1 and recompute_rank1:
        norms = np.linalg.norm(B[:, :, 0], axis=1)
        if np.any(norms == 0):
            raise ValueError("zero B direction")
        u = B[:, :, 0] / norms[:, None]
        A = -(scale / norms)[:, None, None] * np.einsum("eo,eoi->ei", u, W)[:, None, :]
    else:
        A = A * scale
    return (W + np.einsum("eor,eri->eoi", B, A)).reshape(original_shape).astype(np.float32)


def encode_q8(W):
    if W.ndim == 2:
        return F.encode_q8_0(W)
    return np.stack([F.encode_q8_0(row) for row in W])


def decode_q8(raw, n_in, shape):
    return np.stack([F.decode(F.Q8_0, row, n_in)
                     for row in raw.reshape(-1, raw.shape[-1])]).reshape(shape)


def edit_survival(base, edited, encoded):
    """Return alpha_eff and signal; Q(W)-W noise is NOT counted as edit signal."""
    intended = edited - base
    realized = encoded - base
    den = float(np.sum(intended.astype(np.float64) ** 2))
    num = float(np.sum(realized.astype(np.float64) * intended.astype(np.float64)))
    alpha = num / den if den else 1.0
    signal = 100 * num * num / (den * float(np.sum(realized.astype(np.float64) ** 2))) if den and np.any(realized) else 0.0
    rel = float(np.linalg.norm(encoded - edited) / (np.linalg.norm(base) or 1.0))
    return {"alpha_eff": alpha, "signal_pct": signal, "requant_error_rel": rel}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_range(f, offset, nbytes):
    f.seek(offset)
    out = bytearray()
    while len(out) < nbytes:
        b = f.read(min(nbytes - len(out), 16 << 20))
        if not b:
            raise IOError(f"short read at {offset}+{len(out)}")
        out += b
    return bytes(out)


def _split_field(reader, key):
    f = reader.fields.get(key)
    return None if f is None else int(f.contents())


class ModelSet:
    """One GGUF file, or a set of shards that together form one model.

    `tensors` maps every tensor name in the model to its ReaderTensor, and
    `owner` maps that same name to the shard path that physically stores it.
    All reads and writes go through `owner`, so a sharded model is edited in
    place with no merge step (a merge of 188 GB does not fit on this box).
    """

    def __init__(self, paths):
        self.paths = [pathlib.Path(p) for p in paths]
        self.readers = []
        self.tensors = {}
        self.owner = {}
        counts, totals, NOS = {}, {}, {}
        for p in self.paths:
            r = gguf.GGUFReader(str(p))
            self.readers.append((p, r))
            counts[p] = _split_field(r, "split.count")
            totals[p] = _split_field(r, "split.tensors.count")
            NOS[p] = _split_field(r, "split.no")
            for t in r.tensors:
                if t.name in self.tensors:
                    raise SystemExit(f"duplicate tensor {t.name} in {p}")
                self.tensors[t.name] = t
                self.owner[t.name] = p
        declared = next((c for c in counts.values() if c is not None), None)
        if declared is not None:
            if declared != len(self.paths):
                raise SystemExit(
                    f"split.count={declared} but {len(self.paths)} shard file(s) resolved; "
                    f"incomplete download or extra files in the directory")
            if len(set(totals.values())) != 1:
                raise SystemExit(f"shards disagree on split.tensors.count: {totals}")
            nos = sorted(x for x in NOS.values() if x is not None)
            if nos != list(range(declared)):
                raise SystemExit(f"split.no values are {nos}, expected 0..{declared - 1}")
            total = next(iter(totals.values()))
            if len(self.tensors) != total:
                raise SystemExit(f"loaded {len(self.tensors)} tensors, split.tensors.count={total}")
        elif len(self.paths) > 1:
            raise SystemExit("multiple shards but no split.count metadata; refusing")

    @property
    def sharded(self):
        return len(self.paths) > 1

    def path_of(self, name):
        return self.owner[name]

    def shard_counts(self):
        out = {}
        for p in self.paths:
            out[str(p)] = sum(1 for n in self.tensors if self.owner[n] == p)
        return out


def load_model(model=None, model_dir=None):
    if bool(model) == bool(model_dir):
        raise SystemExit("give exactly one of --model or --model-dir")
    if model:
        return ModelSet([model])
    d = pathlib.Path(model_dir)
    if not d.is_dir():
        raise SystemExit(f"--model-dir is not a directory: {d}")
    paths = sorted(p for p in d.glob("*.gguf") if not p.name.endswith(".incomplete"))
    if not paths:
        raise SystemExit(f"no *.gguf shards in {d}")
    return ModelSet(paths)


def load_range_backup(d, model, names):
    """Validate a tensor_backup.py backup against the model, return (blob, index).

    Every tensor in `names` must be covered, and the bytes it covers must still
    be the model's current bytes: that is what makes the backup a rollback for
    an edit that has not happened yet.  Single-file only; see load_shard_backups.
    """
    d = pathlib.Path(d)
    man_path = d / "backup.json"
    if not man_path.is_file():
        raise SystemExit(f"{d}: not a finished backup (no backup.json)")
    man = json.loads(man_path.read_text())
    if pathlib.Path(man["model"]).resolve() != pathlib.Path(model).resolve():
        raise SystemExit(f"backup is for {man['model']}, not {model}")
    blob = pathlib.Path(man["blob"])
    if not blob.is_file():
        raise SystemExit(f"backup blob missing: {blob}")
    by = {t["name"]: t for t in man["tensors"]}
    missing = [n for n in names if n not in by]
    if missing:
        raise SystemExit(f"backup does not cover {len(missing)} target(s), e.g. {missing[:5]}")
    size = os.path.getsize(model)
    for n in names:
        r = by[n]
        if r["offset"] + r["nbytes"] > size:
            raise SystemExit(f"{n}: backup range extends past {model}")
    return blob, {n: by[n] for n in names}


def load_shard_backups(root, m, names):
    """Validate a per-shard range backup (root/shard-NN/backup.json).

    Each shard's backup must cover exactly the targets that live in that shard,
    and every covered range must still equal the shard's current bytes.
    """
    root = pathlib.Path(root)
    by = {}
    for idx, p in enumerate(m.paths):
        shard_names = [n for n in names if m.owner[n] == p]
        if not shard_names:
            continue
        d = root / f"shard-{idx:02d}"
        man_path = d / "backup.json"
        if not man_path.is_file():
            raise SystemExit(f"{d}: no finished backup (backup.json missing)")
        man = json.loads(man_path.read_text())
        if pathlib.Path(man["model"]).resolve() != pathlib.Path(p).resolve():
            raise SystemExit(f"{d}: backup is for {man['model']}, not {p}")
        blob = pathlib.Path(man["blob"])
        if not blob.is_file():
            raise SystemExit(f"{d}: blob missing {blob}")
        recs = {t["name"]: t for t in man["tensors"]}
        have, need = set(recs), set(shard_names)
        if have != need:
            missing = sorted(need - have)
            extra = sorted(have - need)
            raise SystemExit(f"{d}: backup covers {len(have)} tensors, expected {len(need)} "
                             f"(missing {missing[:3]}, extra {extra[:3]})")
        size = pathlib.Path(p).stat().st_size
        with open(blob, "rb") as b, open(p, "rb") as f:
            for n in sorted(need):
                r = recs[n]
                if r["offset"] + r["nbytes"] > size:
                    raise SystemExit(f"{n}: backup range extends past {p}")
                if sha_bytes(read_range(b, r["blob_offset"], r["nbytes"])) != r["sha256"]:
                    raise SystemExit(f"{n}: backup blob is corrupt")
                if read_range(f, r["offset"], r["nbytes"]) != read_range(b, r["blob_offset"], r["nbytes"]):
                    raise SystemExit(f"{n}: model already differs from the backup; refuse to edit")
        for n in need:
            by[n] = {**recs[n], "blob": str(blob)}
    missing = [n for n in names if n not in by]
    if missing:
        raise SystemExit(f"backup does not cover {len(missing)} target(s), e.g. {missing[:5]}")
    return by


def load_range_index(root, m, names):
    if m.sharded:
        return load_shard_backups(root, m, names)
    blob, idx = load_range_backup(root, m.paths[0], names)
    return {n: {**r, "blob": str(blob)} for n, r in idx.items()}


def make_per_shard_backups(m, names, out_root):
    """Create a verified range backup per shard, reusing tensor_backup.py."""
    sys.path.insert(0, str(HERE))
    import tensor_backup as TB  # noqa: E402

    out_root = pathlib.Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    made = []
    for idx, p in enumerate(m.paths):
        shard_names = sorted(n for n in names if m.owner[n] == p)
        if not shard_names:
            print(f"shard {idx}: no targets, skipping backup")
            continue
        manifest = [{"name": n,
                     "offset": int(m.tensors[n].data_offset),
                     "nbytes": int(m.tensors[n].n_bytes),
                     "type": m.tensors[n].tensor_type.name} for n in shard_names]
        mf = out_root / f"shard-{idx:02d}.manifest.json"
        mf.write_text(json.dumps(manifest, indent=1) + "\n")
        out = out_root / f"shard-{idx:02d}"
        TB.backup(str(p), str(mf), str(out), True)
        made.append((p, len(shard_names), out))
        print(f"shard {idx}: {len(shard_names)} tensors backed up -> {out}")
    total = sum(x[1] for x in made)
    if total != len(names):
        raise SystemExit(f"backed up {total} tensors, expected {len(names)}")
    print(f"backup complete: {total} tensors across {len(made)} shard(s)")


class _BackupTensor:
    """A tensor stand-in whose payload comes from a range backup.

    It carries the live tensor's shape/type so `q8_rows` decodes it exactly as
    it decodes the model, without re-deriving gguf-py's byte layout here.
    """

    def __init__(self, t, raw):
        if len(raw) != int(t.n_bytes):
            raise ValueError(f"{t.name}: backup has {len(raw)} bytes, tensor needs {t.n_bytes}")
        self.name = t.name
        self.shape = t.shape
        self.tensor_type = t.tensor_type
        self.data_offset = t.data_offset
        self.data = np.frombuffer(raw, dtype=np.uint8).reshape(t.data.shape)


def check_targets_unchanged(m, state, names):
    """Refuse an edit when any target no longer matches its recorded backup.

    A second application of the same adapter is not idempotent, so this runs
    before the first write whenever there is no journal to resume.
    """
    by_shard = {}
    for n in names:
        by_shard.setdefault(m.owner[n], []).append(n)
    for p, ns in by_shard.items():
        with open(p, "rb") as f:
            for n in sorted(ns):
                t = m.tensors[n]
                want = state["tensor_sha256"][n]
                if sha_bytes(read_range(f, int(t.data_offset), int(t.n_bytes))) != want:
                    raise SystemExit(f"{n} differs from the backup; refusing a double edit")


def _backup_rec_for(state, m, name, cache):
    """Return (blob_path, record) for one target from a prepared range backup."""
    root = pathlib.Path(state["backup"])
    if m.sharded:
        key = m.owner[name]
        d = root / f"shard-{m.paths.index(key):02d}"
    else:
        key = m.paths[0]
        d = root
    if key not in cache:
        man = json.loads((d / "backup.json").read_text())
        cache[key] = (pathlib.Path(man["blob"]), {t["name"]: t for t in man["tensors"]})
    blob, recs = cache[key]
    return blob, recs[name]


def prepare(m, backup, report, pristine=None, pristine_sha=None, range_backup=None, names=None):
    """Record a rollback for an in-place edit.

    --backup hard-links the model on the same filesystem. That is only a
    rollback if the edit target is a SEPARATE file: a hard link shares the
    inode, so an in-place write at data_offset changes both names. `--pristine`
    instead points at an existing independent full copy and proves it is one.
    Both are single-file only; a shard set uses --range-backup.
    """
    if range_backup:
        if not names:
            raise SystemExit("a range backup needs the target tensor list")
        if m.sharded:
            by = load_shard_backups(range_backup, m, names)
            total = sum(r["nbytes"] for r in by.values())
            info = {"schema": 1, "state": "prepared",
                    "shards": [str(p.resolve()) for p in m.paths],
                    "backup": str(pathlib.Path(range_backup).resolve()),
                    "rollback_kind": "shard-range-backup",
                    "source_size": {str(p): p.stat().st_size for p in m.paths},
                    "tensor_sha256": {n: r["sha256"] for n, r in by.items()},
                    "created": time.strftime("%F %T")}
            pathlib.Path(report).write_text(json.dumps(info, indent=1) + "\n")
            print(f"rollback verified (shard-range-backup): {info['backup']}\n"
                  f"{len(by)} tensors, {total / 2**30:.3f} GiB across {len(m.paths)} shards")
            return
        model = m.paths[0]
        blob, idx = load_range_backup(range_backup, model, names)
        print(f"verifying {len(names)} covered ranges against {model} "
              f"({sum(r['nbytes'] for r in idx.values()) / 2**30:.3f} GiB)...", flush=True)
        with open(model, "rb") as f, open(blob, "rb") as b:
            for n, r in sorted(idx.items()):
                if sha_bytes(read_range(b, r["blob_offset"], r["nbytes"])) != r["sha256"]:
                    raise SystemExit(f"{n}: backup blob is corrupt")
                if read_range(f, r["offset"], r["nbytes"]) != read_range(b, r["blob_offset"], r["nbytes"]):
                    raise SystemExit(f"{n}: model already differs from the backup; refuse to edit")
        info = {"schema": 1, "state": "prepared",
                "shards": [str(model.resolve())],
                "backup": str(pathlib.Path(range_backup).resolve()),
                "rollback_kind": "range-backup",
                "source_size": model.stat().st_size,
                "tensor_sha256": {n: r["sha256"] for n, r in idx.items()},
                "created": time.strftime("%F %T")}
        pathlib.Path(report).write_text(json.dumps(info, indent=1) + "\n")
        print(f"rollback verified (range-backup): {info['backup']}\n"
              f"{len(idx)} tensors, {sum(r['nbytes'] for r in idx.values()) / 2**30:.3f} GiB covered")
        return
    if m.sharded:
        raise SystemExit("--backup/--pristine are single-file only; use --range-backup for shards")
    model = m.paths[0]
    if pristine:
        pristine = pathlib.Path(pristine)
        if not pristine.is_file():
            raise SystemExit(f"pristine copy missing: {pristine}")
        if pristine.resolve() == model.resolve():
            raise SystemExit("--pristine is the edit target itself")
        a, b = model.stat(), pristine.stat()
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            raise SystemExit("--pristine is a hard link to the edit target; that is not a copy")
        if a.st_size != b.st_size:
            raise SystemExit(f"pristine size {b.st_size} != model size {a.st_size}")
        sha = pristine_sha or digest(pristine)
        got = digest(model)
        if got != sha:
            raise SystemExit(f"model SHA-256 {got} != pristine SHA-256 {sha}")
        info = {"schema": 1, "state": "prepared", "model": str(model),
                "shards": [str(model.resolve())],
                "backup": str(pristine), "rollback_kind": "independent-copy",
                "source_size": a.st_size, "source_sha256": sha,
                "created": time.strftime("%F %T")}
    else:
        backup = pathlib.Path(backup)
        if backup.exists():
            raise SystemExit(f"refusing to overwrite existing rollback {backup}")
        if model.stat().st_dev != backup.parent.stat().st_dev:
            raise SystemExit("rollback must be on the SAME filesystem for a hard link")
        os.link(model, backup)
        if os.stat(model).st_ino != os.stat(backup).st_ino:
            raise SystemExit("rollback is not a hard link")
        info = {"schema": 1, "state": "prepared", "model": str(model),
                "shards": [str(model.resolve())],
                "backup": str(backup), "rollback_kind": "hardlink-same-inode",
                "source_size": model.stat().st_size, "source_sha256": digest(model),
                "created": time.strftime("%F %T")}
    pathlib.Path(report).write_text(json.dumps(info, indent=1) + "\n")
    print(f"rollback verified ({info['rollback_kind']}): {info['backup']}\n"
          f"source SHA-256: {info['source_sha256']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a single GGUF file")
    ap.add_argument("--model-dir", help="a directory of GGUF shards (split.count checked)")
    ap.add_argument("--adapter", default="/media/mal/NVME1TB/Models/Qwen3.8-Flash-Next-Abliterated-Adapter/Q5_K_M/adapter.gguf")
    ap.add_argument("--backup", help="hard link on the same filesystem (only safe if --model is a separate file)")
    ap.add_argument("--pristine", help="independent full copy used as the rollback for an in-place edit")
    ap.add_argument("--pristine-sha256", help="skip re-hashing the pristine copy; must match --model")
    ap.add_argument("--range-backup", help="tensor_backup.py directory covering exactly this edit's tensors")
    ap.add_argument("--make-range-backup", help="create a verified per-shard range backup in this directory")
    ap.add_argument("--report", default="/home/mal/.quetza-data/qwen38-q8-edit.json")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--layers", default="", help="comma list; default all adapter layers (testing only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.model and not a.model_dir:
        ap.error("one of --model or --model-dir is required")
    m = load_model(a.model, a.model_dir)
    tensors = m.tensors
    edit = pairs(a.adapter)
    layers = {int(x) for x in a.layers.split(",") if x.strip()} if a.layers else None
    selected = {n: ab for n, ab in edit.items()
                if layers is None or int(n.split(".")[1]) in layers}
    missing = sorted(set(selected) - set(tensors))
    if missing:
        raise SystemExit(f"model lacks adapter targets: {missing[:5]}")
    for n, t in tensors.items():
        if n not in selected:
            continue
        if t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            raise SystemExit(f"{n}: expected Q8_0, got {t.tensor_type.name}")
        if not np.allclose(t.data_offset % 4, 0):
            raise SystemExit(f"{n}: unaligned data offset")
    per_shard = {}
    for n in selected:
        p = str(m.owner[n])
        per_shard[p] = per_shard.get(p, 0) + 1
    print(f"targets={len(selected)} shards_used={len(m.paths)}")
    for p in m.paths:
        print(f"  {p}: {per_shard.get(str(p), 0)} target(s)")
    if a.make_range_backup:
        make_per_shard_backups(m, sorted(selected), a.make_range_backup)
        return
    if a.prepare:
        kinds = [k for k, v in (("--range-backup", a.range_backup), ("--pristine", a.pristine),
                                ("--backup", a.backup)) if v]
        if len(kinds) != 1:
            ap.error(f"give exactly one rollback: {', '.join(kinds) or 'none given'}")
        prepare(m, a.backup, a.report, a.pristine, a.pristine_sha256,
                a.range_backup, sorted(selected))
        return
    if a.pristine or a.range_backup:
        ap.error("--pristine/--range-backup are only used with --prepare")
    if not a.dry_run:
        report = pathlib.Path(a.report)
        if not report.is_file():
            raise SystemExit("no preparation report; run --prepare first")
        state = json.loads(report.read_text())
        want = [str(p.resolve()) for p in m.paths]
        got = state.get("shards")
        if got is None:
            got = [state["model"]] if state.get("model") else []
        if [str(pathlib.Path(x).resolve()) for x in got] != want:
            raise SystemExit("report belongs to another model")
        kind = state.get("rollback_kind")
        if kind in ("range-backup", "shard-range-backup"):
            if not pathlib.Path(state["backup"]).is_dir():
                raise SystemExit("range backup is missing")
        elif not pathlib.Path(state["backup"]).is_file():
            raise SystemExit("pristine rollback is missing")
        journal = pathlib.Path(str(report) + ".journal.json")
        completed = {}
        rec_cache = {}
        if journal.exists():
            previous = json.loads(journal.read_text())
            if previous.get("rollback") != state["backup"]:
                raise SystemExit("edit journal is for a different rollback")
            if previous.get("adapter") != a.adapter or previous.get("scale") != a.scale:
                raise SystemExit("edit journal is for a different adapter or scale")
            completed = {r["tensor"]: r for r in previous.get("targets", [])}
        elif kind in ("range-backup", "shard-range-backup"):
            print("checking every target still matches its backup...", flush=True)
            check_targets_unchanged(m, state, selected)
        elif digest(m.paths[0]) != state["source_sha256"]:
            raise SystemExit("model differs from the verified pristine source; refusing a double edit")
    else:
        completed, kind, rec_cache = {}, None, {}
    rows = []
    for i, (name, ab) in enumerate(sorted(selected.items()), 1):
        t = tensors[name]
        B, A = ab["b"], ab["a"]
        rank = A.shape[1] if A.ndim == 3 else A.shape[0]
        if name in completed:
            old = completed[name]
            if old.get("rank") != rank:
                raise SystemExit(f"{name}: adapter rank changed since the journal")
            W = q8_rows(t)
            # Recompute the exact output from the pristine rollback to validate
            # the saved payload, not merely a same-sized tensor region.
            if kind in ("range-backup", "shard-range-backup"):
                blob, rec = _backup_rec_for(state, m, name, rec_cache)
                with open(blob, "rb") as bf:
                    base = q8_rows(_BackupTensor(t, read_range(bf, rec["blob_offset"], rec["nbytes"])))
            else:
                pristine = gguf.GGUFReader(state["backup"])
                pt = next(x for x in pristine.tensors if x.name == name)
                base = q8_rows(pt)
                del pristine, pt
            expected = apply_pair(base, B, A, a.scale, recompute_rank1=(rank == 1))
            raw = encode_q8(expected)
            del base, expected
            with open(m.owner[name], "rb") as f:
                f.seek(int(t.data_offset))
                if f.read(raw.nbytes) != raw.tobytes():
                    raise SystemExit(f"{name}: journal payload differs; restore the pristine backup before retrying")
            rows.append(old)
            print(f"[{i}/{len(selected)}] {name}: verified in journal", flush=True)
            continue
        W = q8_rows(t)
        before = W.copy()
        edited = apply_pair(W, B, A, a.scale, recompute_rank1=(rank == 1))
        if not np.all(np.isfinite(edited)):
            raise ValueError(f"{name}: nonfinite edited weights")
        raw = encode_q8(edited)
        got = decode_q8(raw, int(t.shape[0]), edited.shape)
        metric = edit_survival(before, edited, got)
        rec = {"tensor": name, "rank": rank, "n_rows": int(W.shape[0]),
               "shard": str(m.owner[name]),
               "edit_rel": float(np.linalg.norm(edited - before) / (np.linalg.norm(before) or 1.0)),
               **metric}
        rows.append(rec)
        print(f"[{i}/{len(selected)}] {name}: rank={rank} shard={m.paths.index(m.owner[name])} "
              f"edit={rec['edit_rel']:.4%} alpha_eff={metric['alpha_eff']:.5f} signal={metric['signal_pct']:.2f}%",
              flush=True)
        if not a.dry_run:
            with open(m.owner[name], "r+b", buffering=0) as f:
                f.seek(int(t.data_offset))
                f.write(raw.tobytes())
            # Byte-level payload readback; do not declare an in-place write complete from write() alone.
            with open(m.owner[name], "rb") as f:
                f.seek(int(t.data_offset))
                if f.read(raw.nbytes) != raw.tobytes():
                    raise IOError(f"{name}: payload readback differs")
            journal_state = {"rollback": state["backup"],
                             "adapter": a.adapter, "scale": a.scale, "targets": rows}
            tmp = pathlib.Path(str(report) + ".journal.json.tmp")
            tmp.write_text(json.dumps(journal_state, indent=1) + "\n")
            os.replace(tmp, pathlib.Path(str(report) + ".journal.json"))
    if len(rows) != len(selected):
        raise SystemExit(f"applied {len(rows)} of {len(selected)} targets; refusing to report success")
    result = {"schema": 1, "state": "dry-run" if a.dry_run else "edited",
              "shards": [str(p.resolve()) for p in m.paths],
              "shards_used": len(m.paths),
              "adapter": a.adapter, "adapter_sha256": digest(a.adapter),
              "scale": a.scale, "layers": sorted(layers) if layers is not None else "all",
              "targets_per_shard": per_shard,
              "targets": rows}
    if m.sharded:
        result["model"] = None
        if not a.dry_run:
            result["shards_sha256"] = {str(p): digest(p) for p in m.paths}
    else:
        result["model"] = str(m.paths[0])
        if not a.dry_run:
            result["edited_sha256"] = digest(m.paths[0])
    pathlib.Path(a.report).write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {a.report}: {len(rows)} target tensors; state={result['state']}")


if __name__ == "__main__":
    main()
